import torch as th
import torch.nn as nn
import numpy as np
from gym import spaces
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from typing import Union, Type, Optional, Any
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        layers = []

        # input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        # layers.append(nn.LeakyReLU())
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))

        # hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(0.1))

        # output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)

class DQN_PCT_QNetwork(QNetwork):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_dim: int,
    ):
        super().__init__(observation_space, action_space, nn.Identity(), features_dim=features_dim)
        #MARK: CHANGE HERE
        num_layers = 3
        self.q_net = MLP(features_dim, action_space.n, num_layers) 

class DQN_PCT_Policy(DQNPolicy):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_q_net(self):
        features_dim = int(np.prod(self.observation_space.shape))
        return DQN_PCT_QNetwork(
            self.observation_space,
            self.action_space,
            features_dim=features_dim,
        )

class DQN_PCT(DQN):
    def __init__(
        self,
        policy: Union[str, Type[DQN_PCT_Policy]] = DQN_PCT_Policy,
        env: Union[VecEnv, str] = None,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Any] = None,
        replay_buffer_kwargs: Optional[dict] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )


