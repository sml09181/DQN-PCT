from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import DQN

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from src import config
from src.meta.env_stock_trading.env_stocktrading import *
from src.meta.preprocessor.preprocessors import data_split
from src.agents.stablebaselines3.custom_models import * 

# MODELS = {"a2c": A2C, "ppo": PPO, 'dqn': DQN}
MODELS = {"a2c": A2C, "ppo": PPO, 'dqn': DQN_PCT}
MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])

        except BaseException as error:
            try:
                self.logger.record(key="train/reward", value=self.locals["reward"][0])

            except BaseException as inner_error:
                # Handle the case where neither "rewards" nor "reward" is found
                self.logger.record(key="train/reward", value=None)
                # Print the original error and the inner error for debugging
                print("Original Error:", error)
                print("Inner Error:", inner_error)
        return True


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
        tensorboard_log=None,
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        
        # print(policy)
        # return MODELS[model_name](
        #     policy=policy,
        #     env=self.env,
        #     tensorboard_log=tensorboard_log,
        #     verbose=verbose,
        #     policy_kwargs=policy_kwargs,
        #     seed=seed,
        #     **model_kwargs,
        # )
        return MODELS[model_name](
            policy=DQN_PCT_Policy,  
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )


    def load_model(self, model_path, model_name="ppo", verbose=1):
      if model_name not in MODELS:
          raise ValueError(f"Model '{model_name}' not found in MODELS.")
      try:
          # Load the pre-trained model
          model = MODELS[model_name].load(model_path, env=self.env, verbose=verbose)
          print(f"Successfully loaded {model_name} model from {model_path}.")
          return model
      except Exception as error:
          raise ValueError(f"Failed to load the model. Error: {str(error)}") from error


    @staticmethod
    def train_model(
        model, tb_log_name, total_timesteps=5000
    ):  # this function is static method, so it can be called without creating an instance of the class
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        """make a prediction and get results"""
        test_env, test_obs = environment.get_sb_env()
        account_memory = None  # This help avoid unnecessary list creation
        actions_memory = None  # optimize memory consumption
        # state_memory=[] #add memory pool to store states

        test_env.reset()
        max_steps = len(environment.df.index.unique()) - 1

        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            # account_memory = test_env.env_method(method_name="save_asset_memory")
            # actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, rewards, dones, info = test_env.step(action)

            if (
                i == max_steps - 1
            ):  # more descriptive condition for early termination to clarify the logic
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            # add current state to state memory
            # state_memory=test_env.env_method(method_name="save_state_memory")

            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]

    @staticmethod
    def DRL_prediction_load_from_file(model_name, environment, cwd, deterministic=True):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except BaseException as error:
            raise ValueError(f"Failed to load agent. Error: {str(error)}") from error

        # test on the testing env
        state = environment.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [environment.envs[0].initial_amount]
        done = False

        
        while not done:
            action = model.predict(state, deterministic=deterministic)[0]
            state, reward, done, _ = environment.step(action)
            if done: break
            
            total_asset = (
                environment.envs[0].asset_memory[environment.envs[0].day]
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / environment.envs[0].initial_amount
            episode_returns.append(episode_return)

        print("Test Finished!")

        # Calculate performance metrics
        df_total_value = pd.DataFrame(episode_total_assets, columns=["account_value"])
        # Calculate daily returns
        df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
        # Calculate Sharpe ratio
        sharpe_ratio = None
        if len(df_total_value) > 1 and df_total_value["daily_return"].std() != 0:
            sharpe_ratio = (
            (252**0.5)
            * df_total_value["daily_return"].mean()
            / df_total_value["daily_return"].std()
            )
        print("sharpe", sharpe_ratio)
        print("end total asset", episode_total_assets[-1])
        return episode_total_assets, sharpe_ratio