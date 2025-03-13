from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class ContEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = True,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        logger=None,
        pct1=None,
        pct2=None,
        pct3=None
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.iloc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.logger = logger
        self.pct1 = pct1
        self.pct2 = pct2
        self.pct3 = pct3
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('.',[CSVOutputFormat])
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # self.logger.info('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"./plots/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # self.logger.info(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                self.logger.info(f"day: {self.day}, episode: {self.episode}")
                self.logger.info(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                self.logger.info(f"end_total_asset: {end_total_asset:0.2f}")
                self.logger.info(f"total_reward: {tot_reward:0.2f}")
                self.logger.info(f"total_cost: {self.cost:0.2f}")
                self.logger.info(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    self.logger.info(f"Sharpe: {sharpe:0.3f}")
                self.logger.info("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "./actions/actions_{}_{}_ep{}.csv".format(
                        self.mode, self.model_name, self.episode
                    )
                )
                df_total_value.to_csv(
                    "./total_value/account_value_{}_{}_ep{}.csv".format(
                        self.mode, self.model_name, self.episode
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "./rewards/account_rewards_{}_{}_ep{}.csv".format(
                        self.mode, self.model_name, self.episode
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "./asset_memory/account_value_{}_{}_ep{}.png".format(
                        self.mode, self.model_name, self.episode
                    )
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, False, {}

        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1 # [-1,1] → [-hmax, hmax]
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares 
            # 예: hmax=100 → action=0.5 → 50주
            if self.turbulence_threshold is not None:
                # 변동성 임계치 초과 시 전량 매도
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            # reward = 총 자산 변화량
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # self.logger.info("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # self.logger.info(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # self.logger.info(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # self.logger.info(f'take sell action after : {actions[index]}')
                # self.logger.info(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # self.logger.info('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.iloc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # initiate state
        self.day = 0
        self.data = self.df.iloc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data[0].unique()[0]
        else:
            date = self.data[0]
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # self.logger.info(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # self.logger.info(len(date_list))
        # self.logger.info(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

# MARK: Disc3Env
class Disc3Env(ContEnv):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Discrete(3)  

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # self.logger.info(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                self.logger.info(f"day: {self.day}, episode: {self.episode}")
                self.logger.info(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                self.logger.info(f"end_total_asset: {end_total_asset:0.2f}")
                self.logger.info(f"total_reward: {tot_reward:0.2f}")
                self.logger.info(f"total_cost: {self.cost:0.2f}")
                self.logger.info(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    self.logger.info(f"Sharpe: {sharpe:0.3f}")
                self.logger.info("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "./actions/actions_{}_{}_ep{}.csv".format(
                        self.mode, self.model_name, self.episode
                    )
                )
                df_total_value.to_csv(
                    "./total_value/account_value_{}_{}_ep{}.csv".format(
                        self.mode, self.model_name, self.episode
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "./rewards/account_rewards_{}_{}_ep{}.csv".format(
                        self.mode, self.model_name, self.episode
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "./asset_memory/account_value_{}_{}_ep{}.png".format(
                        self.mode, self.model_name, self.episode
                    )
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, False, {}

        else:
            buy_actions = (actions == 0).astype(int)
            sell_actions = (actions == 2).astype(int)
            
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)]) 
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            # Handle turbulence threshold (force sell if above threshold)
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    sell_actions = np.ones(self.stock_dim).astype(int)
                    buy_actions = np.zeros(self.stock_dim).astype(int)

            # Execute buy actions
            for index in np.where(buy_actions)[0]:
                self._buy_stock(index, self.hmax)

            # Execute sell actions
            for index in np.where(sell_actions)[0]:
                self._sell_stock(index, self.hmax)

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.iloc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step

        return self.state, self.reward, self.terminal, False, {}

# MARK: Disc5Env
class Disc5Env(ContEnv):
    """Stock trading environment with 5 discrete actions (buy/sell percentages)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set discrete action space (0-4)
        self.action_space = spaces.Discrete(5)
        
    def _sell_stock(self, index, action_type):
        """Execute sell action based on discrete action type
        Args:
            index: Stock index to sell
            action_type: Discrete action (3: sell 30%, 4: sell 70%)
        Returns:
            Number of shares sold
        """
        def _do_sell_normal():
            current_shares = self.state[index + self.stock_dim + 1]
            sell_pct_dict = {
                3: self.pct1,
                4: self.pct2,
                5: 1, # if turbulence goes over threshold, just clear out all positions 
            }
            sell_pct = sell_pct_dict.get(action_type, 0)
            if (
                    self.state[index + 2 * self.stock_dim + 1] != True
                ):
                if sell_pct > 0 and current_shares > 0:
                    sell_num_shares = min(int(self.hmax * sell_pct), current_shares)
                    sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])
                    
                    # Update portfolio state
                    self.state[0] += sell_amount  # Add cash
                    self.state[index + self.stock_dim + 1] -= sell_num_shares  # Remove shares
                    self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                    self.trades += 1
                    return sell_num_shares
                    # perform sell action based on the sign of the action
            return 0
        
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()
        return sell_num_shares
        

    def _buy_stock(self, index, action_type):
        """Execute buy action based on discrete action type
        Args:
            index: Stock index to buy
            action_type: Discrete action (0: buy 70%, 1: buy 30%)
        Returns:
            Number of shares bought
        """
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # self.logger.info('available_amount:{}'.format(available_amount))
                
                # update balance
                buy_pct_dict = {
                    0: self.pct2,
                    1: self.pct1,
                }
                buy_pct = buy_pct_dict.get(action_type, 0)
                buy_num_shares = min(available_amount, int(self.hmax * buy_pct))
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
                return buy_num_shares
            else: return 0
                
        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares
        
    def step(self, action):
        """
        Execute one market step with discrete action
        Args:
            action: Discrete value (0-4)
                0: Buy 70% more of current holdings for all stocks
                1: Buy 30% more of current holdings for all stocks 
                2: Hold all positions
                3: Sell 30% of current holdings for all stocks
                4: Sell 70% of current holdings for all stocks
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        
        if self.terminal:
            return self._handle_terminal_state()

        # Force sell all if turbulence exceeds threshold
        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            action = 5  # Force maximum sell
        begin_total_asset = self._calculate_total_asset()

        # Process action for each stock
        # 각 주식에 모두 같은 action을 적용한다는 것에 유의 
        for index in range(self.stock_dim):
            if action in [0, 1]:  # Buy actions
                num_shares = self._buy_stock(index, action)
            elif action in [3, 4, 5]:  # Sell actions
                num_shares = self._sell_stock(index, action)
            else: num_shares = 0
        # actions = np.full(actions.size, num_shares)
        self.actions_memory.append(num_shares)

        # Update market state
        self.day += 1
        self.data = self.df.iloc[self.day, :]
        if self.turbulence_threshold is not None:
            if len(self.df.tic.unique()) == 1:
                self.turbulence = self.data[self.risk_indicator_col]
            elif len(self.df.tic.unique()) > 1:
                self.turbulence = self.data[self.risk_indicator_col].values[0]
        self.state = self._update_state()

        # Calculate rewards
        end_total_asset = self._calculate_total_asset()
        self.reward = end_total_asset - begin_total_asset
        self._update_memories(end_total_asset)
        self.reward *= self.reward_scaling
        self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step
        return self.state, self.reward, self.terminal, False, {}
        
    def _calculate_total_asset(self):
        """Calculate total portfolio value"""
        return self.state[0] + sum(
            np.array(self.state[1:(self.stock_dim+1)]) 
            * np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])
        )

    def _update_memories(self, end_total_asset):
        """Update environment memories"""
        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())
        self.rewards_memory.append(self.reward)

    def _handle_terminal_state(self):
        """Handle end-of-episode operations"""
        # self.logger.info(f"Episode: {self.episode}")
        if self.make_plots:
            self._make_plot()
        end_total_asset = self._calculate_total_asset()
        df_total_value = pd.DataFrame(self.asset_memory)
        tot_reward = (
            self.state[0]
            + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            - self.asset_memory[0]
        )  # initial_amount is only cash part of our initial asset
        df_total_value.columns = ["account_value"]
        df_total_value["date"] = self.date_memory
        df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
            1
        )
        if df_total_value["daily_return"].std() != 0:
            sharpe = (
                (252**0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            )
        df_rewards = pd.DataFrame(self.rewards_memory)
        df_rewards.columns = ["account_rewards"]
        df_rewards["date"] = self.date_memory[:-1]
        if self.episode % self.print_verbosity == 0:
            self.logger.info(f"day: {self.day}, episode: {self.episode}")
            self.logger.info(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
            self.logger.info(f"end_total_asset: {end_total_asset:0.2f}")
            self.logger.info(f"total_reward: {tot_reward:0.2f}")
            self.logger.info(f"total_cost: {self.cost:0.2f}")
            self.logger.info(f"total_trades: {self.trades}")
            if df_total_value["daily_return"].std() != 0:
                self.logger.info(f"Sharpe: {sharpe:0.3f}")
            self.logger.info("=================================")

        if (self.model_name != "") and (self.mode != ""):
            df_actions = self.save_action_memory()
            df_actions.to_csv(
                "./actions/actions_{}_{}_ep{}.csv".format(
                    self.mode, self.model_name, self.episode
                )
            )
            df_total_value.to_csv(
                "./total_value/account_value_{}_{}_ep{}.csv".format(
                    self.mode, self.model_name, self.episode
                ),
                index=False,
            )
            df_rewards.to_csv(
                "./rewards/account_rewards_{}_{}_ep{}.csv".format(
                    self.mode, self.model_name, self.episode
                ),
                index=False,
            )
            plt.plot(self.asset_memory, "r")
            plt.savefig(
                "./asset_memory/account_value_{}_{}_ep{}.png".format(
                    self.mode, self.model_name, self.episode
                )
            )
            plt.close()

        # Add outputs to logger interface
        # logger.record("environment/portfolio_value", end_total_asset)
        # logger.record("environment/total_reward", tot_reward)
        # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
        # logger.record("environment/total_cost", self.cost)
        # logger.record("environment/total_trades", self.trades)

        return self.state, self.reward, self.terminal, False, {}

# MARK: Disc7Env    
class Disc7Env(Disc5Env):
    """Stock trading environment with 7 discrete actions (percentage-based)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set discrete action space (0-6)
        self.action_space = spaces.Discrete(7)
    
    def _sell_stock(self, index, action_type):
        """Execute sell action based on discrete action type
        Args:
            index: Stock index to sell
            action_type: Discrete action (4:25%, 5:50%, 6:75%)
        Returns:
            Number of shares sold
        """
        def _do_sell_normal():
            current_shares = self.state[index + self.stock_dim + 1]
            sell_pct = {
                4: self.pct1,
                5: self.pct2,
                6: self.pct3,
                7: 1,
            }.get(action_type, 0.0)
            if (
                    self.state[index + 2 * self.stock_dim + 1] != True
                ):
                if sell_pct > 0 and current_shares > 0:
                    sell_num_shares = min(int(self.hmax * sell_pct), current_shares)
                    sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])
                    
                    # Update portfolio state
                    self.state[0] += sell_amount  # Add cash
                    self.state[index + self.stock_dim + 1] -= sell_num_shares  # Remove shares
                    self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                    self.trades += 1
                    return sell_num_shares
                    # perform sell action based on the sign of the action
            return 0
        
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()
        return sell_num_shares
        

    def _buy_stock(self, index, action_type):
        """Execute buy action based on discrete action type
        Args:
            index: Stock index to buy
            action_type: Discrete action (0:75%, 1:50%, 2:25%)
        Returns:
            Number of shares bought
        """
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # self.logger.info('available_amount:{}'.format(available_amount))
                
                # update balance
                # Map action to buy percentage
                buy_pct = {
                    0: self.pct3,
                    1: self.pct2,
                    2: self.pct1
                }.get(action_type, 0.0)
                buy_num_shares = min(available_amount, int(self.hmax * buy_pct))
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
                return buy_num_shares
            else: return 0
                
        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares
        
    def step(self, action):
        """
        Execute one market step with discrete action
        Args:
            action: Discrete value (0-6)
                0: Buy 75% more of current holdings for all stocks
                1: Buy 50% more of current holdings for all stocks 
                2: Buy 25% more of current holdings for all stocks
                3: Hold all positions
                4: Sell 25% of current holdings for all stocks
                5: Sell 50% of current holdings for all stocks
                6: Sell 75% of current holdings for all stocks
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        
        if self.terminal:
            return self._handle_terminal_state()

        # Force sell 75% if turbulence exceeds threshold
        if self.turbulence_threshold and self.turbulence >= self.turbulence_threshold:
            action = 7  # Force maximum sell
        begin_total_asset = self._calculate_total_asset()
        
        # Process action for each stock
        for index in range(self.stock_dim):
            if 0 <= action <= 2:  # Buy actions
                num_shares = self._buy_stock(index, action)
            elif 4 <= action <= 7:  # Sell actions
                num_shares = self._sell_stock(index, action)
            # Action 3: Hold (no operation)
            else: num_shares = 0
        self.actions_memory.append(num_shares)

        # Update market state
        self.day += 1
        self.data = self.df.iloc[self.day, :]
        if self.turbulence_threshold is not None:
            if len(self.df.tic.unique()) == 1:
                self.turbulence = self.data[self.risk_indicator_col]
            elif len(self.df.tic.unique()) > 1:
                self.turbulence = self.data[self.risk_indicator_col].values[0]
        self.state = self._update_state()

        # Calculate rewards
        end_total_asset = self._calculate_total_asset()
        self.reward = end_total_asset - begin_total_asset
        self._update_memories(end_total_asset)
        self.reward *= self.reward_scaling
        self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step
        return self.state, self.reward, self.terminal, False, {}

