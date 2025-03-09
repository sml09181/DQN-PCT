from __future__ import annotations

import os
from argparse import ArgumentParser
from typing import List

import init_paths # ModuleNotFoundError: No module named 'finrl' 에러 해결

from finrl.config import ALPACA_API_BASE_URL
from finrl.config import DATA_SAVE_DIR
from finrl.config import ERL_PARAMS
from finrl.config import INDICATORS
from finrl.config import RESULTS_DIR
from finrl.config import TENSORBOARD_LOG_DIR
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config import TRADE_END_DATE
from finrl.config import TRADE_START_DATE
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.config import TRAINED_MODEL_DIR
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

# construct environment

# try:
#     from finrl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
# except ImportError:
#     raise FileNotFoundError(
#         "Please set your own ALPACA_API_KEY and ALPACA_API_SECRET in config_private.py"
#     )


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        default="train",
    )
    return parser


# "./" will be added in front of each directory
def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)


def main() -> int:
    parser = build_parser()
    options = parser.parse_args()
    check_and_make_directories(
        ["./trained"]
    ) #[DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]

    if options.mode == "train":
        from finrl import train
        import pandas as pd

        dataset = pd.read_csv("/data/sujin/IPO/data/372320.csv")
        env = StockTradingEnv

        env_kwargs = (
            {"hmax": 100,                           #한 번에 사고팔 수 있는 최대 주
             "initial_amount": 1000000,       # 초기 자본
             "reward_scaling": 1e-4,        #보상 스케일링 (보상이 과도하게 커지는 것 방지)
            }
        )
        agent_kwargs = (
            { "n_steps": 2048,
              "ent_coef": 0.01,
              "learning_rate": 0.00025,
              "batch_size": 64,
              "n_epochs" : 3,}
        )  
        train(
            dataset=dataset,
            technical_indicator_list=INDICATORS,
            env=env,
            model_name="ppo", # DQN, Double DQN, Dueling DQN, A2C, PPO
            model_save_path = "./trained",
            agent_kwargs = agent_kwargs,
            env_kwargs=env_kwargs
        )
    elif options.mode == "test":
        from finrl import test

        env = StockTradingEnv

        # demo for elegantrl
        # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
        kwargs = {}

        account_value_erl = test(  # noqa
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="elegantrl",
            env=env,
            model_name="ppo",
            cwd="./test_ppo",
            net_dimension=512,
            kwargs=kwargs,
        )
    elif options.mode == "trade":
        from finrl import trade

        try:
            from finrl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
        except ImportError:
            raise FileNotFoundError(
                "Please set your own ALPACA_API_KEY and ALPACA_API_SECRET in config_private.py"
            )
        env = StockTradingEnv
        kwargs = {}
        trade(
            start_date=TRADE_START_DATE,
            end_date=TRADE_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="elegantrl",
            env=env,
            model_name="ppo",
            API_KEY=ALPACA_API_KEY,
            API_SECRET=ALPACA_API_SECRET,
            API_BASE_URL=ALPACA_API_BASE_URL,
            trade_mode="paper_trading",
            if_vix=True,
            kwargs=kwargs,
            state_dim=len(DOW_30_TICKER) * (len(INDICATORS) + 3)
            + 3,  # bug fix: for ppo add dimension of state/observations space =  len(stocks)* len(INDICATORS) + 3+ 3*len(stocks)
            action_dim=len(
                DOW_30_TICKER
            ),  # bug fix: for ppo add dimension of action space = len(stocks)
        )
    else:
        raise ValueError("Wrong mode.")
    return 0


# Users can input the following command in terminal
# python main.py --mode=train
# python main.py --mode=test
# python main.py --mode=trade
if __name__ == "__main__":
    raise SystemExit(main())
