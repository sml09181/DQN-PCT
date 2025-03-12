from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from typing import List
import torch
import pandas as pd
import logging

import init_paths

from finrl.config import MODEL_PATH
from finrl.config import GPU_ID
from finrl.config import DATA_SAVE_DIR
from finrl.config import INDICATORS
from finrl.config import RESULTS_DIR
from finrl.config import RESULTS_ROOT
from finrl.config import TENSORBOARD_LOG_DIR
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config import TRADE_END_DATE
from finrl.config import TRADE_START_DATE
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.config import TRAINED_MODEL_DIR
from finrl.config import FEATURE
from finrl.config import DATASET
from finrl.config import MODEL
from finrl.config import ENV
from finrl.config import AGENT_PARAMS
from finrl.config_tickers import DOW_30_TICKER
#from finrl.meta.env_stock_trading.env_stocktrading import 

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
            
def create_log():
    class LoggerStreamHandler(logging.StreamHandler):
        def emit(self, record):
            log_entry = self.format(record)
            sys.stdout.write(log_entry + "\n")
            
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if not logger.handlers:
        file_handler = logging.FileHandler("./log.txt")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    os.chdir(RESULTS_ROOT)
    
    parser = build_parser()
    options = parser.parse_args()
    logger = create_log()
    check_and_make_directories(
        ["./trained"]
    ) #[DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]

    if options.mode == "train":
        from finrl import train
        dataset = pd.read_csv(f"/data/sujin/IPO/data/{FEATURE}/{DATASET}", encoding='euc-kr')
        env_kwargs = (
            {"hmax": 100,                           #한 번에 사고팔 수 있는 최대 주
             "initial_amount": 1000000,       # 초기 자본
             "reward_scaling": 1e-4,        #보상 스케일링 (보상이 과도하게 커지는 것 방지)
            }
        )
        agent_kwargs = (
            AGENT_PARAMS
        )
        logger.info("env_kwargs: %s", env_kwargs)
        logger.info("agent_kwargs: %s", agent_kwargs)
        
        train(
            dataset=dataset,
            technical_indicator_list=INDICATORS,
            env=ENV,
            model_name=MODEL, # DQN, Double DQN, Dueling DQN, A2C, PPO
            model_save_path = "./trained",
            agent_kwargs = agent_kwargs,
            env_kwargs=env_kwargs,
            logger=logger,
        )
    elif options.mode == "test":
        from finrl import test

        dataset = pd.read_csv(f"/data/sujin/IPO/data/{FEATURE}/{DATASET}", encoding='euc-kr')
        env_kwargs = (
            {"hmax": 100,                           #한 번에 사고팔 수 있는 최대 주
             "initial_amount": 1000000,       # 초기 자본
             "reward_scaling": 1e-4,        #보상 스케일링 (보상이 과도하게 커지는 것 방지)
            }
        )

        logger.info("env_kwargs: %s", env_kwargs)

        account_value_erl = test( 
            dataset=dataset,
            technical_indicator_list=INDICATORS,
            env=ENV,
            model_name=MODEL, # DQN, Double DQN, Dueling DQN, A2C, PPO
            model_path = MODEL_PATH,    # 모델이 저장된 폴더 이름
            env_kwargs=env_kwargs,
            logger=logger,
        )
    else:
        raise ValueError("Wrong mode.")
    return 0


# Users can input the following command in terminal
# python main.py --mode=train
# python main.py --mode=test
if __name__ == "__main__":
    raise SystemExit(main())
