from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from typing import List
import torch
import pandas as pd
import logging

import init_paths

from src.config import MODEL_PATH
from src.config import GPU_ID
from src.config import INDICATORS
from src.config import RESULTS_ROOT
from src.config import TEST_END_DATE
from src.config import TEST_START_DATE
from src.config import TRAIN_END_DATE
from src.config import TRAIN_START_DATE
from src.config import FEATURE
from src.config import DATASET
from src.config import MODEL
from src.config import ENV
from src.config import AGENT_PARAMS
from src.config import INITIAL_N_STOCKS
from src.config import RESTART_MODEL_PATH

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
        from src import train
        from src.config import PCT1, PCT2, PCT3
        dataset = pd.read_csv(f"/DQN-P_IPOStockPrediction/data/{FEATURE}/{DATASET}", encoding='euc-kr')
        env_kwargs = (
            {"hmax": 100,                           #한 번에 사고팔 수 있는 최대 주
            "reward_scaling": 1e-1,        # 1e-4 보상 스케일링 (보상이 과도하게 커지는 것 방지)
            "pct1": PCT1,
            "pct2": PCT2,
            "pct3": PCT3,
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
            initial_amount=1000000,
            env=ENV,
            model_name=MODEL, # DQN, Double DQN, Dueling DQN, A2C, PPO
            model_save_path = "./trained",
            agent_kwargs = agent_kwargs,
            env_kwargs=env_kwargs,
            logger=logger,
            init_n_stocks = INITIAL_N_STOCKS,
            restart_model_path = RESTART_MODEL_PATH,
        )
        print(f"[INFO] Training Done: {RESULTS_ROOT}")
    elif options.mode == "test":
        from src import test

        dataset = pd.read_csv(f"/DQN-P_IPOStockPrediction/data/{FEATURE}/{DATASET}", encoding='euc-kr')
        env_kwargs = (
            {"hmax": 100,                           #한 번에 사고팔 수 있는 최대 주
             "reward_scaling": 1e-1,        #보상 스케일링 (보상이 과도하게 커지는 것 방지) #reward 설정 부분
            }
        )

        logger.info("env_kwargs: %s", env_kwargs)

        account_value_erl = test( 
            dataset=dataset,
            technical_indicator_list=INDICATORS,
            initial_amount=1000000,
            env=ENV,
            model_name=MODEL, # DQN, Double DQN, Dueling DQN, A2C, PPO
            model_path = MODEL_PATH,    # 모델이 저장된 폴더 이름
            env_kwargs=env_kwargs,
            logger=logger,
            norm_start=TRAIN_START_DATE,
            norm_end=TRAIN_END_DATE,
            start=TEST_START_DATE,
            end=TEST_END_DATE
        )
    else:
        raise ValueError("Wrong mode.")
    return 0


# Users can input the following command in terminal
# python main.py --mode=train
# python main.py --mode=test
if __name__ == "__main__":
    raise SystemExit(main())
