from __future__ import annotations
import os
import time
import shutil
import importlib.util

from finrl.meta.env_stock_trading.env_stocktrading import *

# general setting 
GPU_ID = 3 # int, str 모두 가능 
MODEL_PATH = '/data/sujin/eunsang/GlobalStockAnalyzer/results/372320/ppo/0311_1518' # Train: ''
WORKING_ROOT = '/data/sujin/eunsang/GlobalStockAnalyzer/' # NOTE: change to your name
# python main.py --mode=train
# python main.py --mode=test

# For Train (ignore when test mode)
if len(MODEL_PATH)==0:
    FEATURE = 'BaseIPO' # BaseIPO, Base
    DATASET = '372320.csv' #372320.csv, 413640.csv, 446540.csv, 451760.csv
    TIMESTAMP = time.strftime('%m%d_%H%M%S')
    MODEL = 'ppo' # ppo, a2c, dqn3, dqn5, dqn7
    RESULTS_ROOT = os.path.join(WORKING_ROOT, f'results/{DATASET[:-4]}/{MODEL}/{TIMESTAMP}')
# For Test
# MODEL이나 ENV 따로 설정 안해도 됨    
else:
    def load_config(config_path):
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config
    config_path = os.path.join(MODEL_PATH, "config.py")
    config = load_config(config_path)
    FEATURE = getattr(config, "FEATURE", None)
    DATASET = getattr(config, "DATASET", None)
    print("FEATURE:", getattr(config, "FEATURE", None))
    print("DATASET:", getattr(config, "DATASET", None))

    MODEL = MODEL_PATH.split('/')[7]
    RESULTS_ROOT = os.path.join(MODEL_PATH, "test")
    if not MODEL_PATH.endswith(f"{MODEL[:3]}.zip"): MODEL_PATH = os.path.join(MODEL_PATH, f"trained/{MODEL[:3]}.zip")
     
if MODEL in ['a2c', 'ppo']: ENV = ContEnv
else: 
    ENV = {
        'dqn3': Disc3Env, # Disc3Env, Disc5Env, Disc7Env
        'dqn5': Disc5Env,
        'dqn7': Disc7Env,
    }.get(MODEL)
    assert ENV is not None
    MODEL = MODEL[:-1]

if not os.path.exists(RESULTS_ROOT):
    os.makedirs(f"{RESULTS_ROOT}/plots")
    os.makedirs(f"{RESULTS_ROOT}/actions")
    os.makedirs(f"{RESULTS_ROOT}/total_value")
    os.makedirs(f"{RESULTS_ROOT}/rewards")
    os.makedirs(f"{RESULTS_ROOT}/asset_memory")
    os.makedirs(f"{RESULTS_ROOT}/agent_log")
    shutil.copy(os.path.join(WORKING_ROOT, 'finrl/config.py'), os.path.join(RESULTS_ROOT, 'config.py'))
DATA_SAVE_DIR = os.path.join(RESULTS_ROOT, "datasets")
TRAINED_MODEL_DIR = os.path.join(RESULTS_ROOT, "trained_models")
TENSORBOARD_LOG_DIR = os.path.join(RESULTS_ROOT, "tensorboard_log")
AGENT_LOG_DIR = os.path.join(RESULTS_ROOT, "agent_log")
    
# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

# Model Parameters
# DQN, Double DQN, Dueling DQN, A2C, PPO
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.00075}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025, # 0.00025
    "batch_size": 64,
    "n_epochs": 20,
}
DQN_PARAMS = {
    "learning_rate": 0.0001, # 0.0001
    "buffer_size": 1000000, 
    "learning_starts": 10, 
    "batch_size": 64,
    "tau": 1.0,
    "gamma": 0.99,
    "target_update_interval": 800, # 10000
    # exploration_fraction=0.1, 
    # exploration_initial_eps=1.0, 
    # exploration_final_eps=0.05, 
    # max_grad_norm=10, 
    # stats_window_size=100, 
}
AGENT_DICT = {
    'ppo': PPO_PARAMS, 
    'a2c': A2C_PARAMS, 
    'dqn': DQN_PARAMS, 
}

# 
AGENT_PARAMS = AGENT_DICT.get(MODEL)

# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2023-11-09"  # bug fix: set Monday right, start date set 2014-01-01 ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 1658 and the array at index 1 has size 1657
TRAIN_END_DATE = "2024-03-25" # 여기만 수정하면 됨(아현) 참고로 이 날은 포함이 안 되더라

TEST_START_DATE = "2024-03-25" 
TEST_END_DATE = "2024-03-31"

TRADE_START_DATE = "2021-11-01"
TRADE_END_DATE = "2021-12-01"

# Possible time zones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
TIME_ZONE_SELFDEFINED = "Asia/SouthKorea"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 1  # 0 (default) or 1 (use the self defined)

# parameters for data sources
ALPACA_API_KEY = "xxx"  # your ALPACA_API_KEY
ALPACA_API_SECRET = "xxx"  # your ALPACA_API_SECRET
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
BINANCE_BASE_URL = "https://data.binance.vision/"  # binance url

ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}