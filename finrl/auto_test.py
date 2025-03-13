from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from typing import List
import torch
import pandas as pd
import time
import logging
import importlib
import re
import warnings
warnings.filterwarnings('ignore')


import init_paths
from finrl.meta.env_stock_trading.env_stocktrading import *

GPU_ID = "1"
folder_path = "/data/sujin/sujin/GlobalStockAnalyzer/results/451760/dqn7"
results_roots = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
#print(results_roots)
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

def extract_last_values(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()
    last_end_total_asset = None
    last_sharpe = None

    end_total_asset_pattern = r"end_total_asset:\s*(\d+\.\d+)"
    sharpe_pattern = r"Sharpe:\s*([\-\d\.]+)"

    for line in lines:
   
        end_total_asset_match = re.search(end_total_asset_pattern, line)
        if end_total_asset_match:
            last_end_total_asset = float(end_total_asset_match.group(1))
        sharpe_match = re.search(sharpe_pattern, line)
        if sharpe_match:
            last_sharpe = float(sharpe_match.group(1))

    return last_end_total_asset, last_sharpe

def main(result_root) -> int:
    from finrl.config import INDICATORS
    print("=================================================")
    print(result_root)
    os.chdir(result_root)
    
    def load_config(config_path):
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config
    config_path = os.path.join(result_root, "config.py")
    config = load_config(config_path)
    FEATURE = getattr(config, "FEATURE", None)
    DATASET = getattr(config, "DATASET", None)
    MODEL = result_root.split('/')[7]
    
     
    if MODEL in ['a2c', 'ppo']: ENV = ContEnv
    else: 
        ENV = {
            'dqn3': Disc3Env, # Disc3Env, Disc5Env, Disc7Env
            'dqn5': Disc5Env,
            'dqn7': Disc7Env,
        }.get(MODEL)
        assert ENV is not None
        MODEL = MODEL[:-1]
    
    AGENT_DICT = {
        'ppo': getattr(config, "PPO_PARAMS", None), 
        'a2c': getattr(config, "A2C_PARAMS", None), 
        'dqn': getattr(config, "DQN_PARAMS", None), 
    }
    PARAMS = AGENT_DICT.get(MODEL)
    
    if result_root.split('/')[7] == "dqn5":
        #PARAMS['env'] = result_root.split('/')[7]
        PCT1 = getattr(config, "pct1", 0.5)
        PCT2 = getattr(config, "pct2", 1)
        PCT3 = None
        PARAMS["pct1"] = PCT1
        PARAMS["pct2"] = PCT2
        PARAMS["pct3"] = PCT3
    elif result_root.split('/')[7] == "dqn7":
        #PARAMS['env'] = result_root.split('/')[7]
        PCT1 = getattr(config, "pct1", 0.5)
        PCT2 = getattr(config, "pct2", 0.7)
        PCT3 = getattr(config, "pct3", 0.9)
        PARAMS["pct1"] = PCT1
        PARAMS["pct2"] = PCT2
        PARAMS["pct3"] = PCT3
    elif result_root.split('/')[7] == "dqn3":
        #PARAMS['env'] = result_root.split('/')[7]
        PCT1 = None
        PCT2 = None
        PCT3 = None
        PARAMS["pct1"] = PCT1
        PARAMS["pct2"] = PCT2
        PARAMS["pct3"] = PCT3
    else:
        PCT1 = None
        PCT2 = None
        PCT3 = None
        
        
    MODEL_PATH = str(os.path.join(result_root, "trained", f"{MODEL}.zip"))
    #MODEL_PATH = "/data/sujin/sujin/GlobalStockAnalyzer/results/451760/a2c/0313_111055/trained/a2c.zip"
    if not os.path.exists(MODEL_PATH): 
        return None, None, None, None, None, None
    #exit(-1)
    #MODEL_PATH = "/data/sujin/sujin/GlobalStockAnalyzer/results/451760/a2c/0313_110955/trained/a2c.zip"

    from finrl import test

    dataset = pd.read_csv(f"/data/sujin/IPO/data/{FEATURE}/{DATASET}", encoding='euc-kr')
    env_kwargs = (
        {"hmax": 100,                           #한 번에 사고팔 수 있는 최대 주
            "initial_amount": 1000000,       # 초기 자본
            "reward_scaling": 1e-4,        #보상 스케일링 (보상이 과도하게 커지는 것 방지)
            "pct1": PCT1,
            "pct2": PCT2,
            "pct3": PCT3,
        }
    )

    account_value_erl, sharpe_ratio = test( 
        dataset=dataset,
        technical_indicator_list=INDICATORS,
        env=ENV,
        model_name=MODEL, # DQN, Double DQN, Dueling DQN, A2C, PPO
        model_path = MODEL_PATH,    # 모델이 저장된 폴더 이름
        env_kwargs=env_kwargs,
        logger=None,
    )
    print(account_value_erl, sharpe_ratio)
    
    
    log_file = os.path.join(result_root, 'log.txt')
    train_end_total_asset, train_sharpe = extract_last_values(log_file)

    return account_value_erl[-1], sharpe_ratio, FEATURE, PARAMS, train_end_total_asset, train_sharpe


# Users can input the following command in terminal
# python main.py --mode=train
# python main.py --mode=test
if __name__ == "__main__":
    TIMESTAMP = time.strftime('%m%d_%H%M%S')
    model_names = ['ppo', 'a2c', 'dqn3', 'dqn5', 'dqn7']
    results = {k: [] for k in model_names}
    for result_root in results_roots:
        end_total_asset, Sharpe, feature, params, train_end_total_asset, train_sharpe = main(result_root)
        if end_total_asset is None or Sharpe is None: continue
        print(end_total_asset, Sharpe)
        end_total_asset = float(end_total_asset)
        Sharpe = float(Sharpe)
        matched_model = [model for model in model_names if model in result_root][0]
        temp = {'te_end_total_asset': end_total_asset, 'te_sharpe': Sharpe, 'tr_end_total_asset': train_end_total_asset, 'tr_sharpe': train_sharpe, 'feature': feature}
        temp.update(params)
        temp.update({'results_path': result_root})
        results[matched_model].append(temp)
        # except: continue
    results = {k: v for k, v in results.items() if len(v)}
    
    for model in model_names:
        if model not in list(results.keys()): continue
        df = pd.DataFrame(results[model])
        output_path = os.path.join("/data/sujin/sujin/GlobalStockAnalyzer/results/test", f"{model}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(df)
        df.to_excel(os.path.join(output_path, f"{TIMESTAMP}.xlsx"), index=False, engine='openpyxl')