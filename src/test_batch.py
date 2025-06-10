from __future__ import annotations
import os
import sys
import shutil
import argparse
from typing import List
import pandas as pd
import time
import importlib
import re
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

import init_paths
from src.meta.env_stock_trading.env_stocktrading import *
from src.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import itertools

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", '--gpu_id', type=str, default = "4")
    parser.add_argument("-i", '--input_folder', type=str, \
        default = "/DQN-P_IPOStockPrediction/results/451760/dqn3") # <- CHANGE HERE
        # 372320 / 413640 / 446540 / 451760
    parser.add_argument('--delete_dead', action='store_true', help="model.zip 없는 폴더 삭제 여부")
    args = parser.parse_args()
    return args

def read_config(config_path):
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config
    
def load_config(result_root, delete_dead):
    os.chdir(result_root)
    config_path = os.path.join(result_root, "config.py")
    config = read_config(config_path)

    MODEL = result_root.split('/')[8]
    if MODEL in ['a2c', 'ppo']: ENV = ContEnv
    else: 
        ENV = {
            'dqn3': Disc3Env, # Disc3Env, Disc5Env, Disc7Env
            'dqn5': Disc5Env,
            'dqn7': Disc7Env,
            'dqn201': Disc201Env,
        }.get(MODEL)
        assert ENV is not None
        if MODEL == 'dqn201':
            MODEL = MODEL[:-3]
        else: MODEL = MODEL[:-1]
    MODEL_PATH = str(os.path.join(result_root, "trained", f"{MODEL}.zip"))
    print(MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        shutil.rmtree(result_root) # delete dead
        print("delete:", result_root)
        return None, None
    
    FEATURE = getattr(config, "FEATURE", None)
    DATASET = getattr(config, "DATASET", None)
    TRAIN_START_DATE = getattr(config, "TRAIN_START_DATE", None)
    TRAIN_END_DATE = getattr(config, "TRAIN_END_DATE")
    TEST_START_DATE = getattr(config, "TEST_START_DATE", None)
    TEST_END_DATE = getattr(config,"TEST_END_DATE", None)
    INDICATORS = getattr(config, "INDICATORS", None)
    AGENT_DICT = {
        'ppo': getattr(config, "PPO_PARAMS", None), 
        'a2c': getattr(config, "A2C_PARAMS", None), 
        'dqn': getattr(config, "DQN_PARAMS", None), 
    }
    model_params = AGENT_DICT.get(MODEL)
    PCT1 = None
    PCT2 = None
    PCT3 = None
    if "dqn5" in result_root:
        PCT1 = getattr(config, "PCT1")
        PCT2 = getattr(config, "PCT2")
        print(PCT1, PCT2)
        model_params["pct1"] = PCT1
        model_params["pct2"] = PCT2
    elif "dqn7" in result_root:
        PCT1 = getattr(config, "PCT1")
        PCT2 = getattr(config, "PCT2")
        PCT3 = getattr(config, "PCT3")
        model_params["pct1"] = PCT1
        model_params["pct2"] = PCT2
        model_params["pct3"] = PCT3

    dataset = pd.read_csv(f"/DQN-P_IPOStockPrediction/data/{FEATURE}/{DATASET}", encoding='euc-kr')
    env_kwargs = (
        {"hmax": 100,                           #한 번에 사고팔 수 있는 최대 주
            "reward_scaling": 1e-1,        #보상 스케일링 (보상이 과도하게 커지는 것 방지)
            "pct1": PCT1,
            "pct2": PCT2,
            "pct3": PCT3,
        })
        
    te_params = {
        "dataset": dataset,
        "technical_indicator_list": INDICATORS,
        "initial_amount": 1000000,
        "env": ENV,
        "model_name": MODEL, # DQN, Double DQN, Dueling DQN, A2C, PPO
        "model_path": MODEL_PATH,    # 모델이 저장된 폴더 이름
        "env_kwargs": env_kwargs,
        "logger": None,
        "norm_start": TRAIN_START_DATE,
        "norm_end": TRAIN_END_DATE,
        "start": TEST_START_DATE,
        "end": TEST_END_DATE 
    }
    return te_params, model_params
    
def save(results, model_names, output_folder):
    for model in model_names:
        if model not in list(results.keys()): continue
        df = pd.DataFrame(results[model])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print(df)
        df.to_excel(os.path.join(output_folder, f"{model}_{time.strftime('%m%d_%H%M%S')}.xlsx"), index=False, engine='openpyxl')

def evaluate(result_root, params):
    from src import test
    account_value_erl, sharpe_ratio = test(**params)
    return account_value_erl[-1], sharpe_ratio

def extract_tr_result(result_root, te_params):
    log_file = os.path.join(result_root, 'log.txt')
    with open(log_file, 'r') as file:
        lines = file.readlines()
        
    last_end_total_asset = None
    last_sharpe = None
    end_total_asset_pattern = r"end_total_asset:\s*(\d+\.\d+)"
    sharpe_pattern = r"Sharpe:\s*([\-\d\.]+)"
    
    for line in lines[-10:]:
        end_total_asset_match = re.search(end_total_asset_pattern, line)
        if end_total_asset_match:
            last_end_total_asset = float(end_total_asset_match.group(1))
        sharpe_match = re.search(sharpe_pattern, line)
        if sharpe_match:
            last_sharpe = float(sharpe_match.group(1))
    
    # scaling train_total_asset
    dataset = re.findall('\d+', result_root)
    if len(dataset)==0:
        exit("dataset is not found")

    data = ""
    for d in dataset:
        if len(d)==6:
            data=d
            break
    
    last_end_total_asset = scaled(data, last_end_total_asset, te_params['norm_start'], te_params['norm_end'])

    return last_end_total_asset[0], last_sharpe

def scaled(dataset, asset, norm_start, norm_end):
    dataset = pd.read_csv(f"/DQN-P_IPOStockPrediction/data/Basechange/{dataset}_short.csv", encoding='euc-kr')

    fe = FeatureEngineer(use_technical_indicator=True,
                     use_vix=False,
                     use_turbulence=False,
                     user_defined_feature = False)

    processed = fe.preprocess_data(dataset)

    # 주가 종목과 날짜의 가능한 모든 조합 구하기
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))
    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    #데이터에 실재로 존재하는 날짜만 남김  
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(by=['date','tic'])
    # NaN은 0으로 채움
    processed_full = processed_full.fillna(0)
    processed_full['date'] = pd.to_datetime(processed_full['date'])
    processed_full['change %'] = processed_full['change %'].str.rstrip('%').astype('float') / 100.0
    norm_standard=data_split(processed_full, norm_start, norm_end)

    # normalization 부분
    scaler = MaxAbsScaler() # MaxAbsScaler MinMaxScaler
    columns_to_scale = [col for col in norm_standard.columns if col!='date' and col!='tic']
    scaler.fit(norm_standard[columns_to_scale])
    
    # 1. episode_total_assets을 2D 배열로 변환
    episode_total_assets_array = np.array(asset).reshape(-1, 1)
    # 2. scaler가 학습한 feature 개수에 맞게 0-padding 추가 (2D 배열)
    num_missing_features = len(columns_to_scale) - 1  # 'initial_amount' 외 나머지 feature 개수
    padding = np.zeros((episode_total_assets_array.shape[0], num_missing_features))  # 2D 형태로 맞춤
    # 3. concatenate 수행 (axis=1 방향으로 결합)
    episode_total_assets_padded = np.concatenate((episode_total_assets_array, padding), axis=1)
    # 4. 역변환 적용 (첫 번째 열만 사용)
    episode_total_assets_original = scaler.inverse_transform(episode_total_assets_padded)[:, 0] -1.1 
    print(episode_total_assets_original)

    return episode_total_assets_original

# MARK: main 
def main(input_folder, delete_dead):
    model_names = ['ppo', 'a2c', 'dqn3', 'dqn5', 'dqn7', 'dqn201']
    results_roots = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    results = {k: [] for k in model_names}
    for result_root in results_roots:
        te_params, model_params = load_config(result_root, delete_dead)
        print(te_params, model_params)
        if te_params is not None:
            try:
                te_end_total_asset, te_sharpe = evaluate(result_root, te_params)
            except: continue
            if te_end_total_asset is None or te_sharpe is None:
                print("killed:", result_root)
                continue
            float(te_end_total_asset), float(te_sharpe)
            tr_end_total_asset, tr_sharpe = extract_tr_result(result_root, te_params)
            temp = {
                'te_end_total_asset': te_end_total_asset, 
                'te_sharpe': te_sharpe, 
                'tr_end_total_asset': tr_end_total_asset, 
                'tr_sharpe': tr_sharpe, 
                }
            temp.update(model_params)
            temp.update({'results_path': result_root})
            results[result_root.split('/')[8]].append(temp) # add model name
    results = {k: v for k, v in results.items() if len(v)}
    save(results, model_names, os.path.join(os.path.dirname(input_folder), "test"))
    
if __name__ == "__main__":
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args.input_folder, args.delete_dead)