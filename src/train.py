from __future__ import annotations

import os
import pandas as pd
from src.meta.preprocessor.preprocessors import FeatureEngineer, data_split
import itertools

from src.meta.env_stock_trading.env_stocktrading import *
from src.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure

from src.config import TRAIN_START_DATE
from src.config import TRAIN_END_DATE
from src.config import AGENT_LOG_DIR

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

class LossLoggingCallback(BaseCallback):
    def __init__(self, log_dir: str, verbose=0):
        super(LossLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "log.txt")
        
    def _on_step(self) -> bool:
        # DQN에서는 loss 값을 직접 추적하는 방법이 없으므로, 손실 계산을 직접 처리해야 함
        if self.model.num_timesteps % 100 == 0:  # 예시: 100 step마다 기록
            # DQN에서 loss 값을 추출하는 방법은 직접적으로 제공되지 않음
            # 대신, `logger`나 내부 훈련 값을 추적하는 방식 사용
            loss = self.model.logger.get("rollout/ep_100/mean")  # 예시, 실제 모델 로깅 항목에 따라 달라질 수 있음
            
            # 손실값 기록
            with open(self.log_file, 'a') as f:
                if loss is not None:
                    f.write(f"Step: {self.model.num_timesteps}, DQN Loss: {loss:.5f}\n")
                else:
                    f.write(f"Step: {self.model.num_timesteps}, DQN Loss: N/A\n")
            
        return True

def train(
    dataset,
    technical_indicator_list,
    initial_amount,
    env,
    model_name,
    model_save_path,
    agent_kwargs,
    env_kwargs,
    logger,
    init_n_stocks,
    if_vix=False, # False로 수정
    restart_model_path = None,
):
    
    fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = technical_indicator_list,
                     use_vix=if_vix,
                     use_turbulence=False,
                     user_defined_feature = False)

    processed = fe.preprocess_data(dataset)
    

    # 주가 종목과 날짜의 가능한 모든 조합 구하기
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    # 전처리 데이터와 위에서 구한 조합 병합하기.
    # 해당 경우에 데이터가 없는 경우 NaN으로 남도록 함 : how="left"
    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    #데이터에 실재로 존재하는 날짜만 남김  
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(by=['date','tic'])
    
    # NaN은 0으로 채움
    processed_full = processed_full.fillna(0)
    processed_full['date'] = pd.to_datetime(processed_full['date'])

    print(processed_full)
    processed_full = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    processed_full['change %'] = processed_full['change %'].str.rstrip('%').astype('float') / 100.0
    columns_to_scale = [col for col in processed_full.columns if col!='date' and col!='tic']
    
    # nomalization 변경하는 부분
    scaler = MaxAbsScaler()
    scaled_features = scaler.fit_transform(processed_full[columns_to_scale])
    processed_full[columns_to_scale] = scaled_features + 1.1
    print(processed_full)

    dataset = processed_full.set_index(processed_full.columns[0])
    dataset.index.names=['']
    # stock_dimension = 종목의 개수
    stock_dimension = len(dataset.tic.unique())
    # 1:현금보유량, 2*stock_dimension:각 주식별 보유 수량+각 주식의 현재 가격(close price), len(INDICATORS)*stock_dimension:각 주식의 기술적 지표표
    state_space = 1 + 2*stock_dimension + len(technical_indicator_list)*stock_dimension

    # 주식 매수/매도 시 수수료 비율. 0.001=0.1%
    buy_cost_list = sell_cost_list = [0.00015] * stock_dimension
    # 초기 주식 보유량
    num_stock_shares = [init_n_stocks] * stock_dimension
    # 초기 자본 스케일링
    initial_amount_scaled = scaler.transform([[initial_amount] + [0] * (len(columns_to_scale) - 1)])[0, 0] 
    initial_amount_scaled = initial_amount_scaled + 1.1 #로 해도 되나
    extra_env_kwargs = {
        "initial_amount" : initial_amount_scaled,
        "num_stock_shares": num_stock_shares,   # 초기 주식 보유량
        "buy_cost_pct": buy_cost_list,          #매도 수수료
        "sell_cost_pct": sell_cost_list,        #매도 수수료
        "state_space": state_space,             #state space의 차원 수
        "stock_dim": stock_dimension,           #종목 수
        "tech_indicator_list": technical_indicator_list,      #사용될 기술 지표 리스트
        "action_space": stock_dimension,        #action space 크기
    }
    # environment 초기화 => stock trading simulation을 위한 세팅 완료
    e_train_gym = env(df = dataset, model_name=model_name, mode='train', logger=logger, **env_kwargs,**extra_env_kwargs)
    #OpenAI Gym의 스타일로 환경 변환
    env_instance, _ = e_train_gym.get_sb_env()

    # read parameters
    cwd = model_save_path+"/"+model_name

    total_timesteps = agent_kwargs.get("total_timesteps", 1e6)
    
    agent = DRLAgent(env=env_instance)
    if restart_model_path is not None:
      model = agent.load_model(model_path=restart_model_path, model_name=model_name)
    else:
        model = agent.get_model(
            model_name=model_name,
            policy_kwargs=None,
            model_kwargs=agent_kwargs,
            verbose =1)
    
    agent_logger = configure(AGENT_LOG_DIR, ["stdout", "csv", "tensorboard"])
    model.set_logger(agent_logger)
    
    
    trained_model = agent.train_model(
        model=model, tb_log_name=model_name, total_timesteps=total_timesteps
    )

    logger.info("Training is finished!")
    trained_model.save(cwd)
    logger.info("Trained model is saved in " + str(os.getcwd()))