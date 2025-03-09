from __future__ import annotations

import os
import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
import itertools

from finrl.meta.env_stock_trading.env_stocktrading import *
from finrl.agents.stablebaselines3.models import DRLAgent
# construct environment

# dataset : pandas
def train(
    dataset,
    technical_indicator_list,
    env,
    model_name,
    model_save_path,
    agent_kwargs,
    env_kwargs,
    logger,
    if_vix=True
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
    
    dataset = processed_full.set_index(processed_full.columns[0])
    dataset.index.names=['']
    # stock_dimension = 종목의 개수
    stock_dimension = len(dataset.tic.unique())
    # 1:현금보유량, 2*stock_dimension:각 주식별 보유 수량+각 주식의 현재 가격(close price), len(INDICATORS)*stock_dimension:각 주식의 기술적 지표표
    state_space = 1 + 2*stock_dimension + len(technical_indicator_list)*stock_dimension

    # 주식 매수/매도 시 수수료 비율. 0.001=0.1%
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    # 초기 주식 보유량
    num_stock_shares = [0] * stock_dimension

    extra_env_kwargs = {
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
    model = agent.get_model(model_name, model_kwargs=agent_kwargs)
    trained_model = agent.train_model(
        model=model, tb_log_name=model_name, total_timesteps=total_timesteps
    )
    logger.info("Training is finished!")
    trained_model.save(cwd)
    logger.info("Trained model is saved in " + str(os.getcwd()))


"""
이전 코드에서 실행하는 법

if __name__ == "__main__":
    env = StockTradingEnv

    kwargs = (
        {}
    )
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        erl_params=ERL_PARAMS,
        break_step=1e5,
        kwargs=kwargs,
    )

"""