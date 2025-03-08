from __future__ import annotations

import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
import itertools

from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
# construct environment

# dataset : pandas
def train(
    dataset,
    time_interval,
    technical_indicator_list,
    env,
    model_name,
    if_vix=True,
    **kwargs,
):
    fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = technical_indicator_list,
                     use_vix=if_vix,
                     use_turbulence=True,
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
    processed_full = processed_full.sort_values(['date','tic'])
    # NaN은 0으로 채움
    processed_full = processed_full.fillna(0)
    
    dataset = processed_full.set_index(processed_full.columns[0])
    dataset.index.names=['']

    # environment 초기화 => stock trading simulation을 위한 세팅 완료
    e_train_gym = StockTradingEnv(df = dataset, **kwargs)
    #OpenAI Gym의 스타일로 환경 변환
    env_instance, _ = e_train_gym.get_sb_env()

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))

    total_timesteps = kwargs.get("total_timesteps", 1e6)
    agent_params = kwargs.get("agent_params")

    

    agent = DRLAgent(env=env_instance)
    model = agent.get_model(model_name, model_kwargs=agent_params)
    trained_model = agent.train_model(
        model=model, tb_log_name=model_name, total_timesteps=total_timesteps
    )
    print("Training is finished!")
    trained_model.save(cwd)
    print("Trained model is saved in " + str(cwd))


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

