from __future__ import annotations

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

import itertools
from finrl.config import INDICATORS
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.meta.env_stock_trading.env_stocktrading import *
from finrl.agents.stablebaselines3.models import DRLAgent


def test(
    dataset,
    technical_indicator_list,
    env,
    model_name, # DQN, Double DQN, Dueling DQN, A2C, PPO
    model_path,
    env_kwargs,
    logger,
    if_vix=True,
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
    
    
    # data_split
    processed_full = data_split(processed_full, TEST_START_DATE, TEST_END_DATE)

    
    dataset = processed_full.set_index(processed_full.columns[0])
    dataset.index.names=['']
    # stock_dimension = 종목의 개수
    stock_dimension = len(dataset.tic.unique())
    # 1:현금보유량, 2*stock_dimension:각 주식별 보유 수량+각 주식의 현재 가격(close price), len(INDICATORS)*stock_dimension:각 주식의 기술적 지표표
    state_space = 1 + 2*stock_dimension + len(technical_indicator_list)*stock_dimension

    # 주식 매수/매도 시 수수료 비율. 0.001=0.1%
    buy_cost_list = sell_cost_list = [0.00015] * stock_dimension
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
    e_test_gym = env(df = dataset, model_name=model_name, risk_indicator_col='vix', mode='test', logger=logger, **env_kwargs,**extra_env_kwargs)
    #OpenAI Gym의 스타일로 환경 변환
    env_instance, _ = e_test_gym.get_sb_env()
    episode_total_assets, sharpe_ratio = DRLAgent.DRL_prediction_load_from_file(
        model_name=model_name, environment=env_instance, cwd=model_path
    )
    #print(episode_total_assets, sharpe_ratio)
    return episode_total_assets, sharpe_ratio     # 각 단계 별 거래 수행 후 자산 리스트


"""
if __name__ == "__main__":
    env = StockTradingEnv

    # # demo for elegantrl
    # kwargs = (
    #     {}
    # )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty

    # account_value_erl = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="elegantrl",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo",
    #     net_dimension=512,
    #     kwargs=kwargs,
    # )

    ## if users want to use rllib, or stable-baselines3, users can remove the following comments

    # # demo for rllib
    # import ray
    # ray.shutdown()  # always shutdown previous session if any
    # account_value_rllib = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo/checkpoint_000030/checkpoint-30",
    #     rllib_params=RLlib_PARAMS,
    # )
    #
    # demo for stable baselines3
    account_value_sb3 = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="stable_baselines3",
        env=env,
        model_name=model_name,
        cwd=f"./test_{model_name}.zip",
    )
"""