from __future__ import annotations

from src.meta.preprocessor.preprocessors import FeatureEngineer, data_split

import itertools
from src.meta.env_stock_trading.env_stocktrading import *
from src.agents.stablebaselines3.models import DRLAgent
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler


def test(
    dataset,
    technical_indicator_list,
    initial_amount,
    env,
    model_name, # DQN, Double DQN, Dueling DQN, A2C, PPO
    model_path,
    env_kwargs,
    logger,
    if_vix=False, # False로 수정
    norm_start=None,
    norm_end=None,
    start=None,
    end=None
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
    
    processed_full['change %'] = processed_full['change %'].str.rstrip('%').astype('float') / 100.0
    norm_standard=data_split(processed_full, norm_start, norm_end)
    processed_full = data_split(processed_full, start, end)

    # normalization 부분
    scaler = MaxAbsScaler() # MinMaxScaler MaxAbsScaler
    columns_to_scale = [col for col in norm_standard.columns if col!='date' and col!='tic']
    scaler.fit(norm_standard[columns_to_scale])

    scaled_features = scaler.fit_transform(processed_full[columns_to_scale])
    processed_full[columns_to_scale] = scaled_features + 1.1
    
    processed_full = pd.concat([processed_full[['date','tic']], pd.DataFrame(scaled_features, columns=columns_to_scale)], axis=1)
    
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
    # 초기 자본 스케일링
    initial_amount = scaler.transform([[initial_amount] + [0] * (len(columns_to_scale) - 1)])[0, 0]
    initial_amount = initial_amount + 1.1 
    extra_env_kwargs = {
        "initial_amount":initial_amount,
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

    # 1. episode_total_assets을 2D 배열로 변환
    episode_total_assets_array = np.array(episode_total_assets).reshape(-1, 1)-1.1
    # 2. scaler가 학습한 feature 개수에 맞게 0-padding 추가 (2D 배열)
    num_missing_features = len(columns_to_scale) - 1  # 'initial_amount' 외 나머지 feature 개수
    padding = np.zeros((episode_total_assets_array.shape[0], num_missing_features))  # 2D 형태로 맞춤
    # 3. concatenate 수행 (axis=1 방향으로 결합)
    episode_total_assets_padded = np.concatenate((episode_total_assets_array, padding), axis=1)
    # 4. 역변환 적용 (첫 번째 열만 사용)
    episode_total_assets_original = scaler.inverse_transform(episode_total_assets_padded)[:, 0]
    return episode_total_assets_original, sharpe_ratio  
