import pandas as pd
import numpy as np
import datetime
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS
import itertools

from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


'''
dataset : training dataset. csv파일을 pandas로 변경해서 사용하는 거로 가정
hmax : 최대 주식 매도 수
initial_amount : 초기 자본
reward_scaling : 리워드 스케일링
TRAINED_MODEL_DIR : 학습된 모델 결과 저장 루트
RESULTS_DIR : 학습 과정, 결과 저장 루트
if_using_a2c : A2C 학습 유무 
if_using_ppo : PPO 학습 유무
'''
def train(dataset, 
            hmax, 
            initial_amount, 
            reward_scaling,
            TRAINED_MODEL_DIR,
            RESULTS_DIR,
            if_using_a2c, 
            if_using_ppo, 
    ):
    
    """
    data preprocessing 
    """
    fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_vix=True,
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
    
    
    """
    training
    """
    dataset = processed_full.set_index(dataset.columns[0])
    dataset.index.names=['']

    # stock_dimension = 종목의 개수
    stock_dimension = len(dataset.tic.unique())
    # 1:현금보유량, 2*stock_dimension:각 주식별 보유 수량+각 주식의 현재 가격(close price), len(INDICATORS)*stock_dimension:각 주식의 기술적 지표표
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension

    # 주식 매수/매도 시 수수료 비율. 0.001=0.1%
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    # 초기 주식 보유량
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": hmax,                           #한 번에 사고팔 수 있는 최대 주
        "initial_amount": initial_amount,       # 초기 자본
        "num_stock_shares": num_stock_shares,   # 초기 주식 보유량
        "buy_cost_pct": buy_cost_list,          #매도 수수료
        "sell_cost_pct": sell_cost_list,        #매도 수수료
        "state_space": state_space,             #state space의 차원 수
        "stock_dim": stock_dimension,           #종목 수
        "tech_indicator_list": INDICATORS,      #사용될 기술 지표 리스트
        "action_space": stock_dimension,        #action space 크기기
        "reward_scaling": reward_scaling        #보상 스케일링 (보상이 과도하게 커지는 것 방지)
    }


    # environment 초기화 => stock trading simulation을 위한 세팅 완료
    e_train_gym = StockTradingEnv(df = dataset, **env_kwargs)
    #OpenAI Gym의 스타일로 환경 변환
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env = env_train)
    
    # a2c
    if if_using_a2c:
        #A2C 모델 생성
        # hyperparameters 설정 가능
        # GPU 사용 설정 시 device='cuda'를 변수로 넣으면 됨
        model_a2c = agent.get_model("a2c")
        # set up logger
        tmp_path = RESULTS_DIR + '/a2c'
        #SB3(Stable Baseline 3)의 로깅 시스템 설정
        # stdout:콘솔 출력, csv:결과를 csv 파일로 저장, tensorboard:텐서보드 시각화를 위한 로그 저장
        # => 학습 과정 모니터링
        new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        # 위에서 정의한 로깅 시스템 연결 -> 학습 시 reward, loss, 학습 속도 로깅됨
        model_a2c.set_logger(new_logger_a2c)

        # tb_log_name : 학습 과정 시각화 시 사용할 로그 파일 이름 지정
        trained_a2c = agent.train_model(model=model_a2c, 
                                    tb_log_name='a2c',
                                    total_timesteps=50000)
        
        trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c") 

   

    # ddpg
    if if_using_ddpg:
        model_ddpg = agent.get_model("ddpg")
        # set up logger
        tmp_path = RESULTS_DIR + '/ddpg'
        new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_ddpg.set_logger(new_logger_ddpg)
    
        trained_ddpg = agent.train_model(model=model_ddpg, 
                                tb_log_name='ddpg',
                                total_timesteps=50000)
        
        trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg") 

    # ppo
    if if_using_ppo:
        PPO_PARAMS = {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128,
        }
        model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
        # set up logger
        tmp_path = RESULTS_DIR + '/ppo'
        new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_ppo.set_logger(new_logger_ppo)

        trained_ppo = agent.train_model(model=model_ppo, 
                                tb_log_name='ppo',
                                total_timesteps=200000) 
        trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo") 