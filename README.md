TODO 


A2C, PPO for continuous actions
DQN, DoubleDQN, D3QN for discrete actions

[참고 Notebook](https://github.com/AI4Finance-Foundation/FinRL-Tutorials/blob/master/2-Advance/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb)

### Todo

- [ ] `/finrl/meta/env_stock_trading` 정하기: np 버전 or vanilla 버전?
- [ ] `action`, `reward` 수정   
- [ ] `trainV2.py` 완성 



This folder has three subfolders:
+ applications: trading tasks,
+ agents: DRL algorithms, from ElegantRL, RLlib, or Stable Baselines 3 (SB3). Users can plug in any DRL lib and play.
+ meta: market environments, we merge the stable ones from the active [FinRL-Meta repo](https://github.com/AI4Finance-Foundation/FinRL-Meta).

Then, we employ a train-test-trade pipeline by three files: train.py, test.py, and trade.py.

```
FinRL
├── finrl (this folder)
│   ├── applications
│   	├── cryptocurrency_trading
│   	├── high_frequency_trading
│   	├── portfolio_allocation
│   	└── stock_trading
│   ├── agents
│   	├── elegantrl
│   	├── rllib
│   	└── stablebaseline3
│   ├── meta
│   	├── data_processors
│   	├── env_cryptocurrency_trading
│   	├── env_portfolio_allocation
│   	├── env_stock_trading
│   	├── preprocessor
│   	├── data_processor.py
│   	└── finrl_meta_config.py
│   ├── config.py
│   ├── config_tickers.py
│   ├── main.py
│   ├── train.py
│   ├── test.py
│   ├── trade.py
│   └── plot.py
```


self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)

def _initiate_state(self):
    return [
        self.initial_amount,
        *self.data.close.values,
        *self.num_stock_shares,
        *[self.data[tech] for tech in self.tech_indicator_list]
    ]

# env_stocktrading.py의 reward 계산
self.reward = end_total_asset - begin_total_asset
self.reward = self.reward * self.reward_scaling


initial_amount = 0
1. 연속형
2. 이산형
    - 3가지
    - 5가지 