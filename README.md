# DQN-P_IPOStockPrediction

This repository presents **DQN-P**, a Deep Q-Network-based reinforcement learning framework tailored for IPO (Initial Public Offering) stock prediction in the Korean stock market. It builds upon and improves the FinRL framework to better handle the unique challenges posed by IPO stocks, such as short historical data and high volatility.

## Abstract

Recently, stocks of companies newly listed through Initial Public Offering (IPO) have gained significant attention. However, due to their limited historical data and high volatility, it is difficult to directly apply existing reinforcement learning frameworks like FinRL to these stocks.  
To address this issue, we propose enhancements to the Q-network architecture of Deep Q-Network (DQN) within the FinRL framework, along with a newly designed percentage-based discrete action space. Our proposed framework, **DQN-P**, is the first application of reinforcement learning to Korean IPO stocks and demonstrates an average profit improvement of 50,000 KRW compared to the baseline FinRL.

## Project Structure
```
DQN-P_IPOStockPrediction
├── src
│ ├── agents
│ │ └── stablebaseline3 
│ ├── meta
│ │ ├── data_processors # Modules for data preprocessing
│ │ ├── env_stock_trading # Custom stock trading environment
│ │ ├── preprocessor # Raw data preprocessing tools
│ │ ├── data_processor.py # Data handling utilities
│ │ └── meta_config.py # Meta configuration for datasets/environments
│ ├── config.py # Model and environment configuration
│ ├── main.py # Entry point for training
│ ├── train.py # Training pipeline
│ ├── test.py # Single instance testing
│ ├── test_batch.py # Batch testing script
```


## How to Run

- **Training**  
    Run the training pipeline using the `main.py` script:
    ```bash
    python src/main.py
- **Testing**
    For batch testing, use:
    ```bash
    python src/test_batch.py

## Results

### Stock Price Trends
<img width="600" alt="Image" src="https://github.com/user-attachments/assets/275534f4-b145-4071-88fa-f0ccc0ec7b56" />

Each IPO company's daily closing prices in KRW. The dotted line indicates the boundary between the training and testing periods. In this study, models are trained using historical data before the boundary and evaluated on data after it.

### Final Asset Distribution
<img width="600" alt="Image" src="https://github.com/user-attachments/assets/5e6b7c71-fce8-4d6d-98db-f202590382e9" />

The distribution of final assets for each model: DQN-K3, DQN-K201, DQN-P3, DQN-P5, and DQN-P7, respectively. DQN-P series overperform DQN-K series. 


## Reference
This project is based on and extends the open-source [FinRL](https://github.com/AI4Finance-Foundation/FinRL) framework by AI4Finance Foundation.

