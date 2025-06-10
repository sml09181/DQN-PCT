from gymnasium.envs.registration import register

register(
    id="StockTradingContEnv-v0",
    entry_point="env_stock_trading.env_stocktrading:ContEnv",
    max_episode_steps=1000,
)

register(
    id="StockTradingDisc3Env-v0",
    entry_point="env_stock_trading.env_stocktrading:Disc3Env",
    max_episode_steps=1000,
)

register(
    id="StockTradingDisc5Env-v0",
    entry_point="env_stock_trading.env_stocktrading:Disc5Env",
    max_episode_steps=1000,
)

register(
    id="StockTradingDisc7Env-v0",
    entry_point="env_stock_trading.env_stocktrading:Disc7Env",
    max_episode_steps=1000,
)
