import pandas as pd
import numpy as np

prices = pd.read_csv(
    "data/price_data.csv",
    index_col=0,
    parse_dates=True
)

daily_returns = prices.pct_change()

momentum_12m = prices / prices.shift(252) - 1

mean_reversion_1m = -(prices / prices.shift(21) - 1)

forward_returns_1m = prices.shift(-21) / prices - 1

daily_returns.to_csv("data/daily_returns.csv")
momentum_12m.to_csv("data/momentum_12m.csv")
mean_reversion_1m.to_csv("data/mean_reversion_1m.csv")
forward_returns_1m.to_csv("data/forward_returns_1m.csv")

print("Factor calculations completed.")
print("Saved momentum_12m, mean_reversion_1m, and forward_returns_1m.")
