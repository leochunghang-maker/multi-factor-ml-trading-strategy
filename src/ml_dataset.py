import pandas as pd
import numpy as np

momentum = pd.read_csv(
    "data/momentum_12m.csv",
    index_col=0,
    parse_dates=True
)

mean_reversion = pd.read_csv(
    "data/mean_reversion_1m.csv",
    index_col=0,
    parse_dates=True
)

forward_returns = pd.read_csv(
    "data/forward_returns_1m.csv",
    index_col=0,
    parse_dates=True
)

prices = pd.read_csv(
    "data/price_data.csv",
    index_col=0,
    parse_dates=True
)

volatility = prices.pct_change().rolling(21).std()

dataset_rows = []

for date in momentum.index:

    for ticker in momentum.columns:

        mom = momentum.loc[date, ticker]
        mr = mean_reversion.loc[date, ticker]
        vol = volatility.loc[date, ticker]
        target = forward_returns.loc[date, ticker]

        if pd.notna(mom) and pd.notna(mr) and pd.notna(vol) and pd.notna(target):

            dataset_rows.append({
                "date": date,
                "ticker": ticker,
                "momentum_12m": mom,
                "short_term_momentum": -mr,
                "volatility_1m": vol,
                "target_return_1m": target
            })

ml_dataset = pd.DataFrame(dataset_rows)

ml_dataset.to_csv(
    "data/ml_dataset.csv",
    index=False
)

print("ML dataset created.")
print()

print(ml_dataset.head())
print()
print("Dataset shape:", ml_dataset.shape)
