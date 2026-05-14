import pandas as pd
from scipy.stats import spearmanr

momentum = pd.read_csv("data/momentum_12m.csv", index_col=0, parse_dates=True)
forward_returns = pd.read_csv("data/forward_returns_1m.csv", index_col=0, parse_dates=True)

ic_results = []

for date in momentum.index:
    x = momentum.loc[date]
    y = forward_returns.loc[date]

    valid = x.notna() & y.notna()

    if valid.sum() >= 10:
        ic, p_value = spearmanr(x[valid], y[valid])
        ic_results.append({
            "date": date,
            "information_coefficient": ic,
            "p_value": p_value
        })

ic_df = pd.DataFrame(ic_results)
ic_df.to_csv("results/momentum_ic_results.csv", index=False)

print("Momentum IC test completed.")
print()
print(ic_df.describe())
print()
print("Average IC:", ic_df["information_coefficient"].mean())
print("Positive IC Ratio:", (ic_df["information_coefficient"] > 0).mean())
