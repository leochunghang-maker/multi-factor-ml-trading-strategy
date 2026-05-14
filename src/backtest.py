import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

momentum = pd.read_csv("data/momentum_12m.csv", index_col=0, parse_dates=True)
forward_returns = pd.read_csv("data/forward_returns_1m.csv", index_col=0, parse_dates=True)

monthly_dates = momentum.resample("ME").last().index

portfolio_returns = []

for date in monthly_dates:
    if date not in momentum.index:
        continue

    scores = momentum.loc[date]
    future_returns = forward_returns.loc[date]

    valid = scores.notna() & future_returns.notna()

    if valid.sum() >= 10:
        scores = scores[valid]
        future_returns = future_returns[valid]

        long_stocks = scores.nlargest(5).index
        short_stocks = scores.nsmallest(5).index

        long_return = future_returns[long_stocks].mean()
        short_return = future_returns[short_stocks].mean()

        portfolio_return = long_return - short_return

        portfolio_returns.append({
            "date": date,
            "long_return": long_return,
            "short_return": short_return,
            "long_short_return": portfolio_return
        })

results = pd.DataFrame(portfolio_returns)

results["date"] = pd.to_datetime(results["date"])
results.set_index("date", inplace=True)

monthly_returns = results["long_short_return"]

equity_curve = (1 + monthly_returns).cumprod()

annual_return = monthly_returns.mean() * 12
annual_volatility = monthly_returns.std() * np.sqrt(12)
sharpe_ratio = annual_return / annual_volatility
max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()

results.to_csv("results/momentum_backtest.csv")

plt.figure(figsize=(12, 6))
plt.plot(equity_curve)
plt.title("Momentum Strategy Equity Curve")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)

plt.savefig("results/equity_curve.png")

print("Monthly Momentum Long-Short Backtest Completed")
print()
print("Annual Return:", annual_return)
print("Annual Volatility:", annual_volatility)
print("Sharpe Ratio:", sharpe_ratio)
print("Max Drawdown:", max_drawdown)
print()
print("Number of Monthly Trades:", len(results))
print()
print("Equity curve saved to results/equity_curve.png")
