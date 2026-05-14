import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

predictions = pd.read_csv(
    "results/ml_predictions.csv",
    parse_dates=["date"]
)

portfolio_returns = []

monthly_dates = sorted(predictions["date"].unique())

for date in monthly_dates:

    monthly_data = predictions[
        predictions["date"] == date
    ]

    long_stocks = monthly_data.nlargest(
        10,
        "predicted_return"
    )

    short_stocks = monthly_data.nsmallest(
        10,
        "predicted_return"
    )

    long_return = long_stocks[
        "target_return_1m"
    ].mean()

    short_return = short_stocks[
        "target_return_1m"
    ].mean()

    portfolio_return = long_return - short_return

    portfolio_returns.append({
        "date": date,
        "portfolio_return": portfolio_return
    })

results = pd.DataFrame(portfolio_returns)

results["date"] = pd.to_datetime(
    results["date"]
)

results.set_index("date", inplace=True)

returns = results["portfolio_return"]

equity_curve = (1 + returns).cumprod()

annual_return = returns.mean() * 12
annual_volatility = returns.std() * np.sqrt(12)
sharpe_ratio = annual_return / annual_volatility

max_drawdown = (
    equity_curve / equity_curve.cummax() - 1
).min()

print("ML Strategy Backtest Completed")
print()

print("Annual Return:", annual_return)
print("Annual Volatility:", annual_volatility)
print("Sharpe Ratio:", sharpe_ratio)
print("Max Drawdown:", max_drawdown)

plt.figure(figsize=(12, 6))

plt.plot(
    equity_curve.index,
    equity_curve.values
)

plt.title("ML Strategy Equity Curve")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)

plt.savefig(
    "results/ml_strategy_equity_curve.png"
)

print()
print(
    "Equity curve saved to results/ml_strategy_equity_curve.png"
)
