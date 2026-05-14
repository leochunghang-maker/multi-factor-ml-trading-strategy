import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# Load dataset
df = pd.read_csv(
    "data/ml_dataset.csv",
    parse_dates=["date"]
)

# Sort chronologically
df = df.sort_values("date")

# Features
FEATURES = [
    "momentum_12m",
    "short_term_momentum",
    "volatility_1m"
]

TARGET = "target_return_1m"

# Train/Test split
split_date = "2024-01-01"

train_df = df[df["date"] < split_date]
test_df = df[df["date"] >= split_date]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

print("Training model...")

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
rank_ic, _ = spearmanr(y_test, predictions)

print()
print("ML Model Evaluation")
print()
print("Mean Squared Error:", mse)
print("Rank IC:", rank_ic)

# Feature importance
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print()
print("Feature Importances:")
print()
print(importance_df)

# Save predictions
test_df = test_df.copy()
test_df["predicted_return"] = predictions

test_df.to_csv(
    "results/ml_predictions.csv",
    index=False
)

print()
print("Predictions saved to results/ml_predictions.csv")

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

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

monthly_dates = momentum.resample("ME").last().index

portfolio_returns = []
TRANSACTION_COST = 0.001

for date in monthly_dates:

    if date not in momentum.index:
        continue

    mom = momentum.loc[date]
    mr = mean_reversion.loc[date]
    future_returns = forward_returns.loc[date]

    valid = mom.notna() & mr.notna() & future_returns.notna()

    if valid.sum() >= 20:

        mom = mom[valid]
        mr = mr[valid]
        future_returns = future_returns[valid]

        mom_z = (mom - mom.mean()) / mom.std()
        mr_z = (mr - mr.mean()) / mr.std()

        composite_score = mom_z - mr_z

        long_stocks = composite_score.nlargest(10).index
        short_stocks = composite_score.nsmallest(10).index

        long_return = future_returns[long_stocks].mean()
        short_return = future_returns[short_stocks].mean()

        gross_return = long_return - short_return
        turnover_cost = TRANSACTION_COST * 2
        portfolio_return = gross_return - turnover_cost

        portfolio_returns.append({
            "date": date,
            "portfolio_return": portfolio_return
        })

results = pd.DataFrame(portfolio_returns)

results["date"] = pd.to_datetime(results["date"])
results.set_index("date", inplace=True)

returns = results["portfolio_return"]

equity_curve = (1 + returns).cumprod()

annual_return = returns.mean() * 12
annual_volatility = returns.std() * np.sqrt(12)
sharpe_ratio = annual_return / annual_volatility

max_drawdown = (
    equity_curve / equity_curve.cummax() - 1
).min()

benchmark = yf.download(
    "SPY",
    start=results.index.min(),
    end=results.index.max(),
    auto_adjust=True,
    progress=False
)["Close"]

benchmark_returns = benchmark.resample("ME").last().pct_change().dropna()
benchmark_equity_curve = (1 + benchmark_returns).cumprod()

print("Multi-Factor Backtest Completed")
print()
print("Annual Return:", annual_return)
print("Annual Volatility:", annual_volatility)
print("Sharpe Ratio:", sharpe_ratio)
print("Max Drawdown:", max_drawdown)
print("Transaction Cost Per Trade:", TRANSACTION_COST)

plt.figure(figsize=(12, 6))
plt.plot(equity_curve.index, equity_curve.values, label="Strategy")
plt.plot(benchmark_equity_curve.index, benchmark_equity_curve.values, label="SPY Benchmark")

plt.title("Strategy vs SPY Benchmark")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)

plt.savefig("results/strategy_vs_benchmark.png")
plt.close()

results.to_csv("results/multi_factor_results.csv")
equity_curve.to_csv("results/multi_factor_equity_curve.csv")

print("Benchmark chart saved to results/strategy_vs_benchmark.png")