import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv(
    "data/ml_dataset.csv",
    parse_dates=["date"]
)

df = df.sort_values("date")

FEATURES = [
    "momentum_12m",
    "short_term_momentum",
    "volatility_1m"
]

TARGET = "target_return_1m"

TRANSACTION_COST = 0.001

portfolio_returns = []

test_years = [2022, 2023, 2024, 2025]

for test_year in test_years:

    print(f"Running walk-forward test for {test_year}...")

    train_df = df[
        df["date"].dt.year < test_year
    ]

    test_df = df[
        df["date"].dt.year == test_year
    ]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    test_df = test_df.copy()

    test_df["predicted_return"] = model.predict(
        test_df[FEATURES]
    )

    monthly_dates = sorted(
        test_df["date"].unique()
    )

    for date in monthly_dates:

        monthly_data = test_df[
            test_df["date"] == date
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
            TARGET
        ].mean()

        short_return = short_stocks[
            TARGET
        ].mean()

        gross_return = (
            long_return - short_return
        )

        net_return = (
            gross_return - TRANSACTION_COST * 2
        )

        portfolio_returns.append({
            "date": date,
            "portfolio_return": net_return
        })

results = pd.DataFrame(portfolio_returns)

results["date"] = pd.to_datetime(
    results["date"]
)

results.set_index("date", inplace=True)

returns = results["portfolio_return"]

equity_curve = (
    1 + returns
).cumprod()

annual_return = (
    returns.mean() * 12
)

annual_volatility = (
    returns.std() * np.sqrt(12)
)

sharpe_ratio = (
    annual_return / annual_volatility
)

max_drawdown = (
    equity_curve /
    equity_curve.cummax() - 1
).min()

# Benchmark
benchmark = yf.download(
    "SPY",
    start=results.index.min(),
    end=results.index.max(),
    auto_adjust=True,
    progress=False
)["Close"]

benchmark_returns = (
    benchmark
    .resample("ME")
    .last()
    .pct_change()
    .dropna()
)

benchmark_equity_curve = (
    1 + benchmark_returns
).cumprod()

print()
print("Walk-Forward ML Backtest Completed")
print()

print("Annual Return:", annual_return)
print("Annual Volatility:", annual_volatility)
print("Sharpe Ratio:", sharpe_ratio)
print("Max Drawdown:", max_drawdown)

plt.figure(figsize=(12, 6))

plt.plot(
    equity_curve.index,
    equity_curve.values,
    label="Walk-Forward ML Strategy"
)

plt.plot(
    benchmark_equity_curve.index,
    benchmark_equity_curve.values,
    label="SPY Benchmark"
)

plt.title(
    "Walk-Forward ML Strategy vs SPY"
)

plt.xlabel("Date")
plt.ylabel("Portfolio Value")

plt.legend()
plt.grid(True)

plt.savefig(
    "results/walk_forward_ml_strategy.png"
)

print()
print(
    "Chart saved to results/walk_forward_ml_strategy.png"
)
