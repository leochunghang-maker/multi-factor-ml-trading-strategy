from pathlib import Path

import pandas as pd

from src.config import (
    FEATURE_STABILITY_REPORT_PATH,
    FEATURES,
    ML_EMBARGO_DAYS,
    ML_LONG_COUNT,
    ML_PREDICTIONS_PATH,
    ML_SHORT_COUNT,
    PREDICTION_COLUMN,
    WALK_FORWARD_DECILE_REPORT_PATH,
    TARGET,
    TRANSACTION_COST,
    WALK_FORWARD_TEST_YEARS,
)
from src.data.io import load_prediction_dataset, load_ml_dataset
from src.models.validation import decile_spread_analysis, feature_stability_report
from src.models.ml import train_random_forest
from src.portfolio import build_prediction_long_short_returns
from src.reporting.metrics import calculate_performance_metrics, equity_curve


def run_prediction_backtest(
    predictions_path: str = ML_PREDICTIONS_PATH,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    predictions = load_prediction_dataset(predictions_path)
    results = build_prediction_long_short_returns(
        predictions,
        score_column=PREDICTION_COLUMN,
        target_column=TARGET,
        long_count=ML_LONG_COUNT,
        short_count=ML_SHORT_COUNT,
    )
    returns = results["portfolio_return"]
    return results, equity_curve(returns), calculate_performance_metrics(returns)


def run_walk_forward_ml_backtest(
    transaction_cost: float = TRANSACTION_COST,
    test_years: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    df = load_ml_dataset().sort_values("date")
    test_years = test_years or WALK_FORWARD_TEST_YEARS
    yearly_predictions = []

    for test_year in test_years:
        print(f"Running walk-forward test for {test_year}...")
        # Walk-forward validation trains only on dates before the test year.
        # That keeps the model from seeing future market information.
        test_start = pd.Timestamp(year=test_year, month=1, day=1)
        embargo_start = test_start - pd.Timedelta(days=ML_EMBARGO_DAYS)
        train_df = df[df["date"] < embargo_start]
        test_df = df[df["date"].dt.year == test_year].copy()
        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]
        model = train_random_forest(X_train, y_train)

        test_df[PREDICTION_COLUMN] = model.predict(test_df[FEATURES])
        yearly_predictions.append(test_df)

    predictions = pd.concat(yearly_predictions, ignore_index=True)
    Path(WALK_FORWARD_DECILE_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    decile_spread_analysis(
        predictions,
        date_column="date",
        score_column=PREDICTION_COLUMN,
        target_column=TARGET,
    ).to_csv(WALK_FORWARD_DECILE_REPORT_PATH, index=False)
    feature_stability_report(
        df,
        date_column="date",
        feature_columns=FEATURES,
    ).to_csv(FEATURE_STABILITY_REPORT_PATH, index=False)
    results = build_prediction_long_short_returns(
        predictions,
        score_column=PREDICTION_COLUMN,
        target_column=TARGET,
        long_count=ML_LONG_COUNT,
        short_count=ML_SHORT_COUNT,
        transaction_cost=transaction_cost,
    )
    returns = results["portfolio_return"]
    return results, equity_curve(returns), calculate_performance_metrics(returns)
