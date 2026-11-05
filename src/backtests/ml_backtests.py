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
    RESEARCH_LONG_SHORT_GROSS_EXPOSURE,
    WALK_FORWARD_DECILE_REPORT_PATH,
    TARGET,
    TRANSACTION_COST,
    WALK_FORWARD_TEST_YEARS,
)
from src.data.io import load_prediction_dataset, load_ml_dataset, load_price_data
from src.models.validation import (
    assert_no_label_overlap,
    decile_spread_analysis,
    feature_stability_report,
    purged_embargo_year_splits,
    training_rows_without_label_overlap,
)
from src.models.ml import train_random_forest
from src.portfolio import (
    build_prediction_long_short_returns,
    month_end_holding_period_returns,
)
from src.reporting.metrics import calculate_performance_metrics, equity_curve


def run_prediction_backtest(
    predictions_path: str = ML_PREDICTIONS_PATH,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    predictions = load_prediction_dataset(predictions_path)
    holding_returns = month_end_holding_period_returns(load_price_data())
    results = build_prediction_long_short_returns(
        predictions,
        score_column=PREDICTION_COLUMN,
        target_column=TARGET,
        long_count=ML_LONG_COUNT,
        short_count=ML_SHORT_COUNT,
        realized_returns=holding_returns,
        rebalance_on_month_ends=True,
        gross_exposure=RESEARCH_LONG_SHORT_GROSS_EXPOSURE,
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

    splits = purged_embargo_year_splits(
        df["date"],
        test_years=test_years,
        purge_months=1,
        embargo_months=1,
    )
    for split in splits:
        test_year = int(split["test_year"])
        print(f"Running walk-forward test for {test_year}...")
        # Purge training rows whose ~1m/21d label window would overlap the test
        # fold; embargo is encoded in the split boundaries for later folds.
        train_df = training_rows_without_label_overlap(
            df,
            date_column="date",
            train_end_before_purge=split["train_end_before_purge"],
            test_start=split["test_start"],
            label_horizon_days=ML_EMBARGO_DAYS,
        )
        test_df = df[
            (df["date"] >= split["test_start"]) & (df["date"] <= split["test_end"])
        ].copy()
        if train_df.empty or test_df.empty:
            continue

        assert_no_label_overlap(
            train_df["date"],
            test_start=split["test_start"],
            label_horizon_days=ML_EMBARGO_DAYS,
        )

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
    # Month-end rebalances with month-end→next-month-end holding returns so
    # overlapping daily 21-day labels are not compounded as independent periods.
    holding_returns = month_end_holding_period_returns(load_price_data())
    results = build_prediction_long_short_returns(
        predictions,
        score_column=PREDICTION_COLUMN,
        target_column=TARGET,
        long_count=ML_LONG_COUNT,
        short_count=ML_SHORT_COUNT,
        transaction_cost=transaction_cost,
        realized_returns=holding_returns,
        rebalance_on_month_ends=True,
        gross_exposure=RESEARCH_LONG_SHORT_GROSS_EXPOSURE,
    )
    returns = results["portfolio_return"]
    return results, equity_curve(returns), calculate_performance_metrics(returns)
