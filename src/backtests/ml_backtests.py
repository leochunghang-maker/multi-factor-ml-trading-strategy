from pathlib import Path

import pandas as pd

from src.config import (
    FEATURE_STABILITY_REPORT_PATH,
    FEATURES,
    ML_EMBARGO_DAYS,
    ML_LONG_COUNT,
    ML_PREDICTIONS_PATH,
    ML_SHORT_COUNT,
    NESTED_MIN_TRAIN_YEARS,
    NESTED_SELECTION_REPORT_PATH,
    NESTED_INNER_SCORE,
    NESTED_USE_MONTH_END_SUBSET,
    NESTED_WALK_FORWARD,
    PREDICTION_COLUMN,
    RESEARCH_LONG_SHORT_GROSS_EXPOSURE,
    WALK_FORWARD_DECILE_REPORT_PATH,
    TARGET,
    TRANSACTION_COST,
    WALK_FORWARD_TEST_YEARS,
)
from src.data.io import load_prediction_dataset, load_ml_dataset, load_price_data
from src.data.universe import load_default_eligibility_mask
from src.models.validation import (
    assert_no_label_overlap,
    decile_spread_analysis,
    feature_stability_report,
    purged_embargo_year_splits,
    training_rows_without_label_overlap,
)
from src.models.ml import train_random_forest
from src.models.nested import default_nested_candidates, select_model_nested
from src.execution.borrow_tiers import research_borrow_bps_panel
from src.portfolio import (
    build_prediction_long_short_returns,
    month_end_holding_period_returns,
)
from src.reporting.metrics import calculate_performance_metrics, equity_curve


def run_prediction_backtest(
    predictions_path: str = ML_PREDICTIONS_PATH,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    predictions = load_prediction_dataset(predictions_path)
    prices = load_price_data()
    holding_returns = month_end_holding_period_returns(prices)
    eligibility = load_default_eligibility_mask(prices)
    borrow_panel = research_borrow_bps_panel(prices)
    results = build_prediction_long_short_returns(
        predictions,
        score_column=PREDICTION_COLUMN,
        target_column=TARGET,
        long_count=ML_LONG_COUNT,
        short_count=ML_SHORT_COUNT,
        realized_returns=holding_returns,
        rebalance_on_month_ends=True,
        gross_exposure=RESEARCH_LONG_SHORT_GROSS_EXPOSURE,
        eligibility=eligibility,
        borrow_bps_panel=borrow_panel,
    )
    returns = results["portfolio_return"]
    return results, equity_curve(returns), calculate_performance_metrics(returns)


def run_walk_forward_ml_backtest(
    transaction_cost: float = TRANSACTION_COST,
    test_years: list[int] | None = None,
    nested: bool | None = None,
    nested_candidates: list | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    df = load_ml_dataset().sort_values("date")
    test_years = test_years or WALK_FORWARD_TEST_YEARS
    use_nested = NESTED_WALK_FORWARD if nested is None else bool(nested)
    yearly_predictions = []
    selection_rows: list[dict] = []

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
        # Nested selection may only see purged training rows; never the outer test.
        assert train_df["date"].max() < split["test_start"]

        if use_nested:
            model, meta = select_model_nested(
                train_df,
                feature_columns=FEATURES,
                target_column=TARGET,
                candidates=nested_candidates or default_nested_candidates(),
                label_horizon_days=ML_EMBARGO_DAYS,
                min_train_years=NESTED_MIN_TRAIN_YEARS,
                use_month_end_subset=NESTED_USE_MONTH_END_SUBSET,
                outer_test_year=test_year,
                score_metric=NESTED_INNER_SCORE,
            )
            print(
                f"  nested choice for {test_year}: {meta['chosen_model']} "
                f"({meta['selection_mode']})"
            )
            flat = {
                "outer_test_year": meta["outer_test_year"],
                "chosen_model": meta["chosen_model"],
                "selection_mode": meta["selection_mode"],
                "score_metric": meta["score_metric"],
                "n_train_rows": meta["n_train_rows"],
                "train_start": meta["train_start"],
                "train_end": meta["train_end"],
            }
            for score_row in meta["candidate_scores"]:
                name = score_row["name"]
                flat[f"score_{name}"] = score_row["inner_score"]
                flat[f"ic_{name}"] = score_row["mean_inner_rank_ic"]
                flat[f"icir_{name}"] = score_row["inner_icir"]
                flat[f"cost_sharpe_{name}"] = score_row["inner_cost_aware_sharpe"]
            selection_rows.append(flat)
        else:
            model = train_random_forest(train_df[FEATURES], train_df[TARGET])

        test_df[PREDICTION_COLUMN] = model.predict(test_df[FEATURES])
        yearly_predictions.append(test_df)

    predictions = pd.concat(yearly_predictions, ignore_index=True)
    if selection_rows:
        Path(NESTED_SELECTION_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(selection_rows).to_csv(NESTED_SELECTION_REPORT_PATH, index=False)
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
    prices = load_price_data()
    holding_returns = month_end_holding_period_returns(prices)
    eligibility = load_default_eligibility_mask(prices)
    borrow_panel = research_borrow_bps_panel(prices)
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
        eligibility=eligibility,
        borrow_bps_panel=borrow_panel,
    )
    returns = results["portfolio_return"]
    return results, equity_curve(returns), calculate_performance_metrics(returns)
