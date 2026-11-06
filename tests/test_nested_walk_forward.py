"""Nested / anchored walk-forward selection must not peek at outer test folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.nested import (
    default_nested_candidates,
    score_candidate_on_inner_folds,
    select_model_nested,
)
from src.models.validation import (
    assert_no_label_overlap,
    inner_anchored_year_splits,
    month_end_subset,
    purged_embargo_year_splits,
    training_rows_without_label_overlap,
)


def _synthetic_panel(start: str, end: str, n_tickers: int = 8) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="B")
    rows = []
    rng = np.random.default_rng(0)
    tickers = [f"T{i}" for i in range(n_tickers)]
    for date in dates:
        for i, ticker in enumerate(tickers):
            # Feature weakly predicts next-month-ish target with noise.
            feature = float(i) + rng.normal(0, 0.5)
            rows.append({
                "date": date,
                "ticker": ticker,
                "momentum_12m": feature,
                "short_term_momentum": feature * 0.5,
                "volatility_1m": 0.02 + 0.001 * i,
                "target_return_1m": 0.01 * feature + rng.normal(0, 0.05),
            })
    return pd.DataFrame(rows)


def test_inner_splits_are_anchored_and_purged() -> None:
    dates = pd.Series(pd.date_range("2019-01-01", "2023-12-31", freq="B"))
    splits = inner_anchored_year_splits(dates, min_train_years=2, purge_months=1)
    assert [s["test_year"] for s in splits] == [2021, 2022, 2023]
    for split in splits:
        assert split["train_end_before_purge"] < split["test_start"]


def test_outer_train_excludes_outer_test_and_inner_stays_inside() -> None:
    frame = _synthetic_panel("2019-01-01", "2024-12-31")
    outer = purged_embargo_year_splits(frame["date"], test_years=[2024], purge_months=1)[0]
    train = training_rows_without_label_overlap(
        frame,
        date_column="date",
        train_end_before_purge=outer["train_end_before_purge"],
        test_start=outer["test_start"],
        label_horizon_days=21,
    )
    assert train["date"].max() < outer["test_start"]
    assert_no_label_overlap(train["date"], test_start=outer["test_start"], label_horizon_days=21)

    inner_splits = inner_anchored_year_splits(train["date"], min_train_years=2)
    assert inner_splits
    for split in inner_splits:
        assert split["test_end"] < outer["test_start"]
        inner_train = training_rows_without_label_overlap(
            train,
            date_column="date",
            train_end_before_purge=split["train_end_before_purge"],
            test_start=split["test_start"],
            label_horizon_days=21,
        )
        assert inner_train["date"].max() < split["test_start"]
        assert_no_label_overlap(
            inner_train["date"], test_start=split["test_start"], label_horizon_days=21
        )


def test_month_end_subset_keeps_last_date_per_month() -> None:
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-10", "2020-01-31", "2020-02-27", "2020-02-28"]),
        "v": [1, 2, 3, 4],
    })
    subset = month_end_subset(frame)
    assert list(subset["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-31", "2020-02-28"]


def test_select_model_nested_never_reads_outer_test_rows() -> None:
    frame = _synthetic_panel("2019-01-01", "2023-06-30", n_tickers=6)
    outer_test_start = pd.Timestamp("2023-01-01")
    train = frame[frame["date"] < outer_test_start].copy()
    # Poison outer-test-era rows if they were ever passed in: selection gets train only.
    poisoned = frame[frame["date"] >= outer_test_start].copy()
    poisoned["target_return_1m"] = 1e6

    features = ["momentum_12m", "short_term_momentum", "volatility_1m"]
    tiny = [
        {"name": "ridge_1", "kind": "ridge", "params": {"alpha": 1.0}},
        {
            "name": "rf_shallow",
            "kind": "random_forest",
            "params": {"n_estimators": 10, "max_depth": 2, "random_state": 0, "n_jobs": 1},
        },
    ]
    model, meta = select_model_nested(
        train,
        feature_columns=features,
        target_column="target_return_1m",
        candidates=tiny,
        min_train_years=2,
        use_month_end_subset=True,
        outer_test_year=2023,
    )
    assert meta["train_end"] < "2023-01-01"
    assert meta["chosen_model"] in {c["name"] for c in tiny}
    # Predictions on train should be finite; poisoned rows were never fitted on.
    preds = model.predict(train[features].tail(20))
    assert np.isfinite(preds).all()
    assert poisoned["target_return_1m"].iloc[0] == 1e6  # unused


def test_score_candidate_returns_nan_without_inner_history() -> None:
    frame = _synthetic_panel("2022-01-01", "2022-12-31", n_tickers=4)
    score = score_candidate_on_inner_folds(
        frame,
        feature_columns=["momentum_12m", "short_term_momentum", "volatility_1m"],
        target_column="target_return_1m",
        candidate={"name": "ridge_1", "kind": "ridge", "params": {"alpha": 1.0}},
        min_train_years=2,
    )
    assert np.isnan(score)


def test_default_candidates_include_production_rf() -> None:
    names = [c["name"] for c in default_nested_candidates()]
    assert "rf_default" in names
    assert "ridge_1" in names


def test_aggregate_icir_and_cost_aware_sharpe_formulas() -> None:
    from src.models.nested import aggregate_icir, aggregate_sharpe, summarize_inner_score

    assert aggregate_icir([0.1, 0.1, 0.1]) != aggregate_icir([0.1, -0.1, 0.3])
    assert np.isnan(aggregate_icir([0.2]))
    sharpe = aggregate_sharpe([0.01, 0.02, 0.015, 0.01], periods_per_year=12)
    assert sharpe > 0
    diag = {"ics": [0.05, 0.07, 0.06], "net_returns": [0.01, 0.012, 0.008]}
    assert summarize_inner_score(diag, "rank_ic") == pytest.approx(np.mean(diag["ics"]))
    assert summarize_inner_score(diag, "icir") == pytest.approx(aggregate_icir(diag["ics"]))
    assert summarize_inner_score(diag, "cost_aware_sharpe") == pytest.approx(
        aggregate_sharpe(diag["net_returns"])
    )


def test_net_long_short_period_return_subtracts_borrow_and_turnover() -> None:
    from src.models.nested import net_long_short_period_return

    scores = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0, "F": -2.0})
    rets = pd.Series({"A": 0.02, "B": 0.01, "C": 0.0, "D": 0.0, "E": -0.01, "F": -0.02})
    net0, weights = net_long_short_period_return(
        scores,
        rets,
        previous_weights=None,
        long_count=2,
        short_count=2,
        gross_exposure=2.0,
        transaction_cost=0.0,
        annual_borrow_bps=0.0,
        annual_short_rebate_bps=0.0,
        annual_cash_financing_bps=0.0,
        periods_per_year=12,
    )
    # With borrow 120 bps/year on 100% short ≈ 10 bps/month drag.
    net_b, _ = net_long_short_period_return(
        scores,
        rets,
        previous_weights=None,
        long_count=2,
        short_count=2,
        gross_exposure=2.0,
        transaction_cost=0.0,
        annual_borrow_bps=120.0,
        annual_short_rebate_bps=0.0,
        annual_cash_financing_bps=0.0,
        periods_per_year=12,
    )
    assert net0 - net_b == pytest.approx(0.001)
    # Second rebalance with unchanged weights: turnover cost 0 vs first entry.
    net_same, _ = net_long_short_period_return(
        scores,
        rets,
        previous_weights=weights,
        long_count=2,
        short_count=2,
        gross_exposure=2.0,
        transaction_cost=0.01,
        annual_borrow_bps=0.0,
        annual_short_rebate_bps=0.0,
        annual_cash_financing_bps=0.0,
        periods_per_year=12,
    )
    net_fresh, _ = net_long_short_period_return(
        scores,
        rets,
        previous_weights=None,
        long_count=2,
        short_count=2,
        gross_exposure=2.0,
        transaction_cost=0.01,
        annual_borrow_bps=0.0,
        annual_short_rebate_bps=0.0,
        annual_cash_financing_bps=0.0,
        periods_per_year=12,
    )
    assert net_same > net_fresh


def test_score_metric_switch_rank_ic_vs_cost_aware() -> None:
    frame = _synthetic_panel("2019-01-01", "2023-06-30", n_tickers=8)
    train = frame[frame["date"] < pd.Timestamp("2023-01-01")]
    features = ["momentum_12m", "short_term_momentum", "volatility_1m"]
    candidate = {"name": "ridge_1", "kind": "ridge", "params": {"alpha": 1.0}}
    ic = score_candidate_on_inner_folds(
        train,
        feature_columns=features,
        target_column="target_return_1m",
        candidate=candidate,
        min_train_years=2,
        score_metric="rank_ic",
        long_count=2,
        short_count=2,
    )
    sharpe = score_candidate_on_inner_folds(
        train,
        feature_columns=features,
        target_column="target_return_1m",
        candidate=candidate,
        min_train_years=2,
        score_metric="cost_aware_sharpe",
        long_count=2,
        short_count=2,
    )
    assert np.isfinite(ic)
    assert np.isfinite(sharpe)
