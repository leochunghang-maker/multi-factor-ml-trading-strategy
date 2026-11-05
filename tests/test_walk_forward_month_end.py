"""Guards against compounding overlapping daily forward returns as independent P&L."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio import (
    build_prediction_long_short_returns,
    month_end_holding_period_returns,
    month_end_trading_dates,
)


def _daily_predictions_with_overlapping_labels() -> pd.DataFrame:
    """Synthetic daily panel: every day shares a correlated 21-day forward label."""
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    rows = []
    rng = np.random.default_rng(0)
    for date in dates:
        # Overlapping labels: same shock reused across nearby days.
        shared = float(rng.normal(0.01, 0.02))
        for i, ticker in enumerate(tickers):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "predicted_return": float(i) + shared,
                    "target_return_1m": shared + 0.001 * i,
                }
            )
    return pd.DataFrame(rows)


def test_prediction_portfolio_rebalances_only_on_month_ends() -> None:
    predictions = _daily_predictions_with_overlapping_labels()
    prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 110, len(predictions["date"].unique())),
            "BBB": np.linspace(100, 105, len(predictions["date"].unique())),
            "CCC": np.linspace(100, 102, len(predictions["date"].unique())),
            "DDD": np.linspace(100, 101, len(predictions["date"].unique())),
        },
        index=pd.DatetimeIndex(sorted(predictions["date"].unique())),
    )
    holding = month_end_holding_period_returns(prices)

    results = build_prediction_long_short_returns(
        predictions,
        score_column="predicted_return",
        target_column="target_return_1m",
        long_count=1,
        short_count=1,
        realized_returns=holding,
        rebalance_on_month_ends=True,
    )

    expected_month_ends = month_end_trading_dates(prices)
    # Last month-end has no next-month holding return, so it is dropped.
    assert list(results.index) == list(expected_month_ends[:-1])
    assert len(results) < predictions["date"].nunique()
    assert len(results) == expected_month_ends.to_period("M").nunique() - 1


def test_daily_overlapping_forward_returns_are_not_independent_periods() -> None:
    """Fail if someone treats every daily overlapping forward label as a period return."""
    predictions = _daily_predictions_with_overlapping_labels()

    naive = build_prediction_long_short_returns(
        predictions,
        score_column="predicted_return",
        target_column="target_return_1m",
        long_count=1,
        short_count=1,
        rebalance_on_month_ends=False,
    )
    # Naive path keeps nearly every trading day as an independent "period".
    assert len(naive) == predictions["date"].nunique()
    assert naive.index.to_period("M").nunique() < len(naive)

    prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 110, len(predictions["date"].unique())),
            "BBB": np.linspace(100, 105, len(predictions["date"].unique())),
            "CCC": np.linspace(100, 102, len(predictions["date"].unique())),
            "DDD": np.linspace(100, 101, len(predictions["date"].unique())),
        },
        index=pd.DatetimeIndex(sorted(predictions["date"].unique())),
    )
    holding = month_end_holding_period_returns(prices)
    corrected = build_prediction_long_short_returns(
        predictions,
        score_column="predicted_return",
        target_column="target_return_1m",
        long_count=1,
        short_count=1,
        realized_returns=holding,
        rebalance_on_month_ends=True,
    )

    # Corrected path must not keep overlapping daily periods.
    assert len(corrected) < len(naive) / 5
    assert corrected.index.to_period("M").nunique() == len(corrected)
