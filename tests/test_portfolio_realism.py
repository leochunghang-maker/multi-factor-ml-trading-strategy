import pandas as pd

from src.portfolio import (
    build_factor_long_short_returns,
    month_end_trading_dates,
    transaction_cost_from_weight_change,
)


def test_month_end_trading_dates_uses_last_available_observation() -> None:
    dates = pd.to_datetime(["2024-01-30", "2024-02-28", "2024-03-29"])
    frame = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0]}, index=dates)

    selected = month_end_trading_dates(frame)

    assert list(selected) == list(dates)


def test_transaction_cost_depends_on_weight_change() -> None:
    current = pd.Series({"AAPL": 0.5, "MSFT": -0.5})
    previous = pd.Series({"AAPL": 0.5, "MSFT": -0.5})

    cost, traded_gross_exposure = transaction_cost_from_weight_change(
        current,
        previous,
        transaction_cost=0.001,
    )

    assert cost == 0.0
    assert traded_gross_exposure == 0.0


def test_factor_backtest_does_not_skip_non_calendar_month_end() -> None:
    dates = pd.to_datetime(["2024-01-30", "2024-02-28"])
    scores = pd.DataFrame(
        {
            "A": [3.0, 3.0],
            "B": [2.0, 2.0],
            "C": [1.0, 1.0],
            "D": [0.0, 0.0],
        },
        index=dates,
    )
    forward_returns = pd.DataFrame(
        {
            "A": [0.10, 0.10],
            "B": [0.05, 0.05],
            "C": [-0.01, -0.01],
            "D": [-0.02, -0.02],
        },
        index=dates,
    )

    results = build_factor_long_short_returns(
        scores,
        forward_returns,
        long_count=1,
        short_count=1,
        min_assets=4,
        transaction_cost=0.001,
    )

    assert list(results.index) == list(dates)
    assert results.loc[dates[0], "traded_gross_exposure"] == 2.0
    assert results.loc[dates[1], "traded_gross_exposure"] == 0.0
