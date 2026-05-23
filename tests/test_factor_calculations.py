import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.factors import calculate_factor_frames


def test_factor_frames_use_expected_return_formulas() -> None:
    dates = pd.date_range("2024-01-01", periods=280, freq="D")
    prices = pd.DataFrame(
        {
            "AAA": [100 + index for index in range(len(dates))],
            "BBB": [200 + 2 * index for index in range(len(dates))],
        },
        index=dates,
    )

    factors = calculate_factor_frames(prices)

    assert_frame_equal(factors["daily_returns"], prices.pct_change())
    assert_frame_equal(factors["momentum_12m"], prices / prices.shift(252) - 1)
    assert_frame_equal(factors["mean_reversion_1m"], -(prices / prices.shift(21) - 1))
    assert_frame_equal(factors["forward_returns_1m"], prices.shift(-21) / prices - 1)


def test_forward_returns_are_future_labels_not_current_returns() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    prices = pd.DataFrame({"AAA": range(100, 130)}, index=dates)

    factors = calculate_factor_frames(prices)

    assert factors["forward_returns_1m"].loc[dates[0], "AAA"] == prices.loc[dates[21], "AAA"] / prices.loc[dates[0], "AAA"] - 1
    assert pd.isna(factors["forward_returns_1m"].loc[dates[-1], "AAA"])
