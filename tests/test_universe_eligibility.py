"""Survivorship / PIT universe eligibility tests."""

from pathlib import Path

import pandas as pd

from src.config import (
    MIN_UNIVERSE_HISTORY_DAYS,
    REQUIRE_INDEX_MEMBERSHIP,
    UNIVERSE_MEMBERSHIP_PATH,
)
from src.data.universe import (
    build_eligibility_mask,
    build_membership_mask,
    build_tradability_mask,
    detect_delist_dates,
    drop_empty_price_columns,
    load_membership_table,
)
from src.portfolio.portfolio import build_multi_factor_long_short_returns


def test_drop_empty_price_columns() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame(
        {"KEEP": [1.0, 2.0, 3.0, 4.0, 5.0], "DEAD": [None, None, None, None, None]},
        index=dates,
    )
    cleaned = drop_empty_price_columns(prices)
    assert list(cleaned.columns) == ["KEEP"]


def test_detect_delist_and_tradability() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    prices = pd.DataFrame(
        {
            "LIVE": [10.0] * 10,
            "IPO": [None, None, None, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6],
            "DEAD": [8.0, 8.1, 8.2, 8.3, None, None, None, None, None, None],
        },
        index=dates,
    )
    delisted = detect_delist_dates(prices)
    assert pd.isna(delisted["LIVE"])
    assert delisted["DEAD"] == dates[4]

    tradable = build_tradability_mask(prices, min_history_days=3)
    # LIVE becomes tradable once 3 valid prints exist.
    assert not bool(tradable.iloc[1]["LIVE"])
    assert bool(tradable.iloc[2]["LIVE"])
    # IPO needs 3 valid prints after listing.
    assert not bool(tradable.loc[dates[4], "IPO"])
    assert bool(tradable.loc[dates[6], "IPO"])
    # Dead name is not tradable on/after delist date.
    assert not bool(tradable.loc[dates[4], "DEAD"])
    assert not bool(tradable.loc[dates[-1], "DEAD"])


def test_membership_asof_excludes_pre_join_names() -> None:
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    membership = pd.DataFrame(
        {
            "date": [dates[1], dates[1], dates[2], dates[2]],
            "ticker": ["AAPL", "MSFT", "AAPL", "PLTR"],
        }
    )
    mask = build_membership_mask(dates, ["AAPL", "MSFT", "PLTR"], membership=membership)
    assert not bool(mask.loc[dates[0], "AAPL"])
    assert bool(mask.loc[dates[1], "AAPL"])
    assert bool(mask.loc[dates[1], "MSFT"])
    assert not bool(mask.loc[dates[1], "PLTR"])
    assert bool(mask.loc[dates[2], "PLTR"])


def test_eligibility_combines_tradability_and_membership() -> None:
    dates = pd.date_range("2020-01-02", periods=80, freq="B")
    prices = pd.DataFrame(
        {
            "AAPL": 100.0,
            "PLTR": [None] * 40 + [20.0] * 40,
        },
        index=dates,
    )
    membership = pd.DataFrame(
        {
            "date": [dates[50], dates[50], dates[70], dates[70]],
            "ticker": ["AAPL", "MSFT", "AAPL", "PLTR"],
        }
    )
    eligible = build_eligibility_mask(
        prices,
        membership=membership,
        min_history_days=5,
        require_index_membership=True,
    )
    # AAPL has history but is not a member before first membership snapshot.
    assert not bool(eligible.loc[dates[10], "AAPL"])
    assert bool(eligible.loc[dates[55], "AAPL"])
    # PLTR listed earlier than membership → still excluded until member snapshot.
    assert not bool(eligible.loc[dates[55], "PLTR"])
    assert bool(eligible.loc[dates[75], "PLTR"])


def test_portfolio_skips_ineligible_names() -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    momentum = pd.DataFrame(1.0, index=dates, columns=tickers)
    momentum.loc[dates[1], "AAA"] = 5.0
    mean_reversion = pd.DataFrame(0.0, index=dates, columns=tickers)
    forward = pd.DataFrame(0.01, index=dates, columns=tickers)
    eligibility = pd.DataFrame(True, index=dates, columns=tickers)
    eligibility.loc[dates[1], "AAA"] = False  # top momentum name blocked

    results = build_multi_factor_long_short_returns(
        momentum,
        mean_reversion,
        forward,
        long_count=2,
        short_count=2,
        min_assets=4,
        transaction_cost=0.0,
        eligibility=eligibility,
    )
    assert not results.empty


def test_checked_in_membership_file_covers_late_joiners() -> None:
    assert Path(UNIVERSE_MEMBERSHIP_PATH).exists()
    membership = load_membership_table()
    assert REQUIRE_INDEX_MEMBERSHIP is True
    assert MIN_UNIVERSE_HISTORY_DAYS >= 1
    assert not membership.empty
    pltr = membership.loc[membership["ticker"] == "PLTR", "date"]
    assert not pltr.empty
    assert pltr.min() >= pd.Timestamp("2024-09-01")
    uber = membership.loc[membership["ticker"] == "UBER", "date"]
    assert uber.min() >= pd.Timestamp("2023-12-01")
