"""NYSE holiday calendar used by the freshness/alignment checks."""

from datetime import date

import pandas as pd
import pytest

from src.data.market_calendar import (
    easter_sunday,
    holidays_between,
    is_trading_day,
    nyse_holidays,
    trading_days,
    trading_days_between,
)
from src.data.quality import DEFAULT_DATA_CHECKS, check_freshness, load_dated_csv

S = dict(DEFAULT_DATA_CHECKS)


@pytest.mark.parametrize("day, name", [
    ("2024-11-28", "Thanksgiving Day"),
    ("2025-11-27", "Thanksgiving Day"),
    ("2026-11-26", "Thanksgiving Day"),
    ("2024-03-29", "Good Friday"),
    ("2025-04-18", "Good Friday"),
    ("2026-04-03", "Good Friday"),
    ("2026-01-19", "Martin Luther King Jr. Day"),
    ("2026-02-16", "Washington's Birthday"),
    ("2026-05-25", "Memorial Day"),
    ("2026-09-07", "Labor Day"),
    ("2022-06-20", "Juneteenth"),          # Sunday June 19 -> Monday
    ("2026-07-03", "Independence Day"),    # Saturday July 4 -> Friday
    ("2022-12-26", "Christmas Day"),       # Sunday -> Monday
    ("2023-01-02", "New Year's Day"),      # Sunday -> Monday
    ("2018-12-05", "National Day of Mourning (George H. W. Bush)"),
    ("2025-01-09", "National Day of Mourning (Jimmy Carter)"),
])
def test_known_nyse_holidays(day, name) -> None:
    d = pd.Timestamp(day).date()
    assert nyse_holidays(d.year)[d] == name
    assert not is_trading_day(d)


def test_observance_edge_cases() -> None:
    assert is_trading_day("2021-12-31")  # Saturday New Year's Day 2022 is not observed on Friday
    assert date(2022, 1, 1) not in nyse_holidays(2022)
    assert is_trading_day("2021-06-18")  # Juneteenth only from 2022
    assert is_trading_day("2026-11-27")  # day after Thanksgiving: early close, still a trading day
    assert not is_trading_day("2026-11-07")  # Saturday
    assert easter_sunday(2026) == date(2026, 4, 5) and easter_sunday(2024) == date(2024, 3, 31)
    assert len(nyse_holidays(2026)) == 10


def test_trading_days_between_skips_holidays_and_weekends() -> None:
    # Wed before Thanksgiving -> Mon: Thu is a holiday, so only Fri and Mon count.
    assert trading_days_between("2025-11-26", "2025-12-01") == 2
    assert trading_days_between("2025-11-26", "2025-12-01", calendar="weekdays") == 3
    # Thu before Good Friday -> Mon after Easter: only Monday.
    assert trading_days_between("2026-04-02", "2026-04-06") == 1
    assert trading_days_between("2026-10-30", "2026-11-09") == 6  # no holiday in between
    assert trading_days_between("2026-11-09", "2026-11-02") == 0
    with pytest.raises(ValueError):
        holidays_between("2026-01-01", "2026-12-31", calendar="LSE")


def test_freshness_does_not_count_holidays_as_missed_days() -> None:
    dates = trading_days("2025-10-01", "2025-11-26")
    panel = pd.DataFrame({"AAA": range(len(dates))}, index=dates, dtype=float)
    # Last bar Wed 2025-11-26; as of Thu 2025-12-04 that is Fri, Mon, Tue, Wed, Thu = 5 trading days.
    assert check_freshness({"price_data": panel}, "2025-12-04", S) == []
    stale_weekdays = check_freshness({"price_data": panel}, "2025-12-04", {**S, "calendar": "weekdays"})
    assert len(stale_weekdays) == 1 and "6 trading days" in stale_weekdays[0].message
    stale = check_freshness({"price_data": panel}, "2025-12-05", S)
    assert len(stale) == 1 and "6 trading days" in stale[0].message


def test_calendar_matches_checked_in_price_history() -> None:
    prices = load_dated_csv("data/price_data.csv")
    expected = trading_days(prices.index.min(), prices.index.max())
    assert prices.index.difference(expected).empty, "price rows on NYSE holidays/weekends"
    assert expected.difference(prices.index).empty, "NYSE trading days missing from price history"
