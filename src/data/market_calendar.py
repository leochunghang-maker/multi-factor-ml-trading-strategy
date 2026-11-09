"""NYSE trading calendar (full-day closures) for freshness and alignment checks.

Rule-based, dependency-free: regular holidays follow NYSE Rule 7.2 observance
(Saturday -> Friday, Sunday -> Monday; New Year's Day on a Saturday is not
observed), plus a checked-in list of one-off closures. Early closes (half days)
are trading days and are not modelled. Verified against the checked-in price
history (2018-01-02 .. 2026-10-30) in tests/test_market_calendar.py.

``last_completed_session`` maps a wall-clock instant to the latest session whose
regular close (16:00 America/New_York) plus a settle buffer has passed. Early
closes are treated as 16:00 closes, which only delays a half day's bar. Paper
refreshes and the data gate use it so a run during US market hours never treats a
partial intraday bar as that day's close.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Unscheduled full-day closures (national days of mourning, weather, 9/11).
SPECIAL_CLOSURES = {
    date(2001, 9, 11): "September 11 attacks",
    date(2001, 9, 12): "September 11 attacks",
    date(2001, 9, 13): "September 11 attacks",
    date(2001, 9, 14): "September 11 attacks",
    date(2004, 6, 11): "National Day of Mourning (Ronald Reagan)",
    date(2007, 1, 2): "National Day of Mourning (Gerald Ford)",
    date(2012, 10, 29): "Hurricane Sandy",
    date(2012, 10, 30): "Hurricane Sandy",
    date(2018, 12, 5): "National Day of Mourning (George H. W. Bush)",
    date(2025, 1, 9): "National Day of Mourning (Jimmy Carter)",
}

CALENDARS = ("NYSE", "weekdays")

EXCHANGE_TZ = ZoneInfo("America/New_York")
REGULAR_CLOSE = time(16, 0)


def easter_sunday(year: int) -> date:
    """Gregorian Easter (anonymous Gregorian / Meeus algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741 - name from the published algorithm
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = (date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=None)
def nyse_holidays(year: int) -> dict[date, str]:
    """Full-day NYSE closures in ``year`` (weekday dates only) -> name."""
    mon, thu = 0, 3
    days: dict[date, str] = {}
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:  # Saturday New Year's Day is not observed on Dec 31
        days[_observed(new_year)] = "New Year's Day"
    if year >= 1998:
        days[_nth_weekday(year, 1, mon, 3)] = "Martin Luther King Jr. Day"
    days[_nth_weekday(year, 2, mon, 3)] = "Washington's Birthday"
    days[easter_sunday(year) - timedelta(days=2)] = "Good Friday"
    days[_last_weekday(year, 5, mon)] = "Memorial Day"
    if year >= 2022:
        days[_observed(date(year, 6, 19))] = "Juneteenth"
    days[_observed(date(year, 7, 4))] = "Independence Day"
    days[_nth_weekday(year, 9, mon, 1)] = "Labor Day"
    days[_nth_weekday(year, 11, thu, 4)] = "Thanksgiving Day"
    days[_observed(date(year, 12, 25))] = "Christmas Day"
    days.update({d: name for d, name in SPECIAL_CLOSURES.items() if d.year == year})
    return {d: name for d, name in sorted(days.items()) if d.weekday() < 5}


def holidays_between(start, end, calendar: str = "NYSE") -> list[date]:
    """Exchange holidays with ``start <= day <= end`` (empty for the weekday calendar)."""
    if calendar not in CALENDARS:
        raise ValueError(f"unknown calendar {calendar!r}; expected one of {CALENDARS}")
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if calendar == "weekdays" or end < start:
        return []
    return [d for y in range(start.year, end.year + 1) for d in nyse_holidays(y) if start <= d <= end]


def is_trading_day(day, calendar: str = "NYSE") -> bool:
    d = pd.Timestamp(day).date()
    return d.weekday() < 5 and not holidays_between(d, d, calendar)


def trading_days_between(start, end, calendar: str = "NYSE") -> int:
    """Trading days strictly after ``start`` up to and including ``end``."""
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if end <= start:
        return 0
    lo, hi = start + timedelta(days=1), end + timedelta(days=1)
    return int(np.busday_count(lo, hi, holidays=holidays_between(lo, end, calendar)))


def trading_days(start, end, calendar: str = "NYSE") -> pd.DatetimeIndex:
    """All trading days in ``[start, end]``."""
    days = pd.bdate_range(start, end)
    closed = set(holidays_between(start, end, calendar))
    return pd.DatetimeIndex([d for d in days if d.date() not in closed])


def previous_trading_day(day, calendar: str = "NYSE") -> pd.Timestamp:
    """The last trading day strictly before ``day``."""
    d = pd.Timestamp(day).date() - timedelta(days=1)
    while not is_trading_day(d, calendar):
        d -= timedelta(days=1)
    return pd.Timestamp(d)


def last_completed_session(now=None, calendar: str = "NYSE", settle_minutes: int = 30) -> pd.Timestamp:
    """Latest trading day whose 16:00 New York close + ``settle_minutes`` is at or before ``now``.

    ``now`` is a timezone-aware datetime/Timestamp (naive values are read as UTC);
    default is the current wall clock. The date is taken in New York, so a run from
    Hong Kong at 01:00 on Tuesday (12:00 Monday in New York) returns the previous
    session, not Monday or Tuesday.
    """
    stamp = pd.Timestamp(now if now is not None else datetime.now(UTC))
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    local = stamp.tz_convert(EXCHANGE_TZ)
    today = local.date()
    cutoff = datetime.combine(today, REGULAR_CLOSE, tzinfo=EXCHANGE_TZ) + timedelta(minutes=int(settle_minutes))
    if is_trading_day(today, calendar) and local.to_pydatetime() >= cutoff:
        return pd.Timestamp(today)
    return previous_trading_day(today, calendar)
