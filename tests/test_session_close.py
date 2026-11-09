"""Paper data must never treat a partial intraday bar as a close.

A run during US market hours (e.g. 01:00 Tuesday in Hong Kong is 12:00 Monday in
New York) used to download Monday's half-finished bar, fill and mark the book at
it, and key the run by Monday so a rerun after the close was skipped.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.market_calendar import last_completed_session, previous_trading_day
from src.data.quality import run_data_checks
from src.data_loader import resolve_end_date
from src.operations.paper_ops import paper_data_gate

NY = "America/New_York"
GATE_CONFIG = {"data_checks": {"manifest_files": [], "manifest_path": "data/data_manifest.json"}}


def _ny(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=NY)


@pytest.mark.parametrize("now, expected", [
    (_ny("2026-11-09 12:10"), "2026-11-06"),   # Monday mid-session -> Friday
    (_ny("2026-11-09 16:15"), "2026-11-06"),   # closed, inside the 30-min settle buffer
    (_ny("2026-11-09 16:30"), "2026-11-09"),   # settle buffer passed
    (_ny("2026-11-10 08:00"), "2026-11-09"),   # pre-open next day
    (_ny("2026-11-08 12:00"), "2026-11-06"),   # Sunday
    (_ny("2026-11-26 18:00"), "2026-11-25"),   # Thanksgiving (closed all day)
    (_ny("2026-11-27 17:00"), "2026-11-27"),   # day after Thanksgiving (early close, read as 16:00)
    (_ny("2026-01-02 09:00"), "2025-12-31"),   # pre-open after New Year's Day
])
def test_last_completed_session(now, expected) -> None:
    assert last_completed_session(now) == pd.Timestamp(expected)


def test_last_completed_session_reads_the_new_york_date_not_the_local_one() -> None:
    hong_kong = pd.Timestamp("2026-11-10 01:08", tz="Asia/Hong_Kong")  # 12:08 Mon in New York
    assert last_completed_session(hong_kong) == pd.Timestamp("2026-11-06")
    assert last_completed_session(pd.Timestamp("2026-11-09 17:08")) == pd.Timestamp("2026-11-06")  # naive = UTC
    assert last_completed_session(_ny("2026-11-09 16:05"), settle_minutes=0) == pd.Timestamp("2026-11-09")


def test_previous_trading_day_skips_weekends_and_holidays() -> None:
    assert previous_trading_day("2026-11-09") == pd.Timestamp("2026-11-06")
    assert previous_trading_day("2026-11-27") == pd.Timestamp("2026-11-25")


def test_paper_refresh_end_excludes_the_open_session() -> None:
    assert resolve_end_date("today", now=_ny("2026-11-09 12:10")) == "2026-11-07"  # exclusive: through Fri
    assert resolve_end_date("today", now=_ny("2026-11-09 17:00")) == "2026-11-10"  # through Mon
    assert resolve_end_date("2026-10-31") == "2026-10-31"


def _write_data(root, end: str) -> None:
    dates = pd.bdate_range(end=end, periods=40)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (40, 2)), axis=0)),
                          index=dates, columns=["AAA", "BBB"])
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    prices.to_csv(data / "price_data.csv")
    pd.DataFrame(1e6, index=dates, columns=["AAA", "BBB"]).to_csv(data / "volume_data.csv")
    prices[["AAA"]].rename(columns={"AAA": "SPY"}).to_csv(data / "benchmark_prices.csv")


def test_gate_blocks_a_partial_intraday_bar(tmp_path) -> None:
    _write_data(tmp_path, "2026-11-09")
    gate = paper_data_gate(config=GATE_CONFIG, root=tmp_path, now=_ny("2026-11-09 12:10"))
    assert gate["blocked"] and gate["status"] == "ERROR"
    assert gate["as_of"] == "2026-11-06" and gate["last_closed_session"] == "2026-11-06"
    assert any("partial intraday bar" in e for e in gate["errors"])

    after_close = paper_data_gate(config=GATE_CONFIG, root=tmp_path, now=_ny("2026-11-09 17:00"))
    assert not after_close["blocked"], after_close["errors"]
    assert after_close["as_of"] == "2026-11-09"


def test_freshness_uses_the_last_closed_session_not_the_local_date(tmp_path) -> None:
    _write_data(tmp_path, "2026-11-06")
    # 01:08 Tuesday in Hong Kong: Friday's data is current, not one session behind.
    findings, summary = run_data_checks(config=GATE_CONFIG, root=tmp_path,
                                        now=pd.Timestamp("2026-11-10 01:08", tz="Asia/Hong_Kong"))
    assert summary["as_of"] == "2026-11-06"
    assert not [f for f in findings if f.check in ("freshness", "incomplete_session") and f.severity != "INFO"]


def test_explicit_as_of_is_a_backdated_check_without_the_wall_clock(tmp_path) -> None:
    _write_data(tmp_path, "2026-11-09")
    findings, summary = run_data_checks(as_of="2026-11-09", config=GATE_CONFIG, root=tmp_path)
    assert summary["last_closed_session"] is None
    assert not [f for f in findings if f.check == "incomplete_session"]
