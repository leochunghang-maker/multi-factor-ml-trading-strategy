"""The frozen research snapshot is checked against its own end date, not the wall clock.

``data/`` is a fixed research snapshot (``src/config.py`` END_DATE, exclusive).
Measuring it against today turned the smoke test, ``make health`` and the data
report into a permanent, growing "stale" WARNING that said nothing about the data.
With ``data_checks.research_snapshot_end`` set, a wall-clock check of the research
data measures freshness against the last session before that end (a truncated
snapshot is still flagged); the paper gate keeps the last closed session.
"""

import numpy as np
import pandas as pd

from src.data.paths import DataPaths
from src.data.quality import research_snapshot_last_session, run_data_checks
from src.operations.paper_ops import paper_data_gate

BASE = {"manifest_files": [], "manifest_path": "data/data_manifest.json"}
SNAPSHOT = {"data_checks": {**BASE, "research_snapshot_end": "2026-11-01"}}
NOW = pd.Timestamp("2026-11-13 01:00", tz="Asia/Hong_Kong")  # 12:00 Thu 12 Nov in New York


def _write_data(root, end: str) -> None:
    dates = pd.bdate_range(end=end, periods=40)
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (40, 2)), axis=0)),
                          index=dates, columns=["AAA", "BBB"])
    root.mkdir(parents=True, exist_ok=True)
    prices.to_csv(root / "price_data.csv")
    pd.DataFrame(1e6, index=dates, columns=["AAA", "BBB"]).to_csv(root / "volume_data.csv")
    prices[["AAA"]].rename(columns={"AAA": "SPY"}).to_csv(root / "benchmark_prices.csv")


def _stale(findings):
    return [f for f in findings if f.check == "freshness" and f.severity != "INFO"]


def test_snapshot_last_session_is_the_session_before_the_exclusive_end() -> None:
    assert research_snapshot_last_session("2026-11-01") == pd.Timestamp("2026-10-30")  # Sunday end
    assert research_snapshot_last_session("2026-11-27") == pd.Timestamp("2026-11-25")  # Thanksgiving


def test_complete_snapshot_is_not_stale_and_says_why(tmp_path) -> None:
    _write_data(tmp_path / "data", "2026-10-30")
    findings, summary = run_data_checks(config=SNAPSHOT, root=tmp_path, now=NOW)
    assert summary["freshness_reference"] == "research_snapshot_end"
    assert summary["as_of"] == "2026-10-30" and summary["last_closed_session"] == "2026-11-11"
    assert not _stale(findings)
    info = [f for f in findings if f.check == "snapshot"]
    assert len(info) == 1 and info[0].severity == "INFO" and "8 trading day(s) later" in info[0].message


def test_truncated_snapshot_is_still_flagged(tmp_path) -> None:
    _write_data(tmp_path / "data", "2026-10-20")
    findings, _ = run_data_checks(config=SNAPSHOT, root=tmp_path, now=NOW)
    stale = _stale(findings)
    assert stale and all("before 2026-10-30" in f.message for f in stale)


def test_without_the_setting_or_with_explicit_as_of_the_old_reference_applies(tmp_path) -> None:
    _write_data(tmp_path / "data", "2026-10-30")
    findings, summary = run_data_checks(config={"data_checks": BASE}, root=tmp_path, now=NOW)
    assert summary["freshness_reference"] == "last_closed_session" and _stale(findings)
    findings, summary = run_data_checks(as_of="2026-11-11", config=SNAPSHOT, root=tmp_path)
    assert summary["freshness_reference"] == "as_of" and _stale(findings)
    assert not [f for f in findings if f.check == "snapshot"]


def test_paper_data_dir_and_paper_gate_ignore_the_snapshot_end(tmp_path) -> None:
    _write_data(tmp_path / "data", "2026-10-30")
    gate = paper_data_gate(config=SNAPSHOT, root=tmp_path, now=NOW)
    assert gate["blocked"] and gate["as_of"] == "2026-11-11"

    paper = DataPaths(str(tmp_path / "paper"), kind="paper")
    _write_data(tmp_path / "paper", "2026-10-30")
    findings, summary = run_data_checks(config=SNAPSHOT, root=".", data_paths=paper, now=NOW)
    assert summary["freshness_reference"] == "last_closed_session" and _stale(findings)
