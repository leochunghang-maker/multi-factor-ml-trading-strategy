"""Data freshness / alignment / quality checks and the data manifest (read-only)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.quality import (
    DEFAULT_DATA_CHECKS,
    build_manifest,
    check_freshness,
    check_manifest,
    check_panel_quality,
    file_sha256,
    overall_status,
    run_data_checks,
    split_like_factor,
    trading_days_between,
)

S = dict(DEFAULT_DATA_CHECKS)


def _panel(n=60, start="2026-01-05", cols=("AAA", "BBB"), seed=0):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    data = {c: 100 * np.cumprod(1 + rng.normal(0, 0.01, n)) for c in cols}
    return pd.DataFrame(data, index=idx)


def _checks(findings, check, severity=None):
    return [f for f in findings if f.check == check and (severity is None or f.severity == severity)]


def test_trading_days_between_counts_weekdays_after_start() -> None:
    assert trading_days_between("2026-10-30", "2026-11-09") == 6  # Fri -> next-next Mon
    assert trading_days_between("2026-10-30", "2026-11-01") == 0  # weekend only
    assert trading_days_between("2026-11-09", "2026-11-02") == 0


def test_freshness_warns_when_stale_and_passes_when_fresh() -> None:
    p = _panel()
    last = p.index[-1]
    fresh = check_freshness({"price_data": p}, last + pd.offsets.BDay(5), S)
    assert fresh == []
    stale = check_freshness({"price_data": p}, last + pd.offsets.BDay(6), S)
    assert len(stale) == 1 and stale[0].severity == "WARNING" and stale[0].check == "freshness"
    strict = check_freshness({"price_data": p}, last + pd.offsets.BDay(6), {**S, "stale_severity": "ERROR"})
    assert strict[0].severity == "ERROR"


def test_alignment_flags_end_gap_and_missing_dates() -> None:
    p = _panel()
    short = p.iloc[:-2]
    holes = p.drop(p.index[10])
    f = check_freshness({"price_data": p, "volume_data": short, "benchmark_prices": holes}, p.index[-1], S)
    al = _checks(f, "alignment", "ERROR")
    assert any(x.file == "volume_data" and "ends" in x.message for x in al)
    assert any(x.file == "benchmark_prices" and "1 price_data dates missing" in x.message for x in al)
    ok = check_freshness({"price_data": p, "volume_data": p.copy()}, p.index[-1], S)
    assert _checks(ok, "alignment") == []


def test_missing_file_is_an_error() -> None:
    f = check_freshness({"price_data": None}, "2026-01-01", S)
    assert f[0].severity == "ERROR"


def test_split_like_factor() -> None:
    assert split_like_factor(0.5, 0.03) == 2
    assert split_like_factor(0.2, 0.03) == 5
    assert split_like_factor(3.02, 0.03) == 3
    assert split_like_factor(0.73, 0.03) is None


def test_quality_detects_flat_split_large_empty_holes_gaps() -> None:
    p = _panel(cols=("AAA", "BBB", "CCC", "EMPTY", "LATE"))
    p["EMPTY"] = np.nan
    p.iloc[:20, p.columns.get_loc("LATE")] = np.nan  # late listing -> INFO only
    p.iloc[30:36, p.columns.get_loc("AAA")] = p["AAA"].iloc[30]  # 6 identical closes
    p.iloc[40:, p.columns.get_loc("BBB")] = p["BBB"].iloc[40:] / 2  # unadjusted 2:1 split
    p.iloc[45:, p.columns.get_loc("CCC")] = p["CCC"].iloc[45:] * 0.7  # -30% move
    p.iloc[50, p.columns.get_loc("CCC")] = np.nan  # hole inside history
    p = p.drop(p.index[[25, 26, 27, 28]])  # 6-day calendar gap
    f = check_panel_quality(p, "price_data", S)
    assert _checks(f, "empty_columns", "WARNING")[0].message.endswith("EMPTY")
    assert any("AAA" in x.message for x in _checks(f, "flat_prices", "WARNING"))
    split = _checks(f, "split_like_jump", "WARNING")
    assert len(split) == 1 and "BBB" in split[0].message and "1/2" in split[0].message
    assert any("CCC" in x.message for x in _checks(f, "large_move", "INFO"))
    assert any("CCC (1)" in x.message for x in _checks(f, "missing_values", "WARNING"))
    assert _checks(f, "missing_rows", "WARNING")
    assert any("LATE" in x.message for x in _checks(f, "coverage", "INFO"))
    assert overall_status(f) == "WARNING"


def test_quality_errors_on_duplicates_and_nonpositive() -> None:
    p = _panel()
    p.iloc[3, 0] = -1.0
    p = pd.concat([p, p.iloc[[5]]]).sort_index()
    f = check_panel_quality(p, "price_data", S)
    assert _checks(f, "dates", "ERROR") and _checks(f, "values", "ERROR")
    assert overall_status(f) == "ERROR"


def test_volume_checks() -> None:
    v = _panel() * 1000
    v.iloc[2, 0] = 0
    f = check_panel_quality(v, "volume_data", S, kind="volume")
    assert _checks(f, "values", "WARNING")
    v.iloc[3, 0] = -5
    assert _checks(check_panel_quality(v, "volume_data", S, kind="volume"), "values", "ERROR")


def test_clean_panel_has_no_warnings() -> None:
    f = check_panel_quality(_panel(), "price_data", S)
    assert overall_status(f) == "OK"


def test_sha256_ignores_line_endings(tmp_path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"Date,X\n2026-01-01,1\n")
    b.write_bytes(b"Date,X\r\n2026-01-01,1\r\n")
    assert file_sha256(a) == file_sha256(b)


def test_manifest_round_trip_and_change_detection(tmp_path) -> None:
    (tmp_path / "data").mkdir()
    csv = tmp_path / "data" / "price_data.csv"
    _panel().to_csv(csv)
    files = ["data/price_data.csv"]
    manifest = build_manifest(files, root=tmp_path)
    assert manifest["files"]["data/price_data.csv"]["rows"] == 60
    (tmp_path / "data" / "data_manifest.json").write_text(json.dumps(manifest))
    assert check_manifest("data/data_manifest.json", files, S, root=tmp_path) == []
    _panel(n=61).to_csv(csv)  # "re-download"
    changed = check_manifest("data/data_manifest.json", files, S, root=tmp_path)
    assert len(changed) == 1 and changed[0].severity == "WARNING" and "rows 60→61" in changed[0].message
    assert check_manifest("data/missing.json", files, S, root=tmp_path)[0].check == "manifest"


def test_run_data_checks_end_to_end_on_tmp_repo(tmp_path) -> None:
    (tmp_path / "data").mkdir()
    p = _panel()
    p.to_csv(tmp_path / "data" / "price_data.csv")
    (p * 1000).to_csv(tmp_path / "data" / "volume_data.csv")
    p[["AAA"]].rename(columns={"AAA": "SPY"}).to_csv(tmp_path / "data" / "benchmark_prices.csv")
    files = ["data/price_data.csv", "data/volume_data.csv", "data/benchmark_prices.csv"]
    cfg = {"data_checks": {"manifest_files": files}}
    (tmp_path / "data" / "data_manifest.json").write_text(json.dumps(build_manifest(files, root=tmp_path)))
    findings, summary = run_data_checks(as_of=p.index[-1], config=cfg, root=tmp_path)
    assert overall_status(findings) == "OK", findings
    assert summary["files"]["price_data"]["rows"] == 60
    findings, _ = run_data_checks(as_of=p.index[-1] + pd.offsets.BDay(10), config=cfg, root=tmp_path)
    assert overall_status(findings) == "WARNING"


def test_checked_in_data_has_no_errors_and_matches_manifest() -> None:
    findings, _ = run_data_checks(as_of="2026-10-30")
    assert overall_status(findings) != "ERROR", [f for f in findings if f.severity == "ERROR"]
    assert not [f for f in findings if f.check == "manifest"], "data changed: review, then --update-manifest"


def test_health_surfaces_data_findings() -> None:
    from src.operations.configuration import load_platform_config
    from src.operations.health import check_data_inputs

    findings, summary = check_data_inputs(load_platform_config(), as_of="2026-12-31")
    assert summary["status"] in {"WARNING", "ERROR"}
    assert any(f.check == "data_freshness" for f in findings)
    assert all(f.severity in {"ERROR", "WARNING"} for f in findings)
