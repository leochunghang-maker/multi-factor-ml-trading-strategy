"""Paper data dir is separate from the checked-in research snapshot under data/."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.data_loader as loader
from src import config as c
from src.data.paths import PAPER_DATA_DIR_ENV, DataPaths
from src.data.quality import run_data_checks
from src.operations.paper_ops import paper_data_gate

RESEARCH_FILES = ["data/price_data.csv", "data/volume_data.csv", "data/benchmark_prices.csv",
                  "data/benchmark_source.json", "data/data_manifest.json"]


def _digest(files) -> dict[str, str]:
    return {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files if Path(f).exists()}


def test_research_paths_are_the_config_constants() -> None:
    r = DataPaths.research()
    assert (r.price_data, r.volume_data, r.benchmark_prices, r.benchmark_source) == (
        c.PRICE_DATA_PATH, c.VOLUME_DATA_PATH, c.BENCHMARK_PRICES_PATH, c.BENCHMARK_SOURCE_PATH)
    assert (r.daily_returns, r.momentum, r.mean_reversion, r.forward_returns, r.ml_dataset) == (
        c.DAILY_RETURNS_PATH, c.MOMENTUM_PATH, c.MEAN_REVERSION_PATH, c.FORWARD_RETURNS_PATH, c.ML_DATASET_PATH)
    assert r.quality_report == "reports/data_quality_report.md"


def test_paper_dir_resolution_and_guard(monkeypatch) -> None:
    monkeypatch.delenv(PAPER_DATA_DIR_ENV, raising=False)
    assert DataPaths.paper().root == "data/paper"
    assert DataPaths.paper(config={"paper_trading": {"data_dir": "/tmp/pp"}}).root == "/tmp/pp"
    monkeypatch.setenv(PAPER_DATA_DIR_ENV, "/tmp/env_paper")
    assert DataPaths.paper(config={"paper_trading": {"data_dir": "/tmp/pp"}}).root == "/tmp/env_paper"
    assert DataPaths.paper("/tmp/explicit").root == "/tmp/explicit"
    for research in ["data", "./data", str(Path("data").resolve())]:
        with pytest.raises(ValueError, match="research data dir"):
            DataPaths.paper(research)
    p = DataPaths.paper("/tmp/explicit")
    assert p.quality_report == "/tmp/explicit/data_quality_report.md"
    assert p.manifest == "/tmp/explicit/data_manifest.json"


def test_paper_data_dir_is_gitignored() -> None:
    out = subprocess.run(["git", "check-ignore", "data/paper/price_data.csv", "data/paper/data_manifest.json"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and len(out.stdout.split()) == 2


def _fake_download(tickers=c.TICKERS, start=None, end=None, progress=True):
    dates = pd.bdate_range("2026-09-01", pd.Timestamp(end) - pd.Timedelta(days=1))
    rng = np.random.default_rng(1)
    names = list(tickers)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), len(names))), axis=0)),
                         index=dates, columns=names)
    volume = pd.DataFrame(1e6, index=dates, columns=names)
    if "BK" in names:
        close["BK"] = 50.0  # would re-enter the universe if not blanked
    return pd.concat({"Close": close, "Volume": volume}, axis=1)


def test_paper_refresh_writes_only_the_paper_dir(tmp_path, monkeypatch) -> None:
    before = _digest(RESEARCH_FILES)
    monkeypatch.setattr(loader, "_download", _fake_download)
    paths = loader.refresh_paper_data(end="2026-11-07", paths=DataPaths.paper(tmp_path / "paper"))
    assert _digest(RESEARCH_FILES) == before
    prices = pd.read_csv(paths.price_data, index_col=0)
    assert prices["BK"].isna().all()  # recorded exclusion applied in the paper dir too
    manifest = json.loads(Path(paths.manifest).read_text())
    assert set(manifest["files"]) == set(paths.dated_inputs().values())
    assert "BK" in manifest["excluded_symbols"]
    findings, summary = run_data_checks(as_of="2026-11-09", data_paths=paths)
    assert summary["data_dir"] == str(paths.root)
    assert not [f for f in findings if f.check == "manifest"]


def test_seed_from_research_copies_without_touching_data(tmp_path) -> None:
    before = _digest(RESEARCH_FILES)
    paths = loader.seed_paper_data_from_research(DataPaths.paper(tmp_path / "paper"))
    assert _digest(RESEARCH_FILES) == before
    assert Path(paths.price_data).read_bytes() == Path(c.PRICE_DATA_PATH).read_bytes()
    gate = paper_data_gate(as_of="2026-10-30", data_paths=paths)
    assert not gate["blocked"] and gate["data_dir"] == str(paths.root) and gate["calendar"] == "NYSE"
    stale = paper_data_gate(as_of="2026-11-09", data_paths=paths)
    assert stale["blocked"] and all("trading days before" in e for e in stale["errors"])

    # The paper manifest catches a paper file changed after the refresh (research manifest untouched).
    pd.read_csv(paths.price_data, index_col=0).iloc[:-5].to_csv(paths.price_data)
    findings, _ = run_data_checks(as_of="2026-10-30", data_paths=paths)
    changed = [f for f in findings if f.check == "manifest"]
    assert changed and all(f.file.startswith(str(paths.root)) for f in changed)


def test_gate_blocks_when_paper_data_missing(tmp_path) -> None:
    gate = paper_data_gate(as_of="2026-11-09", data_paths=DataPaths.paper(tmp_path / "empty"))
    assert gate["blocked"]
    assert any("file missing" in e for e in gate["errors"])


def test_paper_refresh_is_quiet_and_skips_excluded_symbols(tmp_path, monkeypatch) -> None:
    calls = []

    def recording_download(tickers=c.TICKERS, start=None, end=None, progress=True):
        calls.append((list(tickers), progress))
        return _fake_download(tickers, start, end)

    monkeypatch.setattr(loader, "_download", recording_download)
    paths = loader.refresh_paper_data(end="2026-11-07", paths=DataPaths.paper(tmp_path / "paper"))
    assert calls and all(progress is False for _, progress in calls)
    assert "BK" not in calls[0][0]  # excluded symbol is not requested ...
    assert pd.read_csv(paths.price_data, index_col=0)["BK"].isna().all()  # ... and still blank in the file


def _snapshot_inputs(paths: DataPaths) -> dict[str, str]:
    return _digest([
        paths.price_data, paths.volume_data, paths.benchmark_prices,
        paths.benchmark_source, paths.manifest,
    ])


def _assert_no_refresh_dirs(root) -> None:
    leftovers = [p.name for p in Path(root).glob(".refresh-*")]
    assert not leftovers, leftovers


def test_paper_refresh_keeps_previous_snapshot_if_download_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loader, "_download", _fake_download)
    dest = DataPaths.paper(tmp_path / "paper")
    loader.refresh_paper_data(end="2026-11-07", paths=dest)
    (Path(dest.root) / "daily_returns.csv").write_text("derived,should,stay\n")
    before = _snapshot_inputs(dest)
    derived = (Path(dest.root) / "daily_returns.csv").read_bytes()
    calls = {"n": 0}

    def boom(tickers=c.TICKERS, start=None, end=None, progress=True):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("yahoo down")
        return _fake_download(tickers, start, end, progress)

    monkeypatch.setattr(loader, "_download", boom)
    with pytest.raises(RuntimeError, match="yahoo down"):
        loader.refresh_paper_data(end="2026-11-07", paths=dest)
    assert _snapshot_inputs(dest) == before
    assert (Path(dest.root) / "daily_returns.csv").read_bytes() == derived
    _assert_no_refresh_dirs(dest.root)


def test_paper_refresh_rejects_truncated_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loader, "_download", _fake_download)
    dest = DataPaths.paper(tmp_path / "paper")
    loader.refresh_paper_data(end="2026-11-07", paths=dest)
    before = _snapshot_inputs(dest)

    def short(tickers=c.TICKERS, start=None, end=None, progress=True):
        return _fake_download(tickers, start, end, progress).iloc[-5:]

    monkeypatch.setattr(loader, "_download", short)
    with pytest.raises(ValueError, match="dropped .* already-held session"):
        loader.refresh_paper_data(end="2026-11-07", paths=dest)
    assert _snapshot_inputs(dest) == before
    _assert_no_refresh_dirs(dest.root)


def test_paper_refresh_rejects_empty_download(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loader, "_download", _fake_download)
    dest = DataPaths.paper(tmp_path / "paper")
    loader.refresh_paper_data(end="2026-11-07", paths=dest)
    before = _snapshot_inputs(dest)
    monkeypatch.setattr(loader, "_download", lambda *args, **kwargs: pd.DataFrame())
    with pytest.raises(ValueError, match="no price data"):
        loader.refresh_paper_data(end="2026-11-07", paths=dest)
    assert _snapshot_inputs(dest) == before
    _assert_no_refresh_dirs(dest.root)


def test_paper_refresh_cleans_leftover_staging(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "paper"
    leftover = dest / ".refresh-dead"
    leftover.mkdir(parents=True)
    (leftover / "junk.csv").write_text("stale\n")
    monkeypatch.setattr(loader, "_download", _fake_download)
    loader.refresh_paper_data(end="2026-11-07", paths=DataPaths.paper(dest))
    assert not leftover.exists()
    _assert_no_refresh_dirs(dest)
