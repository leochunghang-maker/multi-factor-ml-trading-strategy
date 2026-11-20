"""A failed paper-run step writes the local snapshot, not the committed report."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path


def _load_runner():
    path = Path("scripts/run_daily_paper_trading.py")
    spec = importlib.util.spec_from_file_location("run_daily_paper_trading", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_failure_status_stays_in_local_results(tmp_path, monkeypatch) -> None:
    report = tmp_path / "results" / "paper" / "latest_run.md"
    committed = tmp_path / "reports" / "paper_trading_status.md"
    committed.parent.mkdir(parents=True)
    committed.write_text("pointer only\n")
    monkeypatch.setattr(
        "src.execution.run_daily_simulation.PAPER_TRADING_STATUS_REPORT_PATH",
        str(report),
    )
    runner = _load_runner()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.write_failure_status("Data refresh failed: boom")

    assert caught == []
    text = report.read_text()
    assert "Data refresh failed: boom" in text
    assert "Portfolio value: 0.00" in text
    assert "Price date: unavailable" in text
    assert committed.read_text() == "pointer only\n"
