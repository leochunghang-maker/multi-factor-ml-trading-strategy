"""Scheduled paper day: lock file, rotating backups, weekly review (temp dirs only)."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from src.operations.paper_alerts import read_alerts
from src.operations.paper_backup import prune_backups, verify_backup
from src.operations.paper_daily import LOCKED_EXIT_CODE, run_paper_daily
from src.operations.paper_ops import PaperOpsPaths, PaperRunBlocked, enforce_data_gate
from src.operations.paper_review import build_weekly_review, week_bounds, write_weekly_review
from src.operations.run_lock import LockHeld, RunLock


def _write_lock(path: Path, pid: int, age_seconds: float = 0.0, host: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time() - age_seconds
    path.write_text(json.dumps({"pid": pid, "hostname": host or socket.gethostname(),
                                "started_at": "t", "started_epoch": started}))


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# ------------------------------------------------------------------------ lock


def test_lock_prevents_overlap_and_releases(tmp_path) -> None:
    path = tmp_path / "paper.lock"
    first = RunLock(path).acquire()
    assert json.loads(path.read_text())["pid"] == os.getpid()
    with pytest.raises(LockHeld, match="another paper run"):
        RunLock(path).acquire()
    first.release()
    assert not path.exists()
    with RunLock(path):
        assert path.exists()
    assert not path.exists()


def test_stale_lock_dead_pid_is_taken_over(tmp_path) -> None:
    path = tmp_path / "paper.lock"
    _write_lock(path, _dead_pid())
    messages = []
    lock = RunLock(path, log=messages.append).acquire()
    assert lock.acquired and any("is not running" in m for m in messages)
    lock.release()


def test_stale_lock_by_age_even_on_other_host(tmp_path) -> None:
    path = tmp_path / "paper.lock"
    _write_lock(path, 12345, host="other-host")
    with pytest.raises(LockHeld):
        RunLock(path, stale_after_seconds=3600).acquire()  # other host, fresh: assume live
    _write_lock(path, 12345, age_seconds=7200, host="other-host")
    messages = []
    RunLock(path, stale_after_seconds=3600, log=messages.append).acquire().release()
    assert any("h old" in m for m in messages)


def test_unreadable_lock_is_respected_briefly_then_stale(tmp_path) -> None:
    path = tmp_path / "paper.lock"
    path.write_text("")
    with pytest.raises(LockHeld):
        RunLock(path).acquire()  # may be a lock being written right now
    old = time.time() - 120
    os.utime(path, (old, old))
    RunLock(path).acquire().release()


def test_release_does_not_remove_someone_elses_lock(tmp_path) -> None:
    path = tmp_path / "paper.lock"
    lock = RunLock(path).acquire()
    _write_lock(path, os.getpid() + 1)  # e.g. taken over after we were considered stale
    lock.release()
    assert path.exists()


# ------------------------------------------------------------------ paper-daily


def _settings(tmp_path, keep=2) -> dict:
    return {"backup_keep": keep, "backup_dir": "backups/paper", "lock_path": str(tmp_path / "ops" / "daily.lock"),
            "lock_stale_hours": 6}


def _fake_run_paper(root: Path, code: int = 0):
    calls = []

    def run(args):
        calls.append(args)
        (root / "results/paper").mkdir(parents=True, exist_ok=True)
        (root / "results/paper/ledger.csv").write_text(f"signal_date\n2026-11-0{len(calls)}\n")
        return code

    run.calls = calls
    return run


def test_paper_daily_runs_status_review_and_rotating_backup(tmp_path) -> None:
    paths = PaperOpsPaths(tmp_path / "results/paper")
    run = _fake_run_paper(tmp_path)
    results = []
    for _ in range(4):
        results.append(run_paper_daily(run, ["--no-lock"], paths=paths, settings=_settings(tmp_path),
                                       backup_sources=["results/paper/"], backup_root=tmp_path,
                                       week_of="2026-11-06", log=lambda _: None))
        time.sleep(1.1)  # backup names have second resolution
    assert run.calls == [["--no-lock"]] * 4
    assert all(r["exit_code"] == 0 and r["status"] == "DONE" for r in results)
    assert paths.status_md.exists() and (paths.reviews / "weekly-2026-W45.md").exists()
    archives = sorted((tmp_path / "backups/paper").glob("paper-daily-*.tar.gz"))
    assert len(archives) == 2 and str(archives[-1]) == results[-1]["backup"]
    verify_backup(archives[-1])
    assert len(results[-1]["pruned"]) == 1
    assert not Path(_settings(tmp_path)["lock_path"]).exists()


def test_paper_daily_exits_locked_without_running(tmp_path) -> None:
    paths = PaperOpsPaths(tmp_path / "results/paper")
    lock_path = Path(_settings(tmp_path)["lock_path"])
    _write_lock(lock_path, os.getpid())
    run = _fake_run_paper(tmp_path)
    out = run_paper_daily(run, paths=paths, settings=_settings(tmp_path), backup_root=tmp_path,
                          log=lambda _: None)
    assert out["status"] == "LOCKED" and out["exit_code"] == LOCKED_EXIT_CODE
    assert run.calls == []
    assert lock_path.exists()  # the live owner's lock is untouched
    assert [a["details"]["reason"] for a in read_alerts(paths)] == ["lock_held"]


def test_paper_daily_failure_still_writes_status_and_releases_lock(tmp_path) -> None:
    paths = PaperOpsPaths(tmp_path / "results/paper")

    def broken(args):
        raise RuntimeError("boom")

    out = run_paper_daily(broken, paths=paths, settings=_settings(tmp_path), backup_root=tmp_path,
                          backup_sources=["results/paper/"], log=lambda _: None)
    assert out["exit_code"] == 1 and paths.status_md.exists()
    assert not Path(_settings(tmp_path)["lock_path"]).exists()


def test_prune_only_touches_daily_archives(tmp_path) -> None:
    for name in ["paper-daily-20261101T000000Z", "paper-daily-20261102T000000Z", "paper-daily-20261103T000000Z",
                 "paper-backup-20261101T000000Z", "pre-restore-20261101T000000Z-abc"]:
        (tmp_path / f"{name}.tar.gz").write_bytes(b"x")
    removed = prune_backups(tmp_path, keep=1)
    assert [p.name for p in removed] == ["paper-daily-20261101T000000Z.tar.gz", "paper-daily-20261102T000000Z.tar.gz"]
    assert len(list(tmp_path.glob("*.tar.gz"))) == 3
    with pytest.raises(ValueError):
        prune_backups(tmp_path, keep=0)


# --------------------------------------------------------------- weekly review


def test_week_bounds_are_iso_weeks() -> None:
    assert week_bounds("2026-11-06") == (pd.Timestamp("2026-11-02").date(), pd.Timestamp("2026-11-08").date())
    assert week_bounds("2026-11-02")[0] == pd.Timestamp("2026-11-02").date()


def test_weekly_review_aggregates_journal_alerts_and_gate_history(tmp_path) -> None:
    paths = PaperOpsPaths(tmp_path / "paper")
    paths.root.mkdir(parents=True)
    pd.DataFrame([
        # previous week: sets the starting NAV
        {"signal_date": "2026-10-30", "nav_end": 100_000.0, "pnl_since_previous": 0.0, "turnover": 1.0,
         "trades_filled": 10, "orders_rejected": 0, "drawdown": 0.0, "drawdown_halted": False,
         "capacity_clipped_positions": 0, "order_adv_clips": 0, "reconciliation_status": "OK", "forced": False},
        {"signal_date": "2026-11-03", "nav_end": 101_000.0, "pnl_since_previous": 1_000.0, "turnover": 0.10,
         "trades_filled": 4, "orders_rejected": 1, "drawdown": 0.0, "drawdown_halted": False,
         "capacity_clipped_positions": 1, "order_adv_clips": 0, "reconciliation_status": "OK", "forced": False},
        {"signal_date": "2026-11-05", "nav_end": 99_000.0, "pnl_since_previous": -2_000.0, "turnover": 0.30,
         "trades_filled": 6, "orders_rejected": 0, "drawdown": -0.0198, "drawdown_halted": True,
         "capacity_clipped_positions": 0, "order_adv_clips": 2, "reconciliation_status": "MISMATCH", "forced": True},
    ]).to_csv(paths.journal_csv, index=False)
    alerts = [
        {"timestamp": "2026-11-04T22:00:00+00:00", "kind": "data_gate_blocked", "severity": "ERROR"},
        {"timestamp": "2026-11-05T22:00:00+00:00", "kind": "drawdown_halt", "severity": "WARNING"},
        {"timestamp": "2026-11-09T01:00:00+00:00", "kind": "drawdown_halt", "severity": "WARNING"},  # next week
    ]
    paths.alerts.write_text("".join(json.dumps({**a, "message": "m", "signal_date": None}) + "\n" for a in alerts))
    pd.DataFrame([
        {"checked_at": "2026-11-03T22:00:00+00:00", "as_of": "2026-11-03", "status": "OK", "blocked": False},
        {"checked_at": "2026-11-04T22:00:00+00:00", "as_of": "2026-11-04", "status": "ERROR", "blocked": True},
        {"checked_at": "2026-11-05T22:00:00+00:00", "as_of": "2026-11-05", "status": "OK", "blocked": False},
        {"checked_at": "2026-11-05T23:00:00+00:00", "as_of": "2026-11-05", "status": "OK", "blocked": False},
    ]).to_csv(paths.gate_history, index=False)

    r = build_weekly_review(paths, "2026-11-06")
    assert r["week"] == "2026-W45" and r["runs"] == 2 and r["signal_dates"] == ["2026-11-03", "2026-11-05"]
    assert r["nav_start"] == 100_000.0 and r["nav_end"] == 99_000.0
    assert r["pnl"] == pytest.approx(-1_000.0) and r["return"] == pytest.approx(-0.01)
    assert r["trades_filled"] == 10 and r["orders_rejected"] == 1
    assert r["turnover_total"] == pytest.approx(0.40) and r["turnover_avg"] == pytest.approx(0.20)
    assert r["max_drawdown"] == pytest.approx(-0.0198)
    assert (r["drawdown_halts"], r["reconciliation_mismatches"], r["adv_clips"], r["forced_runs"]) == (1, 1, 3, 1)
    assert r["alerts_by_kind"] == {"data_gate_blocked": 1, "drawdown_halt": 1}
    assert (r["gate_checks"], r["gate_blocked"], r["gate_block_rate"]) == (4, 1, 0.25)

    path, _ = write_weekly_review(paths, "2026-11-06")
    text = path.read_text()
    assert path.name == "weekly-2026-W45.md" and "| Data gate checks / blocked | 4 / 1 (25%) |" in text
    assert json.loads(path.with_suffix(".json").read_text())["pnl"] == pytest.approx(-1_000.0)


def test_weekly_review_of_empty_week_and_gate_history_recording(tmp_path) -> None:
    paths = PaperOpsPaths(tmp_path / "paper")
    r = build_weekly_review(paths, "2026-11-06")
    assert r["runs"] == 0 and r["pnl"] is None and r["gate_block_rate"] is None
    gate = {"checked_at": "2026-11-04T22:00:00+00:00", "as_of": "2026-11-04", "status": "ERROR",
            "blocked": True, "errors": ["stale"]}
    with pytest.raises(PaperRunBlocked):
        enforce_data_gate(paths, gate, log=None)
    enforce_data_gate(paths, {**gate, "checked_at": "2026-11-05T22:00:00+00:00", "status": "OK",
                              "blocked": False, "errors": []})
    history = pd.read_csv(paths.gate_history)
    assert list(history["blocked"]) == [True, False]
    r = build_weekly_review(paths, "2026-11-06")
    assert (r["gate_checks"], r["gate_blocked"]) == (2, 1)
