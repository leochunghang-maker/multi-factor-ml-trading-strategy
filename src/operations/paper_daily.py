"""``make paper-daily``: one scheduled paper day, end to end, never overlapping.

Order: take the lock -> paper run (refresh -> data gate -> paper day, via
scripts/run_daily_paper_trading.py) -> status summary -> weekly review ->
rotating backup (keep the newest N ``paper-daily-*`` archives) -> release the
lock. Status, review and backup are written even when the run was blocked,
so the operator sees why. Simulated paper trading only; local files only.

Exit codes: the paper run's code (0 ok/skipped, 2 blocked, 1 failed), or 3 when
another run holds the lock.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from src.operations.paper_alerts import raise_alert
from src.operations.run_lock import LockHeld, RunLock

LOCKED_EXIT_CODE = 3
DEFAULT_SETTINGS = {
    "backup_keep": 10,
    "backup_dir": "backups/paper",
    "lock_path": "results/operations/paper_daily.lock",
    "lock_stale_hours": 6,
}


def daily_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is None:
        from src.operations.configuration import load_platform_config

        config = load_platform_config()
    return {**DEFAULT_SETTINGS, **(config.get("paper_trading") or {})}


def paper_lock(settings: dict[str, Any] | None = None, log: Callable[[str], None] = print) -> RunLock:
    settings = settings or daily_settings()
    return RunLock(Path(settings["lock_path"]), float(settings["lock_stale_hours"]) * 3600, log=log)


def run_paper_daily(
    run_paper: Callable[[list[str]], int],
    runner_args: list[str] | None = None,
    *,
    paths=None,
    settings: dict[str, Any] | None = None,
    backup: bool = True,
    backup_sources: list[str] | None = None,
    backup_root: str | Path = ".",
    week_of=None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    from src.operations.paper_backup import BackupError, create_backup, prune_backups
    from src.operations.paper_ops import PaperOpsPaths
    from src.operations.paper_review import write_weekly_review
    from src.operations.paper_status import write_paper_status

    paths = paths or PaperOpsPaths.default()
    settings = settings or daily_settings()
    lock = paper_lock(settings, log=log)
    try:
        lock.acquire()
    except LockHeld as exc:
        raise_alert(paths, "run_blocked", str(exc), log=log, details={"reason": "lock_held"})
        write_paper_status(paths)
        return {"status": "LOCKED", "exit_code": LOCKED_EXIT_CODE, "message": str(exc)}

    result: dict[str, Any] = {"status": "DONE"}
    try:
        log("== paper run ==")
        try:
            code = int(run_paper(list(runner_args or [])))
        except Exception as exc:  # keep going: status/backup still useful
            log(f"paper run failed: {exc}")
            code = 1
        result["exit_code"] = code

        log("== status ==")
        status = write_paper_status(paths)
        result["overall_status"] = status["overall_status"]
        log(f"paper status {status['overall_status']} -> {paths.status_md}")

        log("== weekly review ==")
        review_path, _ = write_weekly_review(paths, week_of)
        result["review"] = str(review_path)
        log(f"weekly review -> {review_path}")

        if backup:
            log("== backup ==")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup_dir = Path(backup_root) / settings["backup_dir"]
            try:
                out, manifest = create_backup(backup_dir / f"paper-daily-{stamp}.tar.gz",
                                              sources=backup_sources, root=backup_root)
                removed = prune_backups(backup_dir, int(settings["backup_keep"]))
                result["backup"] = str(out)
                result["pruned"] = [str(p) for p in removed]
                log(f"backup {out} ({len(manifest['files'])} files); kept newest {settings['backup_keep']}, "
                    f"pruned {len(removed)}")
            except BackupError as exc:
                result["backup_error"] = str(exc)
                log(f"backup skipped: {exc}")
    finally:
        lock.release()
    return result
