"""Scheduled daily paper run (``make paper-daily``); simulated paper trading only.

    python scripts/run_paper_daily.py                       # refresh -> gate -> paper day -> status -> review -> backup
    python scripts/run_paper_daily.py --skip-data-refresh --as-of 2026-10-30
    python scripts/run_paper_daily.py --keep 20 --no-backup

A lock file (paper_trading.lock_path) prevents overlapping runs; a stale lock
(dead pid on this host, or older than paper_trading.lock_stale_hours) is
removed with a log line. Exit codes: 0 ok/skipped, 1 failed, 2 blocked, 3 locked.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.operations.paper_daily import daily_settings, run_paper_daily  # noqa: E402
from src.operations.structured_logging import log_event  # noqa: E402
from src.data.required import run_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-data-refresh", action="store_true", help="Passed to the paper run.")
    parser.add_argument("--as-of", default=None, help="Passed to the paper run's data gate.")
    parser.add_argument("--data-dir", default=None, help="Passed to the paper run.")
    parser.add_argument("--keep", type=int, default=None, help="Daily backups to keep (default paper_trading.backup_keep).")
    parser.add_argument("--no-backup", action="store_true", help="Skip the rotating backup.")
    parser.add_argument("--week-of", default=None, help="Date inside the week to review (default today, UTC).")
    args = parser.parse_args(argv)

    settings = daily_settings()
    if args.keep is not None:
        settings["backup_keep"] = args.keep
    runner_args = ["--no-lock"]  # this process already holds the lock
    if args.skip_data_refresh:
        runner_args.append("--skip-data-refresh")
    if args.as_of:
        runner_args += ["--as-of", args.as_of]
    if args.data_dir:
        runner_args += ["--data-dir", args.data_dir]

    def run_paper(extra: list[str]) -> int:
        return subprocess.call([sys.executable, str(ROOT / "scripts" / "run_daily_paper_trading.py"), *extra], cwd=ROOT)

    log_event("paper_daily_started", "Scheduled paper day started.", runner_args=runner_args)
    result = run_paper_daily(run_paper, runner_args, settings=settings, backup=not args.no_backup,
                             week_of=args.week_of)
    code = int(result.get("exit_code", 1))
    log_event("paper_daily_finished", f"Scheduled paper day finished: {result['status']} (exit {code}).",
              level="INFO" if code == 0 else "WARNING", **{k: v for k, v in result.items() if k != "message"})
    print(f"paper-daily: {result['status']} exit {code}"
          + (f" ({result['message']})" if result.get("message") else ""))
    return code


if __name__ == "__main__":
    sys.exit(run_cli(main))
