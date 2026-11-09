# Operational Workflow

This project is designed for local research, simulated paper trading, and operational validation. It does not enable live-money trading.

## Configuration

Use `config/platform_config.json` as the deployment-facing configuration file. It groups:

- strategy parameters,
- transaction costs,
- slippage assumptions,
- rebalance frequency,
- risk limits,
- simulation settings,
- monitoring thresholds.

Optional machine-specific overrides can live in `config/platform_config.local.json`.

## Daily Workflow

Run:

```bash
python scripts/run_daily_paper_trading.py
```

The workflow performs:

1. data refresh into the paper data dir `data/paper/` (`src/data_loader.py --paper`: prices, volume
   and SPY through today plus a paper manifest; skip with `--skip-data-refresh`),
2. pre-run data gate on `data/paper/` (staleness in NYSE trading days and any other data ERROR
   block the run; exit code 2),
3. idempotency check (a completed signal date is skipped; `--force` replaces the latest run),
4. feature generation (factor files and ML dataset written to `data/paper/`),
5. signal generation (from the gated paper prices and paper ML dataset),
6. signal and portfolio safety checks,
7. simulated broker execution with reconciliation against the previous run, a dated snapshot in
   `results/paper/runs/<date>/`, and ledger/journal rows in `results/paper/`,
8. risk report generation,
9. dashboard/report update and system health status update (reconciliation mismatches are ERROR).

Alerts (gate blocked, reconciliation mismatch, drawdown halt, out-of-order run) are written to
`results/paper/alerts.jsonl` and the event log only; `make paper-status` shows them with the latest
run in `results/paper/status.md`. Back up the account with `make paper-backup` and restore with
`python scripts/paper_backup.py restore FILE` (refuses to overwrite without `--force`).

Paper-run state under `results/paper/` and paper data under `data/paper/` are local and
gitignored. The checked-in research snapshot under `data/` is never written by this workflow, so
no restore step is needed before `make research`. See README, "Paper-run operations".

Offline demo without downloading: `python src/data_loader.py --paper --seed-from-research`, then
`python scripts/run_daily_paper_trading.py --skip-data-refresh --as-of 2026-10-30`.

## Scheduling the Paper Day (examples, not installed)

`make paper-daily` (`scripts/run_paper_daily.py`) runs one complete simulated paper day:
lock → data refresh → data gate → paper day → status → weekly review → rotating backup (keeps
the newest `paper_trading.backup_keep` `paper-daily-*.tar.gz`; manual and `pre-restore-*`
backups are never pruned) → unlock. Exit codes: 0 ok or already done, 1 failed, 2 blocked
(see `results/paper/status.md`), 3 another run holds the lock. Nothing is sent anywhere; check
`make paper-status`, `results/paper/alerts.jsonl` and `results/operations/platform_events.jsonl`.

Run it after the US close once Yahoo has the day's bar, e.g. 18:30 New York time, Monday-Friday
(NYSE holidays are harmless: the run finds no new signal date and skips, or the gate blocks).
Adjust `/path/to/repo`; the repo's virtualenv is assumed at `.venv/`.

cron (`crontab -e`; `CRON_TZ` is supported by cronie and most Linux crons; otherwise convert the
time to the machine's zone, e.g. 06:30 or 07:30 Hong Kong time Tue-Sat depending on US DST):

```cron
CRON_TZ=America/New_York
30 18 * * 1-5  cd /path/to/repo && .venv/bin/python scripts/run_paper_daily.py >> results/operations/paper_daily.log 2>&1
0 10 * * 6     cd /path/to/repo && .venv/bin/python scripts/paper_weekly_review.py --week-of "$(date -d yesterday +\%F)" >> results/operations/paper_daily.log 2>&1
```

systemd (user units in `~/.config/systemd/user/`, then `systemctl --user daemon-reload &&
systemctl --user enable --now paper-daily.timer`):

```ini
# paper-daily.service
[Unit]
Description=Simulated paper-trading day (local files only)

[Service]
Type=oneshot
WorkingDirectory=/path/to/repo
ExecStart=/path/to/repo/.venv/bin/python scripts/run_paper_daily.py
# 2 = blocked by the data gate, 3 = locked: expected outcomes, not unit failures
SuccessExitStatus=2 3
TimeoutStartSec=1h
```

```ini
# paper-daily.timer
[Unit]
Description=Run the simulated paper day after the US close

[Timer]
OnCalendar=Mon..Fri 18:30 America/New_York
Persistent=true

[Install]
WantedBy=timers.target
```

The lock makes overlapping triggers safe (a manual `make paper` during a scheduled run exits with
code 3). A crashed run leaves a stale lock that the next run removes automatically (dead pid on
the same host, or older than `paper_trading.lock_stale_hours`).

## Paper vs Backtest Tracking

Every completed paper run updates `results/paper/tracking.csv`: the paper account versus a
frictionless backtest of the previous run's target weights (holding gap), realized versus modeled
cost per traded dollar, and realized versus target weights. Thresholds are fixed in
`paper_tracking` (see README, "Paper-run operations") and were chosen from the simulator design
before any tracking result existed; do not tune them on results. Breaches are `tracking_breach`
alerts (local files only), appear in `make paper-status`, and are summarised in the weekly
review.

## Simulation Workflow

Run:

```bash
python src/simulation/run_multi_day_simulation.py
```

Use this to test whether the operating process remains stable through many dates, not to optimize returns.

## Monitoring

Run:

```bash
python src/operations/health.py
```

The monitor checks stale data, missing files, abnormal turnover, abnormal exposure, excessive drawdown, concentration, and execution failures.

## Reproducibility

Every operational workflow saves:

- run metadata,
- config hash,
- config snapshot,
- structured JSONL logs,
- git commit when available,
- deterministic random seed.

Reproducibility matters because a reviewable research process must be able to explain which data, settings, and code state produced each run.

## Operational Safety

Operational safety matters because failures in quant systems are often mundane: stale files, missing prices, duplicate signals, bad cash accounting, unexpected turnover, or silently changed settings. Monitoring makes these problems visible before capital is at risk.
