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

1. data refresh (`src/data_loader.py` full refresh through today; skip with `--skip-data-refresh`),
2. pre-run data gate (staleness and any other data ERROR block the run; exit code 2),
3. idempotency check (a completed signal date is skipped; `--force` replaces the latest run),
4. feature generation,
5. signal generation (from the gated local prices),
6. signal and portfolio safety checks,
7. simulated broker execution with reconciliation against the previous run, a dated snapshot in
   `results/paper/runs/<date>/`, and ledger/journal rows in `results/paper/`,
8. risk report generation,
9. dashboard/report update and system health status update (reconciliation mismatches are ERROR).

Paper-run state under `results/paper/` is local and gitignored. See README, "Paper-run operations".

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
