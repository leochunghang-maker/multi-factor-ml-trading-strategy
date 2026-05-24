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

1. data refresh,
2. feature generation,
3. signal generation,
4. signal and portfolio safety checks,
5. simulated broker execution,
6. risk report generation,
7. dashboard/report update,
8. system health status update.

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
