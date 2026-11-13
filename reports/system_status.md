# System Status

- Generated at: 2026-11-13T16:56:12+00:00
- Overall status: OK
- Scope: checkout (committed artifacts only; local paper/sim outputs are gitignored)

## Findings

- **INFO** `operational_file`: Local operational file not in this checkout: results/live/latest_signals.csv; create it with `make signals`.
- **INFO** `operational_file`: Local operational file not in this checkout: results/simulation/latest_account_snapshot.csv; create it with `make paper`.
- **INFO** `operational_file`: Local operational file not in this checkout: results/simulation/execution_log.csv; create it with `make paper`.
- **INFO** `operational_file`: Local operational file not in this checkout: results/reports/summary_report.csv; create it with `make report`.
- **INFO** `portfolio_state`: No local simulation snapshot in this checkout (results/simulation/ is gitignored; run `make paper` or the multi-day simulation).
- **INFO** `concentration`: No local position snapshot in this checkout (results/simulation/ is gitignored; run `make paper` or the multi-day simulation).

## Latest Portfolio State

- No local simulation snapshot in this checkout (`results/simulation/` is gitignored).

## Data Inputs

- Data checks: OK (as of 2026-10-30; details: `python src/check_data.py`)

## Latest Execution Status

- Recent rejections: 0
- Recent failures: 0

## Notes

- Paper trading status: `reports/paper_trading_status.md`
- Structured log: `results/operations/platform_events.jsonl`
