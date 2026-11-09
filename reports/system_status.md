# System Status

- Generated at: 2026-11-09T10:39:17+00:00
- Overall status: ERROR

## Findings

- **ERROR** `missing_file`: Missing required file: results/live/latest_signals.csv
- **ERROR** `missing_file`: Missing required file: results/simulation/latest_account_snapshot.csv
- **ERROR** `missing_file`: Missing required file: results/simulation/execution_log.csv
- **WARNING** `data_freshness`: price_data: last date 2026-10-30 is 6 trading days before 2026-11-09 (limit 5)
- **WARNING** `data_freshness`: volume_data: last date 2026-10-30 is 6 trading days before 2026-11-09 (limit 5)
- **WARNING** `data_freshness`: benchmark_prices: last date 2026-10-30 is 6 trading days before 2026-11-09 (limit 5)
- **ERROR** `portfolio_state`: Latest account snapshot is missing.
- **WARNING** `concentration`: No latest position snapshot available.

## Latest Portfolio State

- Portfolio value: 0.00
- Cash: 0.00
- Positions: 0
- Gross exposure: 0.00%
- Net exposure: 0.00%
- Drawdown: 0.00%

## Data Inputs

- Data checks: WARNING (as of 2026-11-09; details: `python src/check_data.py`)

## Latest Execution Status

- Recent rejections: 0
- Recent failures: 0

## Notes

- Paper trading status: `reports/paper_trading_status.md`
- Structured log: `results/operations/platform_events.jsonl`
