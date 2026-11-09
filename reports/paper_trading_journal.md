# Paper Trading Journal

This journal is for ongoing review of the local simulated paper-trading workflow. Add entries after
reviewing `reports/paper_trading_status.md`, `reports/system_status.md`, live signal outputs, and
execution logs.

The daily numbers are journaled automatically: every completed run of
`scripts/run_daily_paper_trading.py` writes one row per signal date to `results/paper/journal.md`
(and `journal.csv`) with NAV, PnL, trades, risk flags (drawdown halt, ADV clips, vol scale),
data-check status and reconciliation status. Those files are local and gitignored (see README,
"Paper-run operations"). Use this file for manual review notes only.

No manual review entries have been added yet.

Use the template in `docs/paper_trading_journal_template.md`.
