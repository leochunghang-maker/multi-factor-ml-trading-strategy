# Reports

This folder contains lightweight Markdown report snapshots that are useful for GitHub review:

- `quant_research_report.md`: research methodology and analytics summary.
- `paper_trading_status.md`: latest local paper-trading operating status.
- `multi_day_simulation_report.md`: latest multi-day paper simulation summary.
- `system_status.md`: latest operational health snapshot.

Generated charts, CSV exports, JSON logs, and model artifacts live under `results/` and are ignored
by Git. Keeping bulky run artifacts out of commits makes the repository easier to review while the
scripts remain responsible for regenerating outputs.
