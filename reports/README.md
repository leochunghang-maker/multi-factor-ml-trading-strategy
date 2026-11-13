# Reports

This folder contains lightweight Markdown report snapshots that are useful for GitHub review:

- `quant_research_report.md`: research methodology and analytics summary.
- `paper_trading_status.md`: latest local paper-trading operating status.
- `multi_day_simulation_report.md`: latest multi-day paper simulation summary.
- `system_status.md`: checkout health snapshot (`make health-checkout`): committed artifacts only. Gitignored paper/sim outputs are listed as INFO when absent, and do not make overall status ERROR. A local `make health` scores those artifacts when they exist on the machine.
- `paper_constraint_binding_report.md`: how often and by how much the paper guardrails bind on the
  reported multi-factor book (`make report`; `tests/test_constraint_binding.py` fails if stale).

- `data_quality_report.md`: data freshness / alignment / quality findings (`make data-check`).

Generated charts, CSV exports, JSON logs, and model artifacts live under `results/`, which is
ignored by Git except for a small set of research artifacts that are force-added so the README
numbers can be checked (`make research` regenerates all of them; `python src/check_doc_metrics.py`
verifies the quoted numbers). Bulky local run outputs stay untracked.
