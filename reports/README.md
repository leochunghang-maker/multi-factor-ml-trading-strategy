# Reports

This folder contains lightweight Markdown report snapshots that are useful for GitHub review:

- `quant_research_report.md`: research methodology and analytics summary.
- `paper_trading_status.md`: pointer to the local paper account files (`results/paper/status.md` via `make paper-status`; gitignored). Not a live NAV snapshot.
- `multi_day_simulation_report.md`: latest multi-day paper simulation summary.
- `system_status.md`: checkout health snapshot (`make health-checkout`): committed artifacts only. Gitignored paper/sim outputs are listed as INFO when absent, and do not make overall status ERROR. A local `make health` scores those artifacts when they exist on the machine. The checkout file has no "Generated at" timestamp, so regenerating it is a no-op when findings are unchanged.
- `paper_constraint_binding_report.md`: how often and by how much the paper guardrails bind on the
  reported multi-factor book (`make report`; `tests/test_constraint_binding.py` fails if stale).

- `data_quality_report.md`: data freshness / alignment / quality findings for the frozen research snapshot (`make data-check`). Snapshot-stable (no wall-clock last-closed-session lag); the live gap is in `results/operations/local_data_quality_report.md`.

Generated charts, CSV exports, JSON logs, and model artifacts live under `results/`. Local
paper, simulation and operations outputs are gitignored; the research artifacts that back
the README numbers are tracked via explicit `.gitignore` exceptions (no `git add -f`).
`make research` regenerates all of them; `python src/check_doc_metrics.py` verifies the
quoted numbers. `tests/test_tracked_research_artifacts.py` fails if a committed file in
`data/` or `results/` is hidden by the ignore rules.
