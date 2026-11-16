# Paper Trading Status

This file is a checkout pointer, not a live account snapshot.

Live paper-account status is local and gitignored:

- `make paper-status` writes `results/paper/status.md` and `status.json` (alerts, last data gate, latest run, journal tail).
- Each simulated rebalance also writes `results/paper/latest_run.md` (top holdings, warnings, rejected orders for that run).

A local paper day, `make health`, or the multi-day simulation must not write NAV or positions into this path, so machine-specific account numbers are not left in the GitHub tree.

Paper trading is local and simulated only. There is no broker connection and no live-money execution.
