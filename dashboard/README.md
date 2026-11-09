# Quant Research and Simulated Paper-Trading Dashboard

A lightweight, read-only Streamlit dashboard over the project's local files. It does not place
orders, connect to a broker, or enable live-money trading. The paper account it shows is the
local simulated broker.

## Run

```bash
streamlit run dashboard/app.py
```

## Views and sources

- **Research backtests (headline metrics):** the multi-factor and walk-forward ML books from
  `results/multi_factor_results.csv` and `results/walk_forward_period_returns.csv` (committed).
  Monthly net returns, annualized. The figures are computed the same way README "Results
  Summary" quotes them; `tests/test_dashboard.py` checks they match. None of the strategies is
  statistically significant after multiple-testing deflation (see README).
- **Multi-factor performance and risk:** the equity curve and drawdown from the same file.
  Rolling 12-month Sharpe, volatility and beta come from `results/reports/` (local, written by
  `python src/generate_report.py` / `make report`).
- **Multi-factor book turnover and exposures:** turnover, sector, top-holdings and
  beta-exposure charts from `results/reports/` (`make report`). They are built from the held
  weights of the reported multi-factor book (same construction as
  `results/multi_factor_results.csv`, so its turnover equals the headline). "Proposed" is the
  research book. "Paper-constrained" is the same weights after the paper guardrails.
- **Simulated paper account:** model signals from `results/live/latest_signals.csv` ("live" is
  only the file name). Simulated positions, fills and the execution log come from
  `results/simulation/`. The Paper Account tab reads `results/paper/` (`status.json`, journal,
  alerts) written by `make paper-daily` / `make paper-status`.

Sections whose local files have not been generated show "No data available".

The dashboard text holds no hard-coded numbers; every figure is read from a result file.
