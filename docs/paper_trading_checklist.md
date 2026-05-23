# Paper-Trading Daily Checklist

This checklist is for the local simulated paper-trading workflow. It does not enable live-money trading and should be completed before any strategy is considered operationally ready.

## Before The Run

- Confirm local price data is fresh.
  Stale data can make signals look valid while the simulated fills use old prices.

- Confirm signal files are not empty.
  Empty files usually mean the model, data vendor, or feature pipeline failed.

- Confirm signal dates are aligned with price dates.
  Signals newer than prices create unrealistic fills; signals older than prices can trade stale recommendations.

- Check for duplicate tickers.
  Duplicate symbols can double-size a position and create unintended concentration.

- Check target weights.
  Weights should be non-negative in long-only paper mode, should not imply leverage, and should not exceed the max position size.

- Check number of target positions.
  Too many positions may indicate a broken ranking filter or an unexpected signal-generation change.

- Check missing prices.
  A trade without an executable price should be rejected, not filled with a placeholder.

- Review previous cash and positions.
  Cash and holdings should reconcile with the latest portfolio snapshot before new trades are simulated.

## During The Run

- Confirm each workflow step completes:
  data refresh, feature generation, signal generation, portfolio checks, simulated execution, risk report, and dashboard update.

- Watch for rejected orders.
  Rejections are useful controls, not just errors. They show where the simulator protected the portfolio.

- Watch turnover.
  Unusually high turnover can indicate a full portfolio flip, stale state, or a signal pipeline problem.

- Confirm long-only behavior.
  The simulator should sell only shares it owns and should never create short positions by default.

- Confirm no leverage.
  Cash should not go negative, and target weights should not exceed available portfolio value.

## After The Run

- Review `reports/paper_trading_status.md`.
  This is the daily operator summary for portfolio value, cash, holdings, turnover, drawdown, warnings, and rejected orders.

- Review `results/simulation/execution_log.csv`.
  Execution logs matter because they create an audit trail for fills, skips, failures, and rejected orders.

- Review `results/simulation/trade_history.csv`.
  Filled trades should have reasonable quantities, fill prices, transaction costs, and cash balances.

- Review `results/simulation/rejected_orders.csv`.
  Every rejected order should have a clear safety reason.

- Review `results/simulation/latest_account_snapshot.csv`.
  Check portfolio value, cash, gross exposure, net exposure, position count, and reconciliation status.

- Review `results/simulation/latest_position_snapshot.csv`.
  Confirm current holdings and top position weights are plausible.

- Review `results/simulation/allocation_summary.csv`.
  Confirm cash utilization, target exposure, realized exposure, concentration, and rejected weight impact.

- Review `results/simulation/realized_vs_target_weights.csv`.
  Constraints change realized positions because cash, transaction costs, position caps, and turnover controls affect what can actually be bought.

- Review `results/simulation/daily_performance.csv`.
  Confirm daily return, turnover, number of trades, drawdown, and risk metrics were recorded.

- Review drawdown.
  Drawdown tells you whether the simulated account is moving into a risk state that deserves reduced exposure or deeper review.

- Review dashboard charts.
  The dashboard should reflect the latest reports without needing manual file edits.

## Why This Matters

Paper trading must be validated before live trading because a strategy can look fine in research and still fail operationally through stale data, bad fills, missing prices, duplicate signals, excessive turnover, or broken cash accounting.

The goal is not to maximize paper returns. The goal is to prove that the daily operating process is explainable, reproducible, logged, and safe.
