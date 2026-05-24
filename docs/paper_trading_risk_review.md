# Paper Trading Risk Review

This guide explains how to review the local simulated paper-trading system before and after each
run. It is practical by design: the goal is to catch stale data, unsafe allocations, rejected orders,
and unusual portfolio behavior before trusting any output.

This project is for local research and simulated paper trading only. Do not use this process to
enable real-money trading.

## Files to Review

- Latest paper-trading status: `reports/paper_trading_status.md`
- System health status: `reports/system_status.md`
- Latest live signals: `results/live/latest_signals.csv`
- Account snapshot: `results/simulation/latest_account_snapshot.csv`
- Position snapshot: `results/simulation/latest_position_snapshot.csv`
- Allocation summary: `results/simulation/allocation_summary.csv`
- Realized vs target weights: `results/simulation/realized_vs_target_weights.csv`
- Execution log: `results/simulation/execution_log.csv`
- Rejected-order history: `results/simulation/rejected_orders.csv`
- Structured operational log: `results/operations/platform_events.jsonl`

## Before Running the System

Check these items before running `scripts/run_daily_paper_trading.py`:

- Confirm the project is still in paper-trading mode.
- Confirm no real-money broker endpoint or live-order mode has been enabled.
- Confirm `.env` credentials, if present, are paper credentials only.
- Check that `config/platform_config.json` has reasonable risk limits.
- Check that `results/live/latest_signals.csv` exists if you are using existing local signals.
- Check the signal date. Stale signals should not be used without understanding why.
- Check that target weights are non-negative and do not imply leverage.
- Check that the expected number of positions matches the strategy expectation.
- Check that price data exists for the signal tickers.
- Read the latest `reports/system_status.md` and understand any warnings before continuing.

If any of these checks look wrong, stop and investigate before running the workflow.

## After Running the System

Start with `reports/paper_trading_status.md`. For the latest reviewed run, it recorded:

- Portfolio value: `113855.54`
- Cash: `1196.86`
- Number of positions: `10`
- Turnover: `0.0000`
- Drawdown: `-0.11%`
- Trades filled: `0`
- Rejected orders: `0`

Then compare that status report with:

- `results/simulation/latest_account_snapshot.csv`
- `results/simulation/allocation_summary.csv`
- `results/simulation/execution_log.csv`
- `reports/system_status.md`

The latest execution log shows the final portfolio run completed with `filled=0`, `rejected=0`,
`pnl=0.00`, and `turnover=0.000000`. That means the latest run did not need trades because positions
were already close to target weights.

## How to Interpret Rejected Orders

Rejected orders are not automatically bad. In this project, they often mean a safety rule worked.

Common examples:

- `Insufficient cash`: the simulated broker blocked a buy because the account did not have enough
  cash.
- `Order would breach max position weight`: the simulated broker blocked a trade because it would
  exceed the configured position limit.
- Missing or stale price: the order should not proceed because the system cannot value the trade
  safely.

Important nuance: `reports/paper_trading_status.md` reports rejected orders for the latest run, while
`results/simulation/rejected_orders.csv` can contain historical rejected orders from earlier runs.
If the status report says `Rejected orders: 0` but the system health report warns about recent
rejections, check timestamps before treating it as a current failure.

Rejected orders require review when:

- They appear in the latest run.
- The same symbol is rejected repeatedly.
- The reason is unclear.
- Rejections cause the realized portfolio to differ materially from target weights.
- Rejections are caused by missing data or stale signals.

## How to Interpret Turnover

Turnover measures how much the portfolio changes during a run. High turnover usually means the
strategy is trading more aggressively, which can increase transaction costs and implementation risk.

In the latest status report, turnover was `0.0000`, which is consistent with the execution log
showing positions were already close to target weights.

Turnover deserves attention when:

- It jumps sharply compared with recent runs.
- It is high because many names changed at once.
- It is high while expected signal edge or signal confidence is weak.
- It is high because of stale data or repeated rebalancing.
- It causes the simulated portfolio to hit risk limits or cash constraints.

Low turnover is usually easier to manage, but it should still be checked. Very low turnover can also
mean the system did not update, signals were stale, or trades were skipped unexpectedly.

## How to Interpret Cash Balance

Cash shows how much of the portfolio is not invested. In the latest account snapshot, cash was about
`1.05%` of portfolio value, close to the expected cash weight of about `1%`.

Cash is healthy when:

- It is close to the configured or expected cash buffer.
- It is enough to absorb small rounding and transaction-cost effects.
- It is not growing because many orders are being rejected.

Cash deserves review when:

- Cash is much higher than expected, which can mean orders failed or target weights were not reached.
- Cash is near zero, which can make new buys fail because of transaction costs or rounding.
- Cash is negative, which would indicate leverage or accounting problems in a long-only simulation.

## How to Interpret Drawdown

Drawdown measures how far the portfolio is below its previous high. It is a practical measure of
pain and path risk.

In the latest paper-trading status report, drawdown was `-0.11%`, which is small. The number should
still be tracked over time because small losses can become meaningful if they persist or accelerate.

Drawdown deserves review when:

- It breaches a pre-defined stop-review threshold.
- It worsens for several runs in a row.
- It is caused by concentrated exposure to one stock, sector, or factor.
- It appears together with high turnover, rejected orders, stale data, or missing prices.
- It differs between the status report, dashboard, and account snapshot.

## When to Stop the Strategy

Stop the simulated workflow and investigate before continuing if any of the following happen:

- Signal dates are stale or in the future.
- Price dates do not match the expected signal date.
- Live-money trading appears enabled or a real-money endpoint is configured.
- API credentials are missing when an external paper broker path is expected.
- A latest run has unexplained rejected orders.
- Cash is unexpectedly high, negative, or inconsistent with target weights.
- Turnover is much higher than expected.
- Drawdown breaches the review threshold you set for paper trading.
- Portfolio value does not reconcile with cash plus positions.
- Any no-shorting, no-leverage, max-position, or stale-signal check fails.
- Reports disagree with execution logs and the reason is not understood.

Stopping the strategy in this context means pausing the local simulated workflow, not liquidating
real positions. This repository does not enable live-money trading.

## Daily Review Checklist

Use this short checklist after each simulated run:

- Status report reviewed.
- System status reviewed.
- Signal date checked.
- Price date checked.
- Portfolio value reconciled.
- Cash balance close to expected buffer.
- Position count as expected.
- Turnover reasonable.
- Drawdown acceptable.
- Latest rejected orders reviewed.
- Historical rejected-order file checked by timestamp.
- Realized weights close to target weights.
- Any warnings copied into `reports/paper_trading_journal.md`.
- Action items recorded before the next run.
