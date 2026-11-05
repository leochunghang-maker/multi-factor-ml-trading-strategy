# Multi-Day Paper-Trading Simulation Report

This report simulates a local paper-trading operating loop. It does not connect to a broker and does not enable live-money trading.

## Configuration

- Rebalance frequency: monthly
- Starting capital: 100000.00
- Transaction cost: 0.1000%
- Slippage: 5.00 bps
- Half-spread: 2.00 bps; sqrt impact coefficient: 100.0 bps x sqrt(notional/ADV)
- Max positions: 10
- Max position weight: 10.00%
- Drawdown halt (skip rebalance at/below): -10.00%

## Latest State

- Latest date: 2026-10-30 00:00:00
- Portfolio value: 459706.56
- Cash: 4712.83
- Drawdown: -18.08%
- Gross exposure: 98.97%
- Net exposure: 98.97%
- Rolling Sharpe: -1.892
- Rolling beta: 0.792

## Stability Metrics

- Total simulated days: 2220
- Total trades: 1391
- Warning count: 1841
- Average daily turnover: 0.0503
- Average gross exposure: 87.86%
- Return volatility: 21.05%

## Warning Breakdown

- CONCENTRATION: 1178
- DRAWDOWN_HALT: 10
- EXCESSIVE_TURNOVER: 69
- POSITION_COUNT_LIMIT: 80
- SIGNAL_WARNING: 252
- STALE_SIGNAL: 252

## Why This Matters

- Long-term simulation matters because a process can work for one day and still fail after weeks of turnover, drift, missing data, and compounding cash effects.
- Operational stability matters because paper trading is a rehearsal for data, signal, sizing, execution, and reporting controls.
- Portfolio drift matters because market moves can push realized weights away from intended target weights between rebalances.
- Turnover monitoring matters because excessive trading can overwhelm expected alpha with implementation costs.

## Outputs

- NAV history: `results/simulation/multi_day/portfolio_nav_history.csv`
- Exposure history: `results/simulation/multi_day/exposure_history.csv`
- Turnover history: `results/simulation/multi_day/turnover_history.csv`
- Trade history: `results/simulation/multi_day/trade_history.csv`
- Warnings: `results/simulation/multi_day/warnings.csv`
