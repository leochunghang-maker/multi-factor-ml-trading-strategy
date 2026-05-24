# Multi-Day Paper-Trading Simulation Report

This report simulates a local paper-trading operating loop. It does not connect to a broker and does not enable live-money trading.

## Configuration

- Rebalance frequency: monthly
- Starting capital: 100000.00
- Transaction cost: 0.1000%
- Slippage: 5.00 bps
- Max positions: 10
- Max position weight: 10.00%

## Latest State

- Latest date: 2026-01-30 00:00:00
- Portfolio value: 104776.84
- Cash: 999.95
- Drawdown: -1.33%
- Gross exposure: 99.05%
- Net exposure: 99.05%
- Rolling Sharpe: N/A
- Rolling beta: N/A

## Stability Metrics

- Total simulated days: 20
- Total trades: 10
- Warning count: 18
- Average daily turnover: 0.0495
- Average gross exposure: 99.03%
- Return volatility: 23.62%

## Warning Breakdown

- CONCENTRATION: 18

## Why This Matters

- Long-term simulation matters because a process can work for one day and still fail after weeks of turnover, drift, missing data, and compounding cash effects.
- Operational stability matters because paper trading is a rehearsal for data, signal, sizing, execution, and reporting controls.
- Portfolio drift matters because market moves can push realized weights away from intended target weights between rebalances.
- Turnover monitoring matters because excessive trading can overwhelm expected signal edge with implementation costs.

## Outputs

- NAV history: `results/simulation/multi_day/portfolio_nav_history.csv`
- Exposure history: `results/simulation/multi_day/exposure_history.csv`
- Turnover history: `results/simulation/multi_day/turnover_history.csv`
- Trade history: `results/simulation/multi_day/trade_history.csv`
- Warnings: `results/simulation/multi_day/warnings.csv`
