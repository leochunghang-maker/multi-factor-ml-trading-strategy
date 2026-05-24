# Paper Trading Journal Template

Use this template after each simulated paper-trading review. The goal is to build operational memory:
what changed, what looked unusual, what should be checked next, and whether the system behaved as
expected.

This journal is for local simulated paper trading only. It does not enable live trading and should
not be treated as trading advice.

## Daily Review Workflow

1. Run or review the paper-trading workflow output.
2. Open `reports/paper_trading_status.md` and record the latest portfolio status.
3. Open `reports/system_status.md` and note any health warnings.
4. Check `results/live/latest_signals.csv` for signal date, target weights, and unusual names.
5. Check `results/simulation/rejected_orders.csv` and `results/simulation/execution_log.csv` for
   rejected orders, missing prices, stale signals, or unexpected fills.
6. Compare current holdings with `results/simulation/latest_position_snapshot.csv`.
7. Record qualitative observations and action items in `reports/paper_trading_journal.md`.
8. Do not override risk checks or enable live-money execution from this journal process.

## Journal Entry Template

```markdown
## YYYY-MM-DD

- Date:
- Signal date:
- Portfolio value:
- Cash:
- Positions:
- Turnover:
- Drawdown:
- Rejected orders:
- Unusual warnings:

### Qualitative Notes

- 

### Action Items

- 
```

## Review Prompts

- Are signal dates current, or are signals stale?
- Did portfolio value, cash, or number of positions move unexpectedly?
- Were any orders rejected? If yes, why?
- Did turnover look unusually high or low?
- Did drawdown change materially?
- Are there missing prices, duplicate tickers, abnormal weights, or stale inputs?
- Are any warnings repeated across multiple days?
- Is the dashboard consistent with the status report and execution logs?
- Is any follow-up needed before the next simulated run?
