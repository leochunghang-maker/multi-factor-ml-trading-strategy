# Institutional Quant Dashboard

This lightweight Streamlit dashboard reads existing project artifacts from:

- `results/`
- `results/reports/`
- `results/simulation/`
- `results/live/`

It is intentionally read-only. It does not place orders, connect to a broker, or enable live-money trading.

## Run

```bash
streamlit run dashboard/app.py
```

## Included Views

- Performance: equity curve, drawdown, rolling Sharpe, rolling volatility.
- Portfolio risk: turnover, sector exposure, top holdings, beta exposure.
- Signals: latest live signal rankings.
- Execution: execution event history and recent simulated trades.
- Audit tables: exposure, turnover, signal, and execution records.

The dashboard is designed for clarity and research review rather than return optimization.
