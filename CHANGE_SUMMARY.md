# Refactor Change Summary

- `src/config.py`: Added centralized configuration for universe, paths, model features, targets, split dates, portfolio sizes, transaction costs, benchmark, and risk settings.
- `src/data/io.py`: Added shared CSV loading helpers for prices, factors, ML datasets, predictions, and return series.
- `src/features/factors.py`: Centralized factor calculations and ML dataset construction with beginner-friendly factor comments.
- `src/models/ml.py`: Centralized model constants, train/test splitting, Random Forest training, XGBoost training, prediction evaluation, and feature importance formatting.
- `src/portfolio/portfolio.py`: Centralized equal-weight long/short selection, factor portfolio construction, multi-factor scoring, and ML prediction portfolio construction.
- `src/reporting/metrics.py`: Centralized equity curve, annual return, annual volatility, Sharpe ratio, max drawdown, and information coefficient calculations.
- `src/reporting/analytics.py`: Added rolling Sharpe, rolling volatility, rolling drawdown, cumulative return, rolling beta, and turnover analytics.
- `src/reporting/charts.py`: Centralized equity curve, drawdown, rolling Sharpe, rolling volatility, and strategy-vs-benchmark chart generation.
- `src/reporting/summary.py`: Added professional summary report generation for institutional portfolio analytics.
- `src/generate_report.py`: Added report entrypoint that saves charts and summary files to `results/reports/`.
- `src/backtests/factor_backtests.py`: Added reusable momentum and multi-factor backtest engines.
- `src/backtests/ml_backtests.py`: Added reusable ML prediction and walk-forward ML backtest engines.
- `src/backtests/benchmark.py`: Added reusable SPY benchmark equity curve loader.
- `src/risk/volatility.py`: Centralized volatility targeting logic with beginner-friendly risk comments.
- `src/*_backtest.py`, `src/train_*.py`, `src/factors.py`, `src/ml_dataset.py`, `src/research_test.py`: Converted scripts into import-safe `main()` entrypoints that preserve existing strategy behavior and outputs.
- `src/multi_factor_backtest.py`: Removed the accidentally embedded ML training block so the file only runs the multi-factor strategy.
- `src/metrics.py` and `src/portfolio.py`: Kept compatibility wrappers for the new shared metrics and portfolio modules.
- `README.md`: Fixed heading and code-block formatting, updated project structure, and documented the new module layout.
