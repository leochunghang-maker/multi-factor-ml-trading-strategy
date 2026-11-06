# Multi-Factor Machine Learning Trading Strategy

## Executive Summary

This project implements a modular quantitative equity research pipeline for a long/short strategy across S&P 100 equities. The research combines traditional cross-sectional factors, machine learning return ranking, walk-forward validation, transaction-cost-aware portfolio construction, and risk reporting.

Headline multi-factor metrics in this checkout are taken from committed
`results/multi_factor_results.csv` over **2019-01-31 → 2026-09-30** with the PIT
eligibility filter: annual return **7.57%**, annual volatility **28.44%**, Sharpe
ratio **0.27**, maximum drawdown **-42.14%**, beta versus SPY about **-0.30**, and
cumulative return **29.53%** (matching ending wealth **1.2953** in
`results/multi_factor_equity_curve.csv`). Corrected walk-forward metrics are in
`results/walk_forward_metrics.csv` (annual return **17.92%**, Sharpe **0.65**,
max drawdown **-25.95%**, cumulative **103.08%** over 57 periods). Older narrative
snapshots (multi-factor Sharpe **1.16** / **0.39**, walk-forward **0.88** / **0.70**)
are superseded.

The results are research outputs from a prototype workflow and should be interpreted with realistic caveats: the base ticker list is still a modern S&P 100-style snapshot (residual survivorship), eligibility uses an S&P 500 membership proxy rather than official S&P 100 PIT constituents, prices come from Yahoo Finance, transaction costs are simplified, and execution assumptions are idealized. They are not investment promises or live-trading claims.

## Research Objective

The objective is to evaluate whether simple equity factors and machine learning forecasts can support a cross-sectional long/short research strategy. The project is designed to demonstrate a disciplined research workflow:

- Build clean market data and factor datasets.
- Test standalone and combined research signals.
- Convert signals into a transparent equal-weight long/short portfolio.
- Evaluate out-of-sample machine learning predictions with walk-forward validation.
- Report performance, risk, drawdown, market exposure, and turnover.

The goal is not to claim production readiness or future returns. The goal is to show disciplined research structure, leakage-aware validation, and honest reporting of limitations.

## Data Pipeline

The project downloads historical adjusted equity prices using Yahoo Finance and stores derived datasets as local CSV files under `data/`. The research universe is described as S&P 100 equities over a 2018-2025 sample period.

The pipeline creates:

- Daily returns.
- 12-month momentum.
- 1-month mean reversion.
- 1-month forward returns.
- Machine learning feature/target datasets.

Forward returns are used only as labels or realized outcomes. They are not used when forming live portfolio scores, which is essential for leakage prevention.

## Alpha Factors

### 12-Month Momentum

Momentum measures whether stocks that have performed well over roughly the prior trading year continue to outperform. The project calculates:

```text
Momentum = Price(t) / Price(t - 252) - 1
```

This is a classic trend-following factor. Positive values indicate stronger trailing performance.

### Short-Term Momentum

Short-term momentum measures the recent 1-month return:

```text
ShortTermMomentum = Price(t) / Price(t - 21) - 1
```

This feature is used in the machine learning dataset as a recent continuation signal.

### Mean Reversion

The multi-factor strategy uses a 1-month mean-reversion input defined as the negative of the recent 1-month return. In the combined score, the strategy rewards stronger 12-month momentum and penalizes the mean-reversion component according to the existing implementation.

### Rolling Volatility

Rolling volatility is calculated as the 21-day standard deviation of daily returns. It is used as a machine learning feature to help the model distinguish between stable and unstable return environments.

## Portfolio Construction

The project uses equal-weight long/short portfolios. At each rebalance date, stocks are ranked cross-sectionally by a signal or model prediction.

For the multi-factor strategy:

- Rank stocks by the composite factor score.
- Go long the top 10 stocks.
- Go short the bottom 10 stocks.
- Rebalance monthly.
- Apply transaction costs of 10 bps per side.

Equal weighting makes the portfolio construction transparent and easy to audit. It also avoids overfitting through complex optimizers, but it does not account for liquidity, borrow costs, factor constraints, sector neutrality, or capacity.

## Machine Learning Framework

The machine learning pipeline trains a Random Forest Regressor to predict 1-month forward returns using:

- 12-month momentum.
- Short-term momentum.
- 21-day rolling volatility.

The model is used to rank stocks, not to forecast exact prices. This distinction matters: in cross-sectional equity research, rank quality is often more important than point forecast accuracy.

The project also includes an XGBoost training entrypoint, but the reported machine learning strategy results in the README are based on the Random Forest walk-forward framework.

## Walk-Forward Validation

Walk-forward validation is used to reduce overfitting and make testing more realistic. The model is trained on historical data available before a test year and then evaluated on that unseen test year.

The reported walk-forward schedule is:

| Train Period | Test Period |
|---|---|
| 2018-2021 | 2022 |
| 2018-2022 | 2023 |
| 2018-2023 | 2024 |
| 2018-2024 | 2025 |
| 2018-2025 | 2026 (through 2026-09-30) |

This expanding-window process simulates how a research model would be retrained as new data becomes available.

Leakage prevention is handled by:

- Sorting observations chronologically.
- Training only on dates before the test period.
- Using forward returns only as labels and realized outcomes.
- Rebuilding predictions for unseen test periods before portfolio formation.

## Risk Management

The project includes a volatility-targeting overlay. Volatility targeting scales exposure based on recently observed portfolio volatility. If realized volatility is high, exposure is reduced; if volatility is low, exposure can increase subject to a leverage cap.

The implementation uses:

- Target volatility: **15%**.
- Lookback window: **6 months**.
- Maximum leverage: **2.0x**.
- A one-period lag on leverage so the current month is not sized using its own realized return.

This is a common risk control pattern, though it remains simplified because it does not model financing costs, margin constraints, borrow constraints, or intramonth risk changes.

## Research Realism & Limitations

### Nested Walk-Forward Selection

Outer year folds still use expanding training with purge/embargo. Inside each outer training
window, an anchored inner year loop scores a small candidate set (`ridge_1`, `rf_shallow`,
`rf_default`) on a month-end subset, then refits the winner on the full outer train sample
before predicting the outer test year.

Default inner score (`research.nested_inner_score=cost_aware_sharpe`):

`score = mean(net_t) / std(net_t) * sqrt(12)`, where each month-end
`net_t = side*(mean r_long - mean r_short) - ||Δw||_1 * transaction_cost - financing(w_t)`
and financing uses the configured stock-borrow / rebate / cash-financing bps. Alternatives:
`icir = mean(IC)/std(IC)`, `rank_ic = mean(IC)`.

Regenerated 2026-11-06 with `cost_aware_sharpe`: **2022–2026 chose `rf_default`** (under raw
`rank_ic`, 2023–2026 had preferred `ridge_1`). This reduces test-year hyperparameter peeking
but does not eliminate candidate-menu or research-iteration bias.

### Survivorship Bias

The base equity list is still today's S&P 100-style snapshot, so names that left the index and
are not in `TICKERS` remain missing. Mitigation now in code (`src/data/universe.py`):

1. Tradability — require a valid price, a minimum history window (63 trading days), and treat
   trailing all-NaN stretches after the last print as delisted.
2. Index membership overlay — month-end S&P 500 membership for the same tickers
   (`data/universe_sp500_membership.csv`, derived from
   [chinobing/historical_sp500_constituents](https://github.com/chinobing/historical_sp500_constituents)).
   Because S&P 100 ⊆ S&P 500, non-members are excluded on that date (e.g. PLTR before Sep 2024,
   UBER before Dec 2023, TSLA before Dec 2020).

This reduces look-ahead membership bias for late joiners but does **not** restore historical
members that are absent from today's list. A production-grade study still needs official
point-in-time S&P 100 constituents (and delisted return series).

### Transaction Cost Simplifications

The strategy applies a flat 10 bps per side transaction cost. Real trading costs vary by liquidity, spread, volatility, market impact, short borrow fees, and order size. The current assumption is useful for research hygiene but is not a complete execution cost model.

### Borrow / Financing Costs

Research long/short portfolios charge a flat annualized stock-borrow fee on short gross exposure (`execution.annual_borrow_bps`, default 100 bps/year ≈ 8.33 bps per month on a 100% short book). Optional short-proceeds rebate and cash financing default to 0. Paper/sim remains long-only, so short borrow is usually zero there. This is not name-level hard-to-borrow pricing.

### Yahoo Finance Limitations

Yahoo Finance is convenient for research prototypes, but it is not institutional market data. Potential issues include missing data, adjusted-price methodology differences, symbol mapping changes, restatements, and survivorship concerns.

### Execution Assumptions

The backtests assume positions can be entered at the rebalance price with equal weights and no capacity limits. They do not model slippage, borrow availability, borrow fees, financing costs, locate failures, intraday execution, corporate action edge cases, or tax effects.

### Model and Validation Constraints

The machine learning model uses a small feature set and a simple target. Walk-forward validation is more realistic than a static split, but it does not fully address regime changes, feature decay, hyperparameter selection bias, or repeated research iteration.

## Performance Metrics

Source of truth for the multi-factor backtest in this repository checkout:
`results/multi_factor_results.csv` (`portfolio_return`), sample
**2019-01-31 → 2026-09-30** (93 month-end rows). Ending wealth in
`results/multi_factor_equity_curve.csv` is **1.3997**.

| Metric | Value from committed multi-factor results |
|---|---:|
| Annual Return | 7.57% |
| Annual Volatility | 28.44% |
| Sharpe Ratio | 0.27 |
| Max Drawdown | -42.14% |
| Cumulative Return | 29.53% |
| Average One-Way Turnover | 134.73% |
| Latest Turnover | 140.00% |
| Beta vs SPY (summary report) | 0.72 |

`make report` regenerates the same family of numbers into local
`results/reports/summary_report.md` / `.csv` (gitignored).

### Walk-forward ML (regenerated 2026-11-05)

From `results/walk_forward_metrics.csv`, regenerated from Yahoo Finance prices (2018-01-02 →
2026-10-30) with month-end rebalancing, non-overlapping month-end → next-month-end returns,
purge/embargo of 21-day labels, and research long/short gross exposure 2.0 (100% long + 100% short, `research.long_short_gross_exposure` in `config/platform_config.json`). This is distinct from the paper-trading cap `MAX_GROSS_EXPOSURE = 1.0`, which these research backtests do not apply.
Sample: **2022-01-31 → 2026-09-30** (57 month-end periods).

| Metric | Walk-forward ML |
|---|---:|
| Annual return | 17.92% |
| Annual volatility | 27.60% |
| Sharpe ratio | 0.65 |
| Max drawdown | -25.95% |
| Cumulative return | 103.08% |

### Older narrative snapshot (not current)

Earlier versions of this report quoted annual return **24.92%**, volatility **21.43%**,
Sharpe **1.16**, max drawdown **-20.43%**, cumulative return **196.10%**, turnover
**137.41%**, latest turnover **110.00%**, beta **-0.42**, plus walk-forward ML
**23.1%** annual return / Sharpe **0.88**. Those figures are retained here only as
historical narrative; they are not reproduced by the regenerated artifacts. Corrected walk-forward metrics are in
`results/walk_forward_metrics.csv`.

Sharpe ratio measures annualized return per unit of annualized volatility. A higher Sharpe indicates more return for each unit of risk, but it does not capture tail risk, liquidity risk, or path dependence.

Drawdown measures the percentage decline from a prior equity high. Maximum drawdown is important because investors experience losses through the path of returns, not just through average return and volatility.

![Strategy vs benchmark equity curve](../results/strategy_vs_benchmark.png)

Risk charts under `results/reports/` are produced locally by `make report` and are not always present on GitHub.

## Risk Analytics

### Drawdown

The drawdown chart shows how far the strategy falls below its previous peak through time. This helps identify periods of stress and recovery.

_Drawdown chart: run `make report` to generate `results/reports/drawdown.png` locally._

### Rolling Sharpe

Rolling Sharpe evaluates whether return efficiency is stable over time. A strategy with a strong full-sample Sharpe can still be unattractive if performance is concentrated in a short subperiod.

_Rolling Sharpe chart: run `make report` to generate `results/reports/rolling_sharpe.png` locally._

### Rolling Volatility

Rolling volatility measures how unstable portfolio returns are over a recent window. Rising volatility can indicate a changing market regime or increased strategy risk.

_Rolling volatility chart: run `make report` to generate `results/reports/rolling_volatility.png` locally._

### Beta

Beta measures sensitivity to SPY. A beta of 1.0 means the strategy tends to move like SPY, while a beta near 0.0 indicates low linear market exposure. The latest rolling beta of about **0.72** (from the committed multi-factor return series versus SPY) suggests negative estimated market sensitivity over the measured window.

### Turnover

Turnover measures how much the portfolio changes between rebalances. Average one-way turnover of about **134.73%** (latest **140.00%**) indicates meaningful monthly portfolio churn. High turnover can erode live performance once spreads, slippage, borrow fees, and market impact are modeled more completely.

## Key Findings

- From regenerated `results/multi_factor_results.csv`, the multi-factor series shows **7.57% annual return**, **0.27 Sharpe**, **-42.14% max drawdown**, and **29.53% cumulative return** over 2019-01-31 → 2026-09-30 (93 periods).
- Corrected walk-forward ML (month-end, purge/embargo, research L/S gross 2.0) shows **17.92% annual return**, **0.65 Sharpe**, **-25.95% max drawdown**, **103.08% cumulative** over 2022-01-31 → 2026-09-30 (57 periods); older **23.1% / 0.88 Sharpe** figures are superseded.
- The architecture now separates data, features, models, portfolio construction, backtests, risk, and reporting into reusable modules.
- Risk analytics add review context by reporting drawdown, rolling Sharpe, rolling volatility, beta, turnover, and cumulative return.
- The research remains a prototype until official S&P 100 PIT constituents (not only an S&P 500 proxy), more realistic trading costs, borrow assumptions, and production-grade data are added.

## Future Improvements

- Extend eligibility to official S&P 100 point-in-time constituents and include delisted/removed names in the price panel.
- Add a realistic transaction cost model based on spreads, ADV, volatility, market impact, and short borrow costs.
- Add sector, industry, beta, and style-factor exposure reporting.
- Add position-level and rebalance-level audit trails.
- Add benchmark-relative attribution and factor contribution analysis.
- Add liquidity and capacity constraints.
- Add robust experiment tracking for model versions, features, parameters, and validation windows.
- Widen the nested candidate menu carefully (still with purge/embargo) or add cost-aware inner scoring beyond rank IC.
- Add production-quality data checks for stale prices, corporate actions, missing data, and symbol changes.
