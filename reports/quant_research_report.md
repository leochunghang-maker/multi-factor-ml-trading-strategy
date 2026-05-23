# Multi-Factor Machine Learning Trading Strategy

## Executive Summary

This project implements a modular quantitative equity research pipeline for a long/short strategy across S&P 100 equities. The research combines traditional cross-sectional alpha factors, machine learning return ranking, walk-forward validation, transaction-cost-aware portfolio construction, and institutional-style risk reporting.

The primary multi-factor strategy reports an annual return of **24.9%**, annual volatility of **21.4%**, Sharpe ratio of **1.16**, maximum drawdown of **-20.4%**, beta versus SPY of **-0.42**, and cumulative return of **196.1%** over the tested sample. The walk-forward machine learning strategy reports an annual return of **23.1%** and Sharpe ratio of **0.88**.

The results are promising as a research prototype, but they should be interpreted with realistic institutional caveats: the universe is based on a modern S&P 100-style equity list, prices come from Yahoo Finance, transaction costs are simplified, and execution assumptions are idealized.

## Research Objective

The objective is to evaluate whether simple equity factors and machine learning forecasts can produce a robust cross-sectional long/short strategy. The project is designed to demonstrate an institutional research workflow:

- Build clean market data and factor datasets.
- Test standalone and combined alpha signals.
- Convert signals into a transparent equal-weight long/short portfolio.
- Evaluate out-of-sample machine learning predictions with walk-forward validation.
- Report performance, risk, drawdown, market exposure, and turnover.

The goal is not to claim production readiness. The goal is to show disciplined research structure, leakage-aware validation, and honest reporting of limitations.

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

This is a realistic institutional risk control pattern, though it remains simplified because it does not model financing costs, margin constraints, borrow constraints, or intramonth risk changes.

## Institutional Realism & Limitations

### Survivorship Bias

The equity universe is based on a current or static S&P 100-style list. This can introduce survivorship bias because companies that were removed, acquired, bankrupt, or underperforming during the historical sample may be missing. A production-grade study should use point-in-time index constituents.

### Transaction Cost Simplifications

The strategy applies a flat 10 bps per side transaction cost. Real trading costs vary by liquidity, spread, volatility, market impact, short borrow fees, and order size. The current assumption is useful for research hygiene but is not a complete execution cost model.

### Yahoo Finance Limitations

Yahoo Finance is convenient for research prototypes, but it is not institutional market data. Potential issues include missing data, adjusted-price methodology differences, symbol mapping changes, restatements, and survivorship concerns.

### Execution Assumptions

The backtests assume positions can be entered at the rebalance price with equal weights and no capacity limits. They do not model slippage, borrow availability, borrow fees, financing costs, locate failures, intraday execution, corporate action edge cases, or tax effects.

### Model and Validation Constraints

The machine learning model uses a small feature set and a simple target. Walk-forward validation is more realistic than a static split, but it does not fully address regime changes, feature decay, hyperparameter selection bias, or repeated research iteration.

## Performance Metrics

The main multi-factor strategy reports:

| Metric | Value |
|---|---:|
| Annual Return | 24.92% |
| Annual Volatility | 21.43% |
| Sharpe Ratio | 1.16 |
| Max Drawdown | -20.43% |
| Cumulative Return | 196.10% |
| Turnover | 137.41% |
| Latest Turnover | 110.00% |
| Beta vs SPY | -0.42 |

The README also reports the walk-forward ML strategy at **23.1% annual return** with a **0.88 Sharpe ratio**.

Sharpe ratio measures annualized return per unit of annualized volatility. A higher Sharpe indicates more return for each unit of risk, but it does not capture tail risk, liquidity risk, or path dependence.

Drawdown measures the percentage decline from a prior equity high. Maximum drawdown is important because investors experience losses through the path of returns, not just through average return and volatility.

![Equity Curve](../results/reports/equity_curve.png)

## Risk Analytics

### Drawdown

The drawdown chart shows how far the strategy falls below its previous peak through time. This helps identify periods of stress and recovery.

![Drawdown](../results/reports/drawdown.png)

### Rolling Sharpe

Rolling Sharpe evaluates whether return efficiency is stable over time. A strategy with a strong full-sample Sharpe can still be unattractive if performance is concentrated in a short subperiod.

![Rolling Sharpe](../results/reports/rolling_sharpe.png)

### Rolling Volatility

Rolling volatility measures how unstable portfolio returns are over a recent window. Rising volatility can indicate a changing market regime or increased strategy risk.

![Rolling Volatility](../results/reports/rolling_volatility.png)

### Beta

Beta measures sensitivity to SPY. A beta of 1.0 means the strategy tends to move like SPY, while a beta near 0.0 indicates low linear market exposure. The reported beta of **-0.42** suggests the strategy had negative estimated market sensitivity over the measured rolling window.

### Turnover

Turnover measures how much the portfolio changes between rebalances. The reported turnover of **137.41%** indicates meaningful monthly portfolio churn. High turnover can erode live performance once spreads, slippage, borrow fees, and market impact are modeled more completely.

## Key Findings

- The multi-factor strategy reports attractive standalone performance with **24.9% annual return**, **1.16 Sharpe**, and **-20.4% max drawdown**.
- The walk-forward machine learning test produces lower but more realistic out-of-sample results: **23.1% annual return** and **0.88 Sharpe**.
- The architecture now separates data, features, models, portfolio construction, backtests, risk, and reporting into reusable modules.
- Risk analytics add a more institutional lens by reporting drawdown, rolling Sharpe, rolling volatility, beta, turnover, and cumulative return.
- The research remains a prototype until point-in-time constituents, more realistic trading costs, borrow assumptions, and production-grade data are added.

## Future Improvements

- Replace the static universe with point-in-time historical constituents to reduce survivorship bias.
- Add a realistic transaction cost model based on spreads, ADV, volatility, market impact, and short borrow costs.
- Add sector, industry, beta, and style-factor exposure reporting.
- Add position-level and rebalance-level audit trails.
- Add benchmark-relative attribution and factor contribution analysis.
- Add liquidity and capacity constraints.
- Add robust experiment tracking for model versions, features, parameters, and validation windows.
- Expand validation with nested cross-validation or anchored walk-forward model selection.
- Add production-quality data checks for stale prices, corporate actions, missing data, and symbol changes.
