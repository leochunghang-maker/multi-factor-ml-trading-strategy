# Quant Interview Preparation Guide

This guide is designed to help explain the project clearly in recruiter screens, quant interviews,
and technical portfolio reviews. The tone should be confident but realistic: this is a research and
engineering project, not a claim of production-ready trading performance or future returns.

## 30-Second Project Pitch

I built a quantitative equity research and simulated paper-trading platform in Python. It
takes market data, builds cross-sectional factors, trains machine learning ranking models, validates
them with walk-forward testing, applies transaction-cost-aware backtests, adds portfolio and risk
constraints, and simulates paper trading with execution logs and monitoring.

The main goal was not just to maximize a backtest. I wanted to demonstrate a full research workflow:
clean data handling, leakage-aware validation, risk reporting, operational checks, and honest
limitations. The platform includes reports, charts, a Streamlit dashboard, automated tests, and a
smoke test runner.

## 2-Minute Technical Explanation

The project starts with an S&P 100-style equity universe and historical price data. From that data,
it calculates daily returns, 12-month momentum, short-term return features, rolling volatility, and
1-month forward returns used only as labels or realized outcomes.

On the research side, I implemented both transparent factor strategies and machine learning ranking
models. The machine learning pipeline uses features such as momentum, short-term momentum, and
volatility to predict 1-month forward returns. The model output is used mainly for cross-sectional
ranking rather than exact price prediction.

For validation, the project uses an expanding-window walk-forward framework. The model is trained on
historical data available before a test year and then evaluated on the next unseen period. That helps
reduce look-ahead bias compared with a simple random train/test split.

The backtesting layer includes transaction costs, turnover tracking, benchmark comparison, drawdown,
Sharpe, volatility, and beta analytics. I also added a risk-managed backtest using volatility-aware
sizing concepts.

The paper-trading side is simulated and long-only by default. It generates live-style signals from
local files, checks signal safety, applies portfolio constraints, routes orders through a simulated
broker, records fills and rejected orders, and writes reports for review. Nothing in the workflow
places live-money trades.

The project is wrapped with documentation, reports, charts, a dashboard, tests, operational health
checks, and a smoke test runner so an interviewer can review both the quant logic and the engineering
discipline.

## Factor Research Explanation

The factor research layer is built around simple, explainable equity signals:

- 12-month momentum measures whether stocks with strong trailing performance continue to outperform.
- Short-term return features capture recent 1-month price movement.
- Rolling volatility measures recent instability and is used as a model feature.
- Forward returns are used as labels for research and model training, not as live inputs.

The reason for starting with simple factors is that they are easier to audit. A strategy with a
small number of interpretable signals is less likely to hide accidental leakage or overfitting than a
very complex feature set.

If asked why these factors were chosen, a strong answer is:

"I chose them because they are standard, interpretable starting points in cross-sectional equity
research. Momentum gives a classic trend signal, short-term returns provide recent price behavior,
and volatility helps the model understand risk or instability. I kept the feature set intentionally
small so I could focus on validation quality and portfolio realism rather than creating many
uncontrolled features."

## Machine Learning Explanation

The machine learning component trains a Random Forest model to predict 1-month forward returns using
factor-style features. The output is used for ranking stocks cross-sectionally, not for claiming
precise return forecasts.

That distinction is important. In many equity strategies, a model does not need to predict the exact
future return of each stock. It needs to rank securities well enough to support a portfolio
construction process after costs and risk controls.

The repository also includes an XGBoost training entrypoint for comparison experiments. The main
walk-forward framework described in the README is based on the Random Forest workflow.

A strong interview explanation:

"I treated machine learning as a ranking tool rather than a price prediction engine. The target is
1-month forward return, but the practical use is to rank stocks for portfolio construction. I also
kept the model relatively simple because, for this project, the more important research question was
whether the signal survives walk-forward validation and transaction costs."

## Walk-Forward Validation Explanation

Walk-forward validation trains only on data that would have been available before the test period.
The project uses an expanding-window setup, where the model is trained on earlier years and tested
on a later unseen year.

This is more realistic than randomly splitting rows because financial observations are time ordered.
A random split can accidentally allow the model to learn from future regimes or from data close to
the test period in a way that would not be available in live trading.

Strong answer:

"I used walk-forward validation to reduce look-ahead bias. The model trains on the past and tests on
a future period, then the training window expands. That better reflects how a strategy would be
researched and retrained over time. It still does not remove every bias, especially research
iteration and hyperparameter selection bias, but it is much stronger than a random split for time
series financial data."

## Risk Management Explanation

The project includes both portfolio constraints and risk analytics.

Portfolio controls include max position sizing, long-only paper-trading constraints, no-shorting
checks in the simulated broker, turnover awareness, exposure diagnostics, concentration reporting,
and liquidity-aware checks when volume data is available.

Risk analytics include drawdown, rolling volatility, rolling Sharpe, beta, turnover, exposure
reports, and risk constraint summaries. The project also includes volatility-adjusted sizing
utilities and a risk-managed backtest.

Strong answer:

"I wanted the project to show that a strategy is more than a research signal. A signal has to become a
portfolio, and a portfolio has to be monitored. That is why I added exposure checks, position
constraints, turnover tracking, volatility analysis, drawdown reporting, and paper-trading safety
checks."

## Paper Trading / Execution Explanation

The paper-trading workflow is simulated and local. It does not send real-money orders. The workflow
can generate live-style signals, validate them, apply portfolio constraints, route target positions
through a simulated broker, and write execution logs and status reports.

The simulated broker tracks cash, positions, fills, rejected orders, slippage-adjusted prices,
transaction costs, portfolio snapshots, and execution logs. The workflow also checks for stale
signals, missing prices, duplicate tickers, abnormal weights, no-shorting behavior, and no-leverage
behavior.

Strong answer:

"The execution layer is intentionally simulated. I wanted to show the operational path from signal
to target weights to orders to fills and logs, without enabling live trading. That lets me test
important safety behavior like stale signals, rejected orders, cash accounting, and no-shorting
constraints."

## Known Limitations

The project is useful as a research and engineering portfolio project, but it has important
limitations:

- The equity universe is static and S&P 100-style, so it is not free of survivorship bias.
- Historical constituents are not point-in-time.
- Yahoo Finance is convenient for research but not institutional-grade market data.
- Transaction costs and slippage are simplified.
- Short borrow costs, financing costs, taxes, locates, and hard-to-borrow constraints are not fully
  modeled.
- Intraday execution, market impact, queue position, venue selection, and partial fills are not fully
  modeled.
- Sector classifications and liquidity assumptions are simplified.
- Walk-forward validation reduces look-ahead bias but does not eliminate research-iteration bias.
- The platform is local research and simulated paper trading, not production deployment.

Strong answer:

"I would not present this as production-ready. The next institutional steps would be point-in-time
constituents, higher-quality market data, a more realistic transaction cost model, borrow and
financing assumptions, stronger capacity analysis, and more robust experiment tracking."

## Questions Interviewers May Ask

1. Why did you choose these factors?
2. How did you avoid look-ahead bias?
3. Why use machine learning instead of a simple factor model?
4. What does walk-forward validation add?
5. How are transaction costs handled?
6. What are the biggest weaknesses of the backtest?
7. Why is the paper-trading workflow long-only?
8. What does the simulated broker test?
9. How would you make this more production-ready?
10. What part of the project are you most proud of?
11. What would you improve first if you had more time?
12. How do you know the model is not just overfitting?

## Strong Answers to Those Questions

### 1. Why did you choose these factors?

I chose simple, well-known factors because they are interpretable and easier to validate. Momentum,
short-term returns, and volatility are standard building blocks in equity research. I wanted to focus
on building a disciplined research process rather than hiding complexity inside a large feature set.

### 2. How did you avoid look-ahead bias?

Forward returns are used as labels or realized outcomes, not as live signal inputs. The machine
learning validation uses walk-forward splits, where the model trains only on past data and tests on a
future period. The live signal workflow generates predictions from available feature data rather than
using future returns.

### 3. Why use machine learning instead of a simple factor model?

The simple factor model is useful as a transparent baseline. Machine learning adds a way to combine
features nonlinearly and rank stocks based on interactions between momentum, short-term return, and
volatility. I do not assume ML is automatically better; the point is to test it with walk-forward
validation and compare it with simpler research outputs.

### 4. What does walk-forward validation add?

It makes the validation closer to how a strategy would be used in practice. Instead of training and
testing on randomly mixed observations, the model learns from earlier periods and is tested on later
periods. That helps reduce look-ahead bias and gives a better sense of out-of-sample behavior.

### 5. How are transaction costs handled?

The backtests include simplified transaction-cost assumptions and turnover tracking. This is useful
for research hygiene, but it is not a complete trading cost model. A production-quality version would
model spread, market impact, liquidity, borrow costs, financing, and order size.

### 6. What are the biggest weaknesses of the backtest?

The biggest weaknesses are survivorship bias from the static universe, non-institutional data
quality, simplified transaction costs, and idealized execution assumptions. Walk-forward validation
helps, but it does not eliminate research iteration bias or guarantee future performance.

### 7. Why is the paper-trading workflow long-only?

The paper-trading workflow is long-only to keep the simulated execution layer safer and easier to
audit. Shorting requires borrow availability, borrow fees, locate failures, margin rules, and more
complex risk checks. I kept those out of scope for the local paper-trading workflow.

### 8. What does the simulated broker test?

It tests the operational mechanics around cash, positions, target weights, fills, rejected orders,
transaction costs, slippage-adjusted prices, and no-shorting behavior. The goal is not to mimic a
real broker perfectly; it is to catch accounting and workflow errors before any live integration.

### 9. How would you make this more production-ready?

I would start with point-in-time constituents, institutional data, stronger data validation, more
realistic cost and liquidity models, borrow and financing assumptions, capacity analysis, experiment
tracking, and stricter deployment controls. I would also separate research, staging, and production
configurations more formally.

### 10. What part of the project are you most proud of?

I am most proud of connecting the research workflow to operational controls. The project does not
stop at a backtest; it includes reporting, a dashboard, simulated execution, health checks, tests,
and a smoke test runner. That makes it closer to how real quant research has to be reviewed and
operated.

### 11. What would you improve first if you had more time?

I would improve the data layer first, especially point-in-time universe membership and data quality
checks. Better data would make every later result more credible. After that, I would improve the
transaction cost model and add more detailed factor exposure analysis.

### 12. How do you know the model is not just overfitting?

I cannot prove it is not overfitting. What I can do is reduce the risk: use simple features, keep the
model role focused on ranking, validate with walk-forward splits, include transaction costs, compare
with factor baselines, and state the limitations clearly. A stronger next step would be more
out-of-sample history, point-in-time constituents, and stricter model selection procedures.
