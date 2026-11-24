# Quant Interview Preparation Guide

This guide is designed to help explain the project clearly in recruiter screens, quant interviews,
and technical portfolio reviews. The tone should be confident but realistic: this is a research and
engineering project, not a claim of production-ready trading performance or future returns.

The description below matches the current `main` branch (updated 2026-11-25). Current performance
numbers live in `README.md` and `reports/quant_research_report.md` and are checked against the
result files; this guide does not repeat those figures, so it will not go stale when artifacts are
regenerated. The headline result is unchanged: none of the current strategies shows statistically
significant alpha.

## 30-Second Project Pitch

I built a quantitative equity research and simulated paper-trading platform in Python. It takes
market data, builds cross-sectional factors, trains machine learning ranking models, and validates
them with leakage-aware walk-forward testing, including nested model selection. It then applies a
documented cost and risk layer, reports significance tests that account for the number of research
trials, and simulates paper trading with execution logs, a data gate, and operational monitoring.

The main goal was not just to maximize a backtest. I wanted to demonstrate a full research workflow:
clean data handling, leakage-aware validation, honest cost and risk assumptions, operational checks,
and clearly stated limitations. The platform includes reports, charts, a Streamlit dashboard,
automated tests, and a smoke test runner.

## 2-Minute Technical Explanation

The project starts with an S&P 100-style equity universe and historical price data. The base ticker
list is still a current snapshot, but research and paper/sim construction apply a point-in-time eligibility filter: a name needs a valid price history and month-end S&P 500 membership on that date.
That reduces survivorship bias; it does not remove it, because names that left the index and are
absent from today's ticker list are still missing, and the overlay is S&P 500 rather than official
S&P 100 constituents.

From the price history the platform calculates daily returns, 12-month momentum, short-term return
features, rolling volatility, and 1-month forward returns used only as labels or realized outcomes.

On the research side, I implemented both transparent factor strategies and machine learning ranking
models. The machine learning pipeline uses features such as momentum, short-term momentum, and
volatility to predict 1-month forward returns. The model output is used mainly for cross-sectional
ranking rather than exact price prediction. A nested anchored walk-forward procedure selects among a
small, predeclared candidate menu with a cost-aware inner score and a one-SE rule, so hyperparameters
are not chosen on the outer test years.

Validation uses month-end, non-overlapping holding-period returns, with purged and embargoed splits
around test boundaries. That is more realistic than a random train/test split, and it corrected an
earlier version that compounded overlapping 21-day returns.

Portfolio construction for the research long/short books includes sector-neutral scores with net
sector exposure caps, volatility targeting, and beta-neutral legs. The paper/sim path is long-only
and uses a lower gross exposure cap. Costs include a half-spread, square-root market impact,
name-level short-borrow tiers from an ADV and volatility proxy, financing, and an ADV participation
cap. These are transparent research proxies, not a live prime-broker feed.

The reporting layer includes transaction costs, turnover, benchmark comparison, drawdown, Sharpe,
volatility, and beta analytics, plus block-bootstrap Sharpe intervals, the probabilistic Sharpe
ratio, and a Deflated Sharpe ratio whose trial count comes from a checked-in, pre-logged trial log,
along with benchmark-relative alpha and information ratio. On the current artifacts, nothing shows
statistically significant alpha after costs and that deflation.

The paper-trading side is simulated and long-only by default. It generates signals from local
closed-session prices, checks signal safety, applies portfolio constraints, routes orders through a
simulated broker, and writes reports for review. Runs use only the last completed NYSE session,
including 1:00 p.m. early closes, and never a partial intraday bar. Nothing in the workflow places
live-money trades, and the repository has no broker client.

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
very complex feature set. Those signals then go through sector neutralization, cost-aware sizing,
and the walk-forward process rather than being judged on a raw backtest.

If asked why these factors were chosen, a strong answer is:

"I chose them because they are standard, interpretable starting points in cross-sectional equity
research. Momentum gives a classic trend signal, short-term returns provide recent price behavior,
and volatility helps the model understand risk or instability. I kept the feature set intentionally
small so I could focus on validation quality and portfolio realism rather than creating many
uncontrolled features."

## Machine Learning Explanation

The machine learning component trains models to predict 1-month forward returns using factor-style
features. The output is used for ranking stocks cross-sectionally, not for claiming precise return
forecasts.

That distinction is important. In many equity strategies, a model does not need to predict the exact
future return of each stock. It needs to rank securities well enough to support a portfolio
construction process after costs and risk controls.

The reported walk-forward path is not a single hand-picked Random Forest. Nested walk-forward
selects among a small, predeclared candidate menu that includes Random Forest and regularized linear
models, using a cost-aware inner score and a one-SE selection rule. The repository also includes an
XGBoost training entrypoint for comparison experiments. The main framework described in the README
is the nested walk-forward default.

A strong interview explanation:

"I treated machine learning as a ranking tool rather than a price prediction engine. The target is
1-month forward return, but the practical use is to rank stocks for portfolio construction. I kept
the feature set small and used nested walk-forward with a predeclared candidate menu so that model
choice would not be made on the outer test years. For this project, the more important research
question was whether the signal survives walk-forward validation, costs, and a trial-aware
significance test."

## Walk-Forward Validation Explanation

Walk-forward validation trains only on data that would have been available before the test period.
The project uses an expanding-window, month-end setup: the model is trained on earlier history,
tested on the next unseen period, and evaluated on non-overlapping holding-period returns. Training
splits are purged and embargoed around the test boundary.

A nested inner loop, also restricted to earlier data, does the model-family selection. That is
stronger than a single expanding walk-forward in which the researcher could still pick a model after
seeing the outer-test Sharpe. Research changes that alter results are pre-logged in
`research/trial_log.csv` before they are run; defaults are not chosen from outer-test outcomes.

This is more realistic than randomly splitting rows because financial observations are time ordered.
A random split can accidentally allow the model to learn from future regimes or from data close to
the test period in a way that would not be available in live trading. An earlier version of this
project used overlapping daily 21-day returns, which overstated results; that has been replaced.

Strong answer:

"I used walk-forward validation to reduce look-ahead bias. The model trains on the past and tests on
a later period, on month-end non-overlapping returns, with purge and embargo around the boundary.
Nested selection keeps hyperparameter choice off the outer test years, and a trial log feeds the
Deflated Sharpe ratio so the number of attempts is not ignored. It still does not remove every bias,
especially research-iteration bias and the fact that the candidate menu is still researcher-chosen,
but it is much stronger than a random split for time-series financial data."

## Risk Management Explanation

The project includes both portfolio constraints and risk analytics.

Research long/short books are sector-neutral with net sector exposure caps, use volatility targeting,
and are built to be beta-neutral at the leg level. Paper/sim is long-only, with position limits, a
drawdown circuit breaker that freezes rebalances, and an ADV participation cap on trade size.
Liquidity-aware checks use Yahoo volume when it is available. Name-level borrow tiers apply to
research short market value; paper/sim still accrues financing on margin debit and on any short
market value if it appears.

Risk analytics include drawdown, rolling volatility, rolling Sharpe, beta, turnover, exposure
reports, constraint-binding summaries, and benchmark-relative alpha and information ratio.

Strong answer:

"I wanted the project to show that a strategy is more than a research signal. A signal has to become a
portfolio, and a portfolio has to be monitored. That is why I added sector and beta construction
rules, volatility targeting, position and capacity limits, a drawdown halt, turnover tracking, and
paper-trading safety checks — and why I report residual exposures rather than assuming the
constraints removed them."

## Paper Trading / Execution Explanation

The paper-trading workflow is simulated and local. It does not send real-money orders, and the
repository does not include a broker client. The workflow can generate live-style signals from local
files, validate them, apply portfolio constraints, route target positions through a simulated
broker, and write execution logs and status reports.

Signals and paper runs use only completed NYSE sessions — the regular 4:00 p.m. close, or 1:00 p.m.
on published early-close days — plus a short settle buffer. A data gate blocks a partial intraday
bar. Each rebalance executes sells first and sizes buys from the cash on hand after those sells.
The simulated broker tracks cash, positions, fills, rejected orders, slippage-adjusted prices,
transaction costs, portfolio snapshots, and execution logs. The workflow also checks for stale
signals, missing prices, duplicate tickers, abnormal weights, no-shorting behavior, and no-leverage
behavior. Daily operations include a run lock, rotating backups, and a weekly review; local account
files stay out of the committed checkout reports.

Strong answer:

"The execution layer is intentionally simulated. I wanted to show the operational path from signal
to target weights to orders to fills and logs, without enabling live trading. That lets me test
important safety behavior like closed-session data only, stale signals, rejected orders, cash
accounting, and no-shorting constraints."

## Known Limitations

The project is useful as a research and engineering portfolio project, but it is not production-ready
and has important limitations:

- Residual survivorship bias remains: the base ticker list is today's S&P 100-style snapshot, so
  names that left the index and are absent from that list are still missing. Eligibility uses a
  month-end S&P 500 membership overlay, not official point-in-time S&P 100 constituents.
- Yahoo Finance is convenient for research but not institutional-grade market data, and adjusted
  history can be revised.
- Transaction costs, half-spread, square-root impact, borrow tiers, and financing are simplified
  research proxies, not a live prime-broker locate or HTB feed. Taxes are not modeled.
- Intraday execution, queue position, venue selection, and partial fills are not modeled.
- Sector classifications are a static map, not point-in-time historical GICS.
- Nested walk-forward reduces hyperparameter peeking into each outer test year, but the candidate
  menu is still researcher-chosen, and repeated research iteration can still overfit the process.
- On the current artifacts, none of the strategies shows statistically significant alpha after costs
  and trial-aware deflation.
- The platform is local research and simulated paper trading, not production deployment or
  live-money execution.

Strong answer:

"I would not present this as production-ready, and I would not present the current results as
significant alpha. The next institutional steps would be official point-in-time S&P 100 constituents,
higher-quality market data, a live borrow and locate feed rather than an ADV/volatility proxy,
point-in-time sector classifications, and more realistic intraday execution. Data-quality checks,
experiment tracking, and proxy cost and capacity models are already in the repository."

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

Forward returns are used as labels or realized outcomes, not as live signal inputs. Validation uses
month-end walk-forward splits with purge and embargo, so the model trains only on past data and is
tested on a later period. Nested selection stays inside earlier data. Signals and paper runs use
only the last completed NYSE session, never a partial intraday bar, and they never use future
returns as features.

### 3. Why use machine learning instead of a simple factor model?

The simple factor model is useful as a transparent baseline. Machine learning adds a way to combine
features nonlinearly and rank stocks based on interactions between momentum, short-term return, and
volatility. I do not assume ML is automatically better; the point is to test it with nested
walk-forward validation, after costs, and against a trial-aware significance test, and to compare it
with simpler research outputs.

### 4. What does walk-forward validation add?

It makes the validation closer to how a strategy would be used in practice. Instead of training and
testing on randomly mixed observations, the model learns from earlier periods and is tested on later
periods, on non-overlapping month-end returns. Nested inner selection keeps model choice off those
outer test years. That helps reduce look-ahead bias and gives a better sense of out-of-sample
behavior, but it does not remove research-iteration bias.

### 5. How are transaction costs handled?

Research and paper/sim paths charge a simplified but explicit cost model: a half-spread, square-root
market impact scaled by participation in 63-day ADV, name-level short-borrow tiers from an
ADV/volatility proxy, and financing on margin debit. An ADV participation cap can clip trade size,
with leftover weight left in cash. This is useful for research hygiene, but it is not a complete
live trading cost model. A production-quality version would still need a real spread and impact
calibration, a live locate and HTB feed, and taxes.

### 6. What are the biggest weaknesses of the backtest?

The biggest weaknesses are residual survivorship bias from a static ticker list, non-institutional
Yahoo data, proxy rather than live transaction and borrow costs, a static sector map, and idealized
execution with no intraday queue or partial fills. Walk-forward validation and a pre-logged trial
count help, but they do not eliminate research-iteration bias or guarantee future performance. On
top of that, the current results do not show statistically significant alpha, so I would not sell
the backtest as evidence of an edge.

### 7. Why is the paper-trading workflow long-only?

The paper-trading workflow is long-only to keep the simulated execution layer safer and easier to
audit. Shorting requires borrow availability, borrow fees, locate failures, margin rules, and more
complex risk checks. Research long/short books already charge name-level borrow tiers as a proxy,
but I kept live shorting and locates out of scope for the local paper-trading workflow.

### 8. What does the simulated broker test?

It tests the operational mechanics around cash, positions, target weights, fills, rejected orders,
transaction costs, slippage-adjusted prices, sell-before-buy cash accounting, capacity clips, and
no-shorting behavior. The goal is not to mimic a real broker perfectly; it is to catch accounting
and workflow errors before any live integration, which this repository does not have.

### 9. How would you make this more production-ready?

I would start with official point-in-time S&P 100 constituents rather than an S&P 500 overlay on a
static list, institutional data, a live borrow and locate feed, point-in-time sector maps, and more
realistic intraday execution. Data validation, experiment tracking, and proxy cost and capacity
models are already in place. I would also keep research, staging, and production configurations
separate, and I would keep this repository on local simulated paper trading rather than adding a
live-money path.

### 10. What part of the project are you most proud of?

I am most proud of connecting the research workflow to operational controls and to honest
significance testing. The project does not stop at a backtest; it includes reporting, a dashboard,
simulated execution, health checks, tests, a smoke test runner, and a trial log that feeds a
Deflated Sharpe ratio. That makes it closer to how real quant research has to be reviewed and
operated, including being willing to say that the current results are not significant.

### 11. What would you improve first if you had more time?

I would go deeper on the remaining data-universe gap: official point-in-time S&P 100 constituents,
rather than a current ticker list with an S&P 500 membership overlay, and better-than-Yahoo market
data. Data-quality checks, a trial log, and proxy cost, borrow, and capacity models are already in
the repo, so I would not spend the next unit of time inventing another strategy variant just to add
a trial. After the universe and data quality, a live borrow/locate feed or capacity analysis at a
larger NAV would be more informative than a new signal.

### 12. How do you know the model is not just overfitting?

I cannot prove it is not overfitting. What I can do is reduce the risk: use simple features, keep
the model role focused on ranking, validate with nested walk-forward and a one-SE rule, include
transaction costs, pre-log trials, deflate Sharpe for the number of attempts, compare with factor
baselines, and state the limitations clearly. The current artifacts do not show significant alpha,
which is itself a reason not to overclaim. A stronger next step would be more out-of-sample history
and official point-in-time constituents, not another loosely motivated model variant.
