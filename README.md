# Multi-Factor Machine Learning Trading Strategy

## Project Overview

This project develops a quantitative equity trading strategy across S&P 100 equities using factor investing and machine learning techniques.

The system combines:
- momentum factors,
- short-term continuation signals,
- volatility features,
- machine learning return prediction,
- and walk-forward validation.

The objective is to build a realistic institutional-style quantitative research pipeline with robust out-of-sample testing and transaction-cost-aware portfolio construction.

---

# Strategy Architecture

## Data Pipeline

Historical equity data is downloaded using Yahoo Finance and processed into factor datasets and machine learning features.

Universe:
- S&P 100 equities
- Daily adjusted prices
- 2018–2025 sample period

---

# Alpha Factors

## 1. 12-Month Momentum

Measures long-term trend persistence.

Formula:

Momentum = Price(t) / Price(t-252) - 1

---

## 2. Short-Term Momentum

Measures 1-month continuation effects.

Formula:

ShortTermMomentum = Price(t) / Price(t-21) - 1

---

## 3. Rolling Volatility

21-day rolling standard deviation of returns.

Used as:
- risk feature,
- market regime indicator,
- and ML predictor.

---

# Machine Learning Pipeline

Model:
- Random Forest Regressor

Features:
- 12M momentum
- short-term momentum
- rolling volatility

Target:
- 1-month forward return

The model predicts cross-sectional stock return rankings rather than absolute prices.

---

# Validation Methodology

## Walk-Forward Validation

To avoid overfitting and temporal leakage, the project uses expanding-window walk-forward testing:

| Train Period | Test Period |
|---|---|
| 2018–2021 | 2022 |
| 2018–2022 | 2023 |
| 2018–2023 | 2024 |
| 2018–2024 | 2025 |

At each step:
1. retrain model,
2. predict unseen future returns,
3. rebalance portfolio monthly.

---

# Portfolio Construction

Portfolio:
- Long top 10 predicted stocks
- Short bottom 10 predicted stocks
- Monthly rebalancing
- Equal-weighted positions

Transaction costs:
- 10 bps per side

Benchmark:
- SPY ETF

---

# Results

## Multi-Factor Strategy

- Annual Return: 24.9%
- Sharpe Ratio: 1.16
- Max Drawdown: -20.4%

## Walk-Forward ML Strategy

- Annual Return: 23.1%
- Sharpe Ratio: 0.88

---

# Key Research Findings

- Combining long-term and short-term momentum improved Sharpe ratio significantly versus single-factor approaches.
- Initial ML backtests showed inflated performance due to overlapping exposure bias.
- Walk-forward validation produced more realistic out-of-sample performance estimates.
- Volatility was the most important ML feature in Random Forest feature importance analysis.

---

# Project Structure

```text
src/
├── data_loader.py
├── factors.py
├── research_test.py
├── multi_factor_backtest.py
├── ml_dataset.py
├── train_ml_model.py
├── ml_backtest.py
└── walk_forward_ml_backtest.py