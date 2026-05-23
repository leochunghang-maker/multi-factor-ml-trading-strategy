import numpy as np
import pandas as pd

from src.config import PERIODS_PER_YEAR, RISK_ANALYTICS_WINDOW
from src.reporting.metrics import equity_curve


def cumulative_return(returns: pd.Series) -> pd.Series:
    # Cumulative return shows how one dollar compounds through time.
    # A value of 0.25 means the strategy is up 25% since the start.
    return equity_curve(returns) - 1


def drawdown(returns: pd.Series) -> pd.Series:
    curve = equity_curve(returns)
    # Drawdown measures the percentage loss from the previous equity high.
    # It answers: "How far underwater is the strategy from its best point?"
    return curve / curve.cummax() - 1


def rolling_drawdown(
    returns: pd.Series,
    window: int = RISK_ANALYTICS_WINDOW,
) -> pd.Series:
    curve = equity_curve(returns)
    # Rolling drawdown limits the high-water-mark calculation to a recent
    # window, which helps identify current risk stress rather than lifetime risk.
    rolling_peak = curve.rolling(window).max()
    return curve / rolling_peak - 1


def rolling_volatility(
    returns: pd.Series,
    window: int = RISK_ANALYTICS_WINDOW,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    # Volatility is the standard deviation of returns. Annualizing converts a
    # monthly risk number into a yearly risk number using sqrt(12).
    return returns.rolling(window).std() * np.sqrt(periods_per_year)


def rolling_sharpe_ratio(
    returns: pd.Series,
    window: int = RISK_ANALYTICS_WINDOW,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    # Sharpe ratio compares reward to risk: annualized return divided by
    # annualized volatility. Higher is better if returns are stable.
    rolling_return = returns.rolling(window).mean() * periods_per_year
    rolling_risk = rolling_volatility(returns, window, periods_per_year)
    return rolling_return / rolling_risk


def rolling_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = RISK_ANALYTICS_WINDOW,
) -> pd.Series:
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned.columns = ["strategy", "benchmark"]
    # Beta measures market sensitivity. A beta of 1 moves like SPY, 0 means no
    # linear market exposure, and negative beta tends to move opposite SPY.
    covariance = aligned["strategy"].rolling(window).cov(aligned["benchmark"])
    benchmark_variance = aligned["benchmark"].rolling(window).var()
    return covariance / benchmark_variance


def exposure_summary(weights: pd.Series) -> dict[str, float]:
    long_exposure = weights[weights > 0].sum()
    short_exposure = weights[weights < 0].sum()
    return {
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "gross_exposure": weights.abs().sum(),
        "net_exposure": weights.sum(),
    }


def sector_exposure(
    weights: pd.Series,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for ticker, weight in weights.items():
        rows.append({
            "ticker": ticker,
            "sector": sector_map.get(ticker, "Unknown"),
            "weight": weight,
            "gross_weight": abs(weight),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["net_exposure", "gross_exposure"])
    return frame.groupby("sector").agg(
        net_exposure=("weight", "sum"),
        gross_exposure=("gross_weight", "sum"),
    )


def concentration_summary(weights: pd.Series) -> dict[str, float]:
    gross = weights.abs().sum()
    normalized_abs = weights.abs() / gross if gross else weights.abs()
    return {
        "largest_name_weight": weights.abs().max(),
        "top_5_gross_weight": weights.abs().nlargest(5).sum(),
        "herfindahl_index": (normalized_abs ** 2).sum(),
    }


def concentration_limit_report(
    weights: pd.Series,
    sector_map: dict[str, str],
    max_single_name_weight: float,
    max_sector_gross_exposure: float,
) -> dict[str, float | bool]:
    sector = sector_exposure(weights, sector_map)
    max_sector_gross = sector["gross_exposure"].max() if not sector.empty else np.nan
    largest_name = weights.abs().max() if not weights.empty else np.nan
    return {
        "largest_name_weight": largest_name,
        "max_sector_gross_exposure": max_sector_gross,
        "single_name_limit_breached": bool(largest_name > max_single_name_weight),
        "sector_limit_breached": bool(max_sector_gross > max_sector_gross_exposure),
    }


def rolling_risk_report(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    window: int = RISK_ANALYTICS_WINDOW,
) -> pd.DataFrame:
    report = pd.DataFrame(index=returns.index)
    report["rolling_return"] = returns.rolling(window).mean() * PERIODS_PER_YEAR
    report["rolling_volatility"] = rolling_volatility(returns, window)
    report["rolling_sharpe"] = rolling_sharpe_ratio(returns, window)
    report["rolling_drawdown"] = rolling_drawdown(returns, window)
    if benchmark_returns is not None:
        report["rolling_beta"] = rolling_beta(returns, benchmark_returns, window)
    return report


def turnover_series(weights: pd.DataFrame) -> pd.Series:
    # Turnover measures how much the portfolio changes from one rebalance to
    # the next. The one-way convention divides absolute weight changes by two.
    return weights.diff().abs().sum(axis=1).div(2).dropna()


def turnover_statistics(turnover: pd.Series) -> dict[str, float]:
    # Average turnover describes normal trading intensity; max turnover flags
    # the largest rebalance; latest turnover shows the most recent portfolio churn.
    return {
        "average_turnover": turnover.mean(),
        "median_turnover": turnover.median(),
        "max_turnover": turnover.max(),
        "latest_turnover": turnover.iloc[-1] if not turnover.empty else np.nan,
    }
