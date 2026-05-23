import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def calculate_performance_metrics(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> dict[str, float]:
    curve = equity_curve(returns)
    # Annual return scales the average monthly return to a one-year estimate.
    annual_return = returns.mean() * periods_per_year
    # Annual volatility measures the typical yearly variation in returns.
    annual_volatility = returns.std() * np.sqrt(periods_per_year)
    # Sharpe ratio measures return per unit of volatility risk.
    sharpe_ratio = annual_return / annual_volatility
    # Max drawdown is the worst peak-to-trough loss in the equity curve.
    max_drawdown = (curve / curve.cummax() - 1).min()

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


def information_coefficient(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    return spearmanr(x, y)
