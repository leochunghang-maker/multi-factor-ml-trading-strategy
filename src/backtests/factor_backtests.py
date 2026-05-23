import pandas as pd

from src.config import (
    FORWARD_RETURNS_PATH,
    MEAN_REVERSION_PATH,
    MOMENTUM_LONG_COUNT,
    MOMENTUM_MIN_ASSETS,
    MOMENTUM_PATH,
    MOMENTUM_SHORT_COUNT,
    MULTI_FACTOR_LONG_COUNT,
    MULTI_FACTOR_MIN_ASSETS,
    MULTI_FACTOR_SHORT_COUNT,
    TRANSACTION_COST,
)
from src.data.io import load_factor_frame
from src.portfolio import (
    build_factor_long_short_returns,
    build_multi_factor_long_short_returns,
)
from src.reporting.metrics import calculate_performance_metrics, equity_curve


def run_momentum_backtest() -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    momentum = load_factor_frame(MOMENTUM_PATH)
    forward_returns = load_factor_frame(FORWARD_RETURNS_PATH)

    results = build_factor_long_short_returns(
        momentum,
        forward_returns,
        long_count=MOMENTUM_LONG_COUNT,
        short_count=MOMENTUM_SHORT_COUNT,
        min_assets=MOMENTUM_MIN_ASSETS,
        return_column="long_short_return",
    )
    returns = results["long_short_return"]
    return results, equity_curve(returns), calculate_performance_metrics(returns)


def run_multi_factor_backtest(
    transaction_cost: float = TRANSACTION_COST,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    momentum = load_factor_frame(MOMENTUM_PATH)
    mean_reversion = load_factor_frame(MEAN_REVERSION_PATH)
    forward_returns = load_factor_frame(FORWARD_RETURNS_PATH)

    results = build_multi_factor_long_short_returns(
        momentum,
        mean_reversion,
        forward_returns,
        long_count=MULTI_FACTOR_LONG_COUNT,
        short_count=MULTI_FACTOR_SHORT_COUNT,
        min_assets=MULTI_FACTOR_MIN_ASSETS,
        transaction_cost=transaction_cost,
    )
    returns = results["portfolio_return"]
    return results, equity_curve(returns), calculate_performance_metrics(returns)
