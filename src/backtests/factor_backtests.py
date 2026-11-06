import pandas as pd

from src.config import (
    BORROW_TIER_MAP_PATH,
    FORWARD_RETURNS_PATH,
    MEAN_REVERSION_PATH,
    MOMENTUM_LONG_COUNT,
    MOMENTUM_MIN_ASSETS,
    MOMENTUM_PATH,
    MOMENTUM_SHORT_COUNT,
    MULTI_FACTOR_LONG_COUNT,
    MULTI_FACTOR_MIN_ASSETS,
    MULTI_FACTOR_SHORT_COUNT,
    RESEARCH_LONG_SHORT_GROSS_EXPOSURE,
    TRANSACTION_COST,
)
from src.data.io import load_factor_frame, load_price_data
from src.data.universe import load_default_eligibility_mask
from src.execution.borrow_tiers import research_borrow_bps_panel, snapshot_borrow_tiers
from src.portfolio.constraints import load_optional_volume_data
from src.config import VOLUME_DATA_PATH
from src.portfolio import (
    build_factor_long_short_returns,
    build_multi_factor_long_short_returns,
)
from src.reporting.metrics import calculate_performance_metrics, equity_curve


def run_momentum_backtest() -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    momentum = load_factor_frame(MOMENTUM_PATH)
    forward_returns = load_factor_frame(FORWARD_RETURNS_PATH)
    eligibility = load_default_eligibility_mask()
    borrow_panel = research_borrow_bps_panel()

    results = build_factor_long_short_returns(
        momentum,
        forward_returns,
        long_count=MOMENTUM_LONG_COUNT,
        short_count=MOMENTUM_SHORT_COUNT,
        min_assets=MOMENTUM_MIN_ASSETS,
        return_column="long_short_return",
        eligibility=eligibility,
        borrow_bps_panel=borrow_panel,
    )
    returns = results["long_short_return"]
    return results, equity_curve(returns), calculate_performance_metrics(returns)


def run_multi_factor_backtest(
    transaction_cost: float = TRANSACTION_COST,
    gross_exposure: float = RESEARCH_LONG_SHORT_GROSS_EXPOSURE,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    momentum = load_factor_frame(MOMENTUM_PATH)
    mean_reversion = load_factor_frame(MEAN_REVERSION_PATH)
    forward_returns = load_factor_frame(FORWARD_RETURNS_PATH)
    eligibility = load_default_eligibility_mask()
    prices = load_price_data()
    volumes = load_optional_volume_data(VOLUME_DATA_PATH)
    borrow_panel = research_borrow_bps_panel(prices, volumes)
    snapshot_borrow_tiers(prices, volumes, path=BORROW_TIER_MAP_PATH)

    results = build_multi_factor_long_short_returns(
        momentum,
        mean_reversion,
        forward_returns,
        long_count=MULTI_FACTOR_LONG_COUNT,
        short_count=MULTI_FACTOR_SHORT_COUNT,
        min_assets=MULTI_FACTOR_MIN_ASSETS,
        transaction_cost=transaction_cost,
        gross_exposure=gross_exposure,
        eligibility=eligibility,
        borrow_bps_panel=borrow_panel,
    )
    returns = results["portfolio_return"]
    return results, equity_curve(returns), calculate_performance_metrics(returns)
