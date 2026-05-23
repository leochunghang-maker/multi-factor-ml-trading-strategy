from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    LIQUIDITY_LOOKBACK_DAYS,
    MAX_ADV_PARTICIPATION,
    MAX_GROSS_EXPOSURE,
    MAX_POSITION_WEIGHT,
    MAX_SECTOR_WEIGHT,
    MAX_TURNOVER,
    MIN_AVERAGE_DAILY_DOLLAR_VOLUME,
    PERIODS_PER_YEAR,
    PORTFOLIO_SLIPPAGE_BPS,
    PORTFOLIO_TRANSACTION_COST_BPS,
    SECTOR_MAP,
    TARGET_VOLATILITY,
    VOLATILITY_LOOKBACK,
)


@dataclass(frozen=True)
class TransactionCostModel:
    """Simple configurable cost model used for research diagnostics.

    Institutional desks separate expected transaction costs from alpha because
    a profitable paper signal can disappear after commissions, spread, slippage,
    and market impact. This model is intentionally conservative and transparent.
    """

    transaction_cost_bps: float = PORTFOLIO_TRANSACTION_COST_BPS
    slippage_bps: float = PORTFOLIO_SLIPPAGE_BPS

    @property
    def round_trip_rate(self) -> float:
        return (self.transaction_cost_bps + self.slippage_bps) / 10_000

    def estimate_cost(self, traded_gross_exposure: float) -> float:
        return traded_gross_exposure * self.round_trip_rate


@dataclass(frozen=True)
class PortfolioConstraints:
    """Reusable portfolio guardrails for institutional-style exposure control."""

    max_position_weight: float = MAX_POSITION_WEIGHT
    max_sector_weight: float = MAX_SECTOR_WEIGHT
    max_gross_exposure: float = MAX_GROSS_EXPOSURE
    max_turnover: float = MAX_TURNOVER
    target_volatility: float = TARGET_VOLATILITY
    volatility_lookback: int = VOLATILITY_LOOKBACK
    min_average_daily_dollar_volume: float = MIN_AVERAGE_DAILY_DOLLAR_VOLUME
    liquidity_lookback_days: int = LIQUIDITY_LOOKBACK_DAYS
    max_adv_participation: float = MAX_ADV_PARTICIPATION
    sector_map: dict[str, str] | None = None
    cost_model: TransactionCostModel = TransactionCostModel()


def _align(previous: pd.Series | None, current: pd.Series) -> pd.Series:
    if previous is None:
        return pd.Series(0.0, index=current.index)
    return previous.reindex(current.index).fillna(0.0)


def apply_max_position_size(weights: pd.Series, max_position_weight: float) -> pd.Series:
    # Position caps prevent one stock from dominating portfolio risk if its
    # signal is wrong, suspended, or reprices before the next rebalance.
    return weights.clip(lower=-max_position_weight, upper=max_position_weight)


def apply_sector_limits(
    weights: pd.Series,
    max_sector_weight: float,
    sector_map: dict[str, str] | None = None,
) -> pd.Series:
    # Sector limits matter because stocks in the same industry often share the
    # same macro, regulatory, and earnings-cycle risks. A diversified stock list
    # can still be a concentrated sector bet.
    sectors = sector_map or SECTOR_MAP
    constrained = weights.copy()
    for sector in sorted({sectors.get(symbol, "Unknown") for symbol in constrained.index}):
        names = [symbol for symbol in constrained.index if sectors.get(symbol, "Unknown") == sector]
        sector_gross = constrained.loc[names].abs().sum()
        if sector_gross > max_sector_weight and sector_gross > 0:
            constrained.loc[names] *= max_sector_weight / sector_gross
    return constrained


def apply_gross_exposure_limit(weights: pd.Series, max_gross_exposure: float) -> pd.Series:
    gross = weights.abs().sum()
    if gross > max_gross_exposure and gross > 0:
        return weights * (max_gross_exposure / gross)
    return weights


def apply_turnover_limit(
    target_weights: pd.Series,
    previous_weights: pd.Series | None,
    max_turnover: float,
) -> tuple[pd.Series, float, bool]:
    # Turnover constraints make rebalancing more realistic: large instantaneous
    # portfolio flips can be expensive, hard to execute, and capacity-limited.
    previous = _align(previous_weights, target_weights)
    traded_gross = (target_weights - previous).abs().sum()
    if traded_gross <= max_turnover or traded_gross == 0:
        return target_weights, traded_gross, False
    blend = max_turnover / traded_gross
    constrained = previous + (target_weights - previous) * blend
    return constrained, max_turnover, True


def apply_volatility_adjusted_sizing(
    weights: pd.Series,
    daily_returns: pd.DataFrame,
    as_of_date: pd.Timestamp,
    target_volatility: float,
    lookback: int,
) -> pd.Series:
    # Volatility-adjusted sizing gives less capital to names with unstable
    # recent returns. Funds use this to avoid letting high-volatility positions
    # dominate realized portfolio risk.
    if daily_returns.empty:
        return weights
    history = daily_returns.loc[daily_returns.index <= as_of_date].tail(lookback)
    if len(history) < max(2, lookback // 2):
        return weights
    vol = history.std() * np.sqrt(PERIODS_PER_YEAR * 21)
    vol = vol.reindex(weights.index).replace(0, np.nan)
    raw_scale = target_volatility / vol
    scale = raw_scale.clip(upper=1.0).fillna(1.0)
    return weights * scale


def average_daily_dollar_volume(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None,
    as_of_date: pd.Timestamp,
    lookback: int,
) -> pd.Series:
    if volumes is None or volumes.empty:
        return pd.Series(dtype=float)
    price_window = prices.loc[prices.index <= as_of_date].tail(lookback)
    volume_window = volumes.loc[volumes.index <= as_of_date].tail(lookback)
    common_columns = price_window.columns.intersection(volume_window.columns)
    if common_columns.empty:
        return pd.Series(dtype=float)
    return (price_window[common_columns] * volume_window[common_columns]).mean()


def apply_liquidity_filter(
    weights: pd.Series,
    adv: pd.Series,
    min_average_daily_dollar_volume: float,
) -> tuple[pd.Series, int, bool]:
    # Liquidity filters reduce the chance that a backtest assumes fills in
    # names that an institutional portfolio could not trade at reasonable size.
    if adv.empty:
        return weights, 0, False
    liquid_names = adv[adv >= min_average_daily_dollar_volume].index
    constrained = weights.copy()
    removed = constrained.index.difference(liquid_names)
    constrained.loc[removed] = 0.0
    return constrained, len(removed), True


def normalize_to_original_gross(weights: pd.Series, original_gross: float) -> pd.Series:
    gross = weights.abs().sum()
    if gross == 0 or original_gross == 0:
        return weights
    return weights * min(1.0, original_gross / gross)


def constrain_weights(
    target_weights: pd.Series,
    previous_weights: pd.Series | None = None,
    daily_returns: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
    as_of_date: pd.Timestamp | None = None,
    constraints: PortfolioConstraints | None = None,
) -> tuple[pd.Series, dict[str, float | bool]]:
    constraints = constraints or PortfolioConstraints()
    date = pd.Timestamp(as_of_date) if as_of_date is not None else pd.Timestamp(target_weights.name)
    constrained = target_weights.fillna(0.0).copy()
    starting_gross = constrained.abs().sum()
    adv = pd.Series(dtype=float)
    liquidity_filter_available = False
    liquidity_removed = 0

    if prices is not None:
        adv = average_daily_dollar_volume(
            prices,
            volumes,
            date,
            constraints.liquidity_lookback_days,
        )
        constrained, liquidity_removed, liquidity_filter_available = apply_liquidity_filter(
            constrained,
            adv,
            constraints.min_average_daily_dollar_volume,
        )

    if daily_returns is not None:
        constrained = apply_volatility_adjusted_sizing(
            constrained,
            daily_returns,
            date,
            constraints.target_volatility,
            constraints.volatility_lookback,
        )

    constrained = apply_max_position_size(constrained, constraints.max_position_weight)
    constrained = apply_sector_limits(
        constrained,
        constraints.max_sector_weight,
        constraints.sector_map,
    )
    constrained = apply_gross_exposure_limit(constrained, constraints.max_gross_exposure)
    constrained = normalize_to_original_gross(constrained, starting_gross)
    constrained, traded_gross, turnover_limited = apply_turnover_limit(
        constrained,
        previous_weights,
        constraints.max_turnover,
    )

    cost = constraints.cost_model.estimate_cost(traded_gross)
    diagnostics = {
        "starting_gross_exposure": starting_gross,
        "final_gross_exposure": constrained.abs().sum(),
        "final_net_exposure": constrained.sum(),
        "max_position_weight": constrained.abs().max() if not constrained.empty else 0.0,
        "traded_gross_exposure": traded_gross,
        "turnover_limited": turnover_limited,
        "liquidity_filter_available": liquidity_filter_available,
        "liquidity_removed_positions": liquidity_removed,
        "estimated_transaction_cost": cost,
        "transaction_cost_bps": constraints.cost_model.transaction_cost_bps,
        "slippage_bps": constraints.cost_model.slippage_bps,
    }
    return constrained, diagnostics


def constrain_weight_history(
    weights: pd.DataFrame,
    daily_returns: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
    constraints: PortfolioConstraints | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    constrained_rows = []
    diagnostics = []
    previous = None
    for date, row in weights.iterrows():
        constrained, details = constrain_weights(
            row,
            previous_weights=previous,
            daily_returns=daily_returns,
            prices=prices,
            volumes=volumes,
            as_of_date=pd.Timestamp(date),
            constraints=constraints,
        )
        constrained.name = date
        constrained_rows.append(constrained)
        diagnostics.append({"date": date, **details})
        previous = constrained

    constrained_weights = pd.DataFrame(constrained_rows).fillna(0.0)
    diagnostics_frame = pd.DataFrame(diagnostics).set_index("date")
    return constrained_weights, diagnostics_frame


def load_optional_volume_data(path: str) -> pd.DataFrame | None:
    volume_path = Path(path)
    if not volume_path.exists():
        return None
    return pd.read_csv(volume_path, index_col=0, parse_dates=True)
