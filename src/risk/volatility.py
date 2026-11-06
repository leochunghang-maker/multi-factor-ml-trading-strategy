import numpy as np
import pandas as pd

from src.config import (
    MAX_LEVERAGE,
    PAPER_VOL_LOOKBACK_DAYS,
    PAPER_VOL_MAX_SCALE,
    PAPER_VOL_MIN_SCALE,
    PAPER_VOL_TARGETING_ENABLED,
    TARGET_VOLATILITY,
    VOLATILITY_LOOKBACK,
    VOL_MAX_SCALE,
    VOL_MIN_SCALE,
    VOL_PERIODS_PER_YEAR,
    VOL_TARGETING_ENABLED,
)


def annualized_realized_volatility(
    returns: pd.Series,
    lookback: int,
    periods_per_year: float = 12,
) -> float:
    """Annualized sample stdev over the trailing ``lookback`` observations."""
    clean = returns.dropna().astype(float)
    if lookback <= 1 or len(clean) < lookback:
        return float("nan")
    window = clean.iloc[-lookback:]
    realized = float(window.std(ddof=1) * np.sqrt(periods_per_year))
    if not np.isfinite(realized) or realized <= 0:
        return float("nan")
    return realized


def volatility_target_scale(
    past_returns: pd.Series | list[float] | None,
    *,
    target_volatility: float | None = None,
    lookback: int | None = None,
    periods_per_year: float | None = None,
    min_scale: float | None = None,
    max_scale: float | None = None,
    enabled: bool | None = None,
) -> float:
    """Scale factor ``target_vol / realized_vol`` using only *past* returns.

    Callers must pass returns that exclude the period being sized (no look-ahead).
    Returns 1.0 when disabled, history is too short, or realized vol is undefined.
    """
    use = VOL_TARGETING_ENABLED if enabled is None else bool(enabled)
    if not use:
        return 1.0

    target = TARGET_VOLATILITY if target_volatility is None else float(target_volatility)
    lb = VOLATILITY_LOOKBACK if lookback is None else int(lookback)
    ppy = VOL_PERIODS_PER_YEAR if periods_per_year is None else float(periods_per_year)
    lo = VOL_MIN_SCALE if min_scale is None else float(min_scale)
    hi = VOL_MAX_SCALE if max_scale is None else float(max_scale)
    if target <= 0:
        raise ValueError("target_volatility must be positive")
    if lo < 0 or hi < lo:
        raise ValueError("vol scale bounds must satisfy 0 <= min_scale <= max_scale")

    if past_returns is None:
        return 1.0
    series = pd.Series(past_returns, dtype=float).dropna()
    realized = annualized_realized_volatility(series, lookback=lb, periods_per_year=ppy)
    if not np.isfinite(realized):
        return 1.0
    scale = target / realized
    return float(np.clip(scale, lo, hi))


def scale_weights(weights: pd.Series, scale: float) -> pd.Series:
    """Multiply portfolio weights by a volatility-target scale factor."""
    if weights.empty:
        return weights.copy()
    return weights.astype(float) * float(scale)


def apply_volatility_target(
    returns: pd.Series,
    target_volatility: float,
    lookback: int,
    periods_per_year: int = 12,
    max_leverage: float = MAX_LEVERAGE,
    min_scale: float = 0.0,
) -> pd.Series:
    # Volatility targeting scales exposure after observing recent volatility.
    # The one-period shift avoids using this month's return to size this month.
    rolling_vol = returns.rolling(lookback).std() * np.sqrt(periods_per_year)
    leverage = target_volatility / rolling_vol
    leverage = leverage.clip(lower=min_scale, upper=max_leverage).fillna(1.0)
    return (returns * leverage.shift(1)).dropna()


def paper_volatility_target_scale(
    past_daily_returns: pd.Series | list[float] | None,
    *,
    enabled: bool | None = None,
    target_volatility: float | None = None,
    lookback_days: int | None = None,
    min_scale: float | None = None,
    max_scale: float | None = None,
) -> float:
    """Daily-path vol scale for paper/sim (defaults capped at 1.0 — no leverage)."""
    use = PAPER_VOL_TARGETING_ENABLED if enabled is None else bool(enabled)
    return volatility_target_scale(
        past_daily_returns,
        target_volatility=TARGET_VOLATILITY if target_volatility is None else target_volatility,
        lookback=PAPER_VOL_LOOKBACK_DAYS if lookback_days is None else lookback_days,
        periods_per_year=252.0,
        min_scale=PAPER_VOL_MIN_SCALE if min_scale is None else min_scale,
        max_scale=PAPER_VOL_MAX_SCALE if max_scale is None else max_scale,
        enabled=use,
    )
