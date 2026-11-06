import numpy as np
import pandas as pd

from src.config import (
    MAX_LEVERAGE,
    PAPER_VOL_ESTIMATOR,
    PAPER_VOL_EWMA_HALF_LIFE,
    PAPER_VOL_LOOKBACK_DAYS,
    PAPER_VOL_MAX_SCALE,
    PAPER_VOL_MIN_SCALE,
    PAPER_VOL_TARGETING_ENABLED,
    TARGET_VOLATILITY,
    VOLATILITY_LOOKBACK,
    VOL_ESTIMATOR,
    VOL_EWMA_HALF_LIFE,
    VOL_MAX_SCALE,
    VOL_MIN_SCALE,
    VOL_PERIODS_PER_YEAR,
    VOL_TARGETING_ENABLED,
)

_VALID_ESTIMATORS = frozenset({"realized", "ewma"})


def ewma_lambda_from_half_life(half_life: float) -> float:
    """Map a half-life (in periods) to an EWMA decay factor ``lambda``."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return float(0.5 ** (1.0 / float(half_life)))


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


def annualized_ewma_volatility(
    returns: pd.Series,
    *,
    lam: float | None = None,
    half_life: float | None = None,
    periods_per_year: float = 12,
    min_obs: int = 2,
) -> float:
    """Annualized EWMA volatility forecast from a causal return series.

    Recurrence: ``var_t = lambda * var_{t-1} + (1-lambda) * r_t^2``, seeded with
    the first return squared. The returned value is the forecast after consuming
    every observation in ``returns`` — callers must pass only past returns.
    """
    if lam is None:
        if half_life is None:
            raise ValueError("Provide lam or half_life for EWMA volatility")
        lam = ewma_lambda_from_half_life(half_life)
    lam = float(lam)
    if not (0.0 < lam < 1.0):
        raise ValueError("EWMA lambda must be in (0, 1)")

    clean = returns.dropna().astype(float)
    if min_obs < 2:
        min_obs = 2
    if len(clean) < min_obs:
        return float("nan")

    variance = float(clean.iloc[0] ** 2)
    one_minus = 1.0 - lam
    for value in clean.iloc[1:].to_numpy(dtype=float):
        variance = lam * variance + one_minus * float(value) ** 2

    if not np.isfinite(variance) or variance <= 0:
        return float("nan")
    forecast = float(np.sqrt(variance * periods_per_year))
    if not np.isfinite(forecast) or forecast <= 0:
        return float("nan")
    return forecast


def forecast_volatility(
    past_returns: pd.Series,
    *,
    estimator: str,
    lookback: int,
    periods_per_year: float,
    ewma_half_life: float | None = None,
    ewma_lambda: float | None = None,
) -> float:
    """Dispatch realized vs EWMA vol using only the provided past returns."""
    name = str(estimator).strip().lower()
    if name not in _VALID_ESTIMATORS:
        raise ValueError(f"Unknown vol estimator {estimator!r}; expected realized|ewma")
    if name == "realized":
        return annualized_realized_volatility(
            past_returns,
            lookback=lookback,
            periods_per_year=periods_per_year,
        )
    # Require at least ``lookback`` points before trusting the EWMA forecast,
    # matching the realized estimator's warm-up so early periods stay unscaled.
    if len(past_returns.dropna()) < lookback:
        return float("nan")
    return annualized_ewma_volatility(
        past_returns,
        lam=ewma_lambda,
        half_life=ewma_half_life,
        periods_per_year=periods_per_year,
        min_obs=max(2, lookback),
    )


def volatility_target_scale(
    past_returns: pd.Series | list[float] | None,
    *,
    target_volatility: float | None = None,
    lookback: int | None = None,
    periods_per_year: float | None = None,
    min_scale: float | None = None,
    max_scale: float | None = None,
    enabled: bool | None = None,
    estimator: str | None = None,
    ewma_half_life: float | None = None,
    ewma_lambda: float | None = None,
) -> float:
    """Scale factor ``target_vol / forecast_vol`` using only *past* returns.

    Callers must pass returns that exclude the period being sized (no look-ahead).
    Returns 1.0 when disabled, history is too short, or the forecast is undefined.
    """
    use = VOL_TARGETING_ENABLED if enabled is None else bool(enabled)
    if not use:
        return 1.0

    target = TARGET_VOLATILITY if target_volatility is None else float(target_volatility)
    lb = VOLATILITY_LOOKBACK if lookback is None else int(lookback)
    ppy = VOL_PERIODS_PER_YEAR if periods_per_year is None else float(periods_per_year)
    lo = VOL_MIN_SCALE if min_scale is None else float(min_scale)
    hi = VOL_MAX_SCALE if max_scale is None else float(max_scale)
    est = VOL_ESTIMATOR if estimator is None else str(estimator).strip().lower()
    half_life = VOL_EWMA_HALF_LIFE if ewma_half_life is None else float(ewma_half_life)
    if target <= 0:
        raise ValueError("target_volatility must be positive")
    if lo < 0 or hi < lo:
        raise ValueError("vol scale bounds must satisfy 0 <= min_scale <= max_scale")

    if past_returns is None:
        return 1.0
    series = pd.Series(past_returns, dtype=float).dropna()
    forecast = forecast_volatility(
        series,
        estimator=est,
        lookback=lb,
        periods_per_year=ppy,
        ewma_half_life=half_life,
        ewma_lambda=ewma_lambda,
    )
    if not np.isfinite(forecast):
        return 1.0
    scale = target / forecast
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
    estimator: str = "realized",
    ewma_half_life: float | None = None,
) -> pd.Series:
    # Volatility targeting scales exposure after observing recent volatility.
    # The one-period shift avoids using this month's return to size this month.
    name = str(estimator).strip().lower()
    if name == "ewma":
        half_life = float(ewma_half_life) if ewma_half_life is not None else float(lookback)
        lam = ewma_lambda_from_half_life(half_life)
        clean = returns.astype(float)
        variances = np.empty(len(clean), dtype=float)
        variances[:] = np.nan
        if len(clean) > 0 and np.isfinite(clean.iloc[0]):
            var = float(clean.iloc[0] ** 2)
            variances[0] = var
            one_minus = 1.0 - lam
            for i in range(1, len(clean)):
                r = clean.iloc[i]
                if not np.isfinite(r):
                    variances[i] = var
                    continue
                var = lam * var + one_minus * float(r) ** 2
                variances[i] = var
        rolling_vol = pd.Series(np.sqrt(variances * periods_per_year), index=returns.index)
    else:
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
    estimator: str | None = None,
    ewma_half_life: float | None = None,
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
        estimator=PAPER_VOL_ESTIMATOR if estimator is None else estimator,
        ewma_half_life=(
            PAPER_VOL_EWMA_HALF_LIFE if ewma_half_life is None else ewma_half_life
        ),
    )
