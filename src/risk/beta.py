"""Causal per-name betas and beta-neutral long/short leg scaling (research only).

Design (fixed a priori, see README "Beta-neutral long/short construction"):

- Benchmark SPY; beta = OLS slope of daily stock returns on SPY returns over the
  trailing ``lookback`` trading days ending at the rebalance date (inclusive —
  the close is known when the month-end book is formed), ``min_obs`` required.
- Blume shrinkage toward 1: ``beta = (1 - s) * beta_raw + s * 1`` with s = 1/3.
- ``estimator="ewma"`` (default since trial 43): the same slope estimated by
  weighted least squares with weights ``0.5 ** (k / half_life)``, k = trading days
  before the as-of close (k = 0 newest), renormalised within the same window.
  ``estimator="ols"`` reproduces the equal-weight fe8b173 betas exactly.
- Names without enough history get beta = 1 (the shrinkage target).
- Leg scaling keeps gross and sets long-leg beta equal to short-leg beta, with
  the long/short gross ratio bounded to ``[1/R, R]``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def shrink_beta(raw: pd.Series | float, shrinkage: float):
    """Blume-style shrinkage toward 1: (1 - s) * raw + s."""
    if not 0.0 <= float(shrinkage) <= 1.0:
        raise ValueError("beta shrinkage must be in [0, 1]")
    return (1.0 - float(shrinkage)) * raw + float(shrinkage)


BETA_ESTIMATORS = ("ols", "ewma")


def observation_weights(n: int, *, estimator: str = "ols", half_life: float = 63.0) -> np.ndarray:
    """Window weights, oldest first: flat for OLS, ``0.5 ** (k / half_life)`` for EWMA.

    k counts trading days back from the newest row (k = 0). Normalisation is
    irrelevant to the slope, so the raw kernel is returned.
    """
    if estimator not in BETA_ESTIMATORS:
        raise ValueError(f"beta estimator must be one of {BETA_ESTIMATORS}, got {estimator!r}")
    if n <= 0:
        return np.zeros(0)
    if estimator == "ols":
        return np.ones(n)
    if not float(half_life) > 0:
        raise ValueError("beta EWMA half-life must be positive")
    k = np.arange(n - 1, -1, -1, dtype=float)
    return 0.5 ** (k / float(half_life))


def estimate_betas_as_of(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    as_of_date,
    *,
    lookback: int = 252,
    min_obs: int = 126,
    shrinkage: float = 1.0 / 3.0,
    estimator: str = "ols",
    half_life: float = 63.0,
) -> pd.Series:
    """Shrunk OLS or EWMA-weighted betas using only rows dated on/before ``as_of_date``."""
    as_of = pd.Timestamp(as_of_date)
    bench = benchmark_returns.loc[benchmark_returns.index <= as_of].dropna().tail(int(lookback))
    window = stock_returns.reindex(bench.index)
    x = bench.to_numpy(dtype=float)
    Y = window.to_numpy(dtype=float)
    mask = np.isfinite(Y)
    n = mask.sum(axis=0)
    kernel = observation_weights(len(x), estimator=estimator, half_life=half_life)
    # Per-name weights: zero where the stock has no return; renormalised per column.
    W = np.where(mask, kernel[:, None], 0.0)
    Yz = np.where(mask, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        # Names with no data in the window yield empty columns (→ beta 1 below).
        warnings.simplefilter("ignore", category=RuntimeWarning)
        wsum = W.sum(axis=0)
        x_mean = (W * x[:, None]).sum(axis=0) / wsum
        y_mean = (W * Yz).sum(axis=0) / wsum
        dx = x[:, None] - x_mean
        cov = (W * dx * (Yz - y_mean)).sum(axis=0)
        var = (W * dx**2).sum(axis=0)
        raw = cov / var
    raw = pd.Series(raw, index=stock_returns.columns, dtype=float)
    raw[(n < int(min_obs)) | ~np.isfinite(raw.to_numpy())] = np.nan
    return shrink_beta(raw, shrinkage).fillna(1.0)


def build_beta_panel(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    dates,
    *,
    lookback: int = 252,
    min_obs: int = 126,
    shrinkage: float = 1.0 / 3.0,
    estimator: str = "ols",
    half_life: float = 63.0,
) -> pd.DataFrame:
    """Date x ticker panel of causal shrunk betas at each requested date."""
    stock_returns = prices.pct_change(fill_method=None)
    bench_returns = benchmark_prices.pct_change(fill_method=None)
    rows = [
        estimate_betas_as_of(
            stock_returns, bench_returns, date,
            lookback=lookback, min_obs=min_obs, shrinkage=shrinkage,
            estimator=estimator, half_life=half_life,
        )
        for date in pd.DatetimeIndex(dates)
    ]
    if not rows:
        return pd.DataFrame(columns=prices.columns)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))


def betas_on_date(beta_panel: pd.DataFrame | None, date, universe: pd.Index) -> pd.Series | None:
    """As-of lookup (row on/before ``date``); missing names get beta 1."""
    if beta_panel is None or beta_panel.empty:
        return None
    stamp = pd.Timestamp(date)
    prior = beta_panel.index[beta_panel.index <= stamp]
    if len(prior) == 0:
        return None
    return beta_panel.loc[prior[-1]].reindex(universe).fillna(1.0)


def portfolio_beta(weights: pd.Series, betas: pd.Series) -> float:
    aligned = betas.reindex(weights.index).fillna(1.0)
    return float((weights.astype(float) * aligned).sum())


def beta_neutral_leg_scaling(
    weights: pd.Series,
    betas: pd.Series,
    *,
    max_leg_ratio: float = 2.0,
) -> pd.Series:
    """Rescale long and short legs so long-leg beta == short-leg beta at fixed gross.

    With leg grosses L, S and leg betas B_L = Σ w_i β_i (longs), B_S = Σ |w_j| β_j
    (shorts), the neutral long/short gross ratio is r = (L * B_S) / (S * B_L).
    r is clipped to [1/max_leg_ratio, max_leg_ratio]; gross L + S is preserved.
    Returns the input unchanged when a leg is empty or has non-positive beta.
    """
    if max_leg_ratio < 1.0:
        raise ValueError("max_leg_ratio must be >= 1")
    w = weights.astype(float)
    b = betas.reindex(w.index).fillna(1.0).astype(float)
    long_mask, short_mask = w > 0, w < 0
    L = float(w[long_mask].sum())
    S = float(-w[short_mask].sum())
    if L <= 0 or S <= 0:
        return w.copy()
    B_L = float((w[long_mask] * b[long_mask]).sum())
    B_S = float((-w[short_mask] * b[short_mask]).sum())
    if B_L <= 0 or B_S <= 0:
        return w.copy()
    # Neutral condition: a*B_L = b*B_S with new grosses aL, bS → (aL)/(bS) = (L*B_S)/(S*B_L).
    target_ratio = float(np.clip((L * B_S) / (S * B_L), 1.0 / max_leg_ratio, max_leg_ratio))
    gross = L + S
    new_long = gross * target_ratio / (1.0 + target_ratio)
    new_short = gross / (1.0 + target_ratio)
    out = w.copy()
    out[long_mask] *= new_long / L
    out[short_mask] *= new_short / S
    return out


def beta_neutralize_with_sector_caps(
    weights: pd.Series,
    betas: pd.Series,
    *,
    max_leg_ratio: float,
    apply_sector_caps=None,
    max_rounds: int = 5,
    tol: float = 1e-4,
) -> pd.Series:
    """Alternate beta leg scaling and sector caps; the final step is beta scaling.

    ``apply_sector_caps`` is a callable ``w -> w`` (or None to skip sector caps).
    Stops early once re-applying the caps no longer changes the weights.
    """
    w = beta_neutral_leg_scaling(weights, betas, max_leg_ratio=max_leg_ratio)
    if apply_sector_caps is None:
        return w
    for _ in range(int(max_rounds)):
        capped = apply_sector_caps(w)
        if float((capped - w).abs().sum()) <= tol:
            return w
        w = beta_neutral_leg_scaling(capped, betas, max_leg_ratio=max_leg_ratio)
    return w


def load_benchmark_close(path: str | Path, symbol: str) -> pd.Series | None:
    path = Path(path)
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if symbol not in frame.columns:
        return None
    return frame[symbol].dropna()


def research_beta_panel(
    prices: pd.DataFrame | None = None,
    dates=None,
) -> pd.DataFrame | None:
    """Month-end causal beta panel for research L/S when ``research.beta_neutral``.

    Returns None (dollar-neutral legs) when disabled or the benchmark file is
    missing; run ``python src/data_loader.py --benchmark-only`` to create it.
    """
    from src.config import (
        BENCHMARK_PRICES_PATH,
        BETA_BENCHMARK,
        BETA_ESTIMATOR,
        BETA_EWMA_HALF_LIFE_DAYS,
        BETA_LOOKBACK_DAYS,
        BETA_MIN_OBS,
        BETA_NEUTRAL,
        BETA_SHRINKAGE,
        PRICE_DATA_PATH,
    )

    if not BETA_NEUTRAL:
        return None
    bench = load_benchmark_close(BENCHMARK_PRICES_PATH, BETA_BENCHMARK)
    if bench is None:
        print(f"WARNING: {BENCHMARK_PRICES_PATH} missing {BETA_BENCHMARK}; beta-neutral legs disabled")
        return None
    if prices is None:
        prices = pd.read_csv(PRICE_DATA_PATH, index_col=0, parse_dates=True)
    if dates is None:
        from src.portfolio.portfolio import month_end_trading_dates

        dates = month_end_trading_dates(prices)
    return build_beta_panel(
        prices, bench, dates,
        lookback=BETA_LOOKBACK_DAYS, min_obs=BETA_MIN_OBS, shrinkage=BETA_SHRINKAGE,
        estimator=BETA_ESTIMATOR, half_life=BETA_EWMA_HALF_LIFE_DAYS,
    )
