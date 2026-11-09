"""Statistical significance of backtest Sharpe ratios (how much could be luck).

Implements:

- Stationary (Politis & Romano, 1994) and circular block bootstrap confidence
  intervals for the annualized Sharpe ratio. Block resampling keeps short-range
  autocorrelation and volatility clustering that an i.i.d. bootstrap destroys.
- Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012):

      PSR(SR*) = Φ( (SR - SR*) * sqrt(T - 1)
                    / sqrt(1 - γ3*SR + (γ4 - 1)/4 * SR²) )

  with per-period (non-annualized) SR, sample skewness γ3 and *non-excess*
  kurtosis γ4 of the return series.
- Deflated Sharpe Ratio (Bailey & López de Prado, 2014): PSR evaluated at the
  expected maximum Sharpe of N independent trials under the null of zero skill,

      SR0 = sqrt(V[SR]) * ((1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N e)))

  where V[SR] is the cross-trial variance of per-period Sharpe estimates and
  γ ≈ 0.5772 is the Euler–Mascheroni constant.

All functions are pure and seedable; nothing here touches orders or brokers.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

EULER_MASCHERONI = 0.5772156649015329
SUPPORTED_BOOTSTRAP_METHODS = ("stationary", "circular")


def _clean(returns: Iterable[float]) -> np.ndarray:
    arr = np.asarray(pd.Series(returns, dtype=float).dropna(), dtype=float)
    return arr[np.isfinite(arr)]


def per_period_sharpe(returns: Iterable[float]) -> float:
    """mean / sample std (ddof=1) of per-period returns; NaN if undefined."""
    arr = _clean(returns)
    if arr.size < 2:
        return float("nan")
    sd = float(arr.std(ddof=1))
    if sd <= 1e-15:
        return float("nan")
    return float(arr.mean() / sd)


def annualized_sharpe(returns: Iterable[float], periods_per_year: float) -> float:
    return float(per_period_sharpe(returns) * np.sqrt(float(periods_per_year)))


def default_block_length(n_obs: int) -> int:
    """Rule-of-thumb mean block length ~ T^(1/3) (at least 1)."""
    return max(1, int(round(float(n_obs) ** (1.0 / 3.0))))


def block_bootstrap_indices(
    n_obs: int,
    *,
    block_length: float,
    n_resamples: int,
    method: str = "stationary",
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an (n_resamples, n_obs) array of resampled positions (wrap-around).

    ``stationary``: block lengths are geometric with mean ``block_length``
    (each step starts a new block with probability 1/block_length).
    ``circular``: fixed-length blocks with uniformly random starts.
    """
    if n_obs < 1:
        raise ValueError("n_obs must be positive")
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    name = str(method).strip().lower()
    if name not in SUPPORTED_BOOTSTRAP_METHODS:
        raise ValueError(f"Unknown bootstrap method {method!r}; expected {SUPPORTED_BOOTSTRAP_METHODS}")

    if name == "circular":
        length = int(round(block_length))
        n_blocks = int(np.ceil(n_obs / length))
        starts = rng.integers(0, n_obs, size=(n_resamples, n_blocks))
        offsets = np.arange(length)
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_resamples, -1)
        return idx[:, :n_obs] % n_obs

    p_new = 1.0 / float(block_length)
    idx = np.empty((n_resamples, n_obs), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_obs, size=n_resamples)
    new_block = rng.random((n_resamples, n_obs)) < p_new
    fresh = rng.integers(0, n_obs, size=(n_resamples, n_obs))
    for t in range(1, n_obs):
        idx[:, t] = np.where(new_block[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n_obs)
    return idx


def bootstrap_sharpe_ci(
    returns: Iterable[float],
    *,
    periods_per_year: float,
    block_length: float | None = None,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    method: str = "stationary",
    seed: int = 42,
) -> dict[str, float]:
    """Percentile block-bootstrap CI for the annualized Sharpe ratio."""
    arr = _clean(returns)
    out = {
        "sharpe_annualized": annualized_sharpe(arr, periods_per_year),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "bootstrap_se": float("nan"),
        "prob_sharpe_le_zero": float("nan"),
        "block_length": float("nan"),
        "n_resamples": int(n_resamples),
        "n_obs": int(arr.size),
    }
    if arr.size < 3:
        return out
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    length = float(block_length) if block_length else float(default_block_length(arr.size))
    rng = np.random.default_rng(int(seed))
    idx = block_bootstrap_indices(
        arr.size, block_length=length, n_resamples=int(n_resamples), method=method, rng=rng
    )
    samples = arr[idx]
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = samples.mean(axis=1) / sd * np.sqrt(float(periods_per_year))
    sharpes = sharpes[np.isfinite(sharpes)]
    alpha = (1.0 - confidence) / 2.0
    out.update({
        "ci_lower": float(np.quantile(sharpes, alpha)),
        "ci_upper": float(np.quantile(sharpes, 1.0 - alpha)),
        "bootstrap_se": float(sharpes.std(ddof=1)),
        "prob_sharpe_le_zero": float(np.mean(sharpes <= 0.0)),
        "block_length": length,
    })
    return out


def probabilistic_sharpe_ratio(
    sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skewness: float,
    kurt: float,
) -> float:
    """PSR with per-period Sharpes, sample skew and *non-excess* kurtosis."""
    if n_obs < 2 or not all(np.isfinite(x) for x in (sharpe, benchmark_sharpe, skewness, kurt)):
        return float("nan")
    denom_sq = 1.0 - skewness * sharpe + (kurt - 1.0) / 4.0 * sharpe ** 2
    if denom_sq <= 0:
        return float("nan")
    z = (sharpe - benchmark_sharpe) * np.sqrt(n_obs - 1.0) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """SR0: expected max per-period Sharpe of ``n_trials`` skill-less trials."""
    if n_trials < 1 or not np.isfinite(sharpe_variance) or sharpe_variance < 0:
        return float("nan")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    z = (1.0 - g) * norm.ppf(1.0 - 1.0 / n_trials) + g * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sharpe_variance) * z)


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skewness: float,
    kurt: float,
    *,
    n_trials: int,
    sharpe_variance: float,
) -> tuple[float, float]:
    """Return ``(DSR, SR0)``; per-period inputs (variance of per-period SRs)."""
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    return probabilistic_sharpe_ratio(sharpe, sr0, n_obs, skewness, kurt), sr0


def trial_sharpe_variance(
    annualized_trial_sharpes: Iterable[float],
    periods_per_year: float,
) -> float:
    """Cross-trial variance of *per-period* Sharpes from annualized trial Sharpes."""
    arr = _clean(annualized_trial_sharpes)
    if arr.size < 2:
        return float("nan")
    return float(np.var(arr, ddof=1) / float(periods_per_year))


def significance_summary(
    returns: Iterable[float],
    *,
    periods_per_year: float,
    n_trials: int,
    sharpe_variance: float,
    block_length: float | None = None,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    method: str = "stationary",
    seed: int = 42,
) -> dict[str, float]:
    """Sharpe, bootstrap CI, PSR(0) and DSR for one return series."""
    arr = _clean(returns)
    boot = bootstrap_sharpe_ci(
        arr,
        periods_per_year=periods_per_year,
        block_length=block_length,
        n_resamples=n_resamples,
        confidence=confidence,
        method=method,
        seed=seed,
    )
    sr = per_period_sharpe(arr)
    g3 = float(skew(arr, bias=True)) if arr.size > 2 else float("nan")
    g4 = float(kurtosis(arr, fisher=False, bias=True)) if arr.size > 3 else float("nan")
    psr0 = probabilistic_sharpe_ratio(sr, 0.0, arr.size, g3, g4)
    dsr, sr0 = deflated_sharpe_ratio(
        sr, arr.size, g3, g4, n_trials=n_trials, sharpe_variance=sharpe_variance
    )
    return {
        **boot,
        "sharpe_per_period": sr,
        "skewness": g3,
        "kurtosis": g4,
        "psr_vs_zero": psr0,
        "n_trials": int(n_trials),
        "trial_sharpe_variance_per_period": float(sharpe_variance),
        "sr0_annualized": float(sr0 * np.sqrt(float(periods_per_year))) if np.isfinite(sr0) else float("nan"),
        "deflated_sharpe_ratio": dsr,
    }
