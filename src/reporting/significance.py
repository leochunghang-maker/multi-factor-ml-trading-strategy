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


# ---------------------------------------------------------------------------
# Benchmark-relative tests (alpha / beta with Newey-West SEs, information ratio)
# ---------------------------------------------------------------------------

def newey_west_lags(n_obs: int) -> int:
    """Newey–West (1994) rule of thumb: floor(4 * (T/100)^(2/9))."""
    return max(0, int(np.floor(4.0 * (float(n_obs) / 100.0) ** (2.0 / 9.0))))


def ols_newey_west(y: Iterable[float], x: Iterable[float], lags: int | None = None) -> dict[str, float]:
    """OLS ``y = a + b x + e`` with Bartlett-kernel HAC (Newey–West) standard errors.

    Matches ``statsmodels`` ``OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L,
    "use_correction": False})``: meat = Σ_l w_l Σ_t (X_t e_t)(X_{t-l} e_{t-l})ᵀ,
    w_l = 1 - l/(L+1), no small-sample scaling.
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(x_arr)
    y_arr, x_arr = y_arr[mask], x_arr[mask]
    n = y_arr.size
    nan = float("nan")
    out = {"alpha": nan, "beta": nan, "alpha_se": nan, "beta_se": nan,
           "alpha_t": nan, "beta_t": nan, "n_obs": int(n), "hac_lags": 0, "r_squared": nan}
    if n < 3 or float(np.var(x_arr)) <= 1e-18:
        return out
    L = newey_west_lags(n) if lags is None else int(lags)
    L = max(0, min(L, n - 1))
    X = np.column_stack([np.ones(n), x_arr])
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y_arr
    resid = y_arr - X @ coef
    scores = X * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, L + 1):
        weight = 1.0 - lag / (L + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum())
    out.update({
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "alpha_se": float(se[0]),
        "beta_se": float(se[1]),
        "alpha_t": float(coef[0] / se[0]) if se[0] > 0 else nan,
        "beta_t": float(coef[1] / se[1]) if se[1] > 0 else nan,
        "hac_lags": int(L),
        "r_squared": float(1.0 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else nan,
    })
    return out


def causal_rolling_beta(
    strategy: pd.Series,
    benchmark: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Beta known *before* each period: rolling cov/var over prior periods, shifted 1."""
    min_periods = max(3, window // 2) if min_periods is None else int(min_periods)
    cov = strategy.rolling(window, min_periods=min_periods).cov(benchmark)
    var = benchmark.rolling(window, min_periods=min_periods).var()
    return (cov / var).shift(1)


def benchmark_relative_summary(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    periods_per_year: float,
    n_trials: int,
    ir_sharpe_variance: float,
    hac_lags: int | None = None,
    causal_beta_window: int = 36,
    block_length: float | None = None,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    method: str = "stationary",
    seed: int = 42,
) -> dict[str, float]:
    """Alpha/beta (HAC), information ratio with bootstrap CI / PSR / DSR, hedged Sharpe.

    For long/short books the information ratio versus a long-only benchmark mostly
    measures "short the market"; the alpha t-stat and causal hedged Sharpe are the
    meaningful tests there. ``ir_sharpe_variance`` is also used to deflate the
    causal hedged Sharpe.

    - Regression on aligned per-period returns: ``r_s = a + b r_b + e``;
      alpha is annualized as ``a * periods_per_year``.
    - Active return ``r_s - r_b``; IR = annualized Sharpe of active returns.
      IR bootstrap / PSR / DSR reuse :func:`significance_summary` on that series.
    - ``hedged_sharpe_in_sample``: Sharpe of ``r_s - b̂ r_b`` with the full-sample
      beta (uses look-ahead in the hedge ratio; descriptive only).
    - ``hedged_sharpe_causal``: Sharpe of ``r_s - β_{t-1} r_b`` with a rolling
      beta estimated only from earlier periods (tradable in principle).
    """
    aligned = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    reg = ols_newey_west(aligned["s"], aligned["b"], lags=hac_lags)
    active = aligned["s"] - aligned["b"]
    ir = significance_summary(
        active,
        periods_per_year=periods_per_year,
        n_trials=n_trials,
        sharpe_variance=ir_sharpe_variance,
        block_length=block_length,
        n_resamples=n_resamples,
        confidence=confidence,
        method=method,
        seed=seed,
    )
    hedged_is = aligned["s"] - reg["beta"] * aligned["b"] if np.isfinite(reg["beta"]) else aligned["s"] * np.nan
    beta_t = causal_rolling_beta(aligned["s"], aligned["b"], causal_beta_window)
    hedged_causal = (aligned["s"] - beta_t * aligned["b"]).dropna()
    hedged_causal_summary = significance_summary(
        hedged_causal,
        periods_per_year=periods_per_year,
        n_trials=n_trials,
        sharpe_variance=ir_sharpe_variance,
        block_length=block_length,
        n_resamples=n_resamples,
        confidence=confidence,
        method=method,
        seed=seed,
    )
    return {
        "n_obs": int(len(aligned)),
        "hac_lags": reg["hac_lags"],
        "alpha_annualized": reg["alpha"] * float(periods_per_year),
        "alpha_t_hac": reg["alpha_t"],
        "beta": reg["beta"],
        "beta_se_hac": reg["beta_se"],
        "r_squared": reg["r_squared"],
        "information_ratio": ir["sharpe_annualized"],
        "ir_ci_lower": ir["ci_lower"],
        "ir_ci_upper": ir["ci_upper"],
        "ir_prob_le_zero": ir["prob_sharpe_le_zero"],
        "ir_psr_vs_zero": ir["psr_vs_zero"],
        "ir_sr0_annualized": ir["sr0_annualized"],
        "ir_deflated": ir["deflated_sharpe_ratio"],
        "n_trials": int(n_trials),
        "hedged_sharpe_in_sample": annualized_sharpe(hedged_is, periods_per_year),
        "hedged_sharpe_causal": hedged_causal_summary["sharpe_annualized"],
        "hedged_causal_ci_lower": hedged_causal_summary["ci_lower"],
        "hedged_causal_ci_upper": hedged_causal_summary["ci_upper"],
        "hedged_causal_n_obs": hedged_causal_summary["n_obs"],
        "hedged_causal_psr_vs_zero": hedged_causal_summary["psr_vs_zero"],
        "hedged_causal_deflated": hedged_causal_summary["deflated_sharpe_ratio"],
        "mean_causal_beta": float(beta_t.dropna().mean()) if beta_t.notna().any() else float("nan"),
    }
