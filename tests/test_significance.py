"""Known-answer and reproducibility checks for PSR / DSR / block bootstrap."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from src.reporting.significance import (
    EULER_MASCHERONI,
    block_bootstrap_indices,
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
    significance_summary,
    trial_sharpe_variance,
)


def test_dsr_matches_bailey_lopez_de_prado_numerical_example() -> None:
    # Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", worked example:
    # annual SR 2.5 on T=1250 daily obs, skew -3, kurtosis 10, N=100 trials,
    # V[annual SR]=0.5 → SR0 ≈ 0.1132 (daily) and DSR ≈ 0.9004.
    sr = 2.5 / np.sqrt(250)
    variance = 0.5 / 250
    dsr, sr0 = deflated_sharpe_ratio(sr, 1250, -3.0, 10.0, n_trials=100, sharpe_variance=variance)
    assert sr0 == pytest.approx(0.1132, abs=5e-4)
    assert dsr == pytest.approx(0.9004, abs=2e-3)


def test_psr_reduces_to_normal_formula_for_gaussian_moments() -> None:
    sr, t = 0.2, 61
    expected = norm.cdf(sr * np.sqrt(t - 1) / np.sqrt(1 + 0.5 * sr ** 2))
    assert probabilistic_sharpe_ratio(sr, 0.0, t, 0.0, 3.0) == pytest.approx(expected)
    # PSR at the observed SR itself is exactly 0.5.
    assert probabilistic_sharpe_ratio(sr, sr, t, -1.0, 6.0) == pytest.approx(0.5)


def test_negative_skew_and_fat_tails_lower_psr() -> None:
    base = probabilistic_sharpe_ratio(0.2, 0.0, 60, 0.0, 3.0)
    worse = probabilistic_sharpe_ratio(0.2, 0.0, 60, -1.5, 9.0)
    assert worse < base


def test_expected_max_sharpe_edge_cases_and_monotonicity() -> None:
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert np.isnan(expected_max_sharpe(0, 0.01))
    assert expected_max_sharpe(10, 0.0) == 0.0
    vals = [expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in zip(vals, vals[1:]))
    z = (1 - EULER_MASCHERONI) * norm.ppf(1 - 1 / 10) + EULER_MASCHERONI * norm.ppf(1 - 1 / (10 * np.e))
    assert expected_max_sharpe(10, 0.04) == pytest.approx(0.2 * z)


def test_more_trials_never_raise_dsr() -> None:
    dsrs = [deflated_sharpe_ratio(0.2, 60, 0.0, 3.0, n_trials=n, sharpe_variance=0.002)[0]
            for n in (1, 5, 50, 500)]
    assert all(a >= b for a, b in zip(dsrs, dsrs[1:]))
    # One trial: DSR equals PSR vs zero.
    assert dsrs[0] == pytest.approx(probabilistic_sharpe_ratio(0.2, 0.0, 60, 0.0, 3.0))


def test_trial_sharpe_variance_converts_to_per_period() -> None:
    assert trial_sharpe_variance([0.2, 0.4, 0.6], 12) == pytest.approx(np.var([0.2, 0.4, 0.6], ddof=1) / 12)
    assert np.isnan(trial_sharpe_variance([0.5], 12))


def test_bootstrap_is_reproducible_with_seed_and_changes_with_seed() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.01, 0.04, 80)
    a = bootstrap_sharpe_ci(r, periods_per_year=12, block_length=4, n_resamples=500, seed=7)
    b = bootstrap_sharpe_ci(r, periods_per_year=12, block_length=4, n_resamples=500, seed=7)
    c = bootstrap_sharpe_ci(r, periods_per_year=12, block_length=4, n_resamples=500, seed=8)
    assert a == b
    assert a["ci_lower"] != c["ci_lower"]
    assert a["ci_lower"] < a["sharpe_annualized"] < a["ci_upper"]


@pytest.mark.parametrize("method", ["stationary", "circular"])
def test_block_indices_shape_range_and_contiguity(method: str) -> None:
    rng = np.random.default_rng(1)
    idx = block_bootstrap_indices(30, block_length=5, n_resamples=200, method=method, rng=rng)
    assert idx.shape == (200, 30)
    assert idx.min() >= 0 and idx.max() < 30
    steps = (np.diff(idx, axis=1) % 30) == 1
    # Mean block length ≈ 5 → ~80% of steps continue the current block.
    assert 0.7 < steps.mean() < 0.9


def test_block_length_one_stationary_is_iid_like() -> None:
    rng = np.random.default_rng(2)
    idx = block_bootstrap_indices(50, block_length=1, n_resamples=200, method="stationary", rng=rng)
    steps = (np.diff(idx, axis=1) % 50) == 1
    assert steps.mean() < 0.05


def test_bootstrap_ci_widens_with_longer_blocks_on_autocorrelated_returns() -> None:
    rng = np.random.default_rng(3)
    eps = rng.normal(0, 0.03, 400)
    r = np.empty_like(eps)
    r[0] = eps[0]
    for t in range(1, len(r)):
        r[t] = 0.6 * r[t - 1] + eps[t]
    r += 0.004
    short = bootstrap_sharpe_ci(r, periods_per_year=12, block_length=1, n_resamples=1500, seed=0)
    long = bootstrap_sharpe_ci(r, periods_per_year=12, block_length=12, n_resamples=1500, seed=0)
    assert (long["ci_upper"] - long["ci_lower"]) > (short["ci_upper"] - short["ci_lower"])


def test_edge_cases_short_constant_and_bad_args() -> None:
    assert np.isnan(per_period_sharpe([0.01]))
    assert np.isnan(per_period_sharpe([0.01, 0.01, 0.01]))
    out = bootstrap_sharpe_ci([0.01, 0.02], periods_per_year=12)
    assert np.isnan(out["ci_lower"]) and out["n_obs"] == 2
    with pytest.raises(ValueError):
        block_bootstrap_indices(10, block_length=0.5, n_resamples=2, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        block_bootstrap_indices(10, block_length=2, n_resamples=2, method="iid", rng=np.random.default_rng(0))
    assert np.isnan(probabilistic_sharpe_ratio(0.1, 0.0, 1, 0.0, 3.0))


def test_significance_summary_keys_and_consistency() -> None:
    rng = np.random.default_rng(4)
    r = rng.normal(0.008, 0.04, 60)
    s = significance_summary(r, periods_per_year=12, n_trials=1, sharpe_variance=0.001,
                             block_length=4, n_resamples=300, seed=1)
    assert s["deflated_sharpe_ratio"] == pytest.approx(s["psr_vs_zero"])
    assert s["sharpe_annualized"] == pytest.approx(s["sharpe_per_period"] * np.sqrt(12))
    assert s["sr0_annualized"] == 0.0
