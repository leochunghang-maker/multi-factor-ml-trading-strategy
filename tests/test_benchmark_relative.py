"""Benchmark-relative significance: HAC regression, IR, hedged Sharpe, trial log."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reporting.significance import (
    benchmark_relative_summary,
    causal_rolling_beta,
    newey_west_lags,
    ols_newey_west,
)
from src.reporting.trial_log import count_trials, family_sharpes, load_trial_log, pooled_sharpes


def _ma_errors(rng, n, theta=(1.0, 0.5, 0.3)):
    return np.convolve(rng.normal(0, 1, n + len(theta) - 1), theta, "valid")


def test_ols_newey_west_matches_statsmodels_hac() -> None:
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 250)
    y = 0.05 + 0.8 * x + _ma_errors(rng, 250)
    for lags in (0, 3, 6):
        ours = ols_newey_west(y, x, lags=lags)
        ref = sm.OLS(y, sm.add_constant(x)).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}
        )
        assert ours["alpha"] == pytest.approx(ref.params[0])
        assert ours["beta"] == pytest.approx(ref.params[1])
        assert ours["alpha_se"] == pytest.approx(ref.bse[0], rel=1e-9)
        assert ours["beta_se"] == pytest.approx(ref.bse[1], rel=1e-9)


def test_hac_lag_zero_equals_white_known_answer() -> None:
    # y = [1,2,3,5], x = [0,1,2,3]: OLS b = 1.3, a = 0.8; residuals e.
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 5.0])
    out = ols_newey_west(y, x, lags=0)
    assert out["beta"] == pytest.approx(1.3)
    assert out["alpha"] == pytest.approx(0.8)
    X = np.column_stack([np.ones(4), x])
    e = y - X @ np.array([0.8, 1.3])
    bread = np.linalg.inv(X.T @ X)
    white = bread @ (X.T * e**2) @ X @ bread
    assert out["beta_se"] == pytest.approx(np.sqrt(white[1, 1]))


def test_regression_recovers_alpha_beta_and_handles_degenerate_input() -> None:
    rng = np.random.default_rng(1)
    b = rng.normal(0.0, 0.01, 3000)
    s = 0.0004 + 1.5 * b + rng.normal(0, 0.002, 3000)
    out = ols_newey_west(s, b)
    assert out["beta"] == pytest.approx(1.5, abs=0.02)
    assert out["alpha"] == pytest.approx(0.0004, abs=1e-4)
    assert out["hac_lags"] == newey_west_lags(3000) == 8
    assert out["alpha_t"] > 5
    flat = ols_newey_west([0.1, 0.2, 0.3], [1.0, 1.0, 1.0])
    assert np.isnan(flat["beta"])


def test_newey_west_lag_rule() -> None:
    assert newey_west_lags(57) == 3
    assert newey_west_lags(100) == 4
    assert newey_west_lags(1968) == 7


def test_causal_beta_uses_only_prior_periods() -> None:
    idx = pd.date_range("2020-01-31", periods=40, freq="ME")
    rng = np.random.default_rng(2)
    b = pd.Series(rng.normal(0, 0.04, 40), index=idx)
    s = 2.0 * b
    s.iloc[-1] = 99.0  # a huge last-period shock must not affect the beta used for it
    beta = causal_rolling_beta(s, b, window=12)
    assert beta.iloc[-1] == pytest.approx(2.0)
    assert beta.iloc[:6].isna().all()


def test_pure_beta_strategy_has_zero_alpha_and_zero_hedged_sharpe() -> None:
    idx = pd.date_range("2018-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(3)
    bench = pd.Series(rng.normal(0.008, 0.04, 120), index=idx)
    strat = 0.5 * bench
    out = benchmark_relative_summary(
        strat, bench, periods_per_year=12, n_trials=10, ir_sharpe_variance=0.005,
        causal_beta_window=24, block_length=4, n_resamples=300, seed=0,
    )
    assert out["beta"] == pytest.approx(0.5)
    assert out["alpha_annualized"] == pytest.approx(0.0, abs=1e-12)
    assert abs(out["hedged_sharpe_in_sample"]) < 1e-6 or np.isnan(out["hedged_sharpe_in_sample"])
    # Active return = -0.5 * bench → IR has the opposite sign of the benchmark Sharpe.
    assert np.sign(out["information_ratio"]) == -np.sign(bench.mean())


def test_alpha_strategy_has_positive_ir_and_hedged_sharpe() -> None:
    idx = pd.date_range("2010-01-31", periods=200, freq="ME")
    rng = np.random.default_rng(4)
    bench = pd.Series(rng.normal(0.006, 0.04, 200), index=idx)
    strat = bench + 0.01 + pd.Series(rng.normal(0, 0.01, 200), index=idx)
    out = benchmark_relative_summary(
        strat, bench, periods_per_year=12, n_trials=1, ir_sharpe_variance=0.0,
        causal_beta_window=36, block_length=4, n_resamples=500, seed=1,
    )
    assert out["alpha_t_hac"] > 5
    assert out["information_ratio"] > 2
    assert out["ir_ci_lower"] > 0
    assert out["ir_deflated"] == pytest.approx(out["ir_psr_vs_zero"])
    assert out["hedged_sharpe_causal"] > 2


def test_trial_log_parsing_counts_and_validation(tmp_path) -> None:
    path = tmp_path / "log.csv"
    path.write_text(
        "trial_id,date,commit,family,description,trial_increment,reported_sharpe,notes\n"
        "1,2026-01-01,abc,walk_forward_ml,a,1,0.5,\n"
        "2,2026-01-02,def,walk_forward_ml,b,2,,\n"
        "3,2026-01-03,ghi,multi_factor,c,1,0.2,\n"
    )
    log = load_trial_log(path)
    assert count_trials(log) == 4
    assert count_trials(log, padding=10) == 14
    assert family_sharpes(log, "walk_forward_ml") == [0.5]
    assert pooled_sharpes(log) == [0.5, 0.2]
    with pytest.raises(ValueError):
        count_trials(log, padding=-1)

    bad = tmp_path / "bad.csv"
    bad.write_text(
        "trial_id,date,commit,family,description,trial_increment,reported_sharpe\n"
        "1,2026-01-01,abc,wf,a,-1,\n"
    )
    with pytest.raises(ValueError):
        load_trial_log(bad)
    dup = tmp_path / "dup.csv"
    dup.write_text(
        "trial_id,date,commit,family,description,trial_increment,reported_sharpe\n"
        "1,2026-01-01,abc,wf,a,1,\n1,2026-01-02,abc,wf,b,1,\n"
    )
    with pytest.raises(ValueError):
        load_trial_log(dup)
    zero = tmp_path / "zero.csv"
    zero.write_text(
        "trial_id,date,commit,family,description,trial_increment,reported_sharpe\n"
        "1,2026-01-01,abc,wf,a,1,0.5\n2,2026-01-02,abc,wf,a rerun,0,0.5\n"
    )
    with pytest.raises(ValueError, match="increment 0"):
        load_trial_log(zero)
    missing = tmp_path / "missing.csv"
    missing.write_text("trial_id,date,family\n1,2026-01-01,wf\n")
    with pytest.raises(ValueError):
        load_trial_log(missing)


def test_checked_in_trial_log_is_valid_and_matches_config() -> None:
    from src.config import SIGNIFICANCE_TRIAL_LOG_PATH, SIGNIFICANCE_TRIAL_PADDING

    log = load_trial_log(SIGNIFICANCE_TRIAL_LOG_PATH)
    assert list(log["trial_id"]) == sorted(log["trial_id"])
    assert count_trials(log, SIGNIFICANCE_TRIAL_PADDING) >= 50
    assert len(family_sharpes(log, "walk_forward_ml")) >= 10
