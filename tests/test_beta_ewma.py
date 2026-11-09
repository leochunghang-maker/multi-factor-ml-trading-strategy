"""EWMA beta estimator (trial 43): weights, causality, regime tracking, OLS equivalence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.beta import build_beta_panel, estimate_betas_as_of, observation_weights


def _regime_market(n=300, switch=200, seed=1):
    """Stock beta 0.5 for the first ``switch`` days, 1.5 afterwards."""
    idx = pd.bdate_range("2021-01-01", periods=n)
    rng = np.random.default_rng(seed)
    mkt = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    beta = np.where(np.arange(n) < switch, 0.5, 1.5)
    stock = beta * mkt.to_numpy() + rng.normal(0, 0.002, n)
    rets = pd.DataFrame({"S": stock, "MISSING": np.nan}, index=idx)
    return rets, mkt


def test_observation_weights_flat_and_half_life() -> None:
    assert np.allclose(observation_weights(5, estimator="ols"), 1.0)
    w = observation_weights(127, estimator="ewma", half_life=63)
    assert w[-1] == pytest.approx(1.0)  # newest
    assert w[-64] == pytest.approx(0.5)  # 63 days back
    assert w[0] == pytest.approx(0.25)  # 126 days back
    assert np.all(np.diff(w) > 0)
    with pytest.raises(ValueError):
        observation_weights(5, estimator="garch")
    with pytest.raises(ValueError):
        observation_weights(5, estimator="ewma", half_life=0)


def test_ewma_equals_ols_when_half_life_is_huge() -> None:
    rets, mkt = _regime_market()
    as_of = rets.index[-1]
    kw = dict(lookback=252, min_obs=126, shrinkage=0.0)
    ols = estimate_betas_as_of(rets, mkt, as_of, estimator="ols", **kw)
    ewma = estimate_betas_as_of(rets, mkt, as_of, estimator="ewma", half_life=1e12, **kw)
    assert ewma["S"] == pytest.approx(ols["S"], rel=1e-9)


def test_ols_matches_plain_regression_slope() -> None:
    rets, mkt = _regime_market()
    as_of = rets.index[-1]
    got = estimate_betas_as_of(rets, mkt, as_of, lookback=252, min_obs=126, shrinkage=0.0, estimator="ols")
    x, y = mkt.tail(252), rets["S"].tail(252)
    assert got["S"] == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-9)


def test_ewma_tracks_regime_change_faster_than_ols() -> None:
    rets, mkt = _regime_market()
    as_of = rets.index[-1]  # 100 days into the beta-1.5 regime
    kw = dict(lookback=252, min_obs=126, shrinkage=0.0)
    ols = estimate_betas_as_of(rets, mkt, as_of, estimator="ols", **kw)["S"]
    ewma = estimate_betas_as_of(rets, mkt, as_of, estimator="ewma", half_life=63, **kw)["S"]
    assert abs(ewma - 1.5) < abs(ols - 1.5)
    assert 0.5 < ols < ewma < 1.5


def test_ewma_is_causal_and_keeps_shrinkage_and_fallback() -> None:
    rets, mkt = _regime_market()
    as_of = rets.index[220]
    kw = dict(lookback=252, min_obs=126, shrinkage=1 / 3, estimator="ewma", half_life=63)
    base = estimate_betas_as_of(rets, mkt, as_of, **kw)
    shocked = rets.copy()
    shocked.loc[shocked.index > as_of, "S"] = 0.5  # future rows must not matter
    assert estimate_betas_as_of(shocked, mkt, as_of, **kw)["S"] == pytest.approx(base["S"])
    raw = estimate_betas_as_of(rets, mkt, as_of, **{**kw, "shrinkage": 0.0})["S"]
    assert base["S"] == pytest.approx(2 / 3 * raw + 1 / 3)
    assert base["MISSING"] == 1.0
    early = estimate_betas_as_of(rets, mkt, rets.index[50], **kw)
    assert early["S"] == 1.0  # fewer than min_obs observations


def test_build_beta_panel_passes_estimator() -> None:
    rets, mkt = _regime_market()
    prices = (1 + rets[["S"]]).cumprod() * 100
    bench = (1 + mkt).cumprod() * 100
    dates = prices.index[[-1]]
    ols = build_beta_panel(prices, bench, dates, estimator="ols", shrinkage=0.0)
    ewma = build_beta_panel(prices, bench, dates, estimator="ewma", half_life=63, shrinkage=0.0)
    assert ewma.iloc[0]["S"] > ols.iloc[0]["S"]
