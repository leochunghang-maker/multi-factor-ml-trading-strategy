"""Causal betas, beta-neutral leg scaling, and interaction with sector caps / vol scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.portfolio import apply_sector_net_exposure_limits, equal_weight_long_short_weights
from src.risk.beta import (
    beta_neutral_leg_scaling,
    beta_neutralize_with_sector_caps,
    betas_on_date,
    build_beta_panel,
    estimate_betas_as_of,
    portfolio_beta,
    shrink_beta,
)
from src.risk.volatility import scale_weights


def _market(n=400, seed=0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    mkt = pd.Series(rng.normal(0.0004, 0.01, n), index=idx)
    true = {"LO": 0.5, "MID": 1.0, "HI": 1.8}
    rets = pd.DataFrame({k: b * mkt + rng.normal(0, 0.002, n) for k, b in true.items()})
    prices = (1 + rets).cumprod() * 100
    bench = (1 + mkt).cumprod() * 100
    return prices, bench, true


def test_shrinkage_and_bounds() -> None:
    assert shrink_beta(1.6, 1 / 3) == pytest.approx(1.4)
    assert shrink_beta(0.4, 0.0) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        shrink_beta(1.0, 1.5)


def test_estimated_betas_recover_truth_with_shrinkage() -> None:
    prices, bench, true = _market()
    betas = estimate_betas_as_of(
        prices.pct_change(), bench.pct_change(), prices.index[-1], lookback=252, min_obs=126, shrinkage=0.0
    )
    for k, b in true.items():
        assert betas[k] == pytest.approx(b, abs=0.05)
    shrunk = estimate_betas_as_of(
        prices.pct_change(), bench.pct_change(), prices.index[-1], shrinkage=1 / 3
    )
    assert shrunk["HI"] == pytest.approx(2 / 3 * betas["HI"] + 1 / 3)


def test_betas_are_causal_future_data_cannot_change_them() -> None:
    prices, bench, _ = _market()
    as_of = prices.index[300]
    base = build_beta_panel(prices, bench, [as_of])
    poisoned_prices = prices.copy()
    poisoned_prices.iloc[301:] *= np.linspace(1, 50, len(prices) - 301)[:, None]
    poisoned_bench = bench.copy()
    poisoned_bench.iloc[301:] *= 0.01
    after = build_beta_panel(poisoned_prices, poisoned_bench, [as_of])
    pd.testing.assert_frame_equal(base, after)


def test_short_history_gets_beta_one_and_as_of_lookup() -> None:
    prices, bench, _ = _market()
    prices["NEW"] = np.nan
    prices.iloc[-50:, prices.columns.get_loc("NEW")] = 100.0 + np.arange(50)
    panel = build_beta_panel(prices, bench, [prices.index[-1]])
    assert panel.loc[prices.index[-1], "NEW"] == 1.0
    looked = betas_on_date(panel, prices.index[-1] + pd.Timedelta(days=3), pd.Index(["HI", "ZZZ"]))
    assert looked["ZZZ"] == 1.0 and looked["HI"] == panel.loc[prices.index[-1], "HI"]
    assert betas_on_date(panel, prices.index[0], pd.Index(["HI"])) is None


def test_leg_scaling_zeroes_beta_and_keeps_gross() -> None:
    w = pd.Series({"A": 0.5, "B": 0.5, "C": -0.5, "D": -0.5})
    betas = pd.Series({"A": 1.5, "B": 1.3, "C": 0.8, "D": 0.9})
    out = beta_neutral_leg_scaling(w, betas, max_leg_ratio=2.0)
    assert portfolio_beta(out, betas) == pytest.approx(0.0, abs=1e-12)
    assert out.abs().sum() == pytest.approx(2.0)
    # High-beta longs → long leg shrinks, short leg grows.
    assert out[out > 0].sum() < 1.0 < -out[out < 0].sum()


def test_leg_ratio_bound_leaves_residual_beta() -> None:
    w = pd.Series({"A": 0.5, "B": 0.5, "C": -0.5, "D": -0.5})
    betas = pd.Series({"A": 3.0, "B": 3.0, "C": 0.5, "D": 0.5})  # needs ratio 1/6
    out = beta_neutral_leg_scaling(w, betas, max_leg_ratio=2.0)
    assert out[out > 0].sum() / -out[out < 0].sum() == pytest.approx(0.5)
    assert portfolio_beta(out, betas) > 0
    assert out.abs().sum() == pytest.approx(2.0)


def test_degenerate_legs_unchanged() -> None:
    long_only = pd.Series({"A": 0.5, "B": 0.5})
    pd.testing.assert_series_equal(
        beta_neutral_leg_scaling(long_only, pd.Series({"A": 1.0, "B": 1.0})), long_only
    )
    w = pd.Series({"A": 0.5, "C": -0.5})
    pd.testing.assert_series_equal(beta_neutral_leg_scaling(w, pd.Series({"A": -1.0, "C": 1.0})), w)


def test_vol_scaling_preserves_beta_neutrality() -> None:
    w = pd.Series({"A": 0.5, "B": 0.5, "C": -0.5, "D": -0.5})
    betas = pd.Series({"A": 1.4, "B": 1.2, "C": 0.9, "D": 0.7})
    neutral = beta_neutral_leg_scaling(w, betas)
    assert portfolio_beta(scale_weights(neutral, 0.6), betas) == pytest.approx(0.0, abs=1e-12)


def test_sector_caps_and_beta_alternation_satisfies_both_approximately() -> None:
    sectors = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Util", "E": "Util", "F": "Util"}
    w = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2, "D": -0.4, "E": -0.4, "F": -0.2})
    betas = pd.Series({"A": 1.6, "B": 1.5, "C": 1.3, "D": 0.6, "E": 0.7, "F": 0.6})

    def caps(x):
        return apply_sector_net_exposure_limits(
            x, max_sector_net_exposure=0.15, sector_map=sectors, target_gross_exposure=2.0
        )

    out = beta_neutralize_with_sector_caps(
        caps(w), betas, max_leg_ratio=2.0, apply_sector_caps=caps, max_rounds=20
    )
    assert abs(portfolio_beta(out, betas)) < 0.05
    nets = out.groupby(pd.Series(sectors)).sum()
    assert nets.abs().max() <= 0.15 + 0.05


def test_equal_weight_constructor_beta_switch() -> None:
    scores = pd.Series({"AAPL": 3.0, "MSFT": 2.5, "NVDA": 2.0, "JPM": -1.0, "XOM": -2.0, "PFE": -3.0})
    betas = pd.Series({"AAPL": 1.2, "MSFT": 1.1, "NVDA": 1.7, "JPM": 1.0, "XOM": 0.8, "PFE": 0.6})
    off = equal_weight_long_short_weights(scores, 2, 2, gross_exposure=2.0, betas=betas, beta_neutral=False)
    on = equal_weight_long_short_weights(scores, 2, 2, gross_exposure=2.0, betas=betas, beta_neutral=True)
    none = equal_weight_long_short_weights(scores, 2, 2, gross_exposure=2.0, betas=None, beta_neutral=True)
    pd.testing.assert_series_equal(off, none)
    assert abs(portfolio_beta(on, betas)) < abs(portfolio_beta(off, betas))
    assert abs(portfolio_beta(on, betas)) < 0.05


def _panel(start, end, n_tickers=8, seed=0):
    dates = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(seed)
    rows = []
    for date in dates:
        for i in range(n_tickers):
            f = float(i) + rng.normal(0, 0.5)
            rows.append({"date": date, "ticker": f"T{i}", "momentum_12m": f,
                         "short_term_momentum": f * 0.5, "volatility_1m": 0.02 + 0.001 * i,
                         "target_return_1m": 0.01 * f + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_nested_inner_scoring_uses_beta_panel_causally() -> None:
    from src.models.nested import select_model_nested

    frame = _panel("2018-01-01", "2023-12-31")
    train = frame[frame["date"] < "2022-11-15"]
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    tickers = sorted(frame["ticker"].unique())
    # High-score names get high beta so neutralization materially reshapes the book.
    panel = pd.DataFrame({t: 0.5 + 0.2 * i for i, t in enumerate(tickers)}, index=dates)
    poisoned = panel.copy()
    poisoned.loc[poisoned.index >= "2023-01-01"] = 50.0
    kw = dict(feature_columns=["momentum_12m", "short_term_momentum", "volatility_1m"],
              target_column="target_return_1m",
              candidates=[{"name": "ridge_1", "kind": "ridge", "params": {"alpha": 1.0}}],
              min_train_years=2, long_count=2, short_count=2, vol_targeting=False)
    _, plain = select_model_nested(train, **kw)
    _, with_beta = select_model_nested(train, beta_panel=panel, **kw)
    _, with_poison = select_model_nested(train, beta_panel=poisoned, **kw)
    assert with_beta["inner_beta_neutral"] is True and plain["inner_beta_neutral"] is False
    s_plain = plain["candidate_scores"][0]["inner_score"]
    s_beta = with_beta["candidate_scores"][0]["inner_score"]
    assert s_beta != pytest.approx(s_plain)
    assert with_poison["candidate_scores"][0]["inner_score"] == pytest.approx(s_beta)
