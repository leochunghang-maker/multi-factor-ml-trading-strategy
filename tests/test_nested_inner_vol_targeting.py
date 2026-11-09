"""Nested inner cost-aware scoring mirrors the outer book: sector-aware weights + causal vol scale."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.nested import (
    collect_inner_fold_diagnostics,
    net_long_short_period_return,
    select_model_nested,
)
from src.risk.volatility import volatility_target_scale

FEATURES = ["momentum_12m", "short_term_momentum", "volatility_1m"]
RIDGE = {"name": "ridge_1", "kind": "ridge", "params": {"alpha": 1.0}}


def _panel(start: str, end: str, n_tickers: int = 8, seed: int = 0) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(seed)
    rows = []
    for date in dates:
        for i in range(n_tickers):
            feature = float(i) + rng.normal(0, 0.5)
            rows.append({
                "date": date,
                "ticker": f"T{i}",
                "momentum_12m": feature,
                "short_term_momentum": feature * 0.5,
                "volatility_1m": 0.02 + 0.001 * i,
                "target_return_1m": 0.01 * feature + rng.normal(0, 0.05),
            })
    return pd.DataFrame(rows)


def _kwargs(**overrides):
    base = dict(
        previous_weights=None,
        long_count=2,
        short_count=2,
        gross_exposure=2.0,
        transaction_cost=0.0,
        annual_borrow_bps=0.0,
        annual_short_rebate_bps=0.0,
        annual_cash_financing_bps=0.0,
        periods_per_year=12,
    )
    base.update(overrides)
    return base


def test_vol_scale_multiplies_gross_and_weights() -> None:
    scores = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0, "F": -2.0})
    rets = pd.Series({"A": 0.02, "B": 0.01, "C": 0.0, "D": 0.0, "E": -0.01, "F": -0.02})
    base, w1 = net_long_short_period_return(scores, rets, **_kwargs(), vol_scale=1.0)
    half, w_half = net_long_short_period_return(scores, rets, **_kwargs(), vol_scale=0.5)
    assert half == pytest.approx(0.5 * base)
    assert w_half.abs().sum() == pytest.approx(0.5 * w1.abs().sum())


def test_vol_scale_also_scales_borrow_and_turnover_costs() -> None:
    scores = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0, "F": -2.0})
    flat = pd.Series(0.0, index=scores.index)
    kw = _kwargs(transaction_cost=0.001, annual_borrow_bps=120.0)
    cost_1, _ = net_long_short_period_return(scores, flat, **kw, vol_scale=1.0)
    cost_15, _ = net_long_short_period_return(scores, flat, **kw, vol_scale=1.5)
    assert cost_1 < 0
    assert cost_15 == pytest.approx(1.5 * cost_1)


def test_inner_vol_scale_is_causal_from_prior_inner_net_returns() -> None:
    train = _panel("2018-01-01", "2022-12-31")
    diag = collect_inner_fold_diagnostics(
        train,
        feature_columns=FEATURES,
        target_column="target_return_1m",
        candidate=RIDGE,
        min_train_years=2,
        long_count=2,
        short_count=2,
        vol_targeting=True,
    )
    nets, scales = diag["net_returns"], diag["vol_scales"]
    assert len(nets) == len(scales) > 6
    # Each scale must equal the estimator applied to strictly earlier nets only.
    for t in range(len(nets)):
        assert scales[t] == pytest.approx(volatility_target_scale(nets[:t], enabled=True))
    # Warm-up periods are unscaled; later ones actually move.
    assert scales[0] == 1.0
    assert any(abs(s - 1.0) > 1e-9 for s in scales)


def test_inner_vol_targeting_switch_off_gives_unit_scale() -> None:
    train = _panel("2018-01-01", "2022-12-31")
    diag = collect_inner_fold_diagnostics(
        train,
        feature_columns=FEATURES,
        target_column="target_return_1m",
        candidate=RIDGE,
        min_train_years=2,
        long_count=2,
        short_count=2,
        vol_targeting=False,
    )
    assert diag["vol_scales"] and all(s == 1.0 for s in diag["vol_scales"])


def test_inner_vol_scoring_ignores_outer_test_rows() -> None:
    frame = _panel("2018-01-01", "2023-12-31")
    outer_test_start = pd.Timestamp("2023-01-01")
    train = frame[frame["date"] < outer_test_start - pd.Timedelta(days=45)]
    _, meta_a = select_model_nested(
        train,
        feature_columns=FEATURES,
        target_column="target_return_1m",
        candidates=[RIDGE],
        min_train_years=2,
        long_count=2,
        short_count=2,
        vol_targeting=True,
    )
    # Corrupt everything after the training window: inner scores must not change.
    poisoned = frame.copy()
    mask = poisoned["date"] >= outer_test_start
    poisoned.loc[mask, "target_return_1m"] = 1e6
    train_b = poisoned[poisoned["date"] < outer_test_start - pd.Timedelta(days=45)]
    _, meta_b = select_model_nested(
        train_b,
        feature_columns=FEATURES,
        target_column="target_return_1m",
        candidates=[RIDGE],
        min_train_years=2,
        long_count=2,
        short_count=2,
        vol_targeting=True,
    )
    a = meta_a["candidate_scores"][0]
    b = meta_b["candidate_scores"][0]
    assert meta_a["inner_vol_targeting"] is True
    assert a["inner_score"] == pytest.approx(b["inner_score"])
    assert a["mean_inner_vol_scale"] == pytest.approx(b["mean_inner_vol_scale"])
    assert np.isfinite(a["mean_inner_vol_scale"])


def test_inner_gross_uses_sector_aware_weights_dot_returns() -> None:
    # Mixed-sector names: inner weights come from the outer sector-aware
    # constructor, and gross must equal sum(w * r) on those exact weights.
    scores = pd.Series({"AAPL": 3.0, "MSFT": 2.5, "NVDA": 2.0, "JPM": -1.0, "XOM": -2.0, "PFE": -3.0})
    rets = pd.Series({"AAPL": 0.03, "MSFT": 0.02, "NVDA": 0.05, "JPM": 0.01, "XOM": -0.02, "PFE": 0.0})
    net, weights = net_long_short_period_return(scores, rets, **_kwargs(), vol_scale=1.2)
    assert net == pytest.approx(float((weights * rets.reindex(weights.index)).sum()))
