"""Volatility targeting scale math and no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.volatility import (
    annualized_realized_volatility,
    apply_volatility_target,
    paper_volatility_target_scale,
    scale_weights,
    volatility_target_scale,
)


def test_scale_is_target_over_realized_and_clipped() -> None:
    # Constant monthly returns → realized vol ~ 0, scale falls back to 1.0
    assert volatility_target_scale([0.01] * 10, lookback=6, enabled=True) == 1.0

    rng = np.random.default_rng(0)
    noisy = pd.Series(rng.normal(0.0, 0.05, size=24))
    realized = annualized_realized_volatility(noisy, lookback=6, periods_per_year=12)
    assert realized > 0.05  # annualized from ~5% monthly stdev
    assert realized < 0.05 * np.sqrt(12) * 1.5

    scale = volatility_target_scale(
        noisy,
        target_volatility=0.10,
        lookback=6,
        periods_per_year=12,
        min_scale=0.5,
        max_scale=1.5,
        enabled=True,
    )
    raw = 0.10 / annualized_realized_volatility(noisy, 6, 12)
    assert scale == pytest.approx(float(np.clip(raw, 0.5, 1.5)))


def test_insufficient_history_returns_unit_scale() -> None:
    assert volatility_target_scale([0.01, -0.02], lookback=6, enabled=True) == 1.0
    assert volatility_target_scale(None, enabled=True) == 1.0
    assert volatility_target_scale([0.01] * 20, enabled=False) == 1.0


def test_no_lookahead_in_apply_volatility_target() -> None:
    returns = pd.Series([0.02, -0.01, 0.03, 0.00, -0.02, 0.01, 0.04, -0.03])
    scaled = apply_volatility_target(
        returns,
        target_volatility=0.15,
        lookback=3,
        max_leverage=2.0,
        min_scale=0.0,
    )
    # leverage for period t uses rolling vol ending at t-1 (shift 1)
    rolling = returns.rolling(3).std() * np.sqrt(12)
    expected_lev = (0.15 / rolling).clip(lower=0.0, upper=2.0).fillna(1.0).shift(1)
    aligned = (returns * expected_lev).dropna()
    pd.testing.assert_series_equal(scaled, aligned)


def test_research_builder_scale_excludes_current_return(monkeypatch) -> None:
    """Scale on date t must be computed from returns strictly before t."""
    import src.portfolio.portfolio as portfolio_mod
    from src.portfolio.portfolio import build_multi_factor_long_short_returns

    seen: list[int] = []

    def fake_scale(past, **kwargs):
        seen.append(len(list(past) if past is not None else []))
        return 1.0

    monkeypatch.setattr(portfolio_mod, "volatility_target_scale", fake_scale)
    monkeypatch.setattr(portfolio_mod, "SECTOR_NEUTRAL", False)
    # Avoid sector net cap shrinking gross in this leakage test.
    monkeypatch.setattr(portfolio_mod, "MAX_SECTOR_NET_EXPOSURE", 99.0)

    dates = pd.date_range("2020-01-31", periods=8, freq="ME")
    tickers = [f"X{i}" for i in range(6)]
    momentum = pd.DataFrame({t: float(i) for i, t in enumerate(tickers)}, index=dates)
    for col_i, t in enumerate(tickers):
        momentum[t] = float(col_i) + np.linspace(0, 0.1, len(dates))
    mean_reversion = momentum * 0.1
    forward = pd.DataFrame(0.01, index=dates, columns=tickers)

    results = build_multi_factor_long_short_returns(
        momentum,
        mean_reversion,
        forward,
        long_count=2,
        short_count=2,
        min_assets=4,
        transaction_cost=0.0,
        gross_exposure=2.0,
        annual_borrow_bps=0.0,
    )
    assert len(results) == 8
    # First rebalance: 0 past returns; k-th rebalance: k-1 past returns.
    assert seen == list(range(8))


def test_scale_weights_multiplies_gross() -> None:
    weights = pd.Series({"A": 0.5, "B": -0.5})
    scaled = scale_weights(weights, 0.8)
    assert scaled.abs().sum() == pytest.approx(0.8)
    assert scaled.sum() == pytest.approx(0.0)


def test_paper_scale_defaults_cap_at_one() -> None:
    rng = np.random.default_rng(1)
    # Very quiet daily returns → raw scale would exceed 1, but paper max is 1.0
    quiet = pd.Series(rng.normal(0.0, 0.001, size=100))
    scale = paper_volatility_target_scale(
        quiet,
        enabled=True,
        target_volatility=0.15,
        lookback_days=63,
        min_scale=0.5,
        max_scale=1.0,
    )
    assert scale == pytest.approx(1.0)
