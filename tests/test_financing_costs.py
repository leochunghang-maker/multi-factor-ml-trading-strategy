"""Borrow / cash-financing cost accrual tests."""

import pandas as pd
import pytest

from src import config
from src.execution.financing import (
    daily_financing_dollars,
    financing_cost_from_weights,
    holding_period_financing_cost,
    period_rate_from_annual_bps,
)
from src.execution.simulated_broker import SimulatedBroker
from src.operations.configuration import load_platform_config
from src.portfolio.portfolio import build_multi_factor_long_short_returns


def test_borrow_defaults_mirror_platform_config() -> None:
    execution = load_platform_config()["execution"]
    assert config.ANNUAL_BORROW_BPS == execution["annual_borrow_bps"] == 100.0
    assert config.ANNUAL_SHORT_REBATE_BPS == execution["annual_short_rebate_bps"] == 0.0
    assert config.ANNUAL_CASH_FINANCING_BPS == execution["annual_cash_financing_bps"] == 0.0


def test_monthly_borrow_on_unit_short_gross() -> None:
    # 100 bps / year on 100% short ≈ 8.333 bps per month.
    cost = holding_period_financing_cost(
        short_gross=1.0,
        long_gross=1.0,
        annual_borrow_bps=100.0,
        periods_per_year=12,
    )
    assert cost == pytest.approx(100.0 / 10_000 / 12)
    # Rebate offsets borrow when equal.
    net = holding_period_financing_cost(
        short_gross=1.0,
        long_gross=1.0,
        annual_borrow_bps=100.0,
        annual_short_rebate_bps=100.0,
        periods_per_year=12,
    )
    assert net == pytest.approx(0.0)


def test_cash_financing_only_on_excess_long() -> None:
    # Dollar-neutral: no cash financing even if financing bps > 0.
    neutral = holding_period_financing_cost(
        short_gross=1.0,
        long_gross=1.0,
        annual_borrow_bps=0.0,
        annual_cash_financing_bps=200.0,
        periods_per_year=12,
    )
    assert neutral == pytest.approx(0.0)
    # 50% net long financed.
    net_long = holding_period_financing_cost(
        short_gross=0.5,
        long_gross=1.0,
        annual_borrow_bps=0.0,
        annual_cash_financing_bps=200.0,
        periods_per_year=12,
    )
    assert net_long == pytest.approx(0.5 * period_rate_from_annual_bps(200.0, 12))


def test_financing_from_signed_weights() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5, "C": -0.5, "D": -0.5})
    cost = financing_cost_from_weights(weights, annual_borrow_bps=120.0, periods_per_year=12)
    assert cost == pytest.approx(1.0 * 120.0 / 10_000 / 12)


def test_daily_financing_dollars_and_broker_accrual() -> None:
    # $100k short @ 100 bps / 252 ≈ $3.968/day.
    day_cost = daily_financing_dollars(
        short_market_value=100_000.0,
        margin_debit=0.0,
        annual_borrow_bps=100.0,
    )
    assert day_cost == pytest.approx(100_000.0 * 100.0 / 10_000 / 252)

    broker = SimulatedBroker(
        cash=50_000.0,
        positions={"AAA": 100.0},
        annual_borrow_bps=100.0,
        annual_cash_financing_bps=0.0,
        transaction_cost_rate=0.0,
        slippage_bps=0.0,
        half_spread_bps=0.0,
    )
    # Long-only → zero financing.
    row = broker.accrue_financing({"AAA": 100.0}, timestamp="t0")
    assert row["financing_cost"] == pytest.approx(0.0)
    assert broker.cash == pytest.approx(50_000.0)

    # Synthetic short position for accrual only (paper engine still rejects short sells).
    broker.positions["BBB"] = -200.0
    row = broker.accrue_financing({"AAA": 100.0, "BBB": 50.0}, timestamp="t1")
    assert row["short_market_value"] == pytest.approx(10_000.0)
    assert row["financing_cost"] == pytest.approx(10_000.0 * 100.0 / 10_000 / 252)
    assert broker.cash == pytest.approx(50_000.0 - row["financing_cost"])


def test_research_returns_subtract_monthly_borrow() -> None:
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    tickers = [f"T{i}" for i in range(6)]
    momentum = pd.DataFrame(1.0, index=dates, columns=tickers)
    # Make a stable ranking so weights are well-defined.
    for i, t in enumerate(tickers):
        momentum[t] = float(i)
    mean_reversion = pd.DataFrame(0.0, index=dates, columns=tickers)
    forward = pd.DataFrame(0.01, index=dates, columns=tickers)

    with_borrow = build_multi_factor_long_short_returns(
        momentum,
        mean_reversion,
        forward,
        long_count=2,
        short_count=2,
        min_assets=4,
        transaction_cost=0.0,
        gross_exposure=2.0,
        annual_borrow_bps=120.0,
        annual_short_rebate_bps=0.0,
        annual_cash_financing_bps=0.0,
        periods_per_year=12,
    )
    no_borrow = build_multi_factor_long_short_returns(
        momentum,
        mean_reversion,
        forward,
        long_count=2,
        short_count=2,
        min_assets=4,
        transaction_cost=0.0,
        gross_exposure=2.0,
        annual_borrow_bps=0.0,
        periods_per_year=12,
    )
    assert not with_borrow.empty
    # Short gross = 1.0 → 10 bps/month drag.
    expected_drag = 120.0 / 10_000 / 12
    assert (no_borrow["portfolio_return"] - with_borrow["portfolio_return"]).mean() == pytest.approx(
        expected_drag
    )
    assert with_borrow["financing_cost"].iloc[-1] == pytest.approx(expected_drag)
