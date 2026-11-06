"""ADV participation / capacity constraint tests."""

import math

import pandas as pd
import pytest

from src.config import MAX_ADV_PARTICIPATION, SIMULATED_INITIAL_CASH
from src.execution.simulated_broker import SimulatedBroker, SimulatedOrder
from src.portfolio.allocation import build_paper_target_weights
from src.portfolio.constraints import (
    PortfolioConstraints,
    apply_adv_participation_cap,
    constrain_weights,
)


def test_apply_adv_participation_cap_clips_oversized_trade() -> None:
    target = pd.Series({"AAA": 0.50, "BBB": 0.50})
    previous = pd.Series({"AAA": 0.0, "BBB": 0.0})
    # At NAV=100_000 and 5% participation, ADV$=100_000 => max |Δw| = 0.05.
    adv = pd.Series({"AAA": 100_000.0, "BBB": 10_000_000.0})

    capped, n_clipped, available = apply_adv_participation_cap(
        target,
        previous,
        adv,
        nav=100_000.0,
        max_adv_participation=0.05,
    )

    assert available is True
    assert n_clipped == 1
    assert capped["AAA"] == pytest.approx(0.05)
    assert capped["BBB"] == pytest.approx(0.50)


def test_constrain_weights_reports_capacity_clip() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"AAA": 100.0, "BBB": 100.0}, index=dates)
    # ADV$ = price * volume mean => AAA ~ 100*10=1_000; BBB large.
    volumes = pd.DataFrame({"AAA": 10.0, "BBB": 1_000_000.0}, index=dates)
    target = pd.Series({"AAA": 0.40, "BBB": 0.40}, name=dates[-1])
    constraints = PortfolioConstraints(
        max_position_weight=1.0,
        max_sector_weight=1.0,
        max_gross_exposure=1.0,
        max_turnover=2.0,
        min_average_daily_dollar_volume=0.0,
        max_adv_participation=0.05,
        capacity_nav=1_000.0,
        sector_map={"AAA": "Tech", "BBB": "Tech"},
    )

    constrained, diagnostics = constrain_weights(
        target,
        previous_weights=pd.Series({"AAA": 0.0, "BBB": 0.0}),
        prices=prices,
        volumes=volumes,
        as_of_date=dates[-1],
        constraints=constraints,
    )

    assert diagnostics["capacity_filter_available"] is True
    assert diagnostics["capacity_clipped_positions"] >= 1
    # max Δw for AAA = 0.05 * 1000 / 1000 = 0.05
    assert constrained["AAA"] == pytest.approx(0.05, abs=1e-6)


def test_paper_allocation_leaves_cash_after_capacity_clip() -> None:
    targets = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "target_weight": [0.5, 0.5],
            "signal": ["LONG", "LONG"],
        }
    )
    adv = pd.Series({"AAA": 50_000.0, "BBB": 50_000.0})
    weights, diagnostics = build_paper_target_weights(
        targets,
        positions={},
        prices={"AAA": 100.0, "BBB": 100.0},
        portfolio_value=100_000.0,
        max_position_weight=0.5,
        allocation_buffer=0.0,
        max_turnover=1.0,
        adv_dollars=adv,
        max_adv_participation=0.05,
    )
    # max |Δw| = 0.05 * 50k / 100k = 0.025 each
    assert weights["AAA"] == pytest.approx(0.025)
    assert weights["BBB"] == pytest.approx(0.025)
    assert diagnostics["capacity_clipped_positions"] == 2
    assert diagnostics["expected_cash_weight"] == pytest.approx(0.95)


def test_broker_clips_order_to_adv_participation() -> None:
    broker = SimulatedBroker(
        cash=100_000.0,
        transaction_cost_rate=0.0,
        slippage_bps=0.0,
        half_spread_bps=0.0,
        impact_coefficient_bps=0.0,
        max_adv_participation=0.05,
    )
    broker.adv_dollars = {"AAA": 100_000.0}
    order = SimulatedOrder("AAA", "BUY", quantity=20.0, price=100.0)  # $2,000 > 5%*100k=$5k? 2000 < 5000
    # Make oversized: qty 100 -> $10,000 > $5,000
    order = SimulatedOrder("AAA", "BUY", quantity=100.0, price=100.0)
    clipped = broker.clip_order_to_adv_participation(order)
    assert clipped.quantity == pytest.approx(50.0)
    assert "ADV participation clipped" in clipped.reason

    fill = broker.process_order(
        SimulatedOrder("AAA", "BUY", quantity=100.0, price=100.0),
        timestamp="2024-01-02",
        prices={"AAA": 100.0},
        max_position_weight=1.0,
    )
    assert fill["status"] == "FILLED"
    assert fill["quantity"] == pytest.approx(50.0)
    assert broker.positions["AAA"] == pytest.approx(50.0)


def test_max_adv_participation_config_positive() -> None:
    assert MAX_ADV_PARTICIPATION == pytest.approx(0.05)
    assert SIMULATED_INITIAL_CASH > 0
    assert math.isfinite(MAX_ADV_PARTICIPATION)
