"""Sell-before-buy execution and buy sizing from cash after sells (paper simulator).

Regression for the paper run where BUY orders (GEV, INTC, MU) were rejected
for insufficient cash because exit sells were appended after the buys.
Simulated broker only; nothing is sent anywhere.
"""

import numpy as np
import pandas as pd
import pytest

from src.execution.run_daily_simulation import build_rebalance_orders, execute_rebalance_orders
from src.execution.simulated_broker import SimulatedBroker, SimulatedOrder


def _fully_invested_broker() -> tuple[SimulatedBroker, dict[str, float]]:
    # Ten 9.5% positions plus 5% cash (100k NAV): no room for a new 9.5% buy
    # until an exit sell has filled. Keeps every name under the 10% cap.
    prices = {f"OLD{i}": 100.0 for i in range(10)} | {f"NEW{i}": 50.0 for i in range(3)}
    broker = SimulatedBroker(cash=5_000.0, positions={f"OLD{i}": 95.0 for i in range(10)})
    return broker, prices


def test_buys_listed_before_exit_sells_would_be_rejected_sequentially() -> None:
    # The old execution order: buys first, exits after. Documents the bug.
    broker, prices = _fully_invested_broker()
    buy = SimulatedOrder("NEW0", "BUY", 190.0, prices["NEW0"])
    try:
        broker.validate_order(buy, prices)
    except ValueError as exc:
        assert "Insufficient cash" in str(exc)
    else:
        raise AssertionError("expected an insufficient-cash rejection before sells")


def test_sells_cover_buys_so_nothing_is_rejected_and_cash_stays_non_negative() -> None:
    broker, prices = _fully_invested_broker()
    # Deliberately in the old order: buys first, exits appended after.
    orders = [SimulatedOrder(f"NEW{i}", "BUY", 185.0, prices[f"NEW{i}"]) for i in range(3)]
    orders += [SimulatedOrder(f"OLD{i}", "SELL", 95.0, prices[f"OLD{i}"]) for i in range(3)]
    result = execute_rebalance_orders(broker, orders, prices, "2026-11-06")
    assert result["rejected"] == []
    assert [t["side"] for t in result["trades"]] == ["SELL"] * 3 + ["BUY"] * 3
    assert result["buy_scale"] == 1.0  # cash after sells was enough; quantities untouched
    assert all(broker.positions[f"NEW{i}"] == 185.0 for i in range(3))
    assert not any(f"OLD{i}" in broker.positions for i in range(3))
    assert broker.cash >= 0.0
    assert min(t["cash_after"] for t in result["trades"]) >= 0.0


def test_buys_exceeding_cash_after_sells_are_scaled_not_rejected() -> None:
    broker, prices = _fully_invested_broker()
    orders = [  # ~14.5k cash after one exit vs ~19k of buys
        SimulatedOrder("OLD0", "SELL", 95.0, prices["OLD0"]),
        SimulatedOrder("NEW0", "BUY", 190.0, prices["NEW0"]),
        SimulatedOrder("NEW1", "BUY", 95.0, prices["NEW1"]),
        SimulatedOrder("NEW2", "BUY", 95.0, prices["NEW2"]),
    ]
    result = execute_rebalance_orders(broker, orders, prices, "2026-11-06")
    assert result["rejected"] == []
    assert 0.0 < result["buy_scale"] < 1.0
    assert broker.positions["NEW0"] / broker.positions["NEW1"] == pytest.approx(2.0)
    assert 0.0 <= broker.cash < 1.0


def test_no_cash_and_no_sells_drops_buys_instead_of_borrowing() -> None:
    broker = SimulatedBroker(cash=0.0, positions={"AAA": 10.0})
    result = execute_rebalance_orders(
        broker, [SimulatedOrder("BBB", "BUY", 5.0, 10.0)], {"AAA": 10.0, "BBB": 10.0}, "t")
    assert result["buy_scale"] == 0.0
    assert result["trades"] == [] and result["rejected"] == []
    assert broker.cash == 0.0


def test_cash_never_negative_on_random_rebalances_with_impact_and_adv() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        symbols = [f"S{i}" for i in range(8)]
        prices = {s: float(rng.uniform(5, 500)) for s in symbols}
        held = rng.choice(symbols, size=4, replace=False)
        broker = SimulatedBroker(
            cash=float(rng.uniform(0, 5_000)),
            positions={s: float(rng.uniform(1, 200)) for s in held},
            adv_dollars={s: float(rng.uniform(1e4, 1e6)) for s in symbols},
        )
        orders = []
        for s in symbols:
            if s in broker.positions and rng.random() < 0.6:
                orders.append(SimulatedOrder(s, "SELL", broker.positions[s] * rng.uniform(0.1, 1.0), prices[s]))
            elif rng.random() < 0.6:
                orders.append(SimulatedOrder(s, "BUY", float(rng.uniform(1, 400)), prices[s]))
        rng.shuffle(orders)
        result = execute_rebalance_orders(broker, orders, prices, "t")
        assert broker.cash >= -1e-9
        assert not any("Insufficient cash" in r["message"] for r in result["rejected"])


def test_build_rebalance_orders_puts_exit_sells_before_buys() -> None:
    broker, prices = _fully_invested_broker()
    targets = pd.DataFrame({
        "date": ["2026-11-06"] * 2,
        "ticker": ["NEW0", "NEW1"],
        "signal": ["LONG", "LONG"],
        "target_weight": [0.5, 0.5],
    })
    orders, _, _, _ = build_rebalance_orders(broker, targets, prices, broker.portfolio_value(prices), "t")
    sides = [o.side for o in orders]
    assert "BUY" in sides and "SELL" in sides
    assert sides == sorted(sides, key=lambda side: 0 if side == "SELL" else 1)



def test_shared_helper_passes_max_position_weight() -> None:
    # A 15% buy breaches the default 10% paper cap but fits a 20% cap.
    prices = {"AAA": 100.0}
    for cap, rejected in ((None, 1), (0.20, 0)):
        broker = SimulatedBroker(cash=100_000.0)
        kwargs = {} if cap is None else {"max_position_weight": cap}
        result = execute_rebalance_orders(
            broker, [SimulatedOrder("AAA", "BUY", 150.0, 100.0)], prices, "t", **kwargs)
        assert len(result["rejected"]) == rejected


def test_multi_day_simulator_uses_shared_sell_before_buy_helper() -> None:
    from src.execution import simulated_broker
    from src.simulation import run_multi_day_simulation as sim

    assert sim.execute_rebalance_orders is simulated_broker.execute_rebalance_orders
    # Fully invested book rotating into new names: the multi-day order builder
    # plus the shared helper fill every order and keep cash non-negative.
    broker, prices = _fully_invested_broker()
    targets = pd.Series({f"OLD{i}": 0.095 for i in range(3, 10)} | {f"NEW{i}": 0.095 for i in range(3)})
    nav = broker.portfolio_value(prices)
    orders = sim.build_orders(broker, targets, prices, nav, "t")
    result = execute_rebalance_orders(broker, orders, prices, "t", max_position_weight=0.10)
    assert result["rejected"] == []
    assert {t["symbol"] for t in result["trades"] if t["side"] == "BUY"} == {"NEW0", "NEW1", "NEW2"}
    assert broker.cash >= 0.0
