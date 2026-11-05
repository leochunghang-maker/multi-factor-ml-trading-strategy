import math

import pytest

from src import config
from src.execution.simulated_broker import SimulatedBroker, SimulatedOrder
from src.operations.configuration import load_platform_config


def test_execution_cost_defaults_come_from_platform_config() -> None:
    execution = load_platform_config()["execution"]
    assert config.HALF_SPREAD_BPS == execution["half_spread_bps"]
    assert config.IMPACT_COEFFICIENT_BPS == execution["impact_coefficient_bps"]
    assert config.IMPACT_ADV_LOOKBACK_DAYS == execution["impact_adv_lookback_days"]


def test_square_root_impact_scales_with_participation() -> None:
    broker = SimulatedBroker(cash=1e6, slippage_bps=0.0, half_spread_bps=0.0, impact_coefficient_bps=100.0)
    broker.adv_dollars = {"AAA": 1_000_000.0}
    assert broker.impact_bps("AAA", 10_000.0) == pytest.approx(10.0)  # 1% ADV
    assert broker.impact_bps("AAA", 40_000.0) == pytest.approx(20.0)  # 4x size -> 2x impact
    assert broker.impact_bps("UNKNOWN", 10_000.0) == 0.0


def test_fill_price_includes_spread_slippage_and_impact_side_aware() -> None:
    broker = SimulatedBroker(cash=1e6, slippage_bps=5.0, half_spread_bps=2.0, impact_coefficient_bps=100.0)
    broker.adv_dollars = {"AAA": 1_000_000.0}
    buy = SimulatedOrder("AAA", "BUY", quantity=100.0, price=100.0)
    sell = SimulatedOrder("AAA", "SELL", quantity=100.0, price=100.0)
    total_bps = 5.0 + 2.0 + 10.0
    assert broker.estimate_fill_price(buy) == pytest.approx(100.0 * (1 + total_bps / 1e4))
    assert broker.estimate_fill_price(sell) == pytest.approx(100.0 * (1 - total_bps / 1e4))


def test_buy_with_impact_still_fits_cash_buffer() -> None:
    broker = SimulatedBroker(cash=10_000.0, transaction_cost_rate=0.001)
    broker.adv_dollars = {"AAA": 50_000.0}
    target = 10_000.0 / (1 + broker.buy_fill_buffer("AAA", 10_000.0))
    fill = broker.process_order(SimulatedOrder("AAA", "BUY", target / 100.0, 100.0), "t")
    assert fill["impact_bps"] > 0 and broker.cash >= -1e-6
    assert math.isfinite(fill["fill_price"])


def test_state_roundtrip_keeps_cost_settings() -> None:
    broker = SimulatedBroker(half_spread_bps=3.0, impact_coefficient_bps=50.0)
    restored = SimulatedBroker.from_dict(broker.to_dict())
    assert restored.half_spread_bps == 3.0 and restored.impact_coefficient_bps == 50.0
