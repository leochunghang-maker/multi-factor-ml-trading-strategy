import pytest

from src.execution.simulated_broker import SimulatedBroker, SimulatedOrder


def test_simulated_broker_updates_cash_and_positions_for_buy_and_sell() -> None:
    broker = SimulatedBroker(cash=1_000.0, transaction_cost_rate=0.01, slippage_bps=0.0)

    buy = broker.process_order(
        SimulatedOrder("AAA", "BUY", quantity=5, price=100.0),
        timestamp="2024-01-02",
        prices={"AAA": 100.0},
        max_position_weight=1.0,
    )
    sell = broker.process_order(
        SimulatedOrder("AAA", "SELL", quantity=2, price=110.0),
        timestamp="2024-01-03",
        prices={"AAA": 110.0},
        max_position_weight=1.0,
    )

    assert buy["status"] == "FILLED"
    assert sell["status"] == "FILLED"
    assert broker.positions["AAA"] == pytest.approx(3.0)
    assert broker.cash == pytest.approx(1_000.0 - 500.0 - 5.0 + 220.0 - 2.2)
    assert broker.portfolio_value({"AAA": 110.0}) == pytest.approx(broker.cash + 330.0)


def test_simulated_broker_rejects_short_sales_by_default() -> None:
    broker = SimulatedBroker(cash=1_000.0, positions={"AAA": 1.0})

    with pytest.raises(ValueError, match="Shorting is not allowed"):
        broker.process_order(
            SimulatedOrder("AAA", "SELL", quantity=2.0, price=100.0),
            timestamp="2024-01-02",
            prices={"AAA": 100.0},
        )


def test_simulated_broker_rejects_missing_prices_and_position_breaches() -> None:
    broker = SimulatedBroker(cash=1_000.0, transaction_cost_rate=0.0, slippage_bps=0.0)

    with pytest.raises(ValueError, match="Missing executable price"):
        broker.process_order(
            SimulatedOrder("AAA", "BUY", quantity=1.0, price=100.0),
            timestamp="2024-01-02",
            prices={},
        )

    with pytest.raises(ValueError, match="max position weight"):
        broker.process_order(
            SimulatedOrder("AAA", "BUY", quantity=8.0, price=100.0),
            timestamp="2024-01-02",
            prices={"AAA": 100.0},
            max_position_weight=0.50,
        )
