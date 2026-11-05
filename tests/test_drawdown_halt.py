"""Tests for paper/sim drawdown circuit breaker."""

from types import SimpleNamespace

import pytest

from src.config import MAX_DRAWDOWN_WARNING
from src.operations.configuration import load_platform_config
from src.risk.drawdown import (
    filter_orders_for_drawdown_halt,
    is_drawdown_halted,
    portfolio_drawdown,
)


def test_portfolio_drawdown_from_peak() -> None:
    assert portfolio_drawdown(100.0, 100.0) == 0.0
    assert portfolio_drawdown(90.0, 100.0) == pytest.approx(-0.1)
    assert portfolio_drawdown(50.0, 0.0) == 0.0


def test_is_drawdown_halted_threshold() -> None:
    assert is_drawdown_halted(-0.10, -0.10)
    assert is_drawdown_halted(-0.15, -0.10)
    assert not is_drawdown_halted(-0.05, -0.10)
    assert not is_drawdown_halted(-0.50, 0.0)


def test_filter_orders_keeps_sells_only_when_halted() -> None:
    buy = SimpleNamespace(side="BUY", symbol="AAA")
    sell = SimpleNamespace(side="SELL", symbol="BBB")
    assert filter_orders_for_drawdown_halt([buy, sell], halted=False) == [buy, sell]
    assert filter_orders_for_drawdown_halt([buy, sell], halted=True) == [sell]


def test_max_drawdown_warning_mirrors_platform_config() -> None:
    platform = load_platform_config()
    assert MAX_DRAWDOWN_WARNING == platform["risk_limits"]["max_drawdown_warning"] == -0.1
