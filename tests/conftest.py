"""Shared test fixtures.

Outbound network is blocked for the whole session: tests must use local files,
not live Yahoo downloads. The reported multi-factor book is built once and
shared by tests/test_reported_book.py and tests/test_constraint_binding.py;
tests must not mutate the returned frames.
"""

from __future__ import annotations

import pytest

from tests.network_guard import install as install_network_guard
from tests.network_guard import uninstall as uninstall_network_guard


@pytest.fixture(scope="session", autouse=True)
def _block_outbound_network():
    install_network_guard()
    try:
        yield
    finally:
        uninstall_network_guard()


@pytest.fixture(scope="session")
def reported_book():
    """(results, held weights) of the reported multi-factor book; writes nothing."""
    from src.backtests.factor_backtests import build_reported_multi_factor_book

    return build_reported_multi_factor_book()


@pytest.fixture(scope="session")
def reported_book_constrained(reported_book):
    """Paper guardrails on the reported book, with and without per-name vol sizing."""
    from src.config import DAILY_RETURNS_PATH, PRICE_DATA_PATH, VOLUME_DATA_PATH
    from src.data.io import load_price_data
    from src.portfolio import constrain_weight_history
    from src.portfolio.constraints import load_optional_volume_data
    from src.reporting.summary import load_factor_frame

    _, weights = reported_book
    prices = load_price_data(PRICE_DATA_PATH)
    volumes = load_optional_volume_data(VOLUME_DATA_PATH)
    return {
        "with_vol": constrain_weight_history(
            weights, daily_returns=load_factor_frame(DAILY_RETURNS_PATH), prices=prices, volumes=volumes),
        "without_vol": constrain_weight_history(weights, daily_returns=None, prices=prices, volumes=volumes),
    }
