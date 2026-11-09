"""Shared, read-only fixtures for the slow reported-book rebuild.

Rebuilding the reported multi-factor book from the checked-in data takes a few
seconds, so it is built once per test session and shared by
tests/test_reported_book.py and tests/test_constraint_binding.py. Tests must
not mutate the returned frames.
"""

import pytest


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
