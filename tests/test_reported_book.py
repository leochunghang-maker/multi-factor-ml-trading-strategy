"""The reporting layer (make report) uses the exact reported multi-factor book.

Rebuilds the book's held weights from the checked-in data and checks they
reproduce results/multi_factor_results.csv, so exposure/turnover/sector
reports describe the same book README quotes. Read-only: writes nothing.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import FORWARD_RETURNS_PATH, MULTI_FACTOR_RESULTS_PATH
from src.reporting.analytics import turnover_series, turnover_statistics
from src.reporting.doc_metrics import expected_metrics, pct


@pytest.fixture
def book(reported_book):
    return reported_book  # built once per session in tests/conftest.py


def test_rebuilt_book_reproduces_committed_results(book) -> None:
    results, weights = book
    committed = pd.read_csv(MULTI_FACTOR_RESULTS_PATH, index_col=0, parse_dates=True)
    assert list(results.index) == list(committed.index) == list(weights.index)
    np.testing.assert_allclose(results["portfolio_return"], committed["portfolio_return"], atol=1e-12)


def test_held_weights_explain_gross_return_and_turnover(book) -> None:
    results, weights = book
    forward = pd.read_csv(FORWARD_RETURNS_PATH, index_col=0, parse_dates=True)
    gross = (weights * forward.reindex(index=weights.index, columns=weights.columns).fillna(0.0)).sum(axis=1)
    np.testing.assert_allclose(gross, results["gross_return"], atol=1e-12)
    turnover = turnover_series(weights)
    np.testing.assert_allclose(turnover, results["traded_gross_exposure"].iloc[1:] / 2.0, atol=1e-12)
    stats = turnover_statistics(turnover)
    expected = expected_metrics()
    assert pct(2)(stats["average_turnover"]) == expected["mf.turnover_avg"]
    assert pct(2)(stats["latest_turnover"]) == expected["mf.turnover_latest"]


def test_turnover_series_excludes_first_rebalance() -> None:
    weights = pd.DataFrame({"A": [0.5, 0.0, 0.5], "B": [0.0, 0.5, 0.0]}, index=[1, 2, 3])
    assert turnover_series(weights).tolist() == [0.5, 0.5]
