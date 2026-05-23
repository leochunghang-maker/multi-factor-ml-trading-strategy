import numpy as np
import pandas as pd
import pytest

from src.portfolio.portfolio import transaction_cost_from_weight_change
from src.reporting.metrics import calculate_performance_metrics, equity_curve


def test_performance_metrics_calculate_sharpe_and_drawdown() -> None:
    returns = pd.Series([0.10, -0.05, 0.02])

    metrics = calculate_performance_metrics(returns, periods_per_year=12)
    curve = equity_curve(returns)
    expected_volatility = returns.std() * np.sqrt(12)
    expected_annual_return = returns.mean() * 12

    assert metrics["annual_return"] == pytest.approx(expected_annual_return)
    assert metrics["annual_volatility"] == pytest.approx(expected_volatility)
    assert metrics["sharpe_ratio"] == pytest.approx(expected_annual_return / expected_volatility)
    assert metrics["max_drawdown"] == pytest.approx((curve / curve.cummax() - 1).min())


def test_turnover_is_based_on_absolute_weight_changes() -> None:
    current = pd.Series({"AAA": 0.40, "BBB": -0.20, "CCC": 0.0})
    previous = pd.Series({"AAA": 0.10, "BBB": -0.50, "CCC": 0.20})

    cost, traded_gross_exposure = transaction_cost_from_weight_change(
        current,
        previous,
        transaction_cost=0.001,
    )

    assert traded_gross_exposure == pytest.approx(0.80)
    assert cost == pytest.approx(0.0008)
