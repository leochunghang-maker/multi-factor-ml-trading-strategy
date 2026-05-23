import pandas as pd
import pytest

from src.portfolio.allocation import normalize_long_only_weights
from src.portfolio.constraints import (
    PortfolioConstraints,
    apply_max_position_size,
    constrain_weights,
)
from src.portfolio.portfolio import equal_weight_long_short_weights


def test_equal_weight_long_short_weights_are_balanced() -> None:
    scores = pd.Series({"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0})

    weights = equal_weight_long_short_weights(scores, long_count=2, short_count=2)

    assert weights.loc[["AAA", "BBB"]].tolist() == [0.5, 0.5]
    assert weights.loc[["CCC", "DDD"]].tolist() == [-0.5, -0.5]
    assert weights.sum() == pytest.approx(0.0)
    assert weights.abs().sum() == pytest.approx(2.0)


def test_long_only_weight_normalization_respects_position_cap() -> None:
    raw = pd.Series({"AAA": 0.80, "BBB": 0.10, "CCC": 0.10})

    weights = normalize_long_only_weights(raw, max_position_weight=0.40)

    assert (weights >= 0).all()
    assert weights.max() <= 0.40 + 1e-12
    assert weights.sum() == pytest.approx(1.0)


def test_max_position_constraint_clips_long_and_short_exposure() -> None:
    weights = pd.Series({"AAA": 0.25, "BBB": -0.30, "CCC": 0.05})

    constrained = apply_max_position_size(weights, max_position_weight=0.10)

    assert constrained.loc["AAA"] == pytest.approx(0.10)
    assert constrained.loc["BBB"] == pytest.approx(-0.10)
    assert constrained.abs().max() <= 0.10


def test_constrain_weights_reports_position_and_turnover_limits() -> None:
    target = pd.Series({"AAA": 0.60, "BBB": 0.40}, name=pd.Timestamp("2024-01-31"))
    constraints = PortfolioConstraints(
        max_position_weight=0.30,
        max_sector_weight=1.0,
        max_gross_exposure=1.0,
        max_turnover=0.50,
        sector_map={"AAA": "Tech", "BBB": "Health Care"},
    )

    constrained, diagnostics = constrain_weights(
        target,
        previous_weights=pd.Series({"AAA": 0.0, "BBB": 0.0}),
        constraints=constraints,
    )

    assert constrained.abs().max() <= 0.30 + 1e-12
    assert diagnostics["traded_gross_exposure"] <= 0.50
    assert diagnostics["turnover_limited"] is True
