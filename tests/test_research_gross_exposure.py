import pandas as pd
import pytest

from src import config
from src.operations.configuration import load_platform_config
from src.portfolio.portfolio import (
    build_prediction_long_short_returns,
    equal_weight_long_short_weights,
)


def test_research_gross_is_explicit_and_distinct_from_paper_cap() -> None:
    platform = load_platform_config()
    assert config.RESEARCH_LONG_SHORT_GROSS_EXPOSURE == platform["research"]["long_short_gross_exposure"] == 2.0
    assert config.MAX_GROSS_EXPOSURE == platform["risk_limits"]["max_gross_exposure"] == 1.0


@pytest.mark.parametrize("gross", [1.0, 2.0])
def test_long_short_weights_match_requested_gross(gross: float) -> None:
    scores = pd.Series({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    weights = equal_weight_long_short_weights(scores, 2, 2, gross_exposure=gross)
    assert weights.abs().sum() == pytest.approx(gross)
    assert weights.sum() == pytest.approx(0.0)


def test_prediction_returns_scale_with_gross() -> None:
    date = pd.Timestamp("2024-01-31")
    preds = pd.DataFrame({
        "date": [date] * 4,
        "ticker": ["A", "B", "C", "D"],
        "score": [4.0, 3.0, 2.0, 1.0],
        "target": [0.04, 0.02, -0.01, -0.03],
    })
    full = build_prediction_long_short_returns(preds, "score", "target", 2, 2, gross_exposure=2.0)
    half = build_prediction_long_short_returns(preds, "score", "target", 2, 2, gross_exposure=1.0)
    assert full["gross_return"].iloc[0] == pytest.approx(0.05)
    assert half["gross_return"].iloc[0] == pytest.approx(0.025)
