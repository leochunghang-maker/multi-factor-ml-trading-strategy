"""Sector-neutral ranking and sector net-exposure caps."""

from __future__ import annotations

import pandas as pd

from src.portfolio.portfolio import (
    apply_sector_net_exposure_limits,
    demean_scores_by_sector,
    equal_weight_long_short_weights,
    load_sector_map,
)


SECTORS = {
    "A": "Tech",
    "B": "Tech",
    "C": "Tech",
    "D": "Banks",
    "E": "Banks",
    "F": "Banks",
}


def test_load_sector_map_has_sp100_style_coverage() -> None:
    sector_map = load_sector_map()
    assert sector_map["AAPL"] == "Information Technology"
    assert sector_map["JPM"] == "Financials"
    assert len(sector_map) >= 90


def test_demean_scores_by_sector_zero_means() -> None:
    scores = pd.Series({"A": 10.0, "B": 8.0, "C": 6.0, "D": 3.0, "E": 1.0, "F": -1.0})
    demeaned = demean_scores_by_sector(scores, SECTORS)
    assert abs(demeaned[["A", "B", "C"]].mean()) < 1e-12
    assert abs(demeaned[["D", "E", "F"]].mean()) < 1e-12
    assert demeaned["A"] > demeaned["C"]


def test_equal_weight_sector_neutral_avoids_raw_sector_tilt() -> None:
    # Raw ranking would long all Tech and short all Banks.
    scores = pd.Series({"A": 9.0, "B": 8.0, "C": 7.0, "D": 2.0, "E": 1.0, "F": 0.0})
    raw = equal_weight_long_short_weights(
        scores,
        long_count=2,
        short_count=2,
        gross_exposure=2.0,
        sector_neutral=False,
        max_sector_net_exposure=99.0,
        sector_map=SECTORS,
    )
    assert set(raw[raw > 0].index) == {"A", "B"}
    assert set(raw[raw < 0].index) == {"E", "F"}

    neutral = equal_weight_long_short_weights(
        scores,
        long_count=2,
        short_count=2,
        gross_exposure=2.0,
        sector_neutral=True,
        max_sector_net_exposure=99.0,
        sector_map=SECTORS,
    )
    # Within-sector ranking: A vs C (Tech), D vs F (Banks).
    assert set(neutral[neutral > 0].index) == {"A", "D"}
    assert set(neutral[neutral < 0].index) == {"C", "F"}
    assert abs(neutral.abs().sum() - 2.0) < 1e-9


def test_sector_net_exposure_cap_and_gross_restore() -> None:
    # Pure sector L/S cannot keep |net|<=0.2 and gross=2.0; accept lower gross.
    weights = pd.Series({"A": 0.5, "B": 0.5, "D": -0.5, "E": -0.5})
    capped = apply_sector_net_exposure_limits(
        weights,
        max_sector_net_exposure=0.2,
        sector_map=SECTORS,
        target_gross_exposure=2.0,
    )
    tech_net = float(capped[["A", "B"]].sum())
    banks_net = float(capped[["D", "E"]].sum())
    assert abs(tech_net) <= 0.2 + 1e-9
    assert abs(banks_net) <= 0.2 + 1e-9
    assert float(capped.abs().sum()) <= 0.4 + 1e-9
    assert abs(float(capped.sum())) < 1e-9

    # Diversified book: nets stay under the cap and gross can be restored.
    mixed = pd.Series(
        {
            "A": 0.25,
            "B": 0.25,
            "D": 0.25,
            "E": 0.25,
            "C": -0.25,
            "F": -0.75,
        }
    )
    # Tech net = A+B+C = 0.25; Banks = D+E+F = -0.25. Cap 0.15 should bite.
    capped_mixed = apply_sector_net_exposure_limits(
        mixed,
        max_sector_net_exposure=0.15,
        sector_map=SECTORS,
        target_gross_exposure=2.0,
    )
    assert abs(float(capped_mixed[["A", "B", "C"]].sum())) <= 0.15 + 1e-8
    assert abs(float(capped_mixed[["D", "E", "F"]].sum())) <= 0.15 + 1e-8
    assert abs(float(capped_mixed.abs().sum()) - 2.0) < 1e-9


def test_config_sector_flags_default_on() -> None:
    from src.config import MAX_SECTOR_NET_EXPOSURE, SECTOR_MAP_PATH, SECTOR_NEUTRAL

    assert SECTOR_NEUTRAL is True
    assert MAX_SECTOR_NET_EXPOSURE == 0.15
    assert SECTOR_MAP_PATH.endswith("sector_map.csv")
