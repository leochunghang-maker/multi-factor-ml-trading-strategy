"""Paper-constraint binding diagnostics (descriptive only; weights unchanged)."""

from pathlib import Path

import pandas as pd
import pytest

from src.config import CONSTRAINT_BINDING_REPORT_PATH
from src.portfolio import PortfolioConstraints, constrain_weights
from src.reporting.constraint_binding import (
    binding_summary,
    exposure_comparison,
    render_markdown,
    vol_sizing_comparison,
)

ROOT = Path(__file__).resolve().parents[1]
SECTORS = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Energy", "E": "Energy"}


def test_each_step_records_how_much_it_trimmed() -> None:
    constraints = PortfolioConstraints(sector_map=SECTORS, max_turnover=1.0)
    target = pd.Series({"A": 0.30, "B": 0.08, "C": 0.08, "D": -0.30, "E": 0.05}, name=pd.Timestamp("2026-01-30"))
    weights, d = constrain_weights(target, constraints=constraints)
    # Position cap: A and D clipped from 30% to 10%.
    assert d["bind_position_names_clipped"] == 2
    assert d["bind_position_gross_trimmed"] == pytest.approx(0.40)
    # Sector gross: Tech 0.26 and Energy 0.15 after the position cap, both under 35%.
    assert d["bind_sector_count_scaled"] == 0 and d["bind_sector_gross_trimmed"] == pytest.approx(0.0)
    assert d["bind_sector_max_gross_before"] == pytest.approx(0.26)
    assert d["bind_gross_cap_gross_trimmed"] == pytest.approx(0.0)
    assert d["bind_turnover_wanted_traded_gross"] == pytest.approx(0.41)
    assert not d["turnover_limited"]
    assert weights.abs().max() <= 0.10 + 1e-12


def test_sector_gross_and_turnover_binding_amounts() -> None:
    constraints = PortfolioConstraints(sector_map=SECTORS, max_turnover=0.2)
    target = pd.Series({"A": 0.10, "B": 0.10, "C": 0.10, "D": 0.10, "E": 0.10}, name=pd.Timestamp("2026-01-30"))
    previous = pd.Series(0.0, index=target.index)
    weights, d = constrain_weights(target, previous_weights=previous, constraints=constraints)
    assert d["bind_sector_count_scaled"] == 0  # Tech gross 30% < 35%
    assert d["bind_turnover_wanted_traded_gross"] == pytest.approx(0.50)
    assert d["turnover_limited"] and weights.abs().sum() == pytest.approx(0.20)
    summary = binding_summary(pd.DataFrame([d], index=[target.name]), constraints).set_index("constraint")
    assert summary.loc["Turnover cap", "binding_rebalances"] == 1
    assert summary.loc["Turnover cap", "max"] == pytest.approx(0.30)  # wanted 50% vs 20% cap
    assert summary.loc["Position cap", "binding_rebalances"] == 0

    heavy = pd.Series({"A": 0.10, "B": 0.10, "C": 0.10, "D": 0.0, "E": 0.0}, name=target.name)
    _, d = constrain_weights(heavy, constraints=PortfolioConstraints(sector_map=SECTORS, max_sector_weight=0.15))
    assert d["bind_sector_count_scaled"] == 1
    assert d["bind_sector_gross_trimmed"] == pytest.approx(0.15)


def test_checked_in_binding_report_matches_the_reported_book(reported_book, reported_book_constrained) -> None:
    _, weights = reported_book
    constrained, diagnostics = reported_book_constrained["with_vol"]
    comparison = vol_sizing_comparison(
        (constrained, diagnostics), reported_book_constrained["without_vol"], weights)
    expected = render_markdown(
        binding_summary(diagnostics), exposure_comparison(weights, constrained), diagnostics, comparison)
    checked_in = (ROOT / CONSTRAINT_BINDING_REPORT_PATH).read_text()
    assert checked_in == expected, "stale: run `make report` and commit reports/paper_constraint_binding_report.md"


def test_vol_sizing_counterfactual_changes_only_that_step(reported_book, reported_book_constrained) -> None:
    _, with_diag = reported_book_constrained["with_vol"]
    without_weights, without_diag = reported_book_constrained["without_vol"]
    assert "bind_vol_sizing_gross_trimmed" in with_diag and "bind_vol_sizing_gross_trimmed" not in without_diag
    # Skipping vol sizing feeds more gross into the later caps, never less.
    assert (without_diag["bind_gross_cap_gross_before"] >= with_diag["bind_gross_cap_gross_before"] - 1e-12).all()
    assert (without_weights.abs().sum(axis=1) <= 1.0 + 1e-9).all()
