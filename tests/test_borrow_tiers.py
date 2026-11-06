"""Name-level HTB borrow tier proxy tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.execution.borrow_tiers import (
    assign_borrow_bps,
    classify_borrow_tier,
    load_borrow_tier_overrides,
    tier_rates_bps,
)
from src.execution.financing import financing_cost_from_weights, daily_financing_dollars
from src.operations.configuration import load_platform_config


def test_borrow_tier_config_mirrors_platform() -> None:
    execution = load_platform_config()["execution"]
    assert config.BORROW_TIERING_ENABLED == execution["borrow_tiering_enabled"] is True
    assert config.BORROW_TIER_RATES_BPS["htb"] == execution["borrow_tier_rates_bps"]["htb"]


def test_classify_tiers_from_adv_and_vol() -> None:
    assert classify_borrow_tier(
        1e9, 0.01, adv_p20=5e8, adv_p10=2e8, vol_p80=0.03, vol_p90=0.04
    ) == "gc"
    assert classify_borrow_tier(
        1e8, 0.01, adv_p20=5e8, adv_p10=2e8, vol_p80=0.03, vol_p90=0.04
    ) == "elevated"
    assert classify_borrow_tier(
        1e8, 0.05, adv_p20=5e8, adv_p10=2e8, vol_p80=0.03, vol_p90=0.04
    ) == "htb"
    assert classify_borrow_tier(
        1e9, 0.01, adv_p20=5e8, adv_p10=2e8, vol_p80=0.03, vol_p90=0.04, override_tier="special"
    ) == "special"


def test_assign_borrow_bps_honors_overrides_and_rates() -> None:
    adv = pd.Series({"AAA": 1e9, "BBB": 1e8, "PLTR": 2e9})
    vol = pd.Series({"AAA": 0.01, "BBB": 0.01, "PLTR": 0.02})
    overrides = pd.DataFrame([{"ticker": "PLTR", "tier": "htb"}])
    rates = {"gc": 100.0, "elevated": 300.0, "htb": 1000.0, "special": 3000.0}
    assigned = assign_borrow_bps(adv, vol, overrides=overrides, rates=rates)
    assert assigned.loc["AAA", "tier"] == "gc"
    assert assigned.loc["BBB", "tier"] == "elevated"
    assert assigned.loc["PLTR", "tier"] == "htb"
    assert assigned.loc["PLTR", "annual_borrow_bps"] == 1000.0


def test_name_level_financing_costs_more_for_htb_shorts() -> None:
    weights = pd.Series({"GC": -0.5, "HTB": -0.5, "L1": 0.5, "L2": 0.5})
    flat = financing_cost_from_weights(weights, annual_borrow_bps=100.0, periods_per_year=12)
    tiered = financing_cost_from_weights(
        weights,
        annual_borrow_bps=100.0,
        periods_per_year=12,
        borrow_bps_by_symbol={"GC": 100.0, "HTB": 1000.0, "L1": 100.0, "L2": 100.0},
    )
    # Average short rate 550 bps vs 100 → 4.5x higher borrow drag on 100% short.
    assert tiered == pytest.approx(1.0 * 550.0 / 10_000 / 12)
    assert tiered > flat


def test_daily_financing_name_level() -> None:
    cost = daily_financing_dollars(
        annual_borrow_bps=100.0,
        short_market_values_by_symbol={"AAA": 50_000.0, "BBB": 50_000.0},
        borrow_bps_by_symbol={"AAA": 100.0, "BBB": 1000.0},
    )
    expected = (
        50_000.0 * 100.0 / 10_000 / 252
        + 50_000.0 * 1000.0 / 10_000 / 252
    )
    assert cost == pytest.approx(expected)


def test_overrides_file_loads() -> None:
    overrides = load_borrow_tier_overrides()
    assert not overrides.empty
    assert "PLTR" in set(overrides["ticker"])
    assert tier_rates_bps()["gc"] == config.ANNUAL_BORROW_BPS
