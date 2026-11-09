"""Paper vs frictionless-backtest tracking check (fixed thresholds; temp dirs only)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.operations.paper_ops import PaperOpsPaths
from src.operations.paper_review import build_weekly_review
from src.operations.paper_tracking import (
    DEFAULT_TRACKING,
    compute_tracking,
    evaluate_row,
    tracking_settings,
    weekly_tracking,
    write_tracking,
)

S = {**DEFAULT_TRACKING, "modeled_cost_bps": 17.0}


def _make_run(paths, day, nav_start, nav_end, targets=None, prices=None, positions=None, traded=0.0,
              halted=False):
    d = paths.run_dir(day)
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"status": "COMPLETED", "signal_date": day, "nav_start": nav_start,
                                            "nav_end": nav_end, "snapshot": {"drawdown_halted": halted}}))
    if targets is not None:
        pd.DataFrame([{"ticker": t, "side": "LONG", "target_weight": w} for t, w in targets.items()]
                     + [{"ticker": "ZZZ", "side": "HOLD", "target_weight": 0.0}]).to_csv(d / "targets.csv", index=False)
    pd.DataFrame(sorted((prices or {}).items()), columns=["symbol", "price"]).to_csv(d / "prices.csv", index=False)
    pos = positions or {}
    pd.DataFrame([{"symbol": s, "quantity": q, "price": prices[s], "market_value": q * prices[s]}
                  for s, q in pos.items()], columns=["symbol", "quantity", "price", "market_value"]
                 ).to_csv(d / "positions_end.csv", index=False)
    pd.DataFrame([{"symbol": "A", "side": "BUY", "quantity": 1, "notional": traded, "transaction_cost": 0,
                   "status": "FILLED"}] if traded else [],
                 columns=["symbol", "side", "quantity", "notional", "transaction_cost", "status"]
                 ).to_csv(d / "fills.csv", index=False)


@pytest.fixture()
def paths(tmp_path):
    return PaperOpsPaths(tmp_path / "paper")


T = {"A": 0.5, "B": 0.5}
POS = {"A": 495.0, "B": 495.0}  # 99% invested, 1% cash at 100/100


def test_clean_tracking_has_no_findings(paths) -> None:
    _make_run(paths, "2026-11-02", 100_000, 100_000, T, {"A": 100, "B": 100}, POS)
    # A +10%: paper 1,000 cash + 495*110 + 495*100 = 104,950 (+4.95%) vs shadow +5.00% -> -5 bps (cash drag).
    _make_run(paths, "2026-11-03", 104_950, 104_950 - 17.0, T,
              {"A": 110, "B": 100}, {"A": 450.0, "B": 495.0}, traded=10_000)
    frame = compute_tracking(paths, S)
    assert list(frame["status"]) == ["FIRST_RUN", "OK"]
    row = frame.iloc[1].to_dict()
    assert row["paper_hold_return"] == pytest.approx(0.0495)
    assert row["shadow_hold_return"] == pytest.approx(0.05)
    assert row["hold_gap_bps"] == pytest.approx(-5.0)
    assert row["realized_cost_bps_traded"] == pytest.approx(17.0) and row["cost_ratio"] == pytest.approx(1.0)
    assert evaluate_row(row, S) == []
    assert frame.iloc[0]["weight_deviation"] == pytest.approx(0.01)  # 1% cash vs fully invested targets


@pytest.mark.parametrize("nav_start, severity", [(104_950 - 600, "WARNING"), (104_950 - 2_500, "ERROR")])
def test_holding_gap_thresholds(paths, nav_start, severity) -> None:
    _make_run(paths, "2026-11-02", 100_000, 100_000, T, {"A": 100, "B": 100}, POS)
    _make_run(paths, "2026-11-03", nav_start, nav_start, T, {"A": 110, "B": 100}, POS)
    row = compute_tracking(paths, S).iloc[1].to_dict()
    found = evaluate_row(row, S)
    assert [(s, c) for s, c, _ in found] == [(severity, "tracking_hold_gap")]


def test_cost_and_weight_checks(paths) -> None:
    _make_run(paths, "2026-11-02", 100_000, 100_000 - 60, T, {"A": 100, "B": 100}, {"A": 990.0}, traded=10_000)
    row = compute_tracking(paths, S).iloc[0].to_dict()
    assert row["cost_ratio"] == pytest.approx(60 / 17)
    realized_a = 99_000 / 99_940
    assert row["weight_deviation"] == pytest.approx((realized_a - 0.5) + 0.5)  # A over, B missing
    checks = {c for _, c, _ in evaluate_row(row, S)}
    assert checks == {"tracking_cost", "tracking_weights"}


def test_small_trades_skip_cost_check(paths) -> None:
    _make_run(paths, "2026-11-02", 100_000, 100_000 - 50, T, {"A": 100, "B": 100}, POS, traded=400)  # 0.4% < 0.5%
    row = compute_tracking(paths, S).iloc[0].to_dict()
    assert pd.isna(row["cost_ratio"]) and row["realized_cost_bps_traded"] == pytest.approx(1250)
    assert evaluate_row(row, S) == []


def test_skips_halted_missing_targets_and_missing_prices(paths) -> None:
    _make_run(paths, "2026-11-02", 100_000, 100_000, T, {"A": 100, "B": 100}, POS, halted=True)
    _make_run(paths, "2026-11-03", 90_000, 90_000, None, {"A": 90, "B": 90}, POS)
    _make_run(paths, "2026-11-04", 91_000, 91_000, T, {"A": 91}, {"A": 990.0})
    _make_run(paths, "2026-11-05", 92_000, 92_000, T, {"A": 92}, {"A": 990.0})  # B has no price on either day
    frame = compute_tracking(paths, S)
    assert list(frame["status"]) == ["FIRST_RUN", "PREV_HALTED", "NO_TARGETS", "MISSING_PRICES"]
    assert frame["hold_gap_bps"].isna().all()
    assert pd.isna(frame.iloc[0]["weight_deviation"])  # halted run: targets were not implemented
    found = evaluate_row(frame.iloc[3].to_dict(), S)
    assert ("INFO", "tracking_skipped") in [(s, c) for s, c, _ in found]
    assert "tracking_hold_gap" not in [c for _, c, _ in found]


def test_weekly_cumulative_gap_warning(paths) -> None:
    navs = [100_000.0]
    _make_run(paths, "2026-11-02", 100_000, 100_000, {"A": 1.0}, {"A": 100}, {"A": 1000.0})
    for i, day in enumerate(["2026-11-03", "2026-11-04", "2026-11-05"], start=1):
        nav = navs[-1] * (1 - 0.0040)  # flat prices, paper loses 40 bps per interval (< 50 each)
        navs.append(nav)
        _make_run(paths, day, nav, nav, {"A": 1.0}, {"A": 100}, {"A": nav / 100})
    frame = write_tracking(paths, S)
    assert (paths.root / "tracking.csv").exists()
    week = weekly_tracking(frame, pd.Timestamp("2026-11-02").date(), pd.Timestamp("2026-11-08").date(), S)
    assert week["intervals"] == 3 and week["skipped"] == 1
    assert week["cum_gap_bps"] == pytest.approx(-120.0)
    assert [f["check"] for f in week["findings"]] == ["tracking_weekly_gap"]
    review = build_weekly_review(paths, "2026-11-06")
    assert review["tracking"]["intervals"] == 3


def test_thresholds_come_from_config_and_cost_model() -> None:
    settings = tracking_settings()
    for key, value in DEFAULT_TRACKING.items():
        assert settings[key] == value  # config matches the documented a-priori defaults
    assert settings["modeled_cost_bps"] == pytest.approx(17.0)  # 10 commission + 5 slippage + 2 half-spread
    custom = tracking_settings({"execution": {"transaction_cost": 0.002, "slippage_bps": 1, "half_spread_bps": 1}})
    assert custom["modeled_cost_bps"] == pytest.approx(22.0)
