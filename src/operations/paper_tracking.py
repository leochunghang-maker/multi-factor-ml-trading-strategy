"""Paper vs backtest tracking check (simulated paper account; local files only).

For every completed paper run the check compares what the paper account did
with a **frictionless target-weight backtest** of the same signals:

- **Holding gap** (interval from run *i-1* to run *i*): paper return
  ``nav_start_i / nav_end_{i-1} - 1`` vs the shadow return
  ``sum_k w_k,i-1 * (P_k,i / P_k,i-1 - 1)`` of run *i-1*'s target weights at the
  two runs' prices. The gap captures cash drag, unfilled/clipped orders and
  pricing/pipeline drift (no trading happens between runs).
- **Cost check** (at run *i*): realized rebalance cost ``nav_start - nav_end``
  per traded dollar vs the modeled per-dollar cost (commission + slippage +
  half-spread from the execution config; impact is not in the model, so the
  ratio is expected to be at or slightly above 1).
- **Weight deviation** (after run *i*): ``sum |w_realized - w_target|``.

Inputs come only from ``results/paper/runs/<date>/`` (run.json, targets.csv,
prices.csv, positions_end.csv, fills.csv). Rows are written to
``results/paper/tracking.csv``. Thresholds live in ``paper_tracking`` in the
platform config; they were fixed on 2026-11-09 from the simulator design, before
any tracking result was computed, and must not be tuned on results. This check
does not change any research output and is not a strategy trial.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_TRACKING = {
    "hold_gap_warning_bps": 50.0,
    "hold_gap_error_bps": 200.0,
    "weekly_cum_gap_warning_bps": 100.0,
    "weight_deviation_warning": 0.10,
    "cost_ratio_min": 0.75,
    "cost_ratio_max": 2.0,
    "min_traded_fraction_for_cost_check": 0.005,
    "max_missing_target_weight": 0.05,
}

TRACKING_COLUMNS = [
    "signal_date", "prev_signal_date", "status", "paper_hold_return", "shadow_hold_return", "hold_gap_bps",
    "missing_target_weight", "traded_fraction", "realized_cost_bps_traded", "modeled_cost_bps_traded",
    "cost_ratio", "weight_deviation", "halted",
]


def tracking_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is None:
        from src.operations.configuration import load_platform_config

        config = load_platform_config()
    settings = {**DEFAULT_TRACKING, **(config.get("paper_tracking") or {})}
    if settings.get("modeled_cost_bps") is None:
        execution = config.get("execution") or {}
        settings["modeled_cost_bps"] = (
            float(execution.get("transaction_cost", 0.001)) * 1e4
            + float(execution.get("slippage_bps", 5.0))
            + float(execution.get("half_spread_bps", 2.0))
        )
    return settings


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _targets(run_dir: Path) -> dict[str, float] | None:
    frame = _read_csv(run_dir / "targets.csv")
    if frame.empty or not {"ticker", "target_weight"}.issubset(frame.columns):
        return None
    if "side" in frame.columns:
        frame = frame[frame["side"] == "LONG"]
    frame = frame[pd.to_numeric(frame["target_weight"], errors="coerce") > 0]
    return {str(t): float(w) for t, w in zip(frame["ticker"], frame["target_weight"])}


def _prices(run_dir: Path) -> dict[str, float]:
    frame = _read_csv(run_dir / "prices.csv")
    if frame.empty:
        return {}
    return {str(s): float(p) for s, p in zip(frame["symbol"], frame["price"]) if pd.notna(p)}


def _load_run(paths, day: str) -> dict[str, Any]:
    run_dir = paths.run_dir(day)
    meta = json.loads((run_dir / "run.json").read_text())
    return {
        "meta": meta,
        "nav_start": float(meta.get("nav_start") or 0.0),
        "nav_end": float(meta.get("nav_end") or 0.0),
        "halted": bool((meta.get("snapshot") or {}).get("drawdown_halted", False)),
        "targets": _targets(run_dir),
        "prices": _prices(run_dir),
        "positions": _read_csv(run_dir / "positions_end.csv"),
        "fills": _read_csv(run_dir / "fills.csv"),
    }


def _nan(value) -> float:
    return float("nan") if value is None else float(value)


def compute_tracking(paths=None, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    from src.operations.paper_ops import PaperOpsPaths, completed_runs

    paths = paths or PaperOpsPaths.default()
    settings = settings or tracking_settings()
    rows = []
    prev = None
    for day in completed_runs(paths):
        cur = _load_run(paths, day)
        row: dict[str, Any] = {"signal_date": day, "prev_signal_date": prev["day"] if prev else None,
                               "halted": cur["halted"]}

        # Cost at this run's rebalance.
        fills = cur["fills"]
        traded = 0.0
        if not fills.empty and "notional" in fills.columns:
            filled = fills[fills["status"].astype(str) == "FILLED"] if "status" in fills.columns else fills
            traded = float(pd.to_numeric(filled["notional"], errors="coerce").abs().sum())
        nav_start = cur["nav_start"]
        row["traded_fraction"] = traded / nav_start if nav_start else None
        row["modeled_cost_bps_traded"] = float(settings["modeled_cost_bps"])
        if traded > 0:
            row["realized_cost_bps_traded"] = (nav_start - cur["nav_end"]) / traded * 1e4
            if row["traded_fraction"] and row["traded_fraction"] >= settings["min_traded_fraction_for_cost_check"]:
                row["cost_ratio"] = row["realized_cost_bps_traded"] / row["modeled_cost_bps_traded"]

        # Weight deviation after the rebalance.
        if cur["targets"] is not None and not cur["halted"] and cur["nav_end"] > 0:
            pos = cur["positions"]
            realized = {}
            if not pos.empty and "market_value" in pos.columns:
                realized = {str(s): float(mv) / cur["nav_end"] for s, mv in zip(pos["symbol"], pos["market_value"])
                            if pd.notna(mv)}
            names = set(realized) | set(cur["targets"])
            row["weight_deviation"] = sum(abs(realized.get(n, 0.0) - cur["targets"].get(n, 0.0)) for n in names)

        # Holding interval from the previous run.
        if prev is None:
            row["status"] = "FIRST_RUN"
        elif prev["run"]["targets"] is None:
            row["status"] = "NO_TARGETS"
        elif prev["run"]["halted"]:
            row["status"] = "PREV_HALTED"  # targets were not implemented; the gap is the halt itself
        else:
            p0, p1, w = prev["run"]["prices"], cur["prices"], prev["run"]["targets"]
            have = {k: v for k, v in w.items() if k in p0 and k in p1 and p0[k] > 0}
            missing = sum(w.values()) - sum(have.values())
            row["missing_target_weight"] = missing
            if missing > settings["max_missing_target_weight"]:
                row["status"] = "MISSING_PRICES"
            else:
                shadow = sum(v * (p1[k] / p0[k] - 1.0) for k, v in have.items())
                paper = nav_start / prev["run"]["nav_end"] - 1.0 if prev["run"]["nav_end"] else float("nan")
                row.update({"status": "OK", "paper_hold_return": paper, "shadow_hold_return": shadow,
                            "hold_gap_bps": (paper - shadow) * 1e4})
        rows.append(row)
        prev = {"day": day, "run": cur}
    return pd.DataFrame(rows, columns=TRACKING_COLUMNS)


def evaluate_row(row: dict[str, Any] | pd.Series, settings: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(severity, check, message) for one tracking row against the fixed thresholds."""
    findings = []
    day = row["signal_date"]
    gap = row.get("hold_gap_bps")
    if gap is not None and not (isinstance(gap, float) and math.isnan(gap)):
        if abs(gap) > settings["hold_gap_error_bps"]:
            findings.append(("ERROR", "tracking_hold_gap",
                             f"{row['prev_signal_date']}→{day}: paper {row['paper_hold_return']:.2%} vs backtest "
                             f"{row['shadow_hold_return']:.2%} ({gap:+.0f} bps; error > {settings['hold_gap_error_bps']:.0f})"))
        elif abs(gap) > settings["hold_gap_warning_bps"]:
            findings.append(("WARNING", "tracking_hold_gap",
                             f"{row['prev_signal_date']}→{day}: paper {row['paper_hold_return']:.2%} vs backtest "
                             f"{row['shadow_hold_return']:.2%} ({gap:+.0f} bps; warning > {settings['hold_gap_warning_bps']:.0f})"))
    ratio = row.get("cost_ratio")
    if ratio is not None and not (isinstance(ratio, float) and math.isnan(ratio)):
        if not settings["cost_ratio_min"] <= ratio <= settings["cost_ratio_max"]:
            findings.append(("WARNING", "tracking_cost",
                             f"{day}: realized cost {row['realized_cost_bps_traded']:.1f} bps per traded dollar vs "
                             f"modeled {row['modeled_cost_bps_traded']:.1f} (ratio {ratio:.2f}, band "
                             f"{settings['cost_ratio_min']}–{settings['cost_ratio_max']})"))
    dev = row.get("weight_deviation")
    if dev is not None and not (isinstance(dev, float) and math.isnan(dev)):
        if dev > settings["weight_deviation_warning"]:
            findings.append(("WARNING", "tracking_weights",
                             f"{day}: realized weights deviate from targets by {dev:.1%} of NAV "
                             f"(warning > {settings['weight_deviation_warning']:.0%})"))
    if row.get("status") == "MISSING_PRICES":
        findings.append(("INFO", "tracking_skipped",
                         f"{day}: {row['missing_target_weight']:.1%} of previous target weight has no price; gap not computed"))
    return findings


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            for r in frame.to_dict("records")]


def latest_tracking_findings(paths=None, settings: dict[str, Any] | None = None) -> list[tuple[str, str, str]]:
    from src.operations.paper_ops import PaperOpsPaths, read_csv_safe

    paths = paths or PaperOpsPaths.default()
    settings = settings or tracking_settings()
    frame = read_csv_safe(tracking_path(paths), TRACKING_COLUMNS)
    if frame.empty:
        return []
    return evaluate_row(_records(frame.tail(1))[0], settings)


def weekly_tracking(frame: pd.DataFrame, start, end, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = settings or tracking_settings()
    if frame.empty:
        week = frame
    else:
        d = pd.to_datetime(frame["signal_date"]).dt.date
        week = frame[(d >= start) & (d <= end)]
    gaps = pd.to_numeric(week["hold_gap_bps"], errors="coerce").dropna() if not week.empty else pd.Series(dtype=float)
    ratios = pd.to_numeric(week["cost_ratio"], errors="coerce").dropna() if not week.empty else pd.Series(dtype=float)
    devs = pd.to_numeric(week["weight_deviation"], errors="coerce").dropna() if not week.empty else pd.Series(dtype=float)
    cum = float(gaps.sum()) if len(gaps) else None
    findings = [f for r in _records(week) for f in evaluate_row(r, settings)]
    if cum is not None and abs(cum) > settings["weekly_cum_gap_warning_bps"]:
        findings.append(("WARNING", "tracking_weekly_gap",
                         f"cumulative holding gap {cum:+.0f} bps this week (warning > "
                         f"{settings['weekly_cum_gap_warning_bps']:.0f})"))
    return {
        "intervals": int(len(gaps)),
        "skipped": int(len(week) - len(gaps)),
        "cum_gap_bps": cum,
        "tracking_error_bps": float(gaps.std(ddof=1)) if len(gaps) > 1 else None,
        "max_abs_gap_bps": float(gaps.abs().max()) if len(gaps) else None,
        "mean_cost_ratio": float(ratios.mean()) if len(ratios) else None,
        "max_weight_deviation": float(devs.max()) if len(devs) else None,
        "findings": [{"severity": s, "check": c, "message": m} for s, c, m in findings],
        "thresholds": {k: settings[k] for k in (*DEFAULT_TRACKING, "modeled_cost_bps")},
    }


def tracking_path(paths) -> Path:
    return Path(paths.root) / "tracking.csv"


def write_tracking(paths=None, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    from src.operations.paper_ops import PaperOpsPaths, write_csv_atomic

    paths = paths or PaperOpsPaths.default()
    frame = compute_tracking(paths, settings)
    write_csv_atomic(tracking_path(paths), frame)
    return frame
