"""Weekly review of the simulated paper account, built from local files only.

Sources (all under ``results/paper/``): ``journal.csv`` (one row per signal
date), ``alerts.jsonl`` and ``gate_history.csv``. Output:
``results/paper/reviews/weekly-<ISO year>-W<week>.md`` and ``.json``.

Weeks are ISO weeks (Monday-Sunday). Runs are assigned by signal date; alerts
and gate checks by the UTC date of their timestamp. PnL for the week is the
change from the last NAV before the week (or the first run's starting NAV) to
the last NAV in the week, so it includes costs.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.operations.paper_alerts import read_alerts


def week_bounds(day=None) -> tuple[date, date]:
    d = pd.Timestamp(day or datetime.now(UTC).date()).date()
    start = d - timedelta(days=d.weekday())
    return start, start + timedelta(days=6)


def _utc_date(value) -> date:
    ts = pd.Timestamp(value)
    return (ts.tz_convert("UTC") if ts.tzinfo else ts).date()


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def build_weekly_review(paths=None, week_of=None) -> dict[str, Any]:
    from src.operations.paper_ops import (
        GATE_HISTORY_COLUMNS,
        JOURNAL_COLUMNS,
        PaperOpsPaths,
        read_csv_safe,
    )

    paths = paths or PaperOpsPaths.default()
    start, end = week_bounds(week_of)
    iso = start.isocalendar()
    journal = read_csv_safe(paths.journal_csv, JOURNAL_COLUMNS)
    if not journal.empty:
        journal = journal.assign(_d=pd.to_datetime(journal["signal_date"]).dt.date).sort_values("_d")
        week = journal[(journal["_d"] >= start) & (journal["_d"] <= end)]
        before = journal[journal["_d"] < start]
    else:
        week = before = journal

    runs = int(len(week))
    nav_end = float(week["nav_end"].iloc[-1]) if runs else None
    if not before.empty:
        nav_start = float(before["nav_end"].iloc[-1])
    elif runs:
        first = week.iloc[0]
        nav_start = float(first["nav_end"]) - float(first.get("pnl_since_previous") or 0.0)
    else:
        nav_start = None
    pnl = (nav_end - nav_start) if runs and nav_start is not None else None
    ret = (pnl / nav_start) if pnl is not None and nav_start else None

    alerts = [
        a for a in read_alerts(paths)
        if start <= _utc_date(a.get("timestamp")) <= end
    ]
    by_kind: dict[str, int] = {}
    for a in alerts:
        by_kind[a["kind"]] = by_kind.get(a["kind"], 0) + 1

    gates = read_csv_safe(paths.gate_history, GATE_HISTORY_COLUMNS)
    if not gates.empty:
        gdates = pd.to_datetime(gates["checked_at"], utc=True).dt.date
        gates = gates[(gdates >= start) & (gdates <= end)]
    gate_checks = int(len(gates))
    gate_blocked = int(gates["blocked"].map(_truthy).sum()) if gate_checks else 0

    def col_sum(name: str) -> float:
        return float(pd.to_numeric(week[name], errors="coerce").fillna(0).sum()) if runs else 0.0

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "week": f"{iso.year}-W{iso.week:02d}",
        "start": str(start),
        "end": str(end),
        "runs": runs,
        "signal_dates": [str(d) for d in week["_d"]] if runs else [],
        "nav_start": nav_start,
        "nav_end": nav_end,
        "pnl": pnl,
        "return": ret,
        "trades_filled": int(col_sum("trades_filled")),
        "orders_rejected": int(col_sum("orders_rejected")),
        "turnover_total": col_sum("turnover"),
        "turnover_avg": col_sum("turnover") / runs if runs else 0.0,
        "max_drawdown": float(pd.to_numeric(week["drawdown"], errors="coerce").min()) if runs else None,
        "drawdown_halts": int(week["drawdown_halted"].map(_truthy).sum()) if runs else 0,
        "reconciliation_mismatches": int((week["reconciliation_status"] == "MISMATCH").sum()) if runs else 0,
        "adv_clips": int(col_sum("capacity_clipped_positions") + col_sum("order_adv_clips")),
        "forced_runs": int(week["forced"].map(_truthy).sum()) if runs else 0,
        "alerts_total": len(alerts),
        "alerts_by_kind": dict(sorted(by_kind.items())),
        "gate_checks": gate_checks,
        "gate_blocked": gate_blocked,
        "gate_block_rate": (gate_blocked / gate_checks) if gate_checks else None,
        "alerts": alerts[-20:],
    }


def _fmt(value, kind: str = "num") -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value:.2%}"
    return f"{value:,.2f}"


def render_weekly_review(r: dict[str, Any]) -> str:
    rate = "n/a" if r["gate_block_rate"] is None else f"{r['gate_block_rate']:.0%}"
    lines = [
        f"# Paper Account Weekly Review {r['week']} (simulated)",
        "",
        f"Week {r['start']} → {r['end']} · generated {r['generated_at']} from `results/paper/` "
        "(journal.csv, alerts.jsonl, gate_history.csv).",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Completed runs | {r['runs']} |",
        f"| NAV start → end | {_fmt(r['nav_start'])} → {_fmt(r['nav_end'])} |",
        f"| PnL (incl. costs) | {_fmt(r['pnl'])} |",
        f"| Return | {_fmt(r['return'], 'pct')} |",
        f"| Trades filled / rejected | {r['trades_filled']} / {r['orders_rejected']} |",
        f"| Turnover total / avg per run | {r['turnover_total']:.2%} / {r['turnover_avg']:.2%} |",
        f"| Worst drawdown | {_fmt(r['max_drawdown'], 'pct')} |",
        f"| Drawdown-halted runs | {r['drawdown_halts']} |",
        f"| Reconciliation mismatches | {r['reconciliation_mismatches']} |",
        f"| ADV clips (weights + orders) | {r['adv_clips']} |",
        f"| Forced reruns | {r['forced_runs']} |",
        f"| Data gate checks / blocked | {r['gate_checks']} / {r['gate_blocked']} ({rate}) |",
        f"| Alerts | {r['alerts_total']} |",
        "",
        "## Alerts by kind",
        "",
    ]
    lines += [f"- `{k}`: {v}" for k, v in r["alerts_by_kind"].items()] or ["- None."]
    lines += ["", "## Runs", ""]
    lines.append(", ".join(r["signal_dates"]) if r["signal_dates"] else "No completed runs this week.")
    if r["alerts"]:
        lines += ["", "## Alert log (latest 20)", ""]
        lines += [f"- {a['timestamp']} **{a['severity']}** `{a['kind']}`: {a['message']}" for a in r["alerts"]]
    lines += ["", "Simulated paper trading only.", ""]
    return "\n".join(lines)


def write_weekly_review(paths=None, week_of=None) -> tuple[Path, dict[str, Any]]:
    from src.operations.paper_ops import PaperOpsPaths, write_json_atomic, write_text_atomic

    paths = paths or PaperOpsPaths.default()
    review = build_weekly_review(paths, week_of)
    base = paths.reviews / f"weekly-{review['week']}"
    write_json_atomic(base.with_suffix(".json"), review)
    write_text_atomic(base.with_suffix(".md"), render_weekly_review(review))
    return base.with_suffix(".md"), review
