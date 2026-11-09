"""One-page status of the simulated paper account (``make paper-status``).

Written to ``results/paper/status.md`` and ``status.json`` after every run,
gate decision and alert, and on demand. Local files only.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.operations.paper_alerts import read_alerts

SEVERITY_RANK = {"OK": 0, "INFO": 0, "WARNING": 1, "ERROR": 2}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def build_paper_status(paths=None, recent_alerts: int = 10, journal_rows: int = 5) -> dict[str, Any]:
    from src.operations.paper_ops import (
        JOURNAL_COLUMNS,
        PaperOpsPaths,
        completed_runs,
        paper_health_findings,
        read_csv_safe,
    )

    paths = paths or PaperOpsPaths.default()
    active = [{"severity": s, "check": c, "message": m} for s, c, m in paper_health_findings(paths)]
    journal = read_csv_safe(paths.journal_csv, JOURNAL_COLUMNS)
    latest = None
    if not journal.empty:
        row = journal.sort_values("signal_date").iloc[-1].to_dict()
        latest = {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
        if str(latest.get("drawdown_halted")).lower() in ("true", "1"):
            active.append({
                "severity": "WARNING", "check": "drawdown_halt",
                "message": f"Latest run {latest['signal_date']} was halted at drawdown "
                           f"{float(latest.get('drawdown') or 0.0):.2%} (rebalance skipped).",
            })
    overall = "OK"
    for item in active:
        if SEVERITY_RANK[item["severity"]] > SEVERITY_RANK[overall]:
            overall = item["severity"]
    runs = completed_runs(paths)
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "overall_status": overall,
        "paper_dir": str(paths.root),
        "completed_runs": len(runs),
        "first_run": runs[0] if runs else None,
        "latest_run": latest,
        "last_gate": _read_json(paths.gate),
        "last_reconciliation": _read_json(paths.reconciliation),
        "active_alerts": active,
        "recent_alerts": read_alerts(paths, recent_alerts)[::-1],
        "journal_tail": journal.sort_values("signal_date").tail(journal_rows).to_dict("records")
        if not journal.empty else [],
    }


def render_paper_status(status: dict[str, Any]) -> str:
    lines = [
        "# Paper Account Status (simulated)",
        "",
        f"- Generated at: {status['generated_at']}",
        f"- Overall status: **{status['overall_status']}**",
        f"- Paper state: `{status['paper_dir']}/` · completed runs: {status['completed_runs']}"
        + (f" (since {status['first_run']})" if status["first_run"] else ""),
    ]
    gate = status.get("last_gate")
    if gate:
        files = gate.get("files") or {}
        last_dates = ", ".join(
            f"{name} {info['last_date']}" for name, info in files.items() if isinstance(info, dict)
        )
        lines.append(
            f"- Last data gate: {'BLOCKED' if gate.get('blocked') else 'passed'} ({gate.get('status')}) as of "
            f"{gate.get('as_of')} on `{gate.get('data_dir', '?')}/`" + (f"; last dates: {last_dates}" if last_dates else "")
        )
    else:
        lines.append("- Last data gate: none recorded")
    latest = status.get("latest_run")
    if latest:
        lines.append(
            f"- Latest run: {latest['signal_date']} · NAV {float(latest['nav_end']):,.2f} · PnL vs previous "
            f"{float(latest.get('pnl_since_previous') or 0.0):,.2f} · trades {int(latest.get('trades_filled') or 0)} · "
            f"drawdown {float(latest.get('drawdown') or 0.0):.2%} · reconciliation {latest.get('reconciliation_status')}"
        )
    else:
        lines.append("- Latest run: none")
    lines += ["", "## Active alerts", ""]
    if status["active_alerts"]:
        lines += [f"- **{a['severity']}** `{a['check']}`: {a['message']}" for a in status["active_alerts"]]
    else:
        lines.append("- None.")
    lines += ["", "## Recent alerts (newest first, from alerts.jsonl)", ""]
    if status["recent_alerts"]:
        lines += [
            f"- {a['timestamp']} **{a['severity']}** `{a['kind']}`"
            + (f" [{a['signal_date']}]" if a.get("signal_date") else "")
            + f": {a['message']}"
            for a in status["recent_alerts"]
        ]
    else:
        lines.append("- None.")
    lines += ["", "## Journal (latest rows)", ""]
    if status["journal_tail"]:
        lines += ["| Signal date | NAV | PnL | Trades | Drawdown | Halt | Data | Recon |",
                  "|---|---:|---:|---:|---:|---|---|---|"]
        for r in status["journal_tail"]:
            halted = str(r.get("drawdown_halted")).lower() in ("true", "1")
            lines.append(
                f"| {r['signal_date']} | {float(r['nav_end']):,.2f} | {float(r.get('pnl_since_previous') or 0.0):,.2f} | "
                f"{int(r.get('trades_filled') or 0)} | {float(r.get('drawdown') or 0.0):.2%} | "
                f"{'yes' if halted else 'no'} | {r.get('data_check_status')} | {r.get('reconciliation_status')} |"
            )
    else:
        lines.append("- No runs yet.")
    lines += ["", "Simulated paper trading only; alerts are local files and log lines, not messages.", ""]
    return "\n".join(lines)


def write_paper_status(paths=None) -> dict[str, Any]:
    from src.operations.paper_ops import PaperOpsPaths, write_json_atomic, write_text_atomic

    paths = paths or PaperOpsPaths.default()
    status = build_paper_status(paths)
    write_json_atomic(paths.status_json, status)
    write_text_atomic(paths.status_md, render_paper_status(status))
    return status
