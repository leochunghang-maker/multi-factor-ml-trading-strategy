"""Local alerts for the simulated paper run (files and the structured log only).

Nothing here sends email, chat or any other external message. Alerts are
appended to ``results/paper/alerts.jsonl`` and mirrored to the platform event
log (``results/operations/platform_events.jsonl``) when the default paper
directory is used; ``make paper-status`` renders them.

Alert kinds:

- ``data_gate_blocked``        (ERROR)   pre-run data checks failed; no orders.
- ``reconciliation_mismatch``  (ERROR)   start state or fills do not reconcile.
- ``drawdown_halt``            (WARNING) drawdown breached the halt threshold;
                                         the rebalance was skipped (book frozen).
- ``run_blocked``              (WARNING) e.g. out-of-order signal date.
- ``tracking_breach``          (WARNING/ERROR) paper vs frictionless backtest gap
                                         beyond the fixed thresholds (paper_tracking).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

ALERT_KINDS = {
    "data_gate_blocked": "ERROR",
    "reconciliation_mismatch": "ERROR",
    "drawdown_halt": "WARNING",
    "run_blocked": "WARNING",
    "tracking_breach": "WARNING",
}


def alerts_path(paths) -> Path:
    return Path(paths.root) / "alerts.jsonl"


def _mirror_to_event_log(paths) -> bool:
    from src.config import PAPER_OPS_DIR

    return Path(paths.root).resolve() == Path(PAPER_OPS_DIR).resolve()


def raise_alert(
    paths,
    kind: str,
    message: str,
    *,
    signal_date: str | None = None,
    run_id: str | None = None,
    details: dict[str, Any] | None = None,
    severity: str | None = None,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Append one alert (JSONL), mirror it to the event log, print an ``ALERT`` line."""
    if kind not in ALERT_KINDS:
        raise ValueError(f"unknown alert kind {kind!r}")
    alert = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": kind,
        "severity": severity or ALERT_KINDS[kind],
        "signal_date": signal_date,
        "run_id": run_id,
        "message": message,
        "details": details or {},
    }
    path = alerts_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(alert, sort_keys=True, default=str) + "\n")
    if _mirror_to_event_log(paths):
        from src.operations.structured_logging import log_event

        log_event("paper_alert", message, level=alert["severity"], run_id=run_id,
                  alert_kind=kind, signal_date=signal_date)
    if log is not None:
        log(f"ALERT [{alert['severity']}] {kind}: {message}")
    return alert


def read_alerts(paths, limit: int | None = None) -> list[dict[str, Any]]:
    """Alerts oldest-first (unparseable lines are skipped)."""
    path = alerts_path(paths)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:] if limit else rows
