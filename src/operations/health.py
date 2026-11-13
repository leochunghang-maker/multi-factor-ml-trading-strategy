import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import (
    ALLOCATION_SUMMARY_PATH,
    MULTI_DAY_NAV_PATH,
    PAPER_TRADING_STATUS_REPORT_PATH,
    REALIZED_VS_TARGET_WEIGHTS_PATH,
    SIMULATED_EXECUTION_LOG_PATH,
    SIMULATED_LATEST_SNAPSHOT_PATH,
    SIMULATED_POSITION_SNAPSHOT_PATH,
    STRUCTURED_LOG_PATH,
    SYSTEM_STATUS_JSON_PATH,
    SYSTEM_STATUS_REPORT_PATH,
)
from src.data.required import producer_command
from src.operations.configuration import load_platform_config
from src.operations.structured_logging import read_recent_events

# Fallback when producer_command does not know the artifact.
_OPERATIONAL_HINTS = {
    "results/live/latest_signals.csv": "make signals",
    "results/simulation/latest_account_snapshot.csv": "make paper",
    "results/simulation/execution_log.csv": "make paper",
    "results/reports/summary_report.csv": "make report",
}


@dataclass
class HealthFinding:
    severity: str
    check: str
    message: str


def read_csv(path: str, date_columns: list[str] | None = None) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(file_path)
    for column in date_columns or []:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def operational_file_list(config: dict[str, Any]) -> list[str]:
    return list(config["monitoring"]["operational_files"])


def health_overall_status(findings: list[HealthFinding]) -> str:
    """ERROR/WARNING only; INFO findings (e.g. no local paper run) stay OK."""
    if any(f.severity == "ERROR" for f in findings):
        return "ERROR"
    if any(f.severity == "WARNING" for f in findings):
        return "WARNING"
    return "OK"


def check_required_files(
    config: dict[str, Any],
    *,
    present: Callable[[str], bool] | None = None,
) -> list[HealthFinding]:
    exists = present or (lambda path: Path(path).exists())
    findings = []
    for file_name in config["monitoring"]["required_files"]:
        if not exists(file_name):
            findings.append(HealthFinding("ERROR", "missing_file", f"Missing required file: {file_name}"))
    return findings


def check_operational_files(
    config: dict[str, Any],
    *,
    present: Callable[[str], bool] | None = None,
) -> list[HealthFinding]:
    """Gitignored paper/sim/report artifacts: missing is INFO, not ERROR."""
    exists = present or (lambda path: Path(path).exists())
    findings = []
    for file_name in operational_file_list(config):
        if exists(file_name):
            continue
        command = producer_command(file_name) or _OPERATIONAL_HINTS.get(file_name)
        how = f"; create it with `{command}`" if command else " (gitignored local artifact)"
        findings.append(
            HealthFinding(
                "INFO",
                "operational_file",
                f"Local operational file not in this checkout: {file_name}{how}.",
            )
        )
    return findings


def check_data_inputs(config: dict[str, Any], as_of=None) -> tuple[list[HealthFinding], dict[str, Any]]:
    """Freshness (trading days), alignment, quality and manifest checks on the data inputs.

    Thresholds: ``data_checks`` in the platform config (see ``src/data/quality.py``).
    INFO findings stay in the data-quality report; ERROR/WARNING surface here.
    """
    from src.data.quality import overall_status, run_data_checks

    try:
        data_findings, summary = run_data_checks(as_of=as_of, config=config)
    except Exception as exc:  # pragma: no cover - defensive: health must still render
        return [HealthFinding("ERROR", "data_checks", f"Data checks failed to run: {exc}")], {}
    findings = [
        HealthFinding(f.severity, f"data_{f.check}", f"{f.file}: {f.message}")
        for f in data_findings
        if f.severity in ("ERROR", "WARNING")
    ]
    summary = {**summary, "status": overall_status(data_findings)}
    return findings, summary


def check_paper_operations(paths=None) -> list[HealthFinding]:
    """Paper-run reconciliation (ERROR on mismatch), last data-gate block, ledger duplicates."""
    from src.operations.paper_ops import paper_health_findings

    try:
        return [HealthFinding(sev, check, msg) for sev, check, msg in paper_health_findings(paths)]
    except Exception as exc:  # pragma: no cover - defensive: health must still render
        return [HealthFinding("ERROR", "paper_operations", f"Paper-run checks failed to run: {exc}")]


def check_latest_snapshot(
    config: dict[str, Any],
    *,
    available: bool = True,
) -> tuple[list[HealthFinding], dict[str, Any]]:
    findings = []
    snapshot = read_csv(SIMULATED_LATEST_SNAPSHOT_PATH) if available else pd.DataFrame()
    if snapshot.empty:
        return [
            HealthFinding(
                "INFO",
                "portfolio_state",
                "No local simulation snapshot in this checkout "
                "(results/simulation/ is gitignored; run `make paper` or the multi-day simulation).",
            )
        ], {}
    latest = snapshot.iloc[-1].to_dict()
    gross = float(latest.get("gross_exposure", 0.0))
    turnover = float(latest.get("turnover", 0.0))
    drawdown = float(latest.get("drawdown", 0.0))
    max_gross = config["risk_limits"]["max_gross_exposure"]
    max_turnover = config["risk_limits"]["max_turnover"]
    max_drawdown = config["risk_limits"]["max_drawdown_warning"]

    if gross > max_gross + 1e-9:
        findings.append(HealthFinding("WARNING", "exposure", f"Gross exposure {gross:.2%} exceeds {max_gross:.2%}."))
    if turnover > max_turnover + 1e-9:
        findings.append(HealthFinding("WARNING", "turnover", f"Turnover {turnover:.2%} exceeds {max_turnover:.2%}."))
    if drawdown < max_drawdown:
        findings.append(HealthFinding("WARNING", "drawdown", f"Drawdown {drawdown:.2%} exceeds warning threshold {max_drawdown:.2%} (paper/sim skips rebalance / freezes the book while halted)."))
    return findings, latest


def check_concentration(
    config: dict[str, Any],
    *,
    available: bool = True,
) -> tuple[list[HealthFinding], list[dict[str, Any]]]:
    findings = []
    positions = read_csv(SIMULATED_POSITION_SNAPSHOT_PATH) if available else pd.DataFrame()
    if positions.empty:
        return [
            HealthFinding(
                "INFO",
                "concentration",
                "No local position snapshot in this checkout "
                "(results/simulation/ is gitignored; run `make paper` or the multi-day simulation).",
            )
        ], []
    max_weight = config["risk_limits"]["max_position_weight"]
    top = positions.sort_values("weight", key=lambda series: series.abs(), ascending=False).head(5)
    breaches = top[top["weight"].abs() > max_weight + 0.005]
    if not breaches.empty:
        names = ", ".join(breaches["symbol"].astype(str).tolist())
        findings.append(HealthFinding("WARNING", "concentration", f"Position concentration warning for: {names}."))
    return findings, top.to_dict("records")


def latest_execution_status(*, available: bool = True) -> dict[str, Any]:
    execution = read_csv(SIMULATED_EXECUTION_LOG_PATH, ["timestamp"]) if available else pd.DataFrame()
    if execution.empty:
        return {"latest_event": "unavailable", "recent_rejections": 0}
    recent = execution.tail(50)
    return {
        "latest_event": execution.iloc[-1].to_dict(),
        "recent_rejections": int((recent.get("status", pd.Series(dtype=str)) == "REJECTED").sum()),
        "recent_failures": int((recent.get("event", pd.Series(dtype=str)) == "RUN_FAILED").sum()),
    }


def check_execution_status(*, available: bool = True) -> tuple[list[HealthFinding], dict[str, Any]]:
    status = latest_execution_status(available=available)
    findings = []
    if status.get("recent_failures", 0) > 0:
        findings.append(
            HealthFinding(
                "WARNING",
                "execution_status",
                f"Recent execution log contains {status['recent_failures']} failed run event(s).",
            )
        )
    if status.get("recent_rejections", 0) > 0:
        findings.append(
            HealthFinding(
                "WARNING",
                "execution_status",
                f"Recent execution log contains {status['recent_rejections']} rejected order event(s).",
            )
        )
    return findings, status


def load_optional_operational_tables(*, available: bool = True) -> dict[str, Any]:
    if not available:
        return {"latest_allocation": {}, "latest_realized_gap_count": 0, "latest_multi_day_nav": {}}
    allocation = read_csv(ALLOCATION_SUMMARY_PATH)
    realized = read_csv(REALIZED_VS_TARGET_WEIGHTS_PATH)
    multi_day = read_csv(MULTI_DAY_NAV_PATH, ["date"])
    return {
        "latest_allocation": allocation.iloc[-1].to_dict() if not allocation.empty else {},
        "latest_realized_gap_count": int((realized["absolute_weight_gap"] > 0.01).sum()) if "absolute_weight_gap" in realized.columns else 0,
        "latest_multi_day_nav": multi_day.iloc[-1].to_dict() if not multi_day.empty else {},
    }


def build_health_status(*, checkout_only: bool = False) -> dict[str, Any]:
    # Operational safety matters because quant platforms fail through mundane
    # problems as often as model problems: missing files, stale data, broken
    # execution logs, and drifted risk. Health checks make those visible.
    # checkout_only scores committed artifacts only, so a fresh clone and the
    # committed reports/system_status.md are not a permanent ERROR for
    # gitignored paper/sim outputs that this machine happens to have.
    config = load_platform_config()
    operational = set(operational_file_list(config))
    if checkout_only:
        present = lambda path: False if path in operational else Path(path).exists()
        local_ops = False
    else:
        present = lambda path: Path(path).exists()
        local_ops = True

    findings = []
    findings.extend(check_required_files(config, present=present))
    findings.extend(check_operational_files(config, present=present))
    data_findings, data_summary = check_data_inputs(config)
    findings.extend(data_findings)
    if local_ops:
        findings.extend(check_paper_operations())
    snapshot_findings, latest_snapshot = check_latest_snapshot(config, available=local_ops)
    findings.extend(snapshot_findings)
    concentration_findings, top_positions = check_concentration(config, available=local_ops)
    findings.extend(concentration_findings)
    execution_findings, execution_status = check_execution_status(available=local_ops)
    findings.extend(execution_findings)
    events = read_recent_events(STRUCTURED_LOG_PATH, limit=20) if local_ops else []
    status = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "scope": "checkout" if checkout_only else "local",
        "overall_status": health_overall_status(findings),
        "findings": [finding.__dict__ for finding in findings],
        "latest_portfolio_state": latest_snapshot,
        "latest_risk_state": {
            "top_positions": top_positions,
            **load_optional_operational_tables(available=local_ops),
        },
        "latest_execution_status": execution_status,
        "data_inputs": data_summary,
        "recent_structured_events": events,
    }
    return status


def save_system_status_report(status: dict[str, Any]) -> None:
    Path(SYSTEM_STATUS_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(SYSTEM_STATUS_JSON_PATH).write_text(json.dumps(status, indent=2, sort_keys=True, default=str))

    scope = status.get("scope", "local")
    if scope == "checkout":
        scope_line = (
            "- Scope: checkout (committed artifacts only; local paper/sim outputs are gitignored)"
        )
    else:
        scope_line = "- Scope: local (includes gitignored paper/sim outputs when present)"

    lines = [
        "# System Status",
        "",
        f"- Generated at: {status['generated_at']}",
        f"- Overall status: {status['overall_status']}",
        scope_line,
        "",
        "## Findings",
        "",
    ]
    if status["findings"]:
        lines.extend([
            f"- **{item['severity']}** `{item['check']}`: {item['message']}"
            for item in status["findings"]
        ])
    else:
        lines.append("- None.")

    portfolio = status.get("latest_portfolio_state", {})
    lines.extend([
        "",
        "## Latest Portfolio State",
        "",
    ])
    if portfolio:
        lines.extend([
            f"- Portfolio value: {float(portfolio.get('portfolio_value', 0.0)):.2f}",
            f"- Cash: {float(portfolio.get('cash', 0.0)):.2f}",
            f"- Positions: {int(portfolio.get('number_of_positions', 0)) if portfolio else 0}",
            f"- Gross exposure: {float(portfolio.get('gross_exposure', 0.0)):.2%}",
            f"- Net exposure: {float(portfolio.get('net_exposure', 0.0)):.2%}",
            f"- Drawdown: {float(portfolio.get('drawdown', 0.0)):.2%}",
            "- Note: this snapshot is local-only (`results/simulation/` is gitignored).",
        ])
    else:
        lines.append(
            "- No local simulation snapshot in this checkout (`results/simulation/` is gitignored)."
        )

    lines.extend([
        "",
        "## Data Inputs",
        "",
        f"- Data checks: {status.get('data_inputs', {}).get('status', 'unavailable')} "
        f"(as of {status.get('data_inputs', {}).get('as_of', 'n/a')}; details: `python src/check_data.py`)",
        "",
        "## Latest Execution Status",
        "",
        f"- Recent rejections: {status['latest_execution_status'].get('recent_rejections', 0)}",
        f"- Recent failures: {status['latest_execution_status'].get('recent_failures', 0)}",
        "",
        "## Notes",
        "",
        f"- Paper trading status: `{PAPER_TRADING_STATUS_REPORT_PATH}`",
        f"- Structured log: `{STRUCTURED_LOG_PATH}`",
    ])
    Path(SYSTEM_STATUS_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(SYSTEM_STATUS_REPORT_PATH).write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write reports/system_status.md and results/operations/latest_system_status.json. "
            "Without --checkout, local paper/sim artifacts are scored when present."
        )
    )
    parser.add_argument(
        "--checkout",
        action="store_true",
        help=(
            "Score only committed artifacts, treating gitignored paper/sim outputs as absent. "
            "Use this when writing reports/system_status.md for the repo."
        ),
    )
    args = parser.parse_args(argv)
    status = build_health_status(checkout_only=args.checkout)
    save_system_status_report(status)
    print(f"System status: {status['overall_status']} ({status['scope']})")
    print(f"Report saved to {SYSTEM_STATUS_REPORT_PATH}")


if __name__ == "__main__":
    main()
