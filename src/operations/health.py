import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import (
    ALLOCATION_SUMMARY_PATH,
    MULTI_DAY_NAV_PATH,
    PAPER_TRADING_STATUS_REPORT_PATH,
    PRICE_DATA_PATH,
    REALIZED_VS_TARGET_WEIGHTS_PATH,
    SIMULATED_EXECUTION_LOG_PATH,
    SIMULATED_LATEST_SNAPSHOT_PATH,
    SIMULATED_POSITION_SNAPSHOT_PATH,
    STRUCTURED_LOG_PATH,
    SYSTEM_STATUS_JSON_PATH,
    SYSTEM_STATUS_REPORT_PATH,
)
from src.operations.configuration import load_platform_config
from src.operations.structured_logging import read_recent_events


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


def check_required_files(config: dict[str, Any]) -> list[HealthFinding]:
    findings = []
    for file_name in config["monitoring"]["required_files"]:
        if not Path(file_name).exists():
            findings.append(HealthFinding("ERROR", "missing_file", f"Missing required file: {file_name}"))
    return findings


def check_stale_data(config: dict[str, Any]) -> list[HealthFinding]:
    findings = []
    max_days = config["risk_limits"]["max_data_staleness_days"]
    prices = read_csv(PRICE_DATA_PATH, ["Date"])
    date_column = "Date" if "Date" in prices.columns else prices.columns[0] if not prices.empty else None
    if prices.empty or date_column is None:
        return [HealthFinding("ERROR", "stale_data", "Price data is missing or unreadable.")]
    latest = pd.to_datetime(prices[date_column], errors="coerce").max()
    age_days = (pd.Timestamp.now(tz="UTC").normalize() - latest.tz_localize("UTC").normalize()).days
    if age_days > max_days:
        findings.append(
            HealthFinding(
                "WARNING",
                "stale_data",
                f"Latest price date {latest.date()} is {age_days} days old; limit is {max_days}.",
            )
        )
    return findings


def check_latest_snapshot(config: dict[str, Any]) -> tuple[list[HealthFinding], dict[str, Any]]:
    findings = []
    snapshot = read_csv(SIMULATED_LATEST_SNAPSHOT_PATH)
    if snapshot.empty:
        return [HealthFinding("ERROR", "portfolio_state", "Latest account snapshot is missing.")], {}
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
        findings.append(HealthFinding("WARNING", "drawdown", f"Drawdown {drawdown:.2%} exceeds warning threshold {max_drawdown:.2%}."))
    return findings, latest


def check_concentration(config: dict[str, Any]) -> tuple[list[HealthFinding], list[dict[str, Any]]]:
    findings = []
    positions = read_csv(SIMULATED_POSITION_SNAPSHOT_PATH)
    if positions.empty:
        return [HealthFinding("WARNING", "concentration", "No latest position snapshot available.")], []
    max_weight = config["risk_limits"]["max_position_weight"]
    top = positions.sort_values("weight", key=lambda series: series.abs(), ascending=False).head(5)
    breaches = top[top["weight"].abs() > max_weight + 0.005]
    if not breaches.empty:
        names = ", ".join(breaches["symbol"].astype(str).tolist())
        findings.append(HealthFinding("WARNING", "concentration", f"Position concentration warning for: {names}."))
    return findings, top.to_dict("records")


def latest_execution_status() -> dict[str, Any]:
    execution = read_csv(SIMULATED_EXECUTION_LOG_PATH, ["timestamp"])
    if execution.empty:
        return {"latest_event": "unavailable", "recent_rejections": 0}
    recent = execution.tail(50)
    return {
        "latest_event": execution.iloc[-1].to_dict(),
        "recent_rejections": int((recent.get("status", pd.Series(dtype=str)) == "REJECTED").sum()),
        "recent_failures": int((recent.get("event", pd.Series(dtype=str)) == "RUN_FAILED").sum()),
    }


def check_execution_status() -> tuple[list[HealthFinding], dict[str, Any]]:
    status = latest_execution_status()
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


def load_optional_operational_tables() -> dict[str, Any]:
    allocation = read_csv(ALLOCATION_SUMMARY_PATH)
    realized = read_csv(REALIZED_VS_TARGET_WEIGHTS_PATH)
    multi_day = read_csv(MULTI_DAY_NAV_PATH, ["date"])
    return {
        "latest_allocation": allocation.iloc[-1].to_dict() if not allocation.empty else {},
        "latest_realized_gap_count": int((realized["absolute_weight_gap"] > 0.01).sum()) if "absolute_weight_gap" in realized.columns else 0,
        "latest_multi_day_nav": multi_day.iloc[-1].to_dict() if not multi_day.empty else {},
    }


def build_health_status() -> dict[str, Any]:
    # Operational safety matters because quant platforms fail through mundane
    # problems as often as model problems: missing files, stale data, broken
    # execution logs, and drifted risk. Health checks make those visible.
    config = load_platform_config()
    findings = []
    findings.extend(check_required_files(config))
    findings.extend(check_stale_data(config))
    snapshot_findings, latest_snapshot = check_latest_snapshot(config)
    findings.extend(snapshot_findings)
    concentration_findings, top_positions = check_concentration(config)
    findings.extend(concentration_findings)
    execution_findings, execution_status = check_execution_status()
    findings.extend(execution_findings)
    events = read_recent_events(STRUCTURED_LOG_PATH, limit=20)
    status = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "overall_status": "ERROR" if any(f.severity == "ERROR" for f in findings) else "WARNING" if findings else "OK",
        "findings": [finding.__dict__ for finding in findings],
        "latest_portfolio_state": latest_snapshot,
        "latest_risk_state": {
            "top_positions": top_positions,
            **load_optional_operational_tables(),
        },
        "latest_execution_status": execution_status,
        "recent_structured_events": events,
    }
    return status


def save_system_status_report(status: dict[str, Any]) -> None:
    Path(SYSTEM_STATUS_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(SYSTEM_STATUS_JSON_PATH).write_text(json.dumps(status, indent=2, sort_keys=True, default=str))

    lines = [
        "# System Status",
        "",
        f"- Generated at: {status['generated_at']}",
        f"- Overall status: {status['overall_status']}",
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
        f"- Portfolio value: {float(portfolio.get('portfolio_value', 0.0)):.2f}",
        f"- Cash: {float(portfolio.get('cash', 0.0)):.2f}",
        f"- Positions: {int(portfolio.get('number_of_positions', 0)) if portfolio else 0}",
        f"- Gross exposure: {float(portfolio.get('gross_exposure', 0.0)):.2%}",
        f"- Net exposure: {float(portfolio.get('net_exposure', 0.0)):.2%}",
        f"- Drawdown: {float(portfolio.get('drawdown', 0.0)):.2%}",
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


def main() -> None:
    status = build_health_status()
    save_system_status_report(status)
    print(f"System status: {status['overall_status']}")
    print(f"Report saved to {SYSTEM_STATUS_REPORT_PATH}")


if __name__ == "__main__":
    main()
