"""System health treats gitignored paper/sim artifacts as optional (INFO), not ERROR."""

from src.operations.configuration import load_platform_config
from src.operations.health import (
    HealthFinding,
    build_health_status,
    check_concentration,
    check_latest_snapshot,
    check_operational_files,
    check_required_files,
    health_overall_status,
    operational_file_list,
    save_system_status_report,
)


def test_health_overall_status_ignores_info() -> None:
    assert health_overall_status([]) == "OK"
    assert health_overall_status([HealthFinding("INFO", "operational_file", "missing local file")]) == "OK"
    assert health_overall_status([
        HealthFinding("INFO", "operational_file", "missing"),
        HealthFinding("WARNING", "data_freshness", "stale"),
    ]) == "WARNING"
    assert health_overall_status([
        HealthFinding("INFO", "operational_file", "missing"),
        HealthFinding("ERROR", "missing_file", "price"),
    ]) == "ERROR"


def test_missing_required_file_is_an_error() -> None:
    config = load_platform_config()
    findings = check_required_files(config, present=lambda path: False)
    assert findings
    assert all(f.severity == "ERROR" and f.check == "missing_file" for f in findings)
    assert any("data/price_data.csv" in f.message for f in findings)


def test_missing_operational_files_are_info_not_error() -> None:
    config = load_platform_config()
    files = operational_file_list(config)
    assert files, "monitoring.operational_files should list the gitignored paper/sim artifacts"
    assert "results/live/latest_signals.csv" in files
    findings = check_operational_files(config, present=lambda path: False)
    assert len(findings) == len(files)
    assert all(f.severity == "INFO" and f.check == "operational_file" for f in findings)
    messages = " ".join(f.message for f in findings)
    assert "results/live/latest_signals.csv" in messages
    assert "results/simulation/latest_account_snapshot.csv" in messages
    assert "make signals" in messages


def test_present_operational_files_are_silent() -> None:
    config = load_platform_config()
    assert check_operational_files(config, present=lambda path: True) == []


def test_missing_snapshot_and_positions_are_info_not_error() -> None:
    config = load_platform_config()
    snap_findings, snap = check_latest_snapshot(config, available=False)
    assert snap == {}
    assert len(snap_findings) == 1
    assert snap_findings[0].severity == "INFO"
    assert snap_findings[0].check == "portfolio_state"

    conc_findings, positions = check_concentration(config, available=False)
    assert positions == []
    assert len(conc_findings) == 1
    assert conc_findings[0].severity == "INFO"
    assert conc_findings[0].check == "concentration"


def test_checkout_health_on_committed_tree_is_ok() -> None:
    """A fresh clone (no gitignored paper/sim outputs) is not a permanent ERROR."""
    status = build_health_status(checkout_only=True)
    assert status["scope"] == "checkout"
    assert status["overall_status"] == "OK", status["findings"]
    assert status["latest_portfolio_state"] == {}
    findings = status["findings"]
    assert not any(f["severity"] == "ERROR" for f in findings)
    assert not any(f["severity"] == "WARNING" for f in findings)
    info = [f for f in findings if f["severity"] == "INFO"]
    assert any("latest_signals.csv" in f["message"] for f in info)
    assert any(f["check"] == "portfolio_state" for f in info)


def test_checkout_health_ignores_local_operational_artifacts() -> None:
    local = build_health_status(checkout_only=False)
    checkout = build_health_status(checkout_only=True)
    assert checkout["scope"] == "checkout"
    assert checkout["latest_portfolio_state"] == {}
    # Local may have simulation files on this machine; checkout must not surface them.
    assert checkout["latest_risk_state"]["top_positions"] == []
    if local["latest_portfolio_state"]:
        assert local["scope"] == "local"
        assert local["latest_portfolio_state"] != checkout["latest_portfolio_state"]


def test_checkout_report_does_not_invent_a_zero_portfolio(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    report = tmp_path / "reports" / "system_status.md"
    json_path = tmp_path / "results" / "operations" / "latest_system_status.json"
    monkeypatch.setattr("src.operations.health.SYSTEM_STATUS_REPORT_PATH", str(report))
    monkeypatch.setattr("src.operations.health.SYSTEM_STATUS_JSON_PATH", str(json_path))
    status = {
        "generated_at": "2026-11-14T00:00:00+00:00",
        "scope": "checkout",
        "overall_status": "OK",
        "findings": [
            {
                "severity": "INFO",
                "check": "operational_file",
                "message": "Local operational file not in this checkout: results/live/latest_signals.csv.",
            }
        ],
        "latest_portfolio_state": {},
        "latest_execution_status": {"recent_rejections": 0, "recent_failures": 0},
        "data_inputs": {"status": "OK", "as_of": "2026-10-30"},
    }
    save_system_status_report(status)
    text = report.read_text()
    assert "Overall status: OK" in text
    assert "Scope: checkout" in text
    assert "No local simulation snapshot in this checkout" in text
    assert "Portfolio value: 0.00" not in text
    assert "**INFO** `operational_file`" in text
