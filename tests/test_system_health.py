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


def test_local_health_report_does_not_overwrite_checkout_report(tmp_path, monkeypatch) -> None:
    checkout_report = tmp_path / "reports" / "system_status.md"
    local_report = tmp_path / "results" / "operations" / "local_system_status.md"
    json_path = tmp_path / "results" / "operations" / "latest_system_status.json"
    checkout_report.parent.mkdir(parents=True)
    checkout_report.write_text("CHECKOUT PLACEHOLDER\n")
    monkeypatch.setattr("src.operations.health.SYSTEM_STATUS_REPORT_PATH", str(checkout_report))
    monkeypatch.setattr("src.operations.health.LOCAL_SYSTEM_STATUS_REPORT_PATH", str(local_report))
    monkeypatch.setattr("src.operations.health.SYSTEM_STATUS_JSON_PATH", str(json_path))
    status = {
        "generated_at": "2026-11-17T00:00:00+00:00",
        "scope": "local",
        "overall_status": "OK",
        "findings": [],
        "latest_portfolio_state": {
            "portfolio_value": 123456.78,
            "cash": 1.0,
            "number_of_positions": 3,
            "gross_exposure": 0.9,
            "net_exposure": 0.9,
            "drawdown": -0.02,
        },
        "latest_execution_status": {"recent_rejections": 0, "recent_failures": 0},
        "data_inputs": {"status": "OK", "as_of": "2026-10-30"},
    }
    written = save_system_status_report(status)
    assert written == local_report
    assert checkout_report.read_text() == "CHECKOUT PLACEHOLDER\n"
    text = local_report.read_text()
    assert "Scope: local" in text
    assert "Portfolio value: 123456.78" in text
    assert json_path.exists()


def test_checkout_save_does_not_write_the_local_report(tmp_path, monkeypatch) -> None:
    checkout_report = tmp_path / "reports" / "system_status.md"
    local_report = tmp_path / "results" / "operations" / "local_system_status.md"
    json_path = tmp_path / "results" / "operations" / "latest_system_status.json"
    monkeypatch.setattr("src.operations.health.SYSTEM_STATUS_REPORT_PATH", str(checkout_report))
    monkeypatch.setattr("src.operations.health.LOCAL_SYSTEM_STATUS_REPORT_PATH", str(local_report))
    monkeypatch.setattr("src.operations.health.SYSTEM_STATUS_JSON_PATH", str(json_path))
    status = {
        "generated_at": "2026-11-17T00:00:00+00:00",
        "scope": "checkout",
        "overall_status": "OK",
        "findings": [],
        "latest_portfolio_state": {},
        "latest_execution_status": {"recent_rejections": 0, "recent_failures": 0},
        "data_inputs": {"status": "OK", "as_of": "2026-10-30"},
    }
    written = save_system_status_report(status)
    assert written == checkout_report
    assert "Scope: checkout" in checkout_report.read_text()
    assert not local_report.exists()


def test_committed_paper_trading_status_is_a_pointer_not_a_nav_snapshot() -> None:
    from pathlib import Path

    text = Path("reports/paper_trading_status.md").read_text()
    assert "results/paper/status.md" in text
    assert "make paper-status" in text
    assert "Portfolio value:" not in text
    assert "113855" not in text


def test_paper_run_status_path_is_local() -> None:
    from src.config import PAPER_OPS_DIR, PAPER_TRADING_STATUS_REPORT_PATH

    assert PAPER_TRADING_STATUS_REPORT_PATH.startswith(f"{PAPER_OPS_DIR}/")
    assert not PAPER_TRADING_STATUS_REPORT_PATH.startswith("reports/")

