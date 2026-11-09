"""Paper-trading operations: idempotency, snapshots, reconciliation, journal and data gate.

Everything runs in pytest temp dirs with a fake simulated execution, so these
tests never touch results/ and never place any order anywhere (paper only).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.quality import blank_excluded_symbols, run_data_checks
from src.operations.paper_ops import (
    PaperOpsPaths,
    PaperRunBlocked,
    SimulationFiles,
    compare_states,
    enforce_data_gate,
    paper_data_gate,
    paper_health_findings,
    plan_run,
    reconcile_start,
    replay_fills,
    run_paper_day,
)

FILL_COLUMNS = ["symbol", "side", "quantity", "price", "notional", "transaction_cost", "status"]


class FakeSimulation:
    """Mimics run_daily_simulation: buys 10 AAA per call, updates broker state and history."""

    def __init__(self, root: Path, price: float = 50.0, fail: bool = False) -> None:
        self.root = root
        self.price = price
        self.fail = fail
        self.calls = 0
        self.files = SimulationFiles(
            broker_state=root / "broker_state.json",
            orders=root / "latest_orders.csv",
            fills=root / "latest_fills.csv",
            rejected=root / "latest_rejected.csv",
            state_files=[root / "broker_state.json", root / "trade_history.csv", root / "latest_orders.csv",
                         root / "latest_fills.csv", root / "latest_rejected.csv"],
        )

    def state(self) -> dict:
        path = self.files.broker_state
        return json.loads(path.read_text()) if path.exists() else {"cash": 100_000.0, "positions": {}}

    def __call__(self) -> dict:
        self.calls += 1
        state = self.state()
        start_nav = state["cash"] + sum(q * self.price for q in state["positions"].values())
        qty, cost = 10.0, 1.25
        fill = {"symbol": "AAA", "side": "BUY", "quantity": qty, "price": self.price,
                "notional": qty * self.price, "transaction_cost": cost, "status": "FILLED"}
        state["cash"] -= qty * self.price + cost
        state["positions"]["AAA"] = state["positions"].get("AAA", 0.0) + qty
        self.files.broker_state.write_text(json.dumps(state))
        fills = pd.DataFrame([fill], columns=FILL_COLUMNS)
        fills.to_csv(self.files.fills, index=False)
        fills.to_csv(self.files.orders, index=False)
        pd.DataFrame(columns=FILL_COLUMNS).to_csv(self.files.rejected, index=False)
        history = self.root / "trade_history.csv"
        fills.to_csv(history, mode="a", header=not history.exists(), index=False)
        if self.fail:
            raise RuntimeError("simulated crash after the broker state was written")
        nav = state["cash"] + sum(q * self.price for q in state["positions"].values())
        return {"starting_portfolio_value": start_nav, "portfolio_value": nav, "pnl": nav - start_nav,
                "trades_filled": 1, "orders_rejected": 0, "turnover": 0.005, "drawdown": 0.0,
                "drawdown_halted": False, "vol_scale": 1.0, "capacity_clipped_positions": 0,
                "order_adv_clips": 0, "cash": state["cash"]}


@pytest.fixture()
def env(tmp_path):
    sim_root = tmp_path / "simulation"
    sim_root.mkdir()
    return PaperOpsPaths(tmp_path / "paper"), FakeSimulation(sim_root)


def _run(paths, fake, day, **kwargs):
    return run_paper_day(day, fake, paths=paths, sim=fake.files, prices={"AAA": fake.price},
                         price_date=day, data_gate={"status": "OK"}, log=lambda _: None, **kwargs)


def _history_rows(fake) -> int:
    return len(pd.read_csv(fake.root / "trade_history.csv"))


def test_first_run_writes_dated_snapshot_ledger_and_journal(env) -> None:
    paths, fake = env
    out = _run(paths, fake, "2026-11-02")
    assert out["status"] == "COMPLETED" and out["reconciliation_status"] == "NO_PREVIOUS_RUN"
    run_dir = paths.run_dir("2026-11-02")
    for name in ["run.json", "start_state.json", "end_state.json", "positions_start.csv",
                 "positions_end.csv", "orders.csv", "fills.csv", "rejected.csv"]:
        assert (run_dir / name).exists(), name
    assert json.loads((run_dir / "end_state.json").read_text())["positions"] == {"AAA": 10.0}
    assert len(pd.read_csv(paths.ledger)) == 1
    journal = pd.read_csv(paths.journal_csv)
    assert journal.loc[0, "signal_date"] == "2026-11-02"
    assert "2026-11-02" in paths.journal_md.read_text()
    assert not [p for p in paths.runs.iterdir() if p.name.startswith(".")]  # no staging left behind


def test_rerun_same_signal_date_is_skipped_without_trading(env) -> None:
    paths, fake = env
    _run(paths, fake, "2026-11-02")
    out = _run(paths, fake, "2026-11-02")
    assert out["status"] == "SKIPPED"
    assert fake.calls == 1
    assert fake.state()["positions"] == {"AAA": 10.0}
    assert _history_rows(fake) == 1
    assert len(pd.read_csv(paths.ledger)) == 1 and len(pd.read_csv(paths.journal_csv)) == 1


def test_force_replaces_latest_run_from_its_pre_run_state(env) -> None:
    paths, fake = env
    _run(paths, fake, "2026-11-02")
    _run(paths, fake, "2026-11-03")
    out = _run(paths, fake, "2026-11-03", force=True)
    assert out["action"] == "replace" and out["reconciliation_status"] == "OK"
    # Replay of 11-03 starts from the 11-02 end state again: 20 shares, not 30.
    assert fake.state()["positions"] == {"AAA": 20.0}
    assert _history_rows(fake) == 2
    ledger = pd.read_csv(paths.ledger)
    assert list(ledger["signal_date"]) == ["2026-11-02", "2026-11-03"]
    assert bool(ledger.loc[1, "forced"]) is True
    assert paper_health_findings(paths) == []


def test_force_on_older_run_and_out_of_order_dates_are_blocked(env) -> None:
    paths, fake = env
    _run(paths, fake, "2026-11-02")
    _run(paths, fake, "2026-11-03")
    with pytest.raises(PaperRunBlocked, match="most recent run"):
        plan_run(paths, "2026-11-02", force=True)
    with pytest.raises(PaperRunBlocked, match="out of order"):
        _run(paths, fake, "2026-10-30")
    assert fake.calls == 2


def test_failed_execution_restores_state_and_leaves_no_run(env) -> None:
    paths, fake = env
    _run(paths, fake, "2026-11-02")
    before = fake.files.broker_state.read_text()
    fake.fail = True
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(paths, fake, "2026-11-03")
    assert fake.files.broker_state.read_text() == before
    assert _history_rows(fake) == 1
    assert not paths.run_dir("2026-11-03").exists()
    assert not [p for p in paths.runs.iterdir() if p.name.startswith(".")]
    assert list(pd.read_csv(paths.ledger)["signal_date"]) == ["2026-11-02"]
    fake.fail = False
    assert _run(paths, fake, "2026-11-03")["reconciliation_status"] == "OK"


def test_reconciliation_detects_tampered_starting_state(env) -> None:
    paths, fake = env
    _run(paths, fake, "2026-11-02")
    state = fake.state()
    state["cash"] += 500.0
    state["positions"]["BBB"] = 3.0
    fake.files.broker_state.write_text(json.dumps(state))

    with pytest.raises(PaperRunBlocked, match="reconciliation mismatch"):
        _run(paths, fake, "2026-11-03")
    assert fake.calls == 1  # blocked before any simulated order
    findings = paper_health_findings(paths)
    assert any(sev == "ERROR" and check == "paper_reconciliation" for sev, check, _ in findings)
    recon = json.loads(paths.reconciliation.read_text())
    assert any("cash" in issue for issue in recon["issues"]) and any("BBB" in issue for issue in recon["issues"])

    out = _run(paths, fake, "2026-11-03", allow_reconciliation_mismatch=True)
    assert out["reconciliation_status"] == "MISMATCH"
    assert pd.read_csv(paths.journal_csv).iloc[-1]["reconciliation_status"] == "MISMATCH"


def test_reconciliation_detects_fills_that_do_not_explain_previous_end(env) -> None:
    paths, fake = env
    _run(paths, fake, "2026-11-02")
    fills_path = paths.run_dir("2026-11-02") / "fills.csv"
    fills = pd.read_csv(fills_path)
    fills.loc[0, "quantity"] = 7.0
    fills.to_csv(fills_path, index=False)
    recon = reconcile_start(paths, fake.state())
    assert recon["status"] == "MISMATCH"
    assert any("previous start + fills" in issue for issue in recon["issues"])


def test_replay_fills_matches_broker_cash_rules() -> None:
    start = {"cash": 1_000.0, "positions": {"AAA": 5.0}}
    fills = pd.DataFrame([
        {"symbol": "AAA", "side": "SELL", "quantity": 5.0, "notional": 250.0, "transaction_cost": 1.0, "status": "FILLED"},
        {"symbol": "BBB", "side": "BUY", "quantity": 2.0, "notional": 100.0, "transaction_cost": 0.5, "status": "FILLED"},
        {"symbol": "CCC", "side": "BUY", "quantity": 9.0, "notional": 900.0, "transaction_cost": 1.0, "status": "REJECTED"},
    ])
    end = replay_fills(start, fills)
    assert end == {"cash": pytest.approx(1_148.5), "positions": {"BBB": 2.0}}
    assert compare_states(end, end, "x") == []
    assert compare_states(end, {"cash": end["cash"] + 0.02, "positions": end["positions"]}, "x")


def test_health_flags_duplicate_ledger_rows(env) -> None:
    paths, fake = env
    _run(paths, fake, "2026-11-02")
    ledger = pd.read_csv(paths.ledger)
    pd.concat([ledger, ledger]).to_csv(paths.ledger, index=False)
    assert ("ERROR", "paper_ledger") in [(s, c) for s, c, _ in paper_health_findings(paths)]


# ------------------------------------------------------------------- data gate


def _write_data(root: Path, end: str) -> None:
    dates = pd.bdate_range(end=end, periods=40)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (40, 2)), axis=0)),
                          index=dates, columns=["AAA", "BBB"])
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    prices.to_csv(data / "price_data.csv")
    pd.DataFrame(1e6, index=dates, columns=["AAA", "BBB"]).to_csv(data / "volume_data.csv")
    prices[["AAA"]].rename(columns={"AAA": "SPY"}).to_csv(data / "benchmark_prices.csv")


GATE_CONFIG = {"data_checks": {"manifest_files": [], "manifest_path": "data/data_manifest.json"}}


def test_paper_gate_blocks_stale_data_but_research_only_warns(tmp_path) -> None:
    _write_data(tmp_path, "2026-10-30")
    gate = paper_data_gate(as_of="2026-11-09", config=GATE_CONFIG, root=tmp_path)
    assert gate["blocked"] and gate["status"] == "ERROR"
    assert any("trading days before" in e for e in gate["errors"])

    research, _ = run_data_checks(as_of="2026-11-09", config=GATE_CONFIG, root=tmp_path)
    stale = [f for f in research if f.check == "freshness" and "trading days before" in f.message]
    assert stale and all(f.severity == "WARNING" for f in stale)

    paths = PaperOpsPaths(tmp_path / "paper")
    with pytest.raises(PaperRunBlocked, match="Refresh data"):
        enforce_data_gate(paths, gate)
    assert json.loads(paths.gate.read_text())["blocked"] is True
    assert ("WARNING", "paper_data_gate") in [(s, c) for s, c, _ in paper_health_findings(paths)]


def test_paper_gate_passes_fresh_data(tmp_path) -> None:
    _write_data(tmp_path, "2026-11-06")
    gate = paper_data_gate(as_of="2026-11-09", config=GATE_CONFIG, root=tmp_path)
    assert not gate["blocked"], gate["errors"]
    paths = PaperOpsPaths(tmp_path / "paper")
    enforce_data_gate(paths, gate)
    assert paper_health_findings(paths) == []


# ------------------------------------------------------------- excluded symbols


def test_excluded_symbol_is_recorded_as_info_and_blanked_on_download(tmp_path) -> None:
    _write_data(tmp_path, "2026-11-06")
    prices = pd.read_csv(tmp_path / "data" / "price_data.csv", index_col=0)
    prices["BK"] = 10.0
    blanked = blank_excluded_symbols(prices, {"BK": "kept excluded"})
    assert blanked["BK"].isna().all() and list(blanked.columns) == sorted(blanked.columns)
    blanked.to_csv(tmp_path / "data" / "price_data.csv")
    vol = pd.read_csv(tmp_path / "data" / "volume_data.csv", index_col=0)
    vol["BK"] = np.nan
    vol.sort_index(axis=1).to_csv(tmp_path / "data" / "volume_data.csv")
    config = {"data_checks": {**GATE_CONFIG["data_checks"], "excluded_symbols": {"BK": "kept excluded"}}}
    findings, _ = run_data_checks(as_of="2026-11-09", config=config, root=tmp_path)
    bk = [f for f in findings if "BK" in f.message]
    assert bk and all(f.severity == "INFO" for f in bk)
