"""Daily paper-run operations: idempotency, dated snapshots, reconciliation, journal, data gate.

Simulated paper trading only; nothing here talks to a broker.

Layout (``results/paper/``, gitignored operator-local state)::

    runs/<signal_date>/          one folder per completed run (written atomically)
        run.json                 run id, dates, status, data gate, reconciliation, forced
        start_state.json         simulated broker cash + positions before the run
        end_state.json           ... after the run
        positions_start.csv      symbol, quantity, price, market_value
        positions_end.csv
        orders.csv fills.csv rejected.csv
        pre_run_state/           copies of results/simulation state before the run
                                 (used to replace the run with --force and for audit)
    ledger.csv                   one row per signal date (upserted, never duplicated)
    journal.csv / journal.md     concise daily journal (upserted / re-rendered)
    latest_reconciliation.json   pre-run reconciliation result (read by health)
    last_gate.json               last pre-run data gate result (read by health)

Idempotency key: the signal date. A second run for a completed signal date is
skipped with a log line; ``force=True`` replaces it atomically (only the most
recent run can be replaced, because later runs would build on it).
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

CASH_TOLERANCE = 0.01
QUANTITY_TOLERANCE = 1e-6

LEDGER_COLUMNS = [
    "signal_date", "price_date", "run_id", "status", "started_at", "finished_at", "forced",
    "nav_start", "nav_end", "cash_end", "positions_end", "trades_filled", "orders_rejected",
    "turnover", "rebalance_pnl", "data_check_status", "reconciliation_status",
]
JOURNAL_COLUMNS = [
    "signal_date", "nav_end", "pnl_since_previous", "return_since_previous", "rebalance_pnl",
    "trades_filled", "orders_rejected", "turnover", "positions_end", "cash_end", "drawdown",
    "drawdown_halted", "vol_scale", "capacity_clipped_positions", "order_adv_clips",
    "data_check_status", "reconciliation_status", "forced",
]


class PaperRunBlocked(RuntimeError):
    """Raised when a pre-run gate (data checks, reconciliation, ordering) blocks a paper run."""


@dataclass(frozen=True)
class PaperOpsPaths:
    root: Path

    @classmethod
    def default(cls) -> "PaperOpsPaths":
        from src.config import PAPER_OPS_DIR

        return cls(Path(PAPER_OPS_DIR))

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    def run_dir(self, signal_date) -> Path:
        return self.runs / _date_str(signal_date)

    @property
    def ledger(self) -> Path:
        return self.root / "ledger.csv"

    @property
    def journal_csv(self) -> Path:
        return self.root / "journal.csv"

    @property
    def journal_md(self) -> Path:
        return self.root / "journal.md"

    @property
    def reconciliation(self) -> Path:
        return self.root / "latest_reconciliation.json"

    @property
    def gate(self) -> Path:
        return self.root / "last_gate.json"


def _date_str(value) -> str:
    return str(pd.Timestamp(value).date())


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ----------------------------------------------------------------- file helpers


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    write_text_atomic(path, frame.to_csv(index=False))


def read_csv_safe(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def upsert_row(path: Path, row: dict[str, Any], key: str, columns: list[str]) -> pd.DataFrame:
    """Insert or replace the row whose ``key`` matches, sorted by key; atomic write."""
    frame = read_csv_safe(path, columns)
    if not frame.empty and key in frame.columns:
        frame = frame[frame[key].astype(str) != str(row[key])]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame = frame.reindex(columns=list(dict.fromkeys([*columns, *frame.columns])))
    frame = frame.sort_values(key).reset_index(drop=True)
    write_csv_atomic(path, frame)
    return frame


def capture_state(files: list[str | Path], dest: Path) -> None:
    """Copy the listed files (if present) into ``dest`` and record which existed."""
    dest.mkdir(parents=True, exist_ok=True)
    index = {}
    for i, f in enumerate(files):
        src = Path(f)
        if src.exists():
            target = dest / f"{i:02d}_{src.name}"
            shutil.copy2(src, target)
            index[str(f)] = target.name
        else:
            index[str(f)] = None
    (dest / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))


def restore_state(dest: Path) -> None:
    """Undo everything since ``capture_state``: restore copies, delete files that did not exist."""
    index = json.loads((dest / "index.json").read_text())
    for original, saved in index.items():
        target = Path(original)
        if saved is None:
            if target.exists():
                target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.restore")
        shutil.copy2(dest / saved, tmp)
        os.replace(tmp, target)


# ------------------------------------------------------------------ run history


def completed_runs(paths: PaperOpsPaths) -> list[str]:
    if not paths.runs.exists():
        return []
    dates = []
    for child in paths.runs.iterdir():
        meta = child / "run.json"
        if child.is_dir() and not child.name.startswith(".") and meta.exists():
            if json.loads(meta.read_text()).get("status") == "COMPLETED":
                dates.append(child.name)
    return sorted(dates)


def plan_run(paths: PaperOpsPaths, signal_date, force: bool = False) -> str:
    """Return ``run`` | ``skip`` | ``replace`` for this signal date, or raise PaperRunBlocked."""
    day = _date_str(signal_date)
    done = completed_runs(paths)
    if day in done:
        if not force:
            return "skip"
        if day != done[-1]:
            raise PaperRunBlocked(
                f"--force can only replace the most recent run ({done[-1]}); {day} has later runs built on it."
            )
        return "replace"
    if done and day < done[-1]:
        raise PaperRunBlocked(
            f"signal date {day} is older than the latest completed run {done[-1]}; refusing to run out of order."
        )
    return "run"


# ---------------------------------------------------------------- broker state


def broker_state(path: str | Path) -> dict[str, Any]:
    """Cash and positions of the simulated broker (defaults when no state exists)."""
    from src.execution.simulated_broker import SimulatedBroker

    broker = SimulatedBroker.load(str(path))
    return {"cash": float(broker.cash), "positions": {k: float(v) for k, v in broker.positions.items()}}


def positions_frame(state: dict[str, Any], prices: dict[str, float] | None = None) -> pd.DataFrame:
    rows = []
    for symbol, qty in sorted(state.get("positions", {}).items()):
        price = (prices or {}).get(symbol)
        rows.append({
            "symbol": symbol,
            "quantity": qty,
            "price": price,
            "market_value": qty * price if price is not None else None,
        })
    return pd.DataFrame(rows, columns=["symbol", "quantity", "price", "market_value"])


def replay_fills(start: dict[str, Any], fills: pd.DataFrame) -> dict[str, Any]:
    """Apply filled orders to a starting state (same cash rules as the simulated broker)."""
    cash = float(start.get("cash", 0.0))
    positions = dict(start.get("positions", {}))
    for _, fill in fills.iterrows():
        if str(fill.get("status", "FILLED")) != "FILLED":
            continue
        qty = float(fill["quantity"])
        notional = float(fill["notional"])
        cost = float(fill.get("transaction_cost", 0.0))
        symbol = str(fill["symbol"])
        if str(fill["side"]).upper() == "BUY":
            cash -= notional + cost
            positions[symbol] = positions.get(symbol, 0.0) + qty
        else:
            cash += notional - cost
            remaining = positions.get(symbol, 0.0) - qty
            if remaining <= 1e-9:
                positions.pop(symbol, None)
            else:
                positions[symbol] = remaining
    return {"cash": cash, "positions": positions}


def compare_states(expected: dict[str, Any], actual: dict[str, Any], label: str) -> list[str]:
    issues = []
    cash_gap = float(actual.get("cash", 0.0)) - float(expected.get("cash", 0.0))
    if abs(cash_gap) > CASH_TOLERANCE:
        issues.append(f"{label}: cash {actual.get('cash'):.2f} vs expected {expected.get('cash'):.2f} ({cash_gap:+.2f})")
    exp_pos, act_pos = expected.get("positions", {}), actual.get("positions", {})
    for symbol in sorted(set(exp_pos) | set(act_pos)):
        gap = float(act_pos.get(symbol, 0.0)) - float(exp_pos.get(symbol, 0.0))
        if abs(gap) > QUANTITY_TOLERANCE:
            issues.append(
                f"{label}: {symbol} quantity {act_pos.get(symbol, 0.0):.6f} vs expected "
                f"{exp_pos.get(symbol, 0.0):.6f} ({gap:+.6f})"
            )
    return issues


def reconcile_start(paths: PaperOpsPaths, start: dict[str, Any], before: str | None = None) -> dict[str, Any]:
    """Compare today's starting state with the previous run's end state and fills.

    1. Continuity: start == previous ``end_state.json``.
    2. Fills replay: previous ``start_state.json`` + previous ``fills.csv`` == previous end state.
    ``before`` limits "previous" to runs dated strictly before it (used by --force).
    """
    done = [d for d in completed_runs(paths) if before is None or d < before]
    result: dict[str, Any] = {"checked_at": _now(), "previous_run": done[-1] if done else None, "issues": []}
    if not done:
        result["status"] = "NO_PREVIOUS_RUN"
        return result
    prev = paths.run_dir(done[-1])
    prev_end = json.loads((prev / "end_state.json").read_text())
    prev_start = json.loads((prev / "start_state.json").read_text())
    fills = read_csv_safe(prev / "fills.csv")
    issues = compare_states(prev_end, start, "start vs previous end")
    issues += compare_states(prev_end, replay_fills(prev_start, fills), "previous start + fills vs previous end")
    result["issues"] = issues
    result["status"] = "MISMATCH" if issues else "OK"
    return result


# -------------------------------------------------------------------- data gate


def paper_data_gate(as_of=None, config: dict[str, Any] | None = None, root: str | Path = ".") -> dict[str, Any]:
    """Data checks with paper severities: staleness is ``paper_stale_severity`` (ERROR)."""
    from src.data.quality import data_check_settings, overall_status, run_data_checks

    settings = data_check_settings(config)
    paper_config = dict(config or {})
    if config is None:
        from src.operations.configuration import load_platform_config

        paper_config = load_platform_config()
    paper_config = {
        **paper_config,
        "data_checks": {**settings, "stale_severity": settings.get("paper_stale_severity", "ERROR")},
    }
    findings, summary = run_data_checks(as_of=as_of, config=paper_config, root=root)
    errors = [f"{f.file}: {f.message}" for f in findings if f.severity == "ERROR"]
    return {
        "checked_at": _now(),
        "as_of": summary["as_of"],
        "status": overall_status(findings),
        "blocked": bool(errors),
        "errors": errors,
        "warnings": [f"{f.file}: {f.message}" for f in findings if f.severity == "WARNING"],
    }


def enforce_data_gate(paths: PaperOpsPaths, gate: dict[str, Any]) -> None:
    write_json_atomic(paths.gate, gate)
    if gate["blocked"]:
        raise PaperRunBlocked(
            "data checks failed for the paper run (staleness is an ERROR on this path): "
            + "; ".join(gate["errors"])
            + ". Refresh data (python src/data_loader.py --end today --with-benchmark) and retry."
        )


# -------------------------------------------------------------------- journal


def render_journal_markdown(journal: pd.DataFrame) -> str:
    lines = [
        "# Paper Trading Journal (auto-generated)",
        "",
        "One row per signal date, written by `scripts/run_daily_paper_trading.py`. Simulated paper",
        "trading only. Reruns for the same signal date replace the row instead of adding one.",
        "",
        "| Signal date | NAV | PnL vs prev | Return | Trades | Rejected | Turnover | Drawdown | Halt | Vol scale | ADV clips | Data | Recon |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for _, r in journal.iterrows():
        adv = f"{int(r.get('capacity_clipped_positions', 0) or 0)}/{int(r.get('order_adv_clips', 0) or 0)}"
        lines.append(
            f"| {r['signal_date']} | {float(r['nav_end']):,.2f} | {float(r['pnl_since_previous']):+,.2f} | "
            f"{float(r['return_since_previous']):+.2%} | {int(r['trades_filled'])} | {int(r['orders_rejected'])} | "
            f"{float(r['turnover']):.2%} | {float(r['drawdown']):.2%} | {'yes' if bool(r['drawdown_halted']) else 'no'} | "
            f"{float(r['vol_scale']):.2f} | {adv} | {r['data_check_status']} | {r['reconciliation_status']}"
            f"{' (forced)' if bool(r.get('forced', False)) else ''} |"
        )
    lines += ["", "ADV clips = target weights clipped by the ADV participation cap / orders clipped at fill."]
    return "\n".join(lines) + "\n"


def _journal_row(signal_date: str, snapshot: dict[str, Any], previous_nav: float | None,
                 data_status: str, recon_status: str, forced: bool) -> dict[str, Any]:
    nav = float(snapshot.get("portfolio_value", snapshot.get("ending_portfolio_value", 0.0)))
    base = previous_nav if previous_nav else float(snapshot.get("starting_portfolio_value", nav) or nav)
    return {
        "signal_date": signal_date,
        "nav_end": nav,
        "pnl_since_previous": nav - base,
        "return_since_previous": nav / base - 1 if base else 0.0,
        "rebalance_pnl": float(snapshot.get("pnl", 0.0)),
        "trades_filled": int(snapshot.get("trades_filled", 0)),
        "orders_rejected": int(snapshot.get("orders_rejected", 0)),
        "turnover": float(snapshot.get("turnover", 0.0)),
        "positions_end": int(snapshot.get("number_of_positions", 0)),
        "cash_end": float(snapshot.get("cash", 0.0)),
        "drawdown": float(snapshot.get("drawdown", 0.0)),
        "drawdown_halted": bool(snapshot.get("drawdown_halted", False)),
        "vol_scale": float(snapshot.get("vol_scale", 1.0)),
        "capacity_clipped_positions": int(snapshot.get("capacity_clipped_positions", 0) or 0),
        "order_adv_clips": int(snapshot.get("order_adv_clips", 0) or 0),
        "data_check_status": data_status,
        "reconciliation_status": recon_status,
        "forced": bool(forced),
    }


# ------------------------------------------------------------------ orchestrator


@dataclass
class SimulationFiles:
    """Where the simulated broker keeps state and writes per-run outputs."""

    broker_state: Path
    orders: Path
    fills: Path
    rejected: Path
    state_files: list[Path]

    @classmethod
    def default(cls) -> "SimulationFiles":
        from src import config as c

        state = [
            c.SIMULATED_BROKER_STATE_PATH, c.SIMULATED_TRADE_HISTORY_PATH, c.SIMULATED_REJECTED_ORDERS_PATH,
            c.SIMULATED_PORTFOLIO_HISTORY_PATH, c.SIMULATED_DAILY_PERFORMANCE_PATH,
            c.SIMULATED_EXECUTION_LOG_PATH, c.SIMULATED_LATEST_SNAPSHOT_PATH,
            c.SIMULATED_POSITION_SNAPSHOT_PATH, c.ALLOCATION_SUMMARY_PATH,
            c.REALIZED_VS_TARGET_WEIGHTS_PATH, c.SIMULATED_LATEST_ORDERS_PATH, c.SIMULATED_LATEST_FILLS_PATH,
            c.SIMULATED_LATEST_REJECTED_PATH, c.PAPER_TRADING_STATUS_REPORT_PATH,
        ]
        return cls(
            broker_state=Path(c.SIMULATED_BROKER_STATE_PATH),
            orders=Path(c.SIMULATED_LATEST_ORDERS_PATH),
            fills=Path(c.SIMULATED_LATEST_FILLS_PATH),
            rejected=Path(c.SIMULATED_LATEST_REJECTED_PATH),
            state_files=[Path(p) for p in state],
        )


def run_paper_day(
    signal_date,
    execute: Callable[[], dict[str, Any]],
    *,
    paths: PaperOpsPaths | None = None,
    sim: SimulationFiles | None = None,
    prices: dict[str, float] | None = None,
    price_date=None,
    force: bool = False,
    allow_reconciliation_mismatch: bool = False,
    data_gate: dict[str, Any] | None = None,
    run_id: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run ``execute`` once per signal date with snapshots, reconciliation and journal.

    Returns a summary dict with ``status`` in {COMPLETED, SKIPPED}. Raises
    PaperRunBlocked for gate/order/reconciliation blocks; any exception inside
    ``execute`` restores the simulation state files and leaves no run folder.
    """
    paths = paths or PaperOpsPaths.default()
    sim = sim or SimulationFiles.default()
    day = _date_str(signal_date)
    run_id = run_id or uuid.uuid4().hex[:12]
    action = plan_run(paths, day, force=force)
    if action == "skip":
        log(f"Paper run for signal date {day} already completed "
            f"({paths.run_dir(day)}); skipping. Use --force to replace it.")
        return {"status": "SKIPPED", "signal_date": day, "run_dir": str(paths.run_dir(day))}

    paths.runs.mkdir(parents=True, exist_ok=True)
    staging = paths.runs / f".staging-{day}-{run_id}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    current_backup = staging / "state_before_this_invocation"
    capture_state(sim.state_files, current_backup)

    try:
        if action == "replace":
            log(f"Replacing paper run for {day} (--force): restoring the state from before that run.")
            restore_state(paths.run_dir(day) / "pre_run_state")

        start = broker_state(sim.broker_state)
        recon = reconcile_start(paths, start, before=day)
        write_json_atomic(paths.reconciliation, {**recon, "signal_date": day})
        if recon["status"] == "MISMATCH":
            message = "reconciliation mismatch before the run: " + "; ".join(recon["issues"])
            if not allow_reconciliation_mismatch:
                raise PaperRunBlocked(message + " (investigate, or rerun with --allow-reconciliation-mismatch)")
            log(f"WARNING {message} (continuing: --allow-reconciliation-mismatch)")

        capture_state(sim.state_files, staging / "pre_run_state")
        started = _now()
        snapshot = execute()
        finished = _now()
    except BaseException:
        restore_state(current_backup)
        shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        end = broker_state(sim.broker_state)
        fills = read_csv_safe(sim.fills)
        post_issues = compare_states(end, replay_fills(start, fills), "start + fills vs end")
        for name, src in [("orders.csv", sim.orders), ("fills.csv", sim.fills), ("rejected.csv", sim.rejected)]:
            if Path(src).exists():
                shutil.copy2(src, staging / name)
            else:
                (staging / name).write_text("")
        write_json_atomic(staging / "start_state.json", start)
        write_json_atomic(staging / "end_state.json", end)
        write_csv_atomic(staging / "positions_start.csv", positions_frame(start, prices))
        write_csv_atomic(staging / "positions_end.csv", positions_frame(end, prices))
        recon_status = "MISMATCH" if (recon["status"] == "MISMATCH" or post_issues) else recon["status"]
        meta = {
            "run_id": run_id,
            "signal_date": day,
            "price_date": _date_str(price_date) if price_date is not None else str(snapshot.get("price_date")),
            "status": "COMPLETED",
            "started_at": started,
            "finished_at": finished,
            "forced": action == "replace",
            "data_gate": data_gate,
            "reconciliation_pre_run": recon,
            "reconciliation_post_run": {"status": "MISMATCH" if post_issues else "OK", "issues": post_issues},
            "nav_start": float(snapshot.get("starting_portfolio_value", 0.0)),
            "nav_end": float(snapshot.get("portfolio_value", 0.0)),
            "cash_end": end["cash"],
            "snapshot": {k: v for k, v in snapshot.items() if not isinstance(v, (list, dict))},
        }
        write_json_atomic(staging / "run.json", meta)
        shutil.rmtree(current_backup, ignore_errors=True)

        final = paths.run_dir(day)
        retired = None
        if final.exists():
            retired = paths.runs / f".replaced-{day}-{run_id}"
            os.replace(final, retired)
        os.replace(staging, final)
        if retired is not None:
            shutil.rmtree(retired, ignore_errors=True)
    except BaseException:
        # Persisting failed after the simulation ran: roll the simulation back too.
        if current_backup.exists():
            restore_state(current_backup)
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if post_issues:
        write_json_atomic(paths.reconciliation, {
            "checked_at": _now(), "signal_date": day, "status": "MISMATCH",
            "previous_run": recon.get("previous_run"), "issues": post_issues,
        })

    data_status = (data_gate or {}).get("status", "UNKNOWN")
    upsert_row(paths.ledger, {
        "signal_date": day, "price_date": meta["price_date"], "run_id": run_id, "status": "COMPLETED",
        "started_at": started, "finished_at": finished, "forced": action == "replace",
        "nav_start": meta["nav_start"], "nav_end": meta["nav_end"], "cash_end": end["cash"],
        "positions_end": len(end["positions"]), "trades_filled": int(snapshot.get("trades_filled", 0)),
        "orders_rejected": int(snapshot.get("orders_rejected", 0)),
        "turnover": float(snapshot.get("turnover", 0.0)), "rebalance_pnl": float(snapshot.get("pnl", 0.0)),
        "data_check_status": data_status, "reconciliation_status": recon_status,
    }, key="signal_date", columns=LEDGER_COLUMNS)

    journal = read_csv_safe(paths.journal_csv, JOURNAL_COLUMNS)
    earlier = journal[journal["signal_date"].astype(str) < day] if not journal.empty else journal
    previous_nav = float(earlier.sort_values("signal_date")["nav_end"].iloc[-1]) if not earlier.empty else None
    journal = upsert_row(
        paths.journal_csv,
        _journal_row(day, snapshot, previous_nav, data_status, recon_status, action == "replace"),
        key="signal_date", columns=JOURNAL_COLUMNS,
    )
    write_text_atomic(paths.journal_md, render_journal_markdown(journal))
    log(f"Paper run for signal date {day} {'replaced' if action == 'replace' else 'completed'}: "
        f"{paths.run_dir(day)} (reconciliation {recon_status})")
    return {"status": "COMPLETED", "action": action, "signal_date": day, "run_dir": str(paths.run_dir(day)),
            "reconciliation_status": recon_status, "run_id": run_id}


# ----------------------------------------------------------------------- health


def paper_health_findings(paths: PaperOpsPaths | None = None) -> list[tuple[str, str, str]]:
    """(severity, check, message) tuples for health: reconciliation ERROR, gate block, ledger dupes."""
    paths = paths or PaperOpsPaths.default()
    findings: list[tuple[str, str, str]] = []
    if paths.reconciliation.exists():
        recon = json.loads(paths.reconciliation.read_text())
        if recon.get("status") == "MISMATCH":
            findings.append(("ERROR", "paper_reconciliation",
                             f"Signal date {recon.get('signal_date')}: " + "; ".join(recon.get("issues", [])[:5])))
    done = completed_runs(paths)
    if done:
        meta = json.loads((paths.run_dir(done[-1]) / "run.json").read_text())
        post = meta.get("reconciliation_post_run", {})
        if post.get("status") == "MISMATCH":
            findings.append(("ERROR", "paper_reconciliation",
                             f"Run {done[-1]} fills do not explain its end state: " + "; ".join(post.get("issues", [])[:5])))
    if paths.gate.exists():
        gate = json.loads(paths.gate.read_text())
        if gate.get("blocked"):
            findings.append(("WARNING", "paper_data_gate",
                             f"Last paper run (as of {gate.get('as_of')}) was blocked by data checks: "
                             + "; ".join(gate.get("errors", [])[:3])))
    ledger = read_csv_safe(paths.ledger, LEDGER_COLUMNS)
    if not ledger.empty and ledger["signal_date"].duplicated().any():
        dupes = sorted(ledger.loc[ledger["signal_date"].duplicated(), "signal_date"].astype(str).unique())
        findings.append(("ERROR", "paper_ledger", f"Duplicate ledger rows for signal date(s): {', '.join(dupes)}"))
    return findings
