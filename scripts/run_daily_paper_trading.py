import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    LIVE_SIGNALS_PATH,
    MAX_PAPER_POSITIONS,
    MAX_PAPER_TURNOVER,
    PAPER_TRADING_STATUS_REPORT_PATH,
    SIMULATED_LATEST_SNAPSHOT_PATH,
)
from src.data.io import load_price_data
from src.data.paths import DataPaths
from src.data.required import require_file, run_cli
from src.execution.run_daily_simulation import (
    load_latest_prices,
    run_daily_simulation,
    save_paper_trading_status_report,
)
from src.features.factors import build_ml_dataset, calculate_factor_frames
from src.live.generate_live_signals import generate_live_signals
from src.operations.health import build_health_status, save_system_status_report
from src.operations.structured_logging import log_event
from src.operations.metadata import create_run_metadata, set_deterministic_seed
from src.operations.paper_ops import (
    PaperOpsPaths,
    PaperRunBlocked,
    enforce_data_gate,
    paper_data_gate,
    plan_run,
    run_paper_day,
)
from src.operations.paper_alerts import raise_alert
from src.operations.paper_daily import LOCKED_EXIT_CODE, paper_lock
from src.operations.run_lock import LockHeld
from src.operations.paper_status import write_paper_status
from src.reporting.summary import generate_summary_report


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str


def print_step(index: int, name: str) -> None:
    print(f"{index}. {name}...")


def refresh_price_data(data_paths: DataPaths) -> pd.DataFrame:
    """Full-history refresh through today into the paper data dir (prices, volume, SPY).

    Uses ``src/data_loader.py`` so price/volume/benchmark stay aligned and the
    excluded-symbol decisions (e.g. BK) are applied. Writes only ``data_paths``
    (default ``data/paper/``); the checked-in research snapshot under ``data/``
    is never rewritten by the paper workflow.
    """
    from src.data_loader import refresh_paper_data

    refresh_paper_data(end="today", paths=data_paths)
    return load_price_data(data_paths.price_data)


def generate_features(data_paths: DataPaths) -> None:
    """Rebuild factor files and the ML dataset from the paper prices, inside the paper dir."""
    prices = load_price_data(data_paths.price_data)
    factors = calculate_factor_frames(prices)
    factors["daily_returns"].to_csv(data_paths.daily_returns)
    factors["momentum_12m"].to_csv(data_paths.momentum)
    factors["mean_reversion_1m"].to_csv(data_paths.mean_reversion)
    factors["forward_returns_1m"].to_csv(data_paths.forward_returns)

    ml_dataset = build_ml_dataset(
        factors["momentum_12m"],
        factors["mean_reversion_1m"],
        factors["forward_returns_1m"],
        prices,
    )
    ml_dataset.to_csv(data_paths.ml_dataset, index=False)


def validate_signal_file(path: str = LIVE_SIGNALS_PATH) -> list[str]:
    # Paper trading must be validated before live trading because operational
    # failures usually start as small data problems: stale files, duplicate
    # symbols, bad weights, or unexplained turnover.
    signals = pd.read_csv(require_file(path), parse_dates=["date"])
    warnings = []

    if signals.empty:
        raise RuntimeError("Signal file is empty.")
    if signals["ticker"].duplicated().any():
        duplicates = sorted(signals.loc[signals["ticker"].duplicated(), "ticker"].unique())
        raise RuntimeError(f"Duplicate tickers found in signal file: {duplicates}")

    longs = signals[signals["side"] == "LONG"]
    if longs.empty:
        raise RuntimeError("No LONG signals found.")
    if len(longs) > MAX_PAPER_POSITIONS:
        raise RuntimeError(f"Too many LONG signals: {len(longs)} > {MAX_PAPER_POSITIONS}.")

    target_weight_sum = longs["target_weight"].sum()
    if target_weight_sum > 1.0 + 1e-9:
        raise RuntimeError(f"Target weights imply leverage: {target_weight_sum:.2%}.")
    if target_weight_sum < 0.95:
        warnings.append(f"Target weights sum to {target_weight_sum:.2%}; portfolio may be under-invested.")
    if (longs["target_weight"] < 0).any():
        raise RuntimeError("Negative target weights are not allowed.")
    if longs["target_weight"].max() > 1 / max(1, MAX_PAPER_POSITIONS) + 1e-9:
        warnings.append("At least one target weight is above the equal-weight paper allocation.")
    return warnings


def validate_turnover(snapshot: dict) -> list[str]:
    turnover = float(snapshot.get("turnover", 0.0))
    if turnover > MAX_PAPER_TURNOVER:
        return [f"Turnover {turnover:.2%} is above max paper turnover {MAX_PAPER_TURNOVER:.2%}."]
    return []


def write_failure_status(message: str) -> None:
    snapshot = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "price_date": "unavailable",
        "signal_date": "unavailable",
        "portfolio_value": 0.0,
        "cash": 0.0,
        "number_of_positions": 0,
        "turnover": 0.0,
        "drawdown": 0.0,
        "trades_filled": 0,
        "orders_rejected": 0,
    }
    save_paper_trading_status_report(snapshot, [], [message], [])


def run_step(results: list[StepResult], index: int, name: str, action) -> object:
    print_step(index, name)
    try:
        output = action()
        results.append(StepResult(name, True, "completed"))
        print(f"   ok: {name}")
        return output
    except Exception as exc:
        results.append(StepResult(name, False, str(exc)))
        write_failure_status(f"{name} failed: {exc}")
        log_event("step_failed", f"{name} failed: {exc}", level="ERROR", step=name)
        print(f"   failed: {exc}")
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local daily paper-trading validation workflow."
    )
    parser.add_argument(
        "--skip-data-refresh",
        action="store_true",
        help="Use existing local price data instead of downloading fresh paper-trading prices.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the most recent run for the same signal date (restores its pre-run state first).",
    )
    parser.add_argument(
        "--allow-reconciliation-mismatch",
        action="store_true",
        help="Run even if starting positions/cash do not match the previous run (still flagged ERROR).",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Date used by the data-freshness gate (default: today). Staleness is an ERROR on this path.",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Do not take the paper run lock (used by scripts/run_paper_daily.py, which already holds it).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Paper data dir (default: paper_trading.data_dir = data/paper, or $PAPER_DATA_DIR). "
             "The research data/ snapshot is never written by this workflow.",
    )
    return parser.parse_args(argv)


def blocked(run_id: str, step: str, message: str) -> int:
    print(f"   BLOCKED: {message}")
    print("   Alert recorded in results/paper/alerts.jsonl; see results/paper/status.md (make paper-status).")
    write_failure_status(f"{step} blocked: {message}")
    log_event("paper_run_blocked", message, level="ERROR", run_id=run_id, step=step)
    status = build_health_status()
    save_system_status_report(status)
    print()
    print("Daily paper run blocked; no simulated orders were placed.")
    return 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_lock:
        return run_workflow(args)
    lock = paper_lock(log=lambda line: print(f"   {line}"))
    try:
        lock.acquire()
    except LockHeld as exc:
        print(f"Not starting: {exc}")
        raise_alert(PaperOpsPaths.default(), "run_blocked", str(exc), details={"reason": "lock_held"})
        write_paper_status(PaperOpsPaths.default())
        return LOCKED_EXIT_CODE
    try:
        return run_workflow(args)
    finally:
        lock.release()


def run_workflow(args: argparse.Namespace) -> int:
    results: list[StepResult] = []
    set_deterministic_seed()
    metadata = create_run_metadata(
        "daily_paper_trading",
        {
            "skip_data_refresh": args.skip_data_refresh,
            "force": args.force,
            "allow_reconciliation_mismatch": args.allow_reconciliation_mismatch,
            "as_of": args.as_of,
            "data_dir": args.data_dir,
        },
    )
    run_id = metadata["run_id"]
    paths = PaperOpsPaths.default()
    data_paths = DataPaths.paper(args.data_dir)
    log_event("workflow_started", "Daily paper-trading workflow started.", run_id=run_id)

    print("Daily paper-trading validation workflow")
    print("No real-money trading is enabled. This run uses local files and the simulated broker only.")
    print(f"Paper data dir: {data_paths.root}/ (research data/ is not modified)")
    print()

    if args.skip_data_refresh:
        print("1. Data refresh...")
        print(f"   skipped: using existing paper data in {data_paths.root}/")
        results.append(StepResult("Data refresh", True, "skipped by operator"))
        log_event("step_skipped", "Data refresh skipped by operator.", run_id=run_id, step="Data refresh")
    else:
        prices = run_step(results, 1, "Data refresh", lambda: refresh_price_data(data_paths))
        latest_date = prices.dropna(how="all").index.max()
        print(f"   latest price date: {latest_date.date()}")
        log_event("step_completed", "Data refresh completed.", run_id=run_id, latest_price_date=str(latest_date.date()))

    print_step(2, "Pre-run data gate (staleness is an ERROR on the paper path)")
    gate = paper_data_gate(as_of=args.as_of, data_paths=data_paths)
    try:
        enforce_data_gate(paths, gate, run_id=run_id, log=lambda line: print(f"   {line}"))
    except PaperRunBlocked as exc:
        return blocked(run_id, "Pre-run data gate", str(exc))
    print(f"   ok: data checks {gate['status']} as of {gate['as_of']} ({gate['calendar']} calendar, {gate['data_dir']}/)")
    results.append(StepResult("Pre-run data gate", True, gate["status"]))

    local_prices = load_price_data(data_paths.price_data)
    price_date = local_prices.dropna(how="all").index.max()
    print_step(3, "Idempotency check")
    try:
        action = plan_run(paths, price_date, force=args.force)
    except PaperRunBlocked as exc:
        raise_alert(paths, "run_blocked", str(exc), signal_date=str(price_date.date()), run_id=run_id,
                    log=lambda line: print(f"   {line}"))
        write_paper_status(paths)
        return blocked(run_id, "Idempotency check", str(exc))
    if action == "skip":
        message = (
            f"Paper run for signal date {price_date.date()} already completed "
            f"({paths.run_dir(price_date)}); skipping. Use --force to replace it."
        )
        print(f"   {message}")
        log_event("paper_run_skipped", message, run_id=run_id, signal_date=str(price_date.date()))
        return 0
    print(f"   ok: {action} for signal date {price_date.date()}")

    run_step(results, 4, "Feature generation", lambda: generate_features(data_paths))
    signals = run_step(
        results, 5, "Signal generation",
        lambda: generate_live_signals(prices=local_prices, dataset_path=data_paths.ml_dataset),
    )
    print(f"   generated signals: {len(signals)} rows")

    signal_warnings = run_step(results, 6, "Portfolio constraint and signal safety checks", validate_signal_file)
    for warning in signal_warnings:
        print(f"   warning: {warning}")

    signal_date = pd.to_datetime(signals["date"]).max()
    latest_prices, latest_price_date = load_latest_prices(data_paths.price_data)
    print_step(7, "Simulated broker execution (idempotent, snapshotted, reconciled)")
    try:
        outcome = run_paper_day(
            signal_date,
            lambda: run_daily_simulation(price_path=data_paths.price_data, volume_path=data_paths.volume_data),
            paths=paths,
            prices=latest_prices,
            price_date=latest_price_date,
            force=args.force,
            allow_reconciliation_mismatch=args.allow_reconciliation_mismatch,
            data_gate=gate,
            run_id=run_id,
            log=lambda line: print(f"   {line}"),
        )
    except PaperRunBlocked as exc:
        return blocked(run_id, "Simulated broker execution", str(exc))
    except Exception as exc:
        results.append(StepResult("Simulated broker execution", False, str(exc)))
        write_failure_status(f"Simulated broker execution failed: {exc}")
        log_event("step_failed", f"Simulated broker execution failed: {exc}", level="ERROR", run_id=run_id)
        print(f"   failed (simulation state restored): {exc}")
        raise
    results.append(StepResult("Simulated broker execution", True, outcome["status"].lower()))
    if outcome["status"] == "SKIPPED":
        return 0
    snapshot = pd.read_csv(SIMULATED_LATEST_SNAPSHOT_PATH).iloc[-1].to_dict()
    for warning in validate_turnover(snapshot):
        print(f"   warning: {warning}")
    log_event("paper_run_completed", f"Paper run {outcome['action']} for {outcome['signal_date']}.",
              run_id=run_id, signal_date=outcome["signal_date"], run_dir=outcome["run_dir"],
              reconciliation_status=outcome["reconciliation_status"])

    report = run_step(results, 8, "Risk report generation", generate_summary_report)
    print(f"   report rows: {len(report)}")

    print_step(9, "Dashboard/report update")
    # The Streamlit dashboard is read-only and automatically reflects the CSVs
    # regenerated above. No browser or broker connection is required here.
    print("   ok: dashboard reads the refreshed results/reports files")
    results.append(StepResult("Dashboard/report update", True, "completed"))
    status = build_health_status()
    save_system_status_report(status)
    log_event(
        "health_status",
        f"System health status is {status['overall_status']}.",
        run_id=run_id,
        overall_status=status["overall_status"],
        finding_count=len(status["findings"]),
    )

    print()
    print("Daily paper-trading summary")
    for result in results:
        state = "OK" if result.ok else "FAILED"
        print(f"- {state}: {result.name} ({result.detail})")
    print()
    print(f"Run folder: {outcome['run_dir']}")
    print(f"Journal: {paths.journal_md}")
    print(f"Paper status: {paths.status_md} (make paper-status)")
    print(f"Paper trading status report: {PAPER_TRADING_STATUS_REPORT_PATH}")
    print(f"Run metadata: {metadata['metadata_path']}")
    log_event("workflow_completed", "Daily paper-trading workflow completed.", run_id=run_id)
    return 0


if __name__ == "__main__":
    sys.exit(run_cli(main))
