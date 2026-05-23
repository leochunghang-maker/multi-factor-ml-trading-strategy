import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    DAILY_RETURNS_PATH,
    FORWARD_RETURNS_PATH,
    LIVE_SIGNALS_PATH,
    MAX_PAPER_POSITIONS,
    MAX_PAPER_TURNOVER,
    MEAN_REVERSION_PATH,
    ML_DATASET_PATH,
    MOMENTUM_PATH,
    PAPER_TRADING_STATUS_REPORT_PATH,
    PRICE_DATA_PATH,
)
from src.data.io import load_factor_frame, load_price_data
from src.execution.run_daily_simulation import (
    run_daily_simulation,
    save_paper_trading_status_report,
)
from src.features.factors import build_ml_dataset, calculate_factor_frames
from src.live.generate_live_signals import download_latest_prices, generate_live_signals
from src.operations.health import build_health_status, save_system_status_report
from src.operations.structured_logging import log_event
from src.operations.metadata import create_run_metadata, set_deterministic_seed
from src.reporting.summary import generate_summary_report


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str


def print_step(index: int, name: str) -> None:
    print(f"{index}. {name}...")


def refresh_price_data() -> pd.DataFrame:
    prices = download_latest_prices()
    Path(PRICE_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PRICE_DATA_PATH)
    return prices


def generate_features() -> None:
    prices = load_price_data()
    factors = calculate_factor_frames(prices)
    factors["daily_returns"].to_csv(DAILY_RETURNS_PATH)
    factors["momentum_12m"].to_csv(MOMENTUM_PATH)
    factors["mean_reversion_1m"].to_csv(MEAN_REVERSION_PATH)
    factors["forward_returns_1m"].to_csv(FORWARD_RETURNS_PATH)

    ml_dataset = build_ml_dataset(
        factors["momentum_12m"],
        factors["mean_reversion_1m"],
        factors["forward_returns_1m"],
        prices,
    )
    ml_dataset.to_csv(ML_DATASET_PATH, index=False)


def validate_signal_file(path: str = LIVE_SIGNALS_PATH) -> list[str]:
    # Paper trading must be validated before live trading because operational
    # failures usually start as small data problems: stale files, duplicate
    # symbols, bad weights, or unexplained turnover.
    signals = pd.read_csv(path, parse_dates=["date"])
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local daily paper-trading validation workflow."
    )
    parser.add_argument(
        "--skip-data-refresh",
        action="store_true",
        help="Use existing local price data instead of downloading fresh paper-trading prices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[StepResult] = []
    set_deterministic_seed()
    metadata = create_run_metadata(
        "daily_paper_trading",
        {"skip_data_refresh": args.skip_data_refresh},
    )
    run_id = metadata["run_id"]
    log_event("workflow_started", "Daily paper-trading workflow started.", run_id=run_id)

    print("Daily paper-trading validation workflow")
    print("No real-money trading is enabled. This run uses local files and the simulated broker only.")
    print()

    if args.skip_data_refresh:
        print("1. Data refresh...")
        print("   skipped: using existing local price data")
        results.append(StepResult("Data refresh", True, "skipped by operator"))
        log_event("step_skipped", "Data refresh skipped by operator.", run_id=run_id, step="Data refresh")
    else:
        prices = run_step(results, 1, "Data refresh", refresh_price_data)
        latest_date = prices.dropna(how="all").index.max()
        print(f"   latest price date: {latest_date.date()}")
        log_event("step_completed", "Data refresh completed.", run_id=run_id, latest_price_date=str(latest_date.date()))

    run_step(results, 2, "Feature generation", generate_features)
    signals = run_step(results, 3, "Signal generation", generate_live_signals)
    print(f"   generated signals: {len(signals)} rows")

    signal_warnings = run_step(results, 4, "Portfolio constraint and signal safety checks", validate_signal_file)
    for warning in signal_warnings:
        print(f"   warning: {warning}")

    snapshot = run_step(results, 5, "Simulated broker execution", run_daily_simulation)
    for warning in validate_turnover(snapshot):
        print(f"   warning: {warning}")

    report = run_step(results, 6, "Risk report generation", generate_summary_report)
    print(f"   report rows: {len(report)}")

    print_step(7, "Dashboard/report update")
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
        status = "OK" if result.ok else "FAILED"
        print(f"- {status}: {result.name} ({result.detail})")
    print()
    print(f"Paper trading status report: {PAPER_TRADING_STATUS_REPORT_PATH}")
    print(f"Run metadata: {metadata['metadata_path']}")
    log_event("workflow_completed", "Daily paper-trading workflow completed.", run_id=run_id)


if __name__ == "__main__":
    main()
