from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = RESULTS_DIR / "reports"
SIMULATION_DIR = RESULTS_DIR / "simulation"
LIVE_DIR = RESULTS_DIR / "live"


def read_csv(path: Path, date_columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for column in date_columns or []:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if "Unnamed: 0" in frame.columns:
        frame = frame.rename(columns={"Unnamed: 0": "date"})
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def load_summary_metrics() -> dict[str, float]:
    frame = read_csv(REPORTS_DIR / "summary_report.csv")
    if frame.empty or not {"metric", "value"}.issubset(frame.columns):
        return {}
    return dict(zip(frame["metric"], frame["value"]))


def load_strategy_returns() -> pd.DataFrame:
    return read_csv(RESULTS_DIR / "multi_factor_results.csv", ["date"])


def load_rolling_risk() -> pd.DataFrame:
    return read_csv(REPORTS_DIR / "rolling_risk_report.csv", ["date"])


def load_turnover() -> pd.DataFrame:
    return read_csv(REPORTS_DIR / "turnover_report.csv", ["date"])


def load_portfolio_exposure() -> pd.DataFrame:
    return read_csv(REPORTS_DIR / "portfolio_exposure_report.csv", ["date"])


def load_sector_exposure() -> pd.DataFrame:
    return read_csv(REPORTS_DIR / "sector_exposures.csv", ["date"])


def load_live_signals() -> pd.DataFrame:
    return read_csv(LIVE_DIR / "latest_signals.csv", ["date"])


def load_execution_log() -> pd.DataFrame:
    return read_csv(SIMULATION_DIR / "execution_log.csv", ["timestamp"])


def load_trade_history() -> pd.DataFrame:
    frame = read_csv(SIMULATION_DIR / "trade_history.csv", ["timestamp"])
    if "decision_price" not in frame.columns and "price" in frame.columns:
        frame = frame.rename(columns={"price": "decision_price"})
    return frame


def load_position_snapshot() -> pd.DataFrame:
    return read_csv(SIMULATION_DIR / "latest_position_snapshot.csv", ["timestamp"])


def latest_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.dropna(how="all").iloc[-1]
