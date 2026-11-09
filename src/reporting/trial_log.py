"""Research trial log: the honest count of variants tried (feeds the Deflated Sharpe).

Rule: append a row to ``research/trial_log.csv`` *before* running a new research
variant or test, with ``trial_increment`` (usually 1) and later fill in
``reported_sharpe`` if one is observed. ``n_trials`` for DSR is the sum of
increments plus a fixed, documented padding for unrecorded history.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "trial_id",
    "date",
    "commit",
    "family",
    "description",
    "trial_increment",
    "reported_sharpe",
)


def load_trial_log(path: str | Path) -> pd.DataFrame:
    """Read and validate the trial log (raises on malformed rows)."""
    frame = pd.read_csv(path, dtype={"commit": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"trial log {path} missing columns: {missing}")
    if frame["trial_id"].duplicated().any():
        raise ValueError("trial log has duplicate trial_id values")
    increments = pd.to_numeric(frame["trial_increment"], errors="coerce")
    if increments.isna().any() or (increments < 0).any():
        raise ValueError("trial_increment must be a non-negative number on every row")
    if (increments != np.floor(increments)).any():
        raise ValueError("trial_increment must be an integer")
    frame["trial_increment"] = increments.astype(int)
    frame["reported_sharpe"] = pd.to_numeric(frame["reported_sharpe"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["family"] = frame["family"].astype(str).str.strip()
    return frame


def count_trials(log: pd.DataFrame, padding: int = 0) -> int:
    """Total trials = sum of logged increments + documented padding."""
    if padding < 0:
        raise ValueError("padding must be non-negative")
    return int(log["trial_increment"].sum()) + int(padding)


def family_sharpes(log: pd.DataFrame, family: str) -> list[float]:
    values = log.loc[log["family"] == family, "reported_sharpe"].dropna()
    return [float(x) for x in values]


def pooled_sharpes(log: pd.DataFrame) -> list[float]:
    return [float(x) for x in log["reported_sharpe"].dropna()]
