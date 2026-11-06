"""Nested / anchored walk-forward model selection (no outer-fold peeking)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ML_EMBARGO_DAYS, RANDOM_FOREST_PARAMS, RANDOM_SEED
from src.models.validation import (
    assert_no_label_overlap,
    inner_anchored_year_splits,
    mean_rank_ic,
    month_end_subset,
    training_rows_without_label_overlap,
)


def default_nested_candidates() -> list[dict[str, Any]]:
    """Small, fast candidate set for nested selection.

    Ridge variants are cheap; one shallow RF and the production RF depth give a
    minimal nonlinear choice set without a large grid search.
    """
    seed = int(RANDOM_SEED)
    return [
        {"name": "ridge_1", "kind": "ridge", "params": {"alpha": 1.0}},
        {
            "name": "rf_shallow",
            "kind": "random_forest",
            "params": {
                "n_estimators": 40,
                "max_depth": 3,
                "random_state": seed,
                "n_jobs": -1,
            },
        },
        {
            "name": "rf_default",
            "kind": "random_forest",
            "params": {
                **dict(RANDOM_FOREST_PARAMS),
                "random_state": seed,
                "n_jobs": -1,
            },
        },
    ]


def build_estimator(kind: str, params: dict[str, Any]):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if kind == "random_forest":
        return RandomForestRegressor(**params)
    raise ValueError(f"Unsupported model kind: {kind}")


def fit_estimator(kind: str, params: dict[str, Any], X: pd.DataFrame, y: pd.Series):
    model = build_estimator(kind, params)
    model.fit(X, y)
    return model


def score_candidate_on_inner_folds(
    train_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    candidate: dict[str, Any],
    label_horizon_days: int = ML_EMBARGO_DAYS,
    purge_months: int = 1,
    embargo_months: int = 1,
    min_train_years: int = 2,
    use_month_end_subset: bool = True,
) -> float:
    """Mean inner-fold rank IC for one candidate (NaN if no usable inner fold)."""
    frame = month_end_subset(train_df) if use_month_end_subset else train_df
    splits = inner_anchored_year_splits(
        frame["date"],
        min_train_years=min_train_years,
        purge_months=purge_months,
        embargo_months=embargo_months,
    )
    fold_scores: list[float] = []
    for split in splits:
        inner_train = training_rows_without_label_overlap(
            frame,
            date_column="date",
            train_end_before_purge=split["train_end_before_purge"],
            test_start=split["test_start"],
            label_horizon_days=label_horizon_days,
        )
        inner_valid = frame[
            (frame["date"] >= split["test_start"]) & (frame["date"] <= split["test_end"])
        ].copy()
        if inner_train.empty or inner_valid.empty:
            continue
        assert_no_label_overlap(
            inner_train["date"],
            test_start=split["test_start"],
            label_horizon_days=label_horizon_days,
        )
        # Hard guarantee: nothing from the outer test era can appear here because
        # ``train_df`` is already purged to the outer fold.
        model = fit_estimator(
            candidate["kind"], candidate["params"], inner_train[feature_columns], inner_train[target_column]
        )
        scored = inner_valid.copy()
        scored["_nested_score"] = model.predict(inner_valid[feature_columns])
        ic = mean_rank_ic(
            scored,
            date_column="date",
            score_column="_nested_score",
            target_column=target_column,
        )
        if np.isfinite(ic):
            fold_scores.append(ic)
    if not fold_scores:
        return float("nan")
    return float(np.mean(fold_scores))


def select_model_nested(
    train_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    candidates: list[dict[str, Any]] | None = None,
    label_horizon_days: int = ML_EMBARGO_DAYS,
    purge_months: int = 1,
    embargo_months: int = 1,
    min_train_years: int = 2,
    use_month_end_subset: bool = True,
    outer_test_year: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Choose a candidate using only ``train_df``, then refit on all of it.

    If inner folds are unavailable (short history) or every score is NaN, falls
    back to the first candidate (conventionally the production default when the
    caller puts ``rf_default`` last — here we fall back to the best finite score
    or the last RF-like / first entry).
    """
    candidates = deepcopy(candidates or default_nested_candidates())
    if not candidates:
        raise ValueError("candidates must be non-empty")

    score_rows = []
    for candidate in candidates:
        score = score_candidate_on_inner_folds(
            train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            candidate=candidate,
            label_horizon_days=label_horizon_days,
            purge_months=purge_months,
            embargo_months=embargo_months,
            min_train_years=min_train_years,
            use_month_end_subset=use_month_end_subset,
        )
        score_rows.append({"name": candidate["name"], "mean_inner_rank_ic": score})

    finite = [row for row in score_rows if np.isfinite(row["mean_inner_rank_ic"])]
    if finite:
        best_name = max(finite, key=lambda row: row["mean_inner_rank_ic"])["name"]
        selection_mode = "nested_rank_ic"
    else:
        # Prefer rf_default if present, else last candidate.
        names = [c["name"] for c in candidates]
        best_name = "rf_default" if "rf_default" in names else names[-1]
        selection_mode = "fallback_no_inner_scores"

    winner = next(c for c in candidates if c["name"] == best_name)
    model = fit_estimator(
        winner["kind"], winner["params"], train_df[feature_columns], train_df[target_column]
    )
    meta = {
        "outer_test_year": outer_test_year,
        "chosen_model": best_name,
        "selection_mode": selection_mode,
        "candidate_scores": score_rows,
        "n_train_rows": int(len(train_df)),
        "train_start": str(pd.to_datetime(train_df["date"]).min().date()) if len(train_df) else None,
        "train_end": str(pd.to_datetime(train_df["date"]).max().date()) if len(train_df) else None,
    }
    return model, meta
