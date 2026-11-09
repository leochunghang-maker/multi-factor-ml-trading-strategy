"""Nested / anchored walk-forward model selection (no outer-fold peeking).

Inner scoring (configurable via ``NESTED_INNER_SCORE`` / ``score_metric``):

- ``rank_ic``: mean cross-sectional Spearman IC across validation dates.
- ``icir``: mean(IC) / std(IC) across those dates (information coefficient IR).
- ``cost_aware_sharpe`` (default): annualized Sharpe of equal-weight long/short
  period returns built from model scores on each validation date, after
  subtracting (i) turnover × ``transaction_cost`` and (ii) stock-borrow /
  financing on the signed weight book. Formula per month-end date t:

      gross_t = sum_i w_{i,t} * r_{i,t}
      net_t   = gross_t - |w_t - w_{t-1}|_1 * c - financing(w_t)
      score   = mean(net) / std(net) * sqrt(periods_per_year)

  with base side gross ``gross_exposure / 2`` per leg before scaling. This tracks tradable quality better than
  raw IC alone while staying fast on the month-end subset.

  Inner books use the same construction as the outer research portfolio:
  sector-demeaned ranking plus sector net-exposure caps (``SECTOR_NEUTRAL`` /
  ``MAX_SECTOR_NET_EXPOSURE``), and — when ``NESTED_INNER_VOL_TARGETING`` and
  ``VOL_TARGETING_ENABLED`` are on — causal vol-target scaling of base gross:

      scale_t = clip(target_vol / forecast_vol(net_{<t}), min, max)
      gross_t = sum_i scale_t * w_{i,t} * r_{i,t}

  where ``net_{<t}`` are the candidate's own earlier inner-validation net
  returns (inner-train information only; the outer test fold is never seen).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    ANNUAL_BORROW_BPS,
    ANNUAL_CASH_FINANCING_BPS,
    ANNUAL_SHORT_REBATE_BPS,
    ML_EMBARGO_DAYS,
    ML_LONG_COUNT,
    ML_SHORT_COUNT,
    NESTED_INNER_SCORE,
    NESTED_INNER_VOL_TARGETING,
    PERIODS_PER_YEAR,
    RANDOM_FOREST_PARAMS,
    RANDOM_SEED,
    RESEARCH_LONG_SHORT_GROSS_EXPOSURE,
    TRANSACTION_COST,
    VOL_TARGETING_ENABLED,
)
from src.execution.financing import financing_cost_from_weights
from src.models.validation import (
    assert_no_label_overlap,
    inner_anchored_year_splits,
    month_end_subset,
    training_rows_without_label_overlap,
)
from src.portfolio.portfolio import (
    equal_weight_long_short_weights,
    transaction_cost_from_weight_change,
)
from src.risk.volatility import scale_weights, volatility_target_scale

SUPPORTED_INNER_SCORES = ("rank_ic", "icir", "cost_aware_sharpe")


def default_nested_candidates() -> list[dict[str, Any]]:
    """Small, fast candidate set for nested selection."""
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


def _date_rank_ic(scores: pd.Series, targets: pd.Series) -> float:
    valid = pd.concat([scores, targets], axis=1).dropna()
    if len(valid) < 5 or valid.iloc[:, 0].nunique() < 2:
        return float("nan")
    corr = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).correlation
    if corr is None or not np.isfinite(corr):
        return float("nan")
    return float(corr)


def aggregate_icir(ics: list[float]) -> float:
    """Information-coefficient IR: mean(IC) / std(IC)."""
    arr = np.asarray([x for x in ics if np.isfinite(x)], dtype=float)
    if arr.size < 2:
        return float("nan")
    denom = float(arr.std(ddof=1))
    if denom <= 1e-12:
        return float("nan")
    return float(arr.mean() / denom)


def aggregate_sharpe(returns: list[float], periods_per_year: float = PERIODS_PER_YEAR) -> float:
    arr = np.asarray([x for x in returns if np.isfinite(x)], dtype=float)
    if arr.size < 2:
        return float("nan")
    denom = float(arr.std(ddof=1))
    if denom <= 1e-12:
        return float("nan")
    return float(arr.mean() / denom * np.sqrt(float(periods_per_year)))


def net_long_short_period_return(
    scores: pd.Series,
    realized_returns: pd.Series,
    *,
    previous_weights: pd.Series | None,
    long_count: int,
    short_count: int,
    gross_exposure: float,
    transaction_cost: float,
    annual_borrow_bps: float,
    annual_short_rebate_bps: float,
    annual_cash_financing_bps: float,
    periods_per_year: float,
    vol_scale: float = 1.0,
) -> tuple[float, pd.Series]:
    """One rebalance's net L/S return after turnover costs and financing.

    Weights come from the same sector-aware equal-weight constructor as the
    outer book, are multiplied by ``vol_scale`` (computed by the caller from
    *past* returns only), and gross P&L is ``sum(w * r)`` so sector-cap
    reweighting and vol scaling flow through returns, turnover and financing.
    """
    aligned = pd.concat(
        [scores.rename("score"), realized_returns.rename("ret")],
        axis=1,
    ).dropna()
    if len(aligned) < max(long_count + short_count, 4):
        return float("nan"), previous_weights if previous_weights is not None else pd.Series(dtype=float)

    weights = equal_weight_long_short_weights(
        aligned["score"],
        long_count=long_count,
        short_count=short_count,
        gross_exposure=gross_exposure,
    )
    weights = scale_weights(weights, vol_scale)
    long_idx = weights[weights > 0].index
    short_idx = weights[weights < 0].index
    if long_idx.empty or short_idx.empty:
        return float("nan"), weights

    gross = float((weights * aligned["ret"].reindex(weights.index).fillna(0.0)).sum())
    trade_cost, _ = transaction_cost_from_weight_change(weights, previous_weights, transaction_cost)
    financing = financing_cost_from_weights(
        weights,
        annual_borrow_bps=annual_borrow_bps,
        annual_short_rebate_bps=annual_short_rebate_bps,
        annual_cash_financing_bps=annual_cash_financing_bps,
        periods_per_year=periods_per_year,
    )
    return float(gross - trade_cost - financing), weights


def collect_inner_fold_diagnostics(
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
    long_count: int = ML_LONG_COUNT,
    short_count: int = ML_SHORT_COUNT,
    gross_exposure: float = RESEARCH_LONG_SHORT_GROSS_EXPOSURE,
    transaction_cost: float = TRANSACTION_COST,
    annual_borrow_bps: float = ANNUAL_BORROW_BPS,
    annual_short_rebate_bps: float = ANNUAL_SHORT_REBATE_BPS,
    annual_cash_financing_bps: float = ANNUAL_CASH_FINANCING_BPS,
    periods_per_year: float = PERIODS_PER_YEAR,
    vol_targeting: bool | None = None,
) -> dict[str, list[float]]:
    """Pool per-date ICs and cost-aware net L/S returns across inner folds.

    Inner validation years are consecutive, so the candidate's net-return
    history is a single causal series used for vol-target scaling. It starts
    empty for every candidate and only ever contains inner-validation dates
    that precede the date being sized.
    """
    use_vol = (
        NESTED_INNER_VOL_TARGETING if vol_targeting is None else bool(vol_targeting)
    ) and VOL_TARGETING_ENABLED
    frame = month_end_subset(train_df) if use_month_end_subset else train_df
    splits = inner_anchored_year_splits(
        frame["date"],
        min_train_years=min_train_years,
        purge_months=purge_months,
        embargo_months=embargo_months,
    )
    ics: list[float] = []
    net_returns: list[float] = []
    vol_scales: list[float] = []
    previous_weights: pd.Series | None = None

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
        model = fit_estimator(
            candidate["kind"],
            candidate["params"],
            inner_train[feature_columns],
            inner_train[target_column],
        )
        scored = inner_valid.copy()
        scored["_nested_score"] = model.predict(inner_valid[feature_columns])

        for _date, group in scored.groupby("date"):
            if "ticker" in group.columns:
                keyed = group.drop_duplicates("ticker").set_index("ticker")
            else:
                keyed = group
            scores = keyed["_nested_score"]
            targets = keyed[target_column]

            ic = _date_rank_ic(scores, targets)
            if np.isfinite(ic):
                ics.append(ic)

            vol_scale = (
                volatility_target_scale(net_returns, enabled=True) if use_vol else 1.0
            )
            net, previous_weights = net_long_short_period_return(
                scores,
                targets,
                previous_weights=previous_weights,
                long_count=long_count,
                short_count=short_count,
                gross_exposure=gross_exposure,
                transaction_cost=transaction_cost,
                annual_borrow_bps=annual_borrow_bps,
                annual_short_rebate_bps=annual_short_rebate_bps,
                annual_cash_financing_bps=annual_cash_financing_bps,
                periods_per_year=periods_per_year,
                vol_scale=vol_scale,
            )
            if np.isfinite(net):
                net_returns.append(net)
                vol_scales.append(float(vol_scale))

    return {"ics": ics, "net_returns": net_returns, "vol_scales": vol_scales}


def summarize_inner_score(
    diagnostics: dict[str, list[float]],
    score_metric: str,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> float:
    metric = str(score_metric).strip().lower()
    if metric not in SUPPORTED_INNER_SCORES:
        raise ValueError(
            f"Unsupported nested score_metric={score_metric!r}; "
            f"expected one of {SUPPORTED_INNER_SCORES}"
        )
    ics = diagnostics.get("ics", [])
    nets = diagnostics.get("net_returns", [])
    if metric == "rank_ic":
        arr = [x for x in ics if np.isfinite(x)]
        return float(np.mean(arr)) if arr else float("nan")
    if metric == "icir":
        return aggregate_icir(ics)
    return aggregate_sharpe(nets, periods_per_year=periods_per_year)


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
    score_metric: str = NESTED_INNER_SCORE,
    long_count: int = ML_LONG_COUNT,
    short_count: int = ML_SHORT_COUNT,
    vol_targeting: bool | None = None,
) -> float:
    """Scalar inner score for one candidate (NaN if no usable diagnostics)."""
    diagnostics = collect_inner_fold_diagnostics(
        train_df,
        feature_columns=feature_columns,
        target_column=target_column,
        candidate=candidate,
        label_horizon_days=label_horizon_days,
        purge_months=purge_months,
        embargo_months=embargo_months,
        min_train_years=min_train_years,
        use_month_end_subset=use_month_end_subset,
        long_count=long_count,
        short_count=short_count,
        vol_targeting=vol_targeting,
    )
    return summarize_inner_score(diagnostics, score_metric)


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
    score_metric: str | None = None,
    long_count: int = ML_LONG_COUNT,
    short_count: int = ML_SHORT_COUNT,
    vol_targeting: bool | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Choose a candidate using only ``train_df``, then refit on all of it."""
    candidates = deepcopy(candidates or default_nested_candidates())
    if not candidates:
        raise ValueError("candidates must be non-empty")
    metric = NESTED_INNER_SCORE if score_metric is None else str(score_metric)

    score_rows = []
    for candidate in candidates:
        diagnostics = collect_inner_fold_diagnostics(
            train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            candidate=candidate,
            label_horizon_days=label_horizon_days,
            purge_months=purge_months,
            embargo_months=embargo_months,
            min_train_years=min_train_years,
            use_month_end_subset=use_month_end_subset,
            long_count=long_count,
            short_count=short_count,
            vol_targeting=vol_targeting,
        )
        score = summarize_inner_score(diagnostics, metric)
        score_rows.append({
            "name": candidate["name"],
            "inner_score": score,
            "mean_inner_rank_ic": float(np.mean(diagnostics["ics"])) if diagnostics["ics"] else float("nan"),
            "inner_icir": aggregate_icir(diagnostics["ics"]),
            "inner_cost_aware_sharpe": aggregate_sharpe(diagnostics["net_returns"]),
            "n_inner_ic_dates": len(diagnostics["ics"]),
            "n_inner_return_dates": len(diagnostics["net_returns"]),
            "mean_inner_vol_scale": (
                float(np.mean(diagnostics["vol_scales"]))
                if diagnostics.get("vol_scales")
                else float("nan")
            ),
        })

    finite = [row for row in score_rows if np.isfinite(row["inner_score"])]
    if finite:
        best_name = max(finite, key=lambda row: row["inner_score"])["name"]
        selection_mode = f"nested_{metric}"
    else:
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
        "score_metric": metric,
        "inner_vol_targeting": bool(
            (NESTED_INNER_VOL_TARGETING if vol_targeting is None else vol_targeting)
            and VOL_TARGETING_ENABLED
        ),
        "candidate_scores": score_rows,
        "n_train_rows": int(len(train_df)),
        "train_start": str(pd.to_datetime(train_df["date"]).min().date()) if len(train_df) else None,
        "train_end": str(pd.to_datetime(train_df["date"]).max().date()) if len(train_df) else None,
    }
    return model, meta
