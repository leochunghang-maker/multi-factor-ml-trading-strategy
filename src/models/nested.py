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
  Short borrow uses the same point-in-time name-level HTB tier panel as the
  outer book when ``borrow_bps_panel`` is supplied (row as-of each inner date).

Selection rule (``NESTED_SELECTION_RULE``), fixed a priori — not tuned on outer
results:

- ``one_se_fallback`` (default): keep the pre-declared default model
  (``NESTED_DEFAULT_MODEL``, the production ``rf_default``) unless the best
  inner candidate beats it by more than ``NESTED_SELECTION_SE_THRESHOLD``
  standard errors of the *paired* per-date difference (net returns for
  ``cost_aware_sharpe``, ICs for ``rank_ic`` / ``icir``). This is the one-SE
  rule from cross-validation: inner score gaps inside sampling noise are not
  evidence that a challenger is better.
- ``best``: legacy argmax of the inner score.
- ``ensemble_equal``: equal-weight blend of every candidate refit on the full
  outer train set, each prediction standardized by its own *training-set*
  prediction mean/std (no test-fold statistics).
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
    NESTED_DEFAULT_MODEL,
    NESTED_INNER_SCORE,
    NESTED_INNER_VOL_TARGETING,
    NESTED_SELECTION_RULE,
    NESTED_SELECTION_SE_THRESHOLD,
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
    _borrow_bps_on_date,
    equal_weight_long_short_weights,
    transaction_cost_from_weight_change,
)
from src.risk.volatility import scale_weights, volatility_target_scale

SUPPORTED_INNER_SCORES = ("rank_ic", "icir", "cost_aware_sharpe")
SUPPORTED_SELECTION_RULES = ("one_se_fallback", "best", "ensemble_equal")


class StandardizedEnsemble:
    """Equal-weight blend of fitted models with train-set standardization.

    Each member's predictions are z-scored with the mean/std of its own
    predictions on the training rows it was fit on, so blending never uses
    statistics from the rows being predicted.
    """

    def __init__(self, members: list[tuple[str, Any]], X_train: pd.DataFrame):
        if not members:
            raise ValueError("ensemble needs at least one member")
        self.members = members
        self.names = [name for name, _ in members]
        self.stats: list[tuple[float, float]] = []
        for _, model in members:
            preds = np.asarray(model.predict(X_train), dtype=float)
            std = float(np.nanstd(preds))
            self.stats.append((float(np.nanmean(preds)), std if std > 1e-12 else 1.0))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        blended = np.zeros(len(X), dtype=float)
        for (_, model), (mu, sd) in zip(self.members, self.stats):
            blended += (np.asarray(model.predict(X), dtype=float) - mu) / sd
        return blended / len(self.members)


def paired_difference_tstat(
    challenger: dict[Any, float],
    default: dict[Any, float],
) -> tuple[float, float, int]:
    """Mean paired difference, its standard error, and n over shared dates."""
    shared = [d for d in challenger if d in default]
    diffs = np.asarray(
        [challenger[d] - default[d] for d in shared
         if np.isfinite(challenger[d]) and np.isfinite(default[d])],
        dtype=float,
    )
    if diffs.size < 2:
        return float("nan"), float("nan"), int(diffs.size)
    se = float(diffs.std(ddof=1) / np.sqrt(diffs.size))
    return float(diffs.mean()), se, int(diffs.size)


def one_se_choice(
    score_rows: list[dict[str, Any]],
    per_date: dict[str, dict[Any, float]],
    *,
    default_name: str,
    se_threshold: float,
) -> tuple[str, str, dict[str, float]]:
    """Pick the best inner candidate only if it clears the default by > k·SE.

    Returns ``(chosen_name, selection_mode, stats)`` where stats holds the
    paired mean difference / SE / n for the best challenger vs the default.
    """
    finite = [row for row in score_rows if np.isfinite(row["inner_score"])]
    names = [row["name"] for row in score_rows]
    stats = {"best_challenger": None, "paired_mean_diff": float("nan"),
             "paired_se": float("nan"), "paired_n": 0}
    if default_name not in names:
        # Custom menus without the declared default: no anchor, so use argmax.
        if finite:
            return max(finite, key=lambda row: row["inner_score"])["name"], "one_se_no_default_argmax", stats
        return names[-1], "fallback_no_inner_scores", stats
    if not finite:
        return default_name, "fallback_no_inner_scores", stats
    best = max(finite, key=lambda row: row["inner_score"])
    if best["name"] == default_name:
        return default_name, "one_se_default_is_best", stats
    mean_diff, se, n = paired_difference_tstat(
        per_date.get(best["name"], {}), per_date.get(default_name, {})
    )
    stats.update({"best_challenger": best["name"], "paired_mean_diff": mean_diff,
                  "paired_se": se, "paired_n": n})
    if np.isfinite(mean_diff) and np.isfinite(se) and mean_diff > se_threshold * se:
        return best["name"], "one_se_challenger_significant", stats
    return default_name, "one_se_fallback_to_default", stats


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
    borrow_bps_by_symbol: pd.Series | None = None,
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
        borrow_bps_by_symbol=borrow_bps_by_symbol,
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
    borrow_bps_panel: pd.DataFrame | None = None,
) -> dict[str, list[float]]:
    """Pool per-date ICs and cost-aware net L/S returns across inner folds.

    Inner validation years are consecutive, so the candidate's net-return
    history is a single causal series used for vol-target scaling. It starts
    empty for every candidate and only ever contains inner-validation dates
    that precede the date being sized. ``borrow_bps_panel`` (date × ticker,
    built point-in-time) supplies name-level borrow; rows are looked up as-of
    each inner date, so only information available then is used.
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
    ic_by_date: dict[Any, float] = {}
    net_by_date: dict[Any, float] = {}
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
            # Panel rows on/before the inner date only (as-of lookup).
            name_borrow = (
                _borrow_bps_on_date(borrow_bps_panel, _date, group["ticker"].drop_duplicates())
                if borrow_bps_panel is not None and "ticker" in group.columns
                else None
            )
            if name_borrow is not None:
                name_borrow.index = name_borrow.index.astype(str)
            if "ticker" in group.columns:
                keyed = group.drop_duplicates("ticker").set_index("ticker")
            else:
                keyed = group
            scores = keyed["_nested_score"]
            targets = keyed[target_column]

            ic = _date_rank_ic(scores, targets)
            if np.isfinite(ic):
                ics.append(ic)
                ic_by_date[pd.Timestamp(_date)] = ic

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
                borrow_bps_by_symbol=name_borrow,
            )
            if np.isfinite(net):
                net_returns.append(net)
                vol_scales.append(float(vol_scale))
                net_by_date[pd.Timestamp(_date)] = net

    return {
        "ics": ics,
        "net_returns": net_returns,
        "vol_scales": vol_scales,
        "ic_by_date": ic_by_date,
        "net_by_date": net_by_date,
    }


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
    borrow_bps_panel: pd.DataFrame | None = None,
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
        borrow_bps_panel=borrow_bps_panel,
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
    borrow_bps_panel: pd.DataFrame | None = None,
    selection_rule: str | None = None,
    default_model: str | None = None,
    se_threshold: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Choose a candidate (or ensemble) using only ``train_df``, then refit on all of it."""
    candidates = deepcopy(candidates or default_nested_candidates())
    if not candidates:
        raise ValueError("candidates must be non-empty")
    metric = NESTED_INNER_SCORE if score_metric is None else str(score_metric)
    rule = str(NESTED_SELECTION_RULE if selection_rule is None else selection_rule).strip().lower()
    if rule not in SUPPORTED_SELECTION_RULES:
        raise ValueError(
            f"Unsupported nested selection_rule={rule!r}; expected one of {SUPPORTED_SELECTION_RULES}"
        )
    default_name = NESTED_DEFAULT_MODEL if default_model is None else str(default_model)
    k_se = float(NESTED_SELECTION_SE_THRESHOLD if se_threshold is None else se_threshold)
    per_date: dict[str, dict[Any, float]] = {}

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
            borrow_bps_panel=borrow_bps_panel,
        )
        score = summarize_inner_score(diagnostics, metric)
        per_date[candidate["name"]] = (
            diagnostics["net_by_date"]
            if metric.strip().lower() == "cost_aware_sharpe"
            else diagnostics["ic_by_date"]
        )
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

    names = [c["name"] for c in candidates]
    finite = [row for row in score_rows if np.isfinite(row["inner_score"])]
    se_stats = {"best_challenger": None, "paired_mean_diff": float("nan"),
                "paired_se": float("nan"), "paired_n": 0}
    X_train, y_train = train_df[feature_columns], train_df[target_column]
    if rule == "ensemble_equal":
        members = [
            (c["name"], fit_estimator(c["kind"], c["params"], X_train, y_train))
            for c in candidates
        ]
        model = StandardizedEnsemble(members, X_train)
        best_name = "ensemble_equal(" + "+".join(names) + ")"
        selection_mode = "ensemble_equal"
    else:
        if rule == "one_se_fallback":
            best_name, selection_mode, se_stats = one_se_choice(
                score_rows, per_date, default_name=default_name, se_threshold=k_se
            )
        elif finite:
            best_name = max(finite, key=lambda row: row["inner_score"])["name"]
            selection_mode = f"nested_{metric}"
        else:
            best_name = default_name if default_name in names else names[-1]
            selection_mode = "fallback_no_inner_scores"
        winner = next(c for c in candidates if c["name"] == best_name)
        model = fit_estimator(winner["kind"], winner["params"], X_train, y_train)
    meta = {
        "selection_rule": rule,
        "default_model": default_name,
        "se_threshold": k_se,
        "inner_best_model": (
            max(finite, key=lambda row: row["inner_score"])["name"] if finite else None
        ),
        "best_challenger": se_stats["best_challenger"],
        "paired_mean_diff": se_stats["paired_mean_diff"],
        "paired_se": se_stats["paired_se"],
        "paired_n": se_stats["paired_n"],
        "inner_name_level_borrow": borrow_bps_panel is not None,
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
