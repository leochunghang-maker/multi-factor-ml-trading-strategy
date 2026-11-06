import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def decile_spread_analysis(
    predictions: pd.DataFrame,
    date_column: str,
    score_column: str,
    target_column: str,
    deciles: int = 10,
) -> pd.DataFrame:
    rows = []
    for date, group in predictions.groupby(date_column):
        valid = group[[score_column, target_column]].dropna()
        if valid[score_column].nunique() < deciles or len(valid) < deciles:
            continue
        valid = valid.copy()
        valid["decile"] = pd.qcut(
            valid[score_column],
            q=deciles,
            labels=False,
            duplicates="drop",
        ) + 1
        decile_returns = valid.groupby("decile")[target_column].mean()
        if 1 in decile_returns.index and decile_returns.index.max() in decile_returns.index:
            rows.append({
                date_column: date,
                "top_decile_return": decile_returns.loc[decile_returns.index.max()],
                "bottom_decile_return": decile_returns.loc[1],
                "top_minus_bottom": decile_returns.loc[decile_returns.index.max()]
                - decile_returns.loc[1],
                "rank_ic": spearmanr(valid[score_column], valid[target_column]).correlation,
                "asset_count": len(valid),
            })
    return pd.DataFrame(rows)


def feature_stability_report(
    frame: pd.DataFrame,
    date_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        by_date = frame.groupby(date_column)[feature]
        rows.append({
            "feature": feature,
            "average_cross_sectional_mean": by_date.mean().mean(),
            "average_cross_sectional_std": by_date.std().mean(),
            "missing_rate": frame[feature].isna().mean(),
            "time_series_std_of_mean": by_date.mean().std(),
            "min_value": frame[feature].min(),
            "max_value": frame[feature].max(),
        })
    return pd.DataFrame(rows)


def prediction_stability_report(
    predictions: pd.DataFrame,
    date_column: str,
    symbol_column: str,
    score_column: str,
) -> pd.DataFrame:
    ranks = predictions[[date_column, symbol_column, score_column]].dropna().copy()
    ranks["rank"] = ranks.groupby(date_column)[score_column].rank(pct=True)
    rank_matrix = ranks.pivot(index=date_column, columns=symbol_column, values="rank")
    rank_autocorr = rank_matrix.corrwith(rank_matrix.shift(1), axis=1).dropna()
    return pd.DataFrame({
        "average_rank_autocorrelation": [rank_autocorr.mean() if not rank_autocorr.empty else np.nan],
        "latest_rank_autocorrelation": [rank_autocorr.iloc[-1] if not rank_autocorr.empty else np.nan],
        "rank_autocorrelation_observations": [len(rank_autocorr)],
    })


def purged_embargo_year_splits(
    dates: pd.Series,
    test_years: list[int],
    purge_months: int = 1,
    embargo_months: int = 1,
) -> list[dict[str, pd.Timestamp | int]]:
    """Build year-based walk-forward splits with purge and embargo gaps.

    Sample construction (research hygiene):
    - Test fold = all observations whose date falls in the test calendar year.
    - Purge: drop training rows dated after ``test_start - purge_months`` so a
      trailing label window (e.g. 21 trading days / ~1 month) cannot overlap the
      test period.
    - Embargo: after the test fold ends, wait ``embargo_months`` before those
      dates may re-enter a later training set (expanding walk-forward that only
      trains on the past already respects this by construction).
    """
    unique_dates = pd.Series(pd.to_datetime(dates).dropna().unique()).sort_values()
    splits = []
    for year in test_years:
        test_dates = unique_dates[unique_dates.dt.year == year]
        if test_dates.empty:
            continue
        test_start = pd.Timestamp(test_dates.min())
        test_end = pd.Timestamp(test_dates.max())
        train_end = test_start - pd.DateOffset(months=purge_months)
        next_train_start = test_end + pd.DateOffset(months=embargo_months)
        splits.append({
            "test_year": int(year),
            "train_end_before_purge": pd.Timestamp(train_end),
            "test_start": test_start,
            "test_end": test_end,
            "embargo_lift_date": pd.Timestamp(next_train_start),
        })
    return splits


def training_rows_without_label_overlap(
    frame: pd.DataFrame,
    date_column: str,
    train_end_before_purge: pd.Timestamp,
    test_start: pd.Timestamp,
    label_horizon_days: int = 21,
) -> pd.DataFrame:
    """Keep training rows whose label window ends strictly before ``test_start``.

    Rows after ``train_end_before_purge`` are purged. Rows whose
    ``date + label_horizon_days`` still reaches into the test period are also
    removed so overlapping forward-return labels cannot leak.
    """
    dates = pd.to_datetime(frame[date_column])
    label_end = dates + pd.Timedelta(days=int(label_horizon_days))
    mask = (dates <= pd.Timestamp(train_end_before_purge)) & (
        label_end < pd.Timestamp(test_start)
    )
    return frame.loc[mask].copy()


def assert_no_label_overlap(
    train_dates: pd.Series,
    test_start: pd.Timestamp,
    label_horizon_days: int = 21,
) -> None:
    """Raise if any training label window overlaps the test period."""
    if train_dates.empty:
        return
    latest_label_end = pd.to_datetime(train_dates).max() + pd.Timedelta(
        days=int(label_horizon_days)
    )
    if latest_label_end >= pd.Timestamp(test_start):
        raise AssertionError(
            "training label window overlaps test period: "
            f"latest_label_end={latest_label_end.date()} test_start={pd.Timestamp(test_start).date()}"
        )


def month_end_subset(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Keep the last observation date in each calendar month (faster nested CV)."""
    if frame.empty:
        return frame.copy()
    dates = pd.to_datetime(frame[date_column])
    month_ends = (
        pd.Series(dates.unique(), index=pd.to_datetime(dates.unique()))
        .groupby(lambda d: d.to_period("M"))
        .max()
    )
    keep = set(pd.to_datetime(month_ends.values))
    return frame.loc[dates.isin(keep)].copy()


def inner_anchored_year_splits(
    dates: pd.Series,
    *,
    min_train_years: int = 2,
    purge_months: int = 1,
    embargo_months: int = 1,
) -> list[dict[str, pd.Timestamp | int]]:
    """Expanding (anchored) year folds strictly inside an outer training window.

    The earliest ``min_train_years`` calendar years are used only for training;
    later years become successive inner test folds. Purge/embargo gaps match the
    outer walk-forward hygiene so inner selection cannot peek across label
    horizons either.
    """
    years = sorted({int(y) for y in pd.to_datetime(dates).dropna().dt.year.unique()})
    if len(years) < int(min_train_years) + 1:
        return []
    test_years = years[int(min_train_years) :]
    return purged_embargo_year_splits(
        dates,
        test_years=test_years,
        purge_months=purge_months,
        embargo_months=embargo_months,
    )


def mean_rank_ic(
    frame: pd.DataFrame,
    *,
    date_column: str,
    score_column: str,
    target_column: str,
) -> float:
    """Average cross-sectional Spearman IC across dates (NaNs skipped)."""
    ics = []
    for _, group in frame.groupby(date_column):
        valid = group[[score_column, target_column]].dropna()
        if len(valid) < 5 or valid[score_column].nunique() < 2:
            continue
        corr = spearmanr(valid[score_column], valid[target_column]).correlation
        if corr is not None and np.isfinite(corr):
            ics.append(float(corr))
    if not ics:
        return float("nan")
    return float(np.mean(ics))
