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
) -> list[dict[str, pd.Timestamp]]:
    unique_dates = pd.Series(pd.to_datetime(dates).dropna().unique()).sort_values()
    splits = []
    for year in test_years:
        test_dates = unique_dates[unique_dates.dt.year == year]
        if test_dates.empty:
            continue
        test_start = test_dates.min()
        test_end = test_dates.max()
        train_end = test_start - pd.DateOffset(months=purge_months)
        next_train_start = test_end + pd.DateOffset(months=embargo_months)
        splits.append({
            "test_year": year,
            "train_end_before_purge": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "embargo_lift_date": next_train_start,
        })
    return splits
