import numpy as np
import pandas as pd

from src.config import (
    MAX_GROSS_EXPOSURE,
    MAX_POSITION_WEIGHT,
    MAX_SECTOR_WEIGHT,
    SECTOR_MAP,
)


def gross_exposure(weights: pd.DataFrame) -> pd.Series:
    return weights.abs().sum(axis=1)


def net_exposure(weights: pd.DataFrame) -> pd.Series:
    return weights.sum(axis=1)


def leverage(weights: pd.DataFrame, equity: float = 1.0) -> pd.Series:
    if equity <= 0:
        raise ValueError("Equity must be positive when calculating leverage.")
    return gross_exposure(weights) / equity


def long_exposure(weights: pd.DataFrame) -> pd.Series:
    return weights.clip(lower=0).sum(axis=1)


def short_exposure(weights: pd.DataFrame) -> pd.Series:
    return weights.clip(upper=0).sum(axis=1)


def max_position_concentration(weights: pd.DataFrame) -> pd.Series:
    return weights.abs().max(axis=1)


def concentration_ratio(weights: pd.DataFrame, top_n: int = 5) -> pd.Series:
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    return weights.abs().apply(lambda row: row.nlargest(top_n).sum(), axis=1)


def sector_exposures(
    weights: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    sectors = sector_map or SECTOR_MAP
    renamed = weights.copy()
    renamed.columns = [sectors.get(symbol, "Unknown") for symbol in renamed.columns]
    return renamed.T.groupby(level=0).sum().T.sort_index(axis=1)


def top_holdings(weights: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for date, row in weights.iterrows():
        for rank, (symbol, weight) in enumerate(row.abs().nlargest(top_n).items(), start=1):
            rows.append({
                "date": date,
                "rank": rank,
                "ticker": symbol,
                "weight": row[symbol],
                "absolute_weight": abs(row[symbol]),
                "sector": SECTOR_MAP.get(symbol, "Unknown"),
            })
    return pd.DataFrame(rows)


def top_holdings_label(weights: pd.DataFrame, top_n: int = 5) -> pd.Series:
    labels = {}
    for date, row in weights.iterrows():
        parts = [
            f"{symbol}:{row[symbol]:.2%}"
            for symbol in row.abs().nlargest(top_n).index
            if abs(row[symbol]) > 0
        ]
        labels[date] = "; ".join(parts)
    return pd.Series(labels)


def factor_exposure_summary(
    weights: pd.DataFrame,
    factor_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for date, row in weights.iterrows():
        output = {"date": date}
        active = row[row != 0]
        for name, factors in factor_frames.items():
            if date not in factors.index or active.empty:
                output[f"{name}_exposure"] = np.nan
                continue
            aligned_factor = factors.loc[date].reindex(active.index)
            output[f"{name}_exposure"] = (active * aligned_factor).sum()
        rows.append(output)
    return pd.DataFrame(rows).set_index("date")


def portfolio_beta_exposure(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 252,
) -> pd.Series:
    aligned_benchmark = benchmark_returns.dropna()
    beta_rows = {}
    for date, row in weights.iterrows():
        history = asset_returns.loc[asset_returns.index <= date].tail(window)
        benchmark_history = aligned_benchmark.loc[aligned_benchmark.index <= date].tail(window)
        common_dates = history.index.intersection(benchmark_history.index)
        if len(common_dates) < max(20, window // 4):
            beta_rows[date] = np.nan
            continue
        history = history.loc[common_dates]
        benchmark = benchmark_history.loc[common_dates]
        benchmark_variance = benchmark.var()
        if benchmark_variance == 0 or np.isnan(benchmark_variance):
            beta_rows[date] = np.nan
            continue
        covariances = {}
        for symbol in history.columns:
            pair = pd.concat([history[symbol], benchmark], axis=1).dropna()
            covariances[symbol] = pair.iloc[:, 0].cov(pair.iloc[:, 1]) if len(pair) >= 2 else np.nan
        asset_betas = pd.Series(covariances) / benchmark_variance
        beta_rows[date] = (row.reindex(asset_betas.index).fillna(0.0) * asset_betas).sum()
    return pd.Series(beta_rows, name="portfolio_beta_exposure")


def exposure_report(weights: pd.DataFrame) -> pd.DataFrame:
    tolerance = 1e-9
    sectors = sector_exposures(weights)
    max_sector = sectors.abs().max(axis=1) if not sectors.empty else np.nan
    report = pd.DataFrame(index=weights.index)
    report["gross_exposure"] = gross_exposure(weights)
    report["net_exposure"] = net_exposure(weights)
    report["long_exposure"] = long_exposure(weights)
    report["short_exposure"] = short_exposure(weights)
    report["leverage"] = leverage(weights)
    report["max_position_weight"] = max_position_concentration(weights)
    report["top_5_concentration"] = concentration_ratio(weights, top_n=5)
    report["top_holdings"] = top_holdings_label(weights, top_n=5)
    report["max_abs_sector_exposure"] = max_sector
    report["position_limit_breach"] = (
        report["max_position_weight"] > MAX_POSITION_WEIGHT + tolerance
    )
    report["sector_limit_breach"] = (
        report["max_abs_sector_exposure"] > MAX_SECTOR_WEIGHT + tolerance
    )
    report["gross_limit_breach"] = report["gross_exposure"] > MAX_GROSS_EXPOSURE + tolerance
    return report


def latest_exposure_summary(weights: pd.DataFrame) -> dict[str, float | bool]:
    if weights.empty:
        return {}
    latest = exposure_report(weights).iloc[-1]
    return latest.to_dict()
