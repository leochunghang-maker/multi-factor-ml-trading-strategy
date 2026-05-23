import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    REPORT_DRAWDOWN_PATH,
    REPORT_EQUITY_CURVE_PATH,
    REPORT_MONTHLY_RETURN_HEATMAP_PATH,
    REPORT_ROLLING_SHARPE_PATH,
    REPORT_ROLLING_VOLATILITY_PATH,
)
from src.reporting.analytics import drawdown, rolling_sharpe_ratio, rolling_volatility


def _apply_chart_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 160,
    })


def save_equity_curve(
    curve: pd.Series,
    path: str,
    title: str,
    ylabel: str = "Portfolio Value",
) -> None:
    _apply_chart_style()
    plt.figure(figsize=(12, 6))
    plt.plot(curve, linewidth=2.0)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_series_chart(
    series: pd.Series,
    path: str,
    title: str,
    ylabel: str,
) -> None:
    _apply_chart_style()
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, linewidth=2.0)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_risk_report_charts(returns: pd.Series) -> None:
    curve = (1 + returns).cumprod()
    save_equity_curve(
        curve,
        REPORT_EQUITY_CURVE_PATH,
        "Strategy Equity Curve",
    )
    save_series_chart(
        drawdown(returns),
        REPORT_DRAWDOWN_PATH,
        "Strategy Drawdown",
        "Drawdown",
    )
    save_series_chart(
        rolling_sharpe_ratio(returns),
        REPORT_ROLLING_SHARPE_PATH,
        "Rolling Sharpe Ratio",
        "Sharpe Ratio",
    )
    save_series_chart(
        rolling_volatility(returns),
        REPORT_ROLLING_VOLATILITY_PATH,
        "Rolling Volatility",
        "Annualized Volatility",
    )
    save_monthly_return_heatmap(returns, REPORT_MONTHLY_RETURN_HEATMAP_PATH)


def save_monthly_return_heatmap(
    returns: pd.Series,
    path: str,
    title: str = "Monthly Returns",
) -> None:
    monthly = returns.dropna().copy()
    if monthly.empty:
        return
    _apply_chart_style()
    heatmap = monthly.to_frame("return")
    heatmap["year"] = heatmap.index.year
    heatmap["month"] = heatmap.index.month
    pivot = heatmap.pivot(index="year", columns="month", values="return")
    pivot = pivot.reindex(columns=range(1, 13))

    plt.figure(figsize=(12, max(3, 0.45 * len(pivot))))
    data = pivot.to_numpy()
    limit = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 0.01
    limit = max(limit, 0.01)
    plt.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
    plt.colorbar(label="Return")
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.yticks(range(len(pivot.index)), pivot.index)

    for i, year in enumerate(pivot.index):
        for j, month in enumerate(pivot.columns):
            value = pivot.loc[year, month]
            if pd.notna(value):
                plt.text(j, i, f"{value:.1%}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_strategy_vs_benchmark(
    strategy_curve: pd.Series,
    benchmark_curve: pd.Series,
    path: str,
    title: str,
    strategy_label: str = "Strategy",
    benchmark_label: str = "SPY Benchmark",
    ylabel: str = "Cumulative Return",
) -> None:
    _apply_chart_style()
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_curve.index, strategy_curve.values, label=strategy_label, linewidth=2.0)
    plt.plot(benchmark_curve.index, benchmark_curve.values, label=benchmark_label, linewidth=2.0)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
