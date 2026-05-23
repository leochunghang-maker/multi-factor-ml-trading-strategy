from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dashboard import charts
from dashboard.data import (
    load_execution_log,
    load_live_signals,
    load_portfolio_exposure,
    load_position_snapshot,
    load_rolling_risk,
    load_sector_exposure,
    load_strategy_returns,
    load_summary_metrics,
    load_trade_history,
    load_turnover,
)


st.set_page_config(
    page_title="Institutional Quant Dashboard",
    page_icon=None,
    layout="wide",
)


def format_metric(value: float | int | None, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if percent:
        return f"{value:.2%}"
    return f"{value:.3f}"


def metric_grid(metrics: dict[str, float]) -> None:
    # Summary metrics are the dashboard's first control panel. They compress
    # return, risk, market exposure, and implementation intensity into one scan.
    columns = st.columns(4)
    metric_specs = [
        ("Annual Return", "annual_return", True, "Compounded edge matters, but only after risk and costs are understood."),
        ("Annual Volatility", "annual_volatility", True, "Volatility estimates the normal range of portfolio fluctuation."),
        ("Sharpe Ratio", "sharpe_ratio", False, "Sharpe measures return per unit of volatility risk."),
        ("Max Drawdown", "max_drawdown", True, "Drawdown measures the worst capital loss from a previous high."),
        ("Turnover", "average_one_way_turnover", True, "Turnover shows how much of the book changes at each rebalance."),
        ("Beta", "beta", False, "Beta shows market sensitivity versus SPY."),
        ("Gross Exposure", "gross_exposure", True, "Gross exposure measures total long plus short risk deployed."),
        ("Net Exposure", "net_exposure", True, "Net exposure shows directional long or short market posture."),
    ]
    for index, (label, key, percent, help_text) in enumerate(metric_specs):
        with columns[index % 4]:
            st.metric(label, format_metric(metrics.get(key), percent), help=help_text)


def show_table(title: str, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    st.subheader(title)
    if frame.empty:
        st.info("No data available for this section.")
        return
    display = frame[columns] if columns else frame
    st.dataframe(display, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Institutional Quantitative Trading Dashboard")
    st.caption(
        "Local research and paper-simulation reporting only. This dashboard does not place orders or enable live-money trading."
    )

    returns = load_strategy_returns()
    rolling_risk = load_rolling_risk()
    turnover = load_turnover()
    exposure = load_portfolio_exposure()
    sector = load_sector_exposure()
    signals = load_live_signals()
    execution_log = load_execution_log()
    trade_history = load_trade_history()
    positions = load_position_snapshot()
    metrics = load_summary_metrics()

    metric_grid(metrics)

    st.divider()
    st.header("Performance And Risk")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.equity_curve_chart(returns), use_container_width=True)
        st.plotly_chart(
            charts.rolling_metric_chart(
                rolling_risk,
                "rolling_sharpe",
                "Rolling Sharpe Ratio",
                "Sharpe Ratio",
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(charts.drawdown_chart(returns), use_container_width=True)
        st.plotly_chart(
            charts.rolling_metric_chart(
                rolling_risk,
                "rolling_volatility",
                "Rolling Volatility",
                "Annualized Volatility",
            ),
            use_container_width=True,
        )

    st.header("Portfolio Exposures")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.turnover_chart(turnover), use_container_width=True)
        st.plotly_chart(charts.top_holdings_chart(exposure), use_container_width=True)
    with right:
        st.plotly_chart(charts.sector_exposure_chart(sector), use_container_width=True)
        st.plotly_chart(charts.beta_exposure_chart(exposure), use_container_width=True)

    st.header("Signals And Allocation")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.signal_rankings_chart(signals), use_container_width=True)
    with right:
        st.plotly_chart(charts.allocation_chart(positions), use_container_width=True)

    st.header("Execution Review")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.execution_history_chart(execution_log), use_container_width=True)
    with right:
        show_table(
            "Recent Trades",
            trade_history.tail(25).sort_index(ascending=False),
            [
                column
                for column in [
                    "timestamp",
                    "symbol",
                    "side",
                    "quantity",
                    "decision_price",
                    "fill_price",
                    "notional",
                    "transaction_cost",
                    "status",
                ]
                if column in trade_history.columns
            ],
        )

    st.header("Audit Tables")
    tabs = st.tabs(["Exposure", "Turnover", "Signals", "Execution Log"])
    with tabs[0]:
        show_table("Portfolio Exposure Report", exposure.tail(20).sort_index(ascending=False))
    with tabs[1]:
        show_table("Turnover Report", turnover.tail(20).sort_index(ascending=False))
    with tabs[2]:
        show_table(
            "Latest Signals",
            signals.sort_values("signal_score", ascending=False) if "signal_score" in signals.columns else signals,
        )
    with tabs[3]:
        show_table("Execution Log", execution_log.tail(50).sort_index(ascending=False))


if __name__ == "__main__":
    main()
