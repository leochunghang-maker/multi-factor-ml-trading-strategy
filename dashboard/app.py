from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dashboard import charts
from dashboard.data import (
    RESEARCH_BOOKS,
    book_headline_metrics,
    load_execution_log,
    load_live_signals,
    load_paper_journal,
    load_paper_status,
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
    page_title="Quant Research & Simulated Paper Dashboard",
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
    # Headline metrics of one research book, computed from its committed result
    # file exactly as README "Results Summary" quotes them (checked by doc metrics).
    columns = st.columns(6)
    metric_specs = [
        ("Annual Return", "annual_return", True, "Mean monthly net return x 12 (after modeled costs)."),
        ("Annual Volatility", "annual_volatility", True, "Monthly return standard deviation x sqrt(12)."),
        ("Sharpe Ratio", "sharpe_ratio", False, "Annual return / annual volatility. Not significant after deflation (see README)."),
        ("Max Drawdown", "max_drawdown", True, "Worst peak-to-trough loss of the net equity curve."),
        ("Cumulative Return", "cumulative_return", True, "Compounded net return over the whole sample."),
        ("Avg One-Way Turnover", "average_one_way_turnover", True,
         "Mean traded gross exposure / 2 per rebalance, after the first (rebalance log)."),
    ]
    for index, (label, key, percent, help_text) in enumerate(metric_specs):
        with columns[index % 6]:
            st.metric(label, format_metric(metrics.get(key), percent), help=help_text)


def show_table(title: str, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    st.subheader(title)
    if frame.empty:
        st.info("No data available for this section.")
        return
    display = frame[columns] if columns else frame
    st.dataframe(display, hide_index=True, key=f"table:{title}")


def main() -> None:
    st.title("Quant Research and Simulated Paper-Trading Dashboard")
    st.caption(
        "Local research backtests and a simulated paper account only: no broker connection, no orders, "
        "no live-money trading. No strategy shows statistically significant alpha after multiple-testing "
        "deflation; see README \"Results Summary\" for the significance tests."
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

    st.header("Research Backtests (headline metrics)")
    st.caption(
        "Month-end long/short books (point-in-time eligibility, beta-neutral legs), net of the flat research "
        "transaction cost on traded gross exposure and monthly short borrow; monthly returns annualized. "
        "Same values as README \"Results Summary\"."
    )
    for book, file_name in RESEARCH_BOOKS.items():
        book_metrics = book_headline_metrics(load_strategy_returns(file_name))
        months = book_metrics.get("months")
        st.subheader(f"{book} · {int(months)} months" if months else book)
        metric_grid(book_metrics)

    st.divider()
    st.header("Multi-Factor Backtest: Performance And Risk")
    st.caption(
        "From results/multi_factor_results.csv; rolling charts use a 12-month window "
        "(results/reports/, written by python src/generate_report.py). "
        f"Latest 12-month beta vs SPY: {format_metric(metrics.get('beta'))}."
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.equity_curve_chart(returns), key="equity_curve_chart")
        st.plotly_chart(
            charts.rolling_metric_chart(
                rolling_risk,
                "rolling_sharpe",
                "Rolling Sharpe Ratio",
                "Sharpe Ratio",
            ),
            key="rolling_sharpe",
        )
    with right:
        st.plotly_chart(charts.drawdown_chart(returns), key="drawdown_chart")
        st.plotly_chart(
            charts.rolling_metric_chart(
                rolling_risk,
                "rolling_volatility",
                "Rolling Volatility",
                "Annualized Volatility",
            ),
            key="rolling_volatility",
        )

    st.header("Multi-Factor Book: Turnover And Exposures")
    st.caption(
        "From results/reports/ (make report), built from the held weights of the reported multi-factor book "
        "(same construction as results/multi_factor_results.csv). \"Proposed\" is that research book; "
        "\"paper-constrained\" is the same weights after the paper guardrails (max gross, position, sector, "
        "turnover and liquidity limits), to compare research against paper implementability."
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.turnover_chart(turnover), key="turnover_chart")
        st.plotly_chart(charts.top_holdings_chart(exposure), key="top_holdings_chart")
    with right:
        st.plotly_chart(charts.sector_exposure_chart(sector), key="sector_exposure_chart")
        st.plotly_chart(charts.beta_exposure_chart(exposure), key="beta_exposure_chart")

    st.header("Simulated Paper Account: Signals And Allocation")
    st.caption(
        "Latest model signals (results/live/latest_signals.csv, the input to the simulated paper run) and the "
        "simulated broker's latest positions. \"live\" is only the file name: nothing is sent to a broker."
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.signal_rankings_chart(signals), key="signal_rankings_chart")
    with right:
        st.plotly_chart(charts.allocation_chart(positions), key="allocation_chart")

    st.header("Simulated Execution Review")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.execution_history_chart(execution_log), key="execution_history_chart")
    with right:
        show_table(
            "Recent Simulated Fills",
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
    tabs = st.tabs(["Exposure", "Turnover", "Signals", "Simulated Execution Log", "Paper Account (simulated)"])
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
        show_table("Simulated Execution Log", execution_log.tail(50).sort_index(ascending=False))
    with tabs[4]:
        show_paper_account(load_paper_status(), load_paper_journal())


def show_paper_account(status: dict, journal: pd.DataFrame) -> None:
    """Read-only view of results/paper/ (simulated paper account; see make paper-status)."""
    if not status:
        st.info("No paper account status yet. Run `make paper-daily` or `make paper-status`.")
        return
    st.caption("Simulated paper account (results/paper/); see make paper-status. No real orders or money.")
    st.markdown(f"**Overall status:** {status.get('overall_status')} · generated {status.get('generated_at')} "
                f"· completed runs {status.get('completed_runs')}")
    for alert in status.get("active_alerts", []):
        (st.error if alert.get("severity") == "ERROR" else st.warning)(f"{alert.get('check')}: {alert.get('message')}")
    if not journal.empty:
        # Explicit x column: an index-based chart makes Streamlit call the deprecated
        # DataFrame.reset_index(inplace=True) under pandas 3.1.
        st.line_chart(journal, x="signal_date", y="nav_end", height=220)
    show_table("Paper Journal", journal.tail(20).sort_index(ascending=False))
    show_table("Recent Alerts", pd.DataFrame(status.get("recent_alerts", [])),
               ["timestamp", "severity", "kind", "signal_date", "message"])


if __name__ == "__main__":
    main()
