import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TEMPLATE = "plotly_white"


def _missing(frame: pd.DataFrame, *columns: str) -> bool:
    return frame.empty or any(column not in frame.columns for column in columns)


def empty_chart(title: str, hint: str = "No data yet: run python src/generate_report.py (make report).") -> go.Figure:
    """Titled placeholder for a chart whose local result file has not been generated."""
    figure = go.Figure()
    figure.update_layout(title=title, template=TEMPLATE, xaxis={"visible": False}, yaxis={"visible": False},
                         annotations=[{"text": hint, "showarrow": False, "xref": "paper", "yref": "paper"}])
    return figure


def equity_curve_chart(returns: pd.DataFrame) -> go.Figure:
    # Equity curves show the compounded investor experience through time.
    # Institutions use them to inspect path dependency, not just final return.
    if _missing(returns, "date", "portfolio_return"):
        return empty_chart("Multi-Factor Backtest: Growth of $1 (net)", "results/multi_factor_results.csv not found.")
    frame = returns.dropna(subset=["date", "portfolio_return"]).copy()
    frame["equity_curve"] = (1 + frame["portfolio_return"]).cumprod()
    return px.line(
        frame,
        x="date",
        y="equity_curve",
        title="Multi-Factor Backtest: Growth of $1 (net)",
        template=TEMPLATE,
        labels={"equity_curve": "Growth of $1", "date": "Date"},
    )


def drawdown_chart(returns: pd.DataFrame) -> go.Figure:
    # Drawdown is the peak-to-trough loss investors must survive. It is often
    # more relevant to capital allocation than average return.
    if _missing(returns, "date", "portfolio_return"):
        return empty_chart("Multi-Factor Backtest: Drawdown", "results/multi_factor_results.csv not found.")
    frame = returns.dropna(subset=["date", "portfolio_return"]).copy()
    curve = (1 + frame["portfolio_return"]).cumprod()
    frame["drawdown"] = curve / curve.cummax() - 1
    return px.area(
        frame,
        x="date",
        y="drawdown",
        title="Multi-Factor Backtest: Drawdown",
        template=TEMPLATE,
        labels={"drawdown": "Drawdown", "date": "Date"},
    )


def rolling_metric_chart(frame: pd.DataFrame, column: str, title: str, ylabel: str) -> go.Figure:
    # Rolling risk charts reveal whether performance is stable or concentrated
    # in a short favorable regime.
    if _missing(frame, "date", column):
        return empty_chart(title)
    clean = frame.dropna(subset=["date", column])
    return px.line(
        clean,
        x="date",
        y=column,
        title=title,
        template=TEMPLATE,
        labels={column: ylabel, "date": "Date"},
    )


def turnover_chart(frame: pd.DataFrame) -> go.Figure:
    # Turnover measures trading intensity. High turnover can turn a good signal
    # into a poor implementable portfolio after costs and capacity limits.
    columns = [
        column
        for column in ["proposed_one_way_turnover", "constrained_one_way_turnover"]
        if column in frame.columns
    ]
    if not columns or _missing(frame, "date"):
        return empty_chart("Multi-Factor Book: One-Way Turnover (proposed vs paper-constrained)")
    clean = frame.dropna(subset=["date"])
    return px.line(
        clean,
        x="date",
        y=columns,
        title="Multi-Factor Book: One-Way Turnover (proposed vs paper-constrained)",
        template=TEMPLATE,
        labels={"value": "One-Way Turnover", "date": "Date", "variable": "Series"},
    )


def sector_exposure_chart(frame: pd.DataFrame) -> go.Figure:
    # Sector exposure identifies hidden common risk: many stocks can still be
    # one concentrated macro or industry bet.
    sector_columns = [column for column in frame.columns if column != "date"]
    if not sector_columns or _missing(frame, "date"):
        return empty_chart("Multi-Factor Book: Sector Net Exposure (paper-constrained)")
    clean = frame.dropna(subset=["date"])
    return px.area(
        clean,
        x="date",
        y=sector_columns,
        title="Multi-Factor Book: Sector Net Exposure (paper-constrained)",
        template=TEMPLATE,
        labels={"value": "Net Weight", "date": "Date", "variable": "Sector"},
    )


def top_holdings_chart(frame: pd.DataFrame) -> go.Figure:
    # Top holdings make concentration visible. Institutions watch this because
    # single-name shocks can dominate realized PnL.
    if _missing(frame, "date", "constrained_top_holdings"):
        return empty_chart("Multi-Factor Book: Top Holdings (paper-constrained, latest)")
    latest = frame.dropna(subset=["date"]).tail(1)
    if latest.empty:
        return empty_chart("Multi-Factor Book: Top Holdings (paper-constrained, latest)")
    holdings = []
    for item in str(latest["constrained_top_holdings"].iloc[0]).split("; "):
        if ":" not in item:
            continue
        ticker, value = item.split(":", 1)
        holdings.append({"ticker": ticker, "weight": float(value.strip("%")) / 100})
    holdings_frame = pd.DataFrame(holdings)
    if holdings_frame.empty:
        return empty_chart("Multi-Factor Book: Top Holdings (paper-constrained, latest)")
    return px.bar(
        holdings_frame,
        x="ticker",
        y="weight",
        title="Multi-Factor Book: Top Holdings (paper-constrained, latest)",
        template=TEMPLATE,
        labels={"ticker": "Ticker", "weight": "Weight"},
    )


def beta_exposure_chart(frame: pd.DataFrame) -> go.Figure:
    # Beta exposure shows broad market sensitivity. It helps separate stock
    # selection skill from simply being long or short the market.
    column = "constrained_beta_exposure"
    if _missing(frame, "date", column):
        return empty_chart("Multi-Factor Book: 36-Month Beta Exposure (paper-constrained)")
    clean = frame.dropna(subset=["date", column])
    return px.line(
        clean,
        x="date",
        y=column,
        title="Multi-Factor Book: 36-Month Beta Exposure (paper-constrained)",
        template=TEMPLATE,
        labels={column: "Beta", "date": "Date"},
    )


def signal_rankings_chart(frame: pd.DataFrame) -> go.Figure:
    # Signal rankings show what the model currently wants to own. This is a
    # useful sanity check before orders or simulations are generated.
    if _missing(frame, "signal_score", "ticker"):
        return empty_chart("Latest Model Signal Rankings", "No signals yet: run make signals (or a paper run).")
    clean = frame.sort_values("signal_score", ascending=False).head(20)
    return px.bar(
        clean,
        x="ticker",
        y="signal_score",
        color="side" if "side" in clean.columns else None,
        title="Latest Model Signal Rankings",
        template=TEMPLATE,
        labels={"signal_score": "Signal Score", "ticker": "Ticker"},
    )


def execution_history_chart(frame: pd.DataFrame) -> go.Figure:
    # Execution history exposes rejected orders, skipped orders, and fills. This
    # is essential for operational controls and post-trade review.
    if _missing(frame, "event"):
        return empty_chart("Simulated Execution Events", "No simulated execution log yet: run make paper.")
    counts = frame["event"].fillna("UNKNOWN").value_counts().reset_index()
    counts.columns = ["event", "count"]
    return px.bar(
        counts,
        x="event",
        y="count",
        title="Simulated Execution Events",
        template=TEMPLATE,
        labels={"event": "Event", "count": "Count"},
    )


def allocation_chart(frame: pd.DataFrame) -> go.Figure:
    # Allocation charts compare current position weights and make accidental
    # cash, concentration, or stale-position issues easier to spot.
    if _missing(frame, "weight", "symbol"):
        return empty_chart("Simulated Paper Allocation (latest)", "No simulated positions yet: run make paper.")
    clean = frame.sort_values("weight", key=lambda series: series.abs(), ascending=False)
    return px.bar(
        clean,
        x="symbol",
        y="weight",
        title="Simulated Paper Allocation (latest)",
        template=TEMPLATE,
        labels={"symbol": "Symbol", "weight": "Weight"},
    )
