import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TEMPLATE = "plotly_white"


def equity_curve_chart(returns: pd.DataFrame) -> go.Figure:
    # Equity curves show the compounded investor experience through time.
    # Institutions use them to inspect path dependency, not just final return.
    frame = returns.dropna(subset=["date", "portfolio_return"]).copy()
    frame["equity_curve"] = (1 + frame["portfolio_return"]).cumprod()
    return px.line(
        frame,
        x="date",
        y="equity_curve",
        title="Equity Curve",
        template=TEMPLATE,
        labels={"equity_curve": "Growth of $1", "date": "Date"},
    )


def drawdown_chart(returns: pd.DataFrame) -> go.Figure:
    # Drawdown is the peak-to-trough loss investors must survive. It is often
    # more relevant to capital allocation than average return.
    frame = returns.dropna(subset=["date", "portfolio_return"]).copy()
    curve = (1 + frame["portfolio_return"]).cumprod()
    frame["drawdown"] = curve / curve.cummax() - 1
    return px.area(
        frame,
        x="date",
        y="drawdown",
        title="Drawdown",
        template=TEMPLATE,
        labels={"drawdown": "Drawdown", "date": "Date"},
    )


def rolling_metric_chart(frame: pd.DataFrame, column: str, title: str, ylabel: str) -> go.Figure:
    # Rolling risk charts reveal whether performance is stable or concentrated
    # in a short favorable regime.
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
    clean = frame.dropna(subset=["date"])
    return px.line(
        clean,
        x="date",
        y=columns,
        title="Portfolio Turnover",
        template=TEMPLATE,
        labels={"value": "One-Way Turnover", "date": "Date", "variable": "Series"},
    )


def sector_exposure_chart(frame: pd.DataFrame) -> go.Figure:
    # Sector exposure identifies hidden common risk: many stocks can still be
    # one concentrated macro or industry bet.
    sector_columns = [column for column in frame.columns if column != "date"]
    clean = frame.dropna(subset=["date"])
    return px.area(
        clean,
        x="date",
        y=sector_columns,
        title="Sector Exposure",
        template=TEMPLATE,
        labels={"value": "Net Weight", "date": "Date", "variable": "Sector"},
    )


def top_holdings_chart(frame: pd.DataFrame) -> go.Figure:
    # Top holdings make concentration visible. Institutions watch this because
    # single-name shocks can dominate realized PnL.
    latest = frame.dropna(subset=["date"]).tail(1)
    if latest.empty or "constrained_top_holdings" not in latest.columns:
        return go.Figure()
    holdings = []
    for item in str(latest["constrained_top_holdings"].iloc[0]).split("; "):
        if ":" not in item:
            continue
        ticker, value = item.split(":", 1)
        holdings.append({"ticker": ticker, "weight": float(value.strip("%")) / 100})
    holdings_frame = pd.DataFrame(holdings)
    if holdings_frame.empty:
        return go.Figure()
    return px.bar(
        holdings_frame,
        x="ticker",
        y="weight",
        title="Top Holdings",
        template=TEMPLATE,
        labels={"ticker": "Ticker", "weight": "Weight"},
    )


def beta_exposure_chart(frame: pd.DataFrame) -> go.Figure:
    # Beta exposure shows broad market sensitivity. It helps separate stock
    # selection skill from simply being long or short the market.
    column = "constrained_beta_exposure"
    clean = frame.dropna(subset=["date", column]) if column in frame.columns else pd.DataFrame()
    return px.line(
        clean,
        x="date",
        y=column,
        title="Portfolio Beta Exposure",
        template=TEMPLATE,
        labels={column: "Beta", "date": "Date"},
    )


def signal_rankings_chart(frame: pd.DataFrame) -> go.Figure:
    # Signal rankings show what the model currently wants to own. This is a
    # useful sanity check before orders or simulations are generated.
    if frame.empty or "signal_score" not in frame.columns:
        return go.Figure()
    clean = frame.sort_values("signal_score", ascending=False).head(20)
    return px.bar(
        clean,
        x="ticker",
        y="signal_score",
        color="side" if "side" in clean.columns else None,
        title="Latest Signal Rankings",
        template=TEMPLATE,
        labels={"signal_score": "Signal Score", "ticker": "Ticker"},
    )


def execution_history_chart(frame: pd.DataFrame) -> go.Figure:
    # Execution history exposes rejected orders, skipped orders, and fills. This
    # is essential for operational controls and post-trade review.
    if frame.empty or "event" not in frame.columns:
        return go.Figure()
    counts = frame["event"].fillna("UNKNOWN").value_counts().reset_index()
    counts.columns = ["event", "count"]
    return px.bar(
        counts,
        x="event",
        y="count",
        title="Execution Events",
        template=TEMPLATE,
        labels={"event": "Event", "count": "Count"},
    )


def allocation_chart(frame: pd.DataFrame) -> go.Figure:
    # Allocation charts compare current position weights and make accidental
    # cash, concentration, or stale-position issues easier to spot.
    if frame.empty or "weight" not in frame.columns:
        return go.Figure()
    clean = frame.sort_values("weight", key=lambda series: series.abs(), ascending=False)
    return px.bar(
        clean,
        x="symbol",
        y="weight",
        title="Latest Portfolio Allocation",
        template=TEMPLATE,
        labels={"symbol": "Symbol", "weight": "Weight"},
    )
