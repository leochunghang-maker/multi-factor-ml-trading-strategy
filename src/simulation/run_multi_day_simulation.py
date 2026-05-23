import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import (
    MAX_PAPER_ALLOCATION,
    MAX_PAPER_POSITIONS,
    MAX_PAPER_TURNOVER,
    MIN_REBALANCE_DOLLARS,
    MULTI_DAY_DRAWDOWN_CHART_PATH,
    MULTI_DAY_EXPOSURE_CHART_PATH,
    MULTI_DAY_EXPOSURE_PATH,
    MULTI_DAY_MONTHLY_RETURNS_PATH,
    MULTI_DAY_NAV_PATH,
    MULTI_DAY_REBALANCE_FREQUENCY,
    MULTI_DAY_ROLLING_WINDOW,
    MULTI_DAY_SIMULATION_DIR,
    MULTI_DAY_SIMULATION_REPORT_PATH,
    MULTI_DAY_STABILITY_METRICS_PATH,
    MULTI_DAY_STARTING_CAPITAL,
    MULTI_DAY_TRADE_HISTORY_PATH,
    MULTI_DAY_TURNOVER_CHART_PATH,
    MULTI_DAY_TURNOVER_PATH,
    MULTI_DAY_WARNING_LOG_PATH,
    PAPER_ALLOCATION_BUFFER,
    PRICE_DATA_PATH,
    SLIPPAGE_BPS,
    TRANSACTION_COST,
)
from src.data.io import load_price_data
from src.execution.simulated_broker import SimulatedBroker, SimulatedOrder
from src.features.factors import calculate_factor_frames
from src.operations.health import build_health_status, save_system_status_report
from src.operations.structured_logging import log_event
from src.operations.metadata import create_run_metadata, set_deterministic_seed
from src.portfolio.allocation import build_paper_target_weights, normalize_long_only_weights


@dataclass(frozen=True)
class SimulationSettings:
    rebalance_frequency: str = MULTI_DAY_REBALANCE_FREQUENCY
    transaction_cost: float = TRANSACTION_COST
    slippage_bps: float = SLIPPAGE_BPS
    max_positions: int = MAX_PAPER_POSITIONS
    max_position_weight: float = MAX_PAPER_ALLOCATION
    allocation_buffer: float = PAPER_ALLOCATION_BUFFER
    max_turnover: float = MAX_PAPER_TURNOVER
    starting_capital: float = MULTI_DAY_STARTING_CAPITAL
    rolling_window: int = MULTI_DAY_ROLLING_WINDOW


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def build_daily_signal(
    date: pd.Timestamp,
    momentum: pd.DataFrame,
    mean_reversion: pd.DataFrame,
    daily_returns: pd.DataFrame,
    settings: SimulationSettings,
) -> tuple[pd.DataFrame, list[str]]:
    warnings = []
    if date not in momentum.index:
        return pd.DataFrame(), [f"No factor row available for {date.date()}."]

    mom = momentum.loc[date]
    short_term_momentum = -mean_reversion.loc[date]
    volatility = daily_returns.loc[:date].tail(21).std()
    valid = mom.notna() & short_term_momentum.notna() & volatility.notna()
    if valid.sum() < settings.max_positions:
        warnings.append(
            f"Only {int(valid.sum())} valid signal names on {date.date()}; required {settings.max_positions}."
        )
    if valid.sum() == 0:
        return pd.DataFrame(), warnings

    # Signals are refreshed every simulated day using only information available
    # through that date. Long-term momentum rewards trend, short-term momentum
    # captures recent strength, and volatility is penalized for risk control.
    score = zscore(mom[valid]) + zscore(short_term_momentum[valid]) - zscore(volatility[valid])
    selected = score.sort_values(ascending=False).head(settings.max_positions)
    effective_weight = max(0.0, settings.max_position_weight - settings.allocation_buffer)
    target_weight = min(effective_weight, 1 / max(1, len(selected)))
    signals = pd.DataFrame({
        "date": date,
        "ticker": selected.index,
        "signal_score": selected.values,
        "side": "LONG",
        "target_weight": target_weight,
    })
    return signals, warnings


def should_rebalance(
    date: pd.Timestamp,
    previous_rebalance_date: pd.Timestamp | None,
    frequency: str,
) -> bool:
    if previous_rebalance_date is None:
        return True
    if frequency == "daily":
        return True
    if frequency == "monthly":
        return date.to_period("M") != previous_rebalance_date.to_period("M")
    raise ValueError("rebalance_frequency must be 'daily' or 'monthly'.")


def build_orders(
    broker: SimulatedBroker,
    target_weights: pd.Series,
    prices: dict[str, float],
    portfolio_value: float,
    timestamp: str,
) -> list[SimulatedOrder]:
    orders = []
    target_symbols = set(target_weights[target_weights > 0].index)

    for symbol, target_weight in target_weights.items():
        price = prices.get(symbol)
        if price is None or price <= 0:
            continue
        current_value = broker.positions.get(symbol, 0.0) * price
        target_value = portfolio_value * target_weight
        if target_value > current_value:
            fill_buffer = broker.transaction_cost_rate + broker.slippage_bps / 10_000
            target_value = target_value / (1 + fill_buffer)
        dollar_delta = target_value - current_value
        if abs(dollar_delta) < MIN_REBALANCE_DOLLARS:
            continue
        orders.append(
            SimulatedOrder(
                symbol=symbol,
                side="BUY" if dollar_delta > 0 else "SELL",
                quantity=abs(dollar_delta) / price,
                price=price,
                reason="multi-day simulation rebalance",
            )
        )

    for symbol, quantity in broker.positions.items():
        if symbol in target_symbols:
            continue
        price = prices.get(symbol)
        if price is None or price <= 0:
            continue
        orders.append(
            SimulatedOrder(
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                price=price,
                reason="remove position outside current target list",
            )
        )

    # Selling first releases cash and reduces legacy concentration before buys
    # are evaluated. That is closer to an institutional rebalance workflow than
    # submitting buys first and creating artificial cash rejections.
    return sorted(orders, key=lambda order: 0 if order.side == "SELL" else 1)


def monthly_return_table(nav: pd.DataFrame) -> pd.DataFrame:
    monthly = nav.set_index("date")["portfolio_value"].resample("ME").last().pct_change()
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    return table.pivot(index="year", columns="month", values="return").reindex(columns=range(1, 13))


def add_rolling_metrics(nav: pd.DataFrame, market_proxy: pd.Series, window: int) -> pd.DataFrame:
    nav = nav.copy()
    returns = nav["portfolio_value"].pct_change().fillna(0.0)
    nav["daily_return"] = returns
    nav["drawdown"] = nav["portfolio_value"] / nav["portfolio_value"].cummax() - 1
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    nav["rolling_sharpe"] = rolling_mean / rolling_vol

    aligned = pd.concat(
        [returns.rename("portfolio"), market_proxy.reindex(nav["date"]).reset_index(drop=True).rename("market")],
        axis=1,
    ).dropna()
    beta = aligned["portfolio"].rolling(window).cov(aligned["market"]) / aligned["market"].rolling(window).var()
    nav["rolling_beta"] = beta.reindex(nav.index)
    return nav


def save_charts(nav: pd.DataFrame, turnover: pd.DataFrame, exposure: pd.DataFrame) -> None:
    Path(MULTI_DAY_SIMULATION_DIR).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(nav["date"], nav["drawdown"])
    plt.title("Multi-Day Simulation Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MULTI_DAY_DRAWDOWN_CHART_PATH)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(turnover["date"], turnover["turnover"])
    plt.title("Turnover Trend")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MULTI_DAY_TURNOVER_CHART_PATH)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(exposure["date"], exposure["gross_exposure"], label="Gross")
    plt.plot(exposure["date"], exposure["net_exposure"], label="Net")
    plt.title("Exposure Trend")
    plt.xlabel("Date")
    plt.ylabel("Exposure")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MULTI_DAY_EXPOSURE_CHART_PATH)
    plt.close()


def save_report(
    nav: pd.DataFrame,
    trades: pd.DataFrame,
    warnings: pd.DataFrame,
    settings: SimulationSettings,
) -> None:
    latest = nav.iloc[-1]
    returns = nav["daily_return"].dropna()
    warning_counts = (
        warnings["type"].value_counts().to_dict()
        if not warnings.empty and "type" in warnings.columns
        else {}
    )
    lines = [
        "# Multi-Day Paper-Trading Simulation Report",
        "",
        "This report simulates a local paper-trading operating loop. It does not connect to a broker and does not enable live-money trading.",
        "",
        "## Configuration",
        "",
        f"- Rebalance frequency: {settings.rebalance_frequency}",
        f"- Starting capital: {settings.starting_capital:.2f}",
        f"- Transaction cost: {settings.transaction_cost:.4%}",
        f"- Slippage: {settings.slippage_bps:.2f} bps",
        f"- Max positions: {settings.max_positions}",
        f"- Max position weight: {settings.max_position_weight:.2%}",
        "",
        "## Latest State",
        "",
        f"- Latest date: {latest['date']}",
        f"- Portfolio value: {latest['portfolio_value']:.2f}",
        f"- Cash: {latest['cash']:.2f}",
        f"- Drawdown: {latest['drawdown']:.2%}",
        f"- Gross exposure: {latest['gross_exposure']:.2%}",
        f"- Net exposure: {latest['net_exposure']:.2%}",
        f"- Rolling Sharpe: {latest['rolling_sharpe']:.3f}" if pd.notna(latest["rolling_sharpe"]) else "- Rolling Sharpe: N/A",
        f"- Rolling beta: {latest['rolling_beta']:.3f}" if pd.notna(latest["rolling_beta"]) else "- Rolling beta: N/A",
        "",
        "## Stability Metrics",
        "",
        f"- Total simulated days: {len(nav)}",
        f"- Total trades: {len(trades)}",
        f"- Warning count: {len(warnings)}",
        f"- Average daily turnover: {nav['turnover'].mean():.4f}",
        f"- Average gross exposure: {nav['gross_exposure'].mean():.2%}",
        f"- Return volatility: {returns.std() * np.sqrt(252):.2%}",
        "",
        "## Warning Breakdown",
        "",
    ]
    if warning_counts:
        lines.extend([f"- {key}: {value}" for key, value in sorted(warning_counts.items())])
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "## Why This Matters",
        "",
        "- Long-term simulation matters because a process can work for one day and still fail after weeks of turnover, drift, missing data, and compounding cash effects.",
        "- Operational stability matters because paper trading is a rehearsal for data, signal, sizing, execution, and reporting controls.",
        "- Portfolio drift matters because market moves can push realized weights away from intended target weights between rebalances.",
        "- Turnover monitoring matters because excessive trading can overwhelm expected alpha with implementation costs.",
        "",
        "## Outputs",
        "",
        f"- NAV history: `{MULTI_DAY_NAV_PATH}`",
        f"- Exposure history: `{MULTI_DAY_EXPOSURE_PATH}`",
        f"- Turnover history: `{MULTI_DAY_TURNOVER_PATH}`",
        f"- Trade history: `{MULTI_DAY_TRADE_HISTORY_PATH}`",
        f"- Warnings: `{MULTI_DAY_WARNING_LOG_PATH}`",
    ])
    Path(MULTI_DAY_SIMULATION_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(MULTI_DAY_SIMULATION_REPORT_PATH).write_text("\n".join(lines) + "\n")


def run_simulation(settings: SimulationSettings, start: str | None = None, end: str | None = None) -> None:
    set_deterministic_seed()
    metadata = create_run_metadata(
        "multi_day_simulation",
        {
            "start": start,
            "end": end,
            "rebalance_frequency": settings.rebalance_frequency,
            "starting_capital": settings.starting_capital,
        },
    )
    run_id = metadata["run_id"]
    log_event("simulation_started", "Multi-day simulation started.", run_id=run_id)
    full_prices = load_price_data(PRICE_DATA_PATH).dropna(how="all")
    factors = calculate_factor_frames(full_prices)
    daily_returns = factors["daily_returns"]
    market_proxy = daily_returns.mean(axis=1)

    prices = full_prices
    if start:
        prices = prices.loc[pd.Timestamp(start):]
    if end:
        prices = prices.loc[:pd.Timestamp(end)]

    broker = SimulatedBroker(
        cash=settings.starting_capital,
        transaction_cost_rate=settings.transaction_cost,
        slippage_bps=settings.slippage_bps,
    )

    nav_rows = []
    exposure_rows = []
    turnover_rows = []
    trade_rows = []
    warning_rows = []
    previous_value = settings.starting_capital
    previous_rebalance_date = None

    for date in prices.index:
        price_row = prices.loc[date].dropna()
        if price_row.empty:
            warning_rows.append({"date": date, "type": "MISSING_DATA", "message": "No prices available."})
            continue
        price_dict = price_row.to_dict()
        portfolio_start = broker.portfolio_value(price_dict)
        traded_notional = 0.0
        trades_filled = 0
        rejected = 0

        missing_position_prices = sorted(set(broker.positions) - set(price_dict))
        if missing_position_prices:
            warning_rows.append({
                "date": date,
                "type": "MISSING_POSITION_PRICE",
                "message": ",".join(missing_position_prices),
            })

        if should_rebalance(date, previous_rebalance_date, settings.rebalance_frequency):
            signals, signal_warnings = build_daily_signal(
                date,
                factors["momentum_12m"],
                factors["mean_reversion_1m"],
                daily_returns,
                settings,
            )
            for message in signal_warnings:
                warning_rows.append({"date": date, "type": "SIGNAL_WARNING", "message": message})
            if signals.empty:
                warning_rows.append({"date": date, "type": "STALE_SIGNAL", "message": "No current signal generated."})
            else:
                target_weights, allocation = build_paper_target_weights(
                    signals,
                    broker.positions,
                    price_dict,
                    portfolio_start,
                    max_position_weight=settings.max_position_weight,
                    allocation_buffer=settings.allocation_buffer,
                    max_turnover=settings.max_turnover,
                )
                if (target_weights > 0).sum() > settings.max_positions:
                    keep = target_weights.nlargest(settings.max_positions).index
                    target_weights = target_weights.loc[keep]
                    target_weights = normalize_long_only_weights(
                        target_weights,
                        max(0.0, settings.max_position_weight - settings.allocation_buffer),
                    )
                    warning_rows.append({
                        "date": date,
                        "type": "POSITION_COUNT_LIMIT",
                        "message": f"Target book pruned to {settings.max_positions} positions after turnover adjustment.",
                    })
                orders = build_orders(broker, target_weights, price_dict, portfolio_start, date.isoformat())
                for order in orders:
                    try:
                        trade = broker.process_order(
                            order,
                            date.isoformat(),
                            prices=price_dict,
                            max_position_weight=settings.max_position_weight,
                        )
                        trade_rows.append(trade)
                        traded_notional += trade["notional"]
                        trades_filled += 1
                    except ValueError as exc:
                        rejected += 1
                        warning_rows.append({
                            "date": date,
                            "type": "FAILED_EXECUTION",
                            "message": f"{order.symbol} {order.side}: {exc}",
                        })
                previous_rebalance_date = date
                if allocation["expected_turnover"] > settings.max_turnover:
                    warning_rows.append({
                        "date": date,
                        "type": "EXCESSIVE_TURNOVER",
                        "message": f"Expected turnover {allocation['expected_turnover']:.4f}",
                    })

        portfolio_value = broker.portfolio_value(price_dict)
        daily_return = portfolio_value / previous_value - 1 if previous_value > 0 else 0.0
        turnover = traded_notional / portfolio_start if portfolio_start > 0 else 0.0
        position_values = broker.position_values(price_dict)
        gross = sum(abs(value) for value in position_values.values()) / portfolio_value if portfolio_value else 0.0
        net = sum(position_values.values()) / portfolio_value if portfolio_value else 0.0
        max_position = max((abs(value) / portfolio_value for value in position_values.values()), default=0.0)
        top_5 = sum(sorted([abs(value) / portfolio_value for value in position_values.values()], reverse=True)[:5]) if portfolio_value else 0.0

        if turnover > settings.max_turnover:
            warning_rows.append({"date": date, "type": "EXCESSIVE_TURNOVER", "message": f"Realized turnover {turnover:.4f}"})
        if gross > 1.0 + 1e-9:
            warning_rows.append({"date": date, "type": "EXPOSURE_LIMIT", "message": f"Gross exposure {gross:.2%}"})
        if max_position > settings.max_position_weight + 0.005:
            warning_rows.append({"date": date, "type": "CONCENTRATION", "message": f"Max position {max_position:.2%}"})
        if len(broker.positions) > settings.max_positions:
            warning_rows.append({
                "date": date,
                "type": "POSITION_COUNT_LIMIT",
                "message": f"Open positions {len(broker.positions)} exceed max {settings.max_positions}.",
            })

        nav_rows.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "cash": broker.cash,
            "daily_return": daily_return,
            "turnover": turnover,
            "trades_filled": trades_filled,
            "orders_rejected": rejected,
            "number_of_positions": len(broker.positions),
            "gross_exposure": gross,
            "net_exposure": net,
            "max_position_weight": max_position,
            "top_5_concentration": top_5,
        })
        exposure_rows.append({
            "date": date,
            "gross_exposure": gross,
            "net_exposure": net,
            "max_position_weight": max_position,
            "top_5_concentration": top_5,
        })
        turnover_rows.append({"date": date, "turnover": turnover, "trades_filled": trades_filled})
        previous_value = portfolio_value

    nav = pd.DataFrame(nav_rows)
    nav = add_rolling_metrics(nav, market_proxy, settings.rolling_window)
    exposure = pd.DataFrame(exposure_rows)
    turnover = pd.DataFrame(turnover_rows)
    trades = pd.DataFrame(trade_rows)
    warnings = pd.DataFrame(warning_rows)
    monthly = monthly_return_table(nav)
    stability = pd.DataFrame([{
        "simulated_days": len(nav),
        "final_nav": nav["portfolio_value"].iloc[-1],
        "max_drawdown": nav["drawdown"].min(),
        "average_turnover": nav["turnover"].mean(),
        "average_gross_exposure": nav["gross_exposure"].mean(),
        "warning_count": len(warnings),
        "trade_count": len(trades),
    }])

    Path(MULTI_DAY_SIMULATION_DIR).mkdir(parents=True, exist_ok=True)
    nav.to_csv(MULTI_DAY_NAV_PATH, index=False)
    exposure.to_csv(MULTI_DAY_EXPOSURE_PATH, index=False)
    turnover.to_csv(MULTI_DAY_TURNOVER_PATH, index=False)
    trades.to_csv(MULTI_DAY_TRADE_HISTORY_PATH, index=False)
    warnings.to_csv(MULTI_DAY_WARNING_LOG_PATH, index=False)
    monthly.to_csv(MULTI_DAY_MONTHLY_RETURNS_PATH)
    stability.to_csv(MULTI_DAY_STABILITY_METRICS_PATH, index=False)
    save_charts(nav, turnover, exposure)
    save_report(nav, trades, warnings, settings)
    status = build_health_status()
    save_system_status_report(status)
    log_event(
        "simulation_completed",
        "Multi-day simulation completed.",
        run_id=run_id,
        simulated_days=len(nav),
        final_nav=float(nav["portfolio_value"].iloc[-1]),
        warning_count=len(warnings),
        health_status=status["overall_status"],
    )

    print("Multi-day paper-trading simulation completed.")
    print(f"Simulated days: {len(nav)}")
    print(f"Final NAV: {nav['portfolio_value'].iloc[-1]:.2f}")
    print(f"Max drawdown: {nav['drawdown'].min():.2%}")
    print(f"Warnings: {len(warnings)}")
    print(f"Report saved to {MULTI_DAY_SIMULATION_REPORT_PATH}")
    print(f"Run metadata: {metadata['metadata_path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run historical multi-day paper simulation.")
    parser.add_argument("--rebalance-frequency", choices=["daily", "monthly"], default=MULTI_DAY_REBALANCE_FREQUENCY)
    parser.add_argument("--starting-capital", type=float, default=MULTI_DAY_STARTING_CAPITAL)
    parser.add_argument("--transaction-cost", type=float, default=TRANSACTION_COST)
    parser.add_argument("--slippage-bps", type=float, default=SLIPPAGE_BPS)
    parser.add_argument("--max-positions", type=int, default=MAX_PAPER_POSITIONS)
    parser.add_argument("--max-position-weight", type=float, default=MAX_PAPER_ALLOCATION)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = SimulationSettings(
        rebalance_frequency=args.rebalance_frequency,
        transaction_cost=args.transaction_cost,
        slippage_bps=args.slippage_bps,
        max_positions=args.max_positions,
        max_position_weight=args.max_position_weight,
        starting_capital=args.starting_capital,
    )
    run_simulation(settings, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
