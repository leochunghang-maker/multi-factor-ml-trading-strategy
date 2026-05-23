import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.config import (
    ALLOCATION_SUMMARY_PATH,
    LIVE_SIGNALS_PATH,
    MAX_PAPER_ALLOCATION,
    MAX_PAPER_POSITIONS,
    MAX_PAPER_TURNOVER,
    MIN_REBALANCE_DOLLARS,
    PAPER_ALLOCATION_BUFFER,
    PAPER_TRADING_STATUS_REPORT_PATH,
    PRICE_DATA_PATH,
    REALIZED_VS_TARGET_WEIGHTS_PATH,
    SIGNAL_STALENESS_DAYS,
    SIMULATED_BROKER_STATE_PATH,
    SIMULATED_DAILY_PERFORMANCE_PATH,
    SIMULATED_EXECUTION_LOG_PATH,
    SIMULATED_LATEST_SNAPSHOT_PATH,
    SIMULATED_POSITION_SNAPSHOT_PATH,
    SIMULATED_PORTFOLIO_HISTORY_PATH,
    SIMULATED_REJECTED_ORDERS_PATH,
    SIMULATED_TRADE_HISTORY_PATH,
)
from src.execution.simulated_broker import SimulatedBroker, SimulatedOrder, append_csv
from src.portfolio.allocation import build_paper_target_weights, realized_vs_target_frame


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_latest_prices(path: str = PRICE_DATA_PATH) -> tuple[dict[str, float], pd.Timestamp]:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    latest_date = prices.dropna(how="all").index.max()
    latest_prices = prices.loc[latest_date].dropna()
    return latest_prices.to_dict(), latest_date


def load_long_only_targets(path: str = LIVE_SIGNALS_PATH) -> pd.DataFrame:
    signals = pd.read_csv(path, parse_dates=["date"])
    # Empty signals are dangerous because they can accidentally trigger a full
    # liquidation or hide a broken model/data pipeline. Stop and investigate.
    if signals.empty:
        raise RuntimeError("Signal file is empty; refusing to rebalance.")
    if signals["ticker"].duplicated().any():
        duplicates = sorted(signals.loc[signals["ticker"].duplicated(), "ticker"].unique())
        raise RuntimeError(f"Duplicate tickers in signal file: {duplicates}")

    longs = signals[signals["side"] == "LONG"].copy()
    if len(longs) > MAX_PAPER_POSITIONS:
        raise RuntimeError(
            f"Too many LONG targets: {len(longs)} > max allowed {MAX_PAPER_POSITIONS}."
        )
    if (longs["target_weight"] > MAX_PAPER_ALLOCATION + 1e-9).any():
        raise RuntimeError(
            f"At least one target weight exceeds max paper allocation {MAX_PAPER_ALLOCATION:.2%}."
        )

    # Long-only rebalancing means the simulator buys only approved LONG names.
    # HOLD names are not opened as new positions.
    longs = longs.sort_values("signal_score", ascending=False).head(MAX_PAPER_POSITIONS)
    if longs.empty:
        raise RuntimeError("Signal file contains no LONG targets; refusing to rebalance.")

    target_weight_sum = longs["target_weight"].sum()
    if target_weight_sum > 1.0 + 1e-9:
        raise RuntimeError("Target weights exceed 100%; refusing to use leverage.")
    if (longs["target_weight"] < 0).any():
        raise RuntimeError("Negative target weights are not allowed in long-only mode.")
    if target_weight_sum < 0.95:
        raise RuntimeError(
            f"Target weights sum to only {target_weight_sum:.2%}; refusing under-invested abnormal signal file."
        )

    return longs


def validate_signal_freshness(
    targets: pd.DataFrame,
    price_date: pd.Timestamp,
    max_staleness_days: int = SIGNAL_STALENESS_DAYS,
) -> pd.Timestamp:
    latest_signal_date = pd.to_datetime(targets["date"]).max()
    staleness_days = (pd.Timestamp(price_date).normalize() - latest_signal_date.normalize()).days
    if staleness_days < 0:
        raise RuntimeError(
            "Latest signal is newer than the latest local price. "
            f"signal_date={latest_signal_date.date()}, price_date={pd.Timestamp(price_date).date()}. "
            "Refresh local prices before simulating fills."
        )
    if staleness_days > max_staleness_days:
        raise RuntimeError(
            "Latest signal is stale relative to local prices: "
            f"signal_date={latest_signal_date.date()}, price_date={pd.Timestamp(price_date).date()}, "
            f"staleness_days={staleness_days}, allowed={max_staleness_days}."
        )
    return latest_signal_date


def build_rebalance_orders(
    broker: SimulatedBroker,
    targets: pd.DataFrame,
    prices: dict[str, float],
    portfolio_value: float,
    timestamp: str,
) -> tuple[list[SimulatedOrder], list[dict]]:
    orders = []
    execution_logs = []
    target_weights, allocation_diagnostics = build_paper_target_weights(
        targets,
        broker.positions,
        prices,
        portfolio_value,
        max_position_weight=MAX_PAPER_ALLOCATION,
        allocation_buffer=PAPER_ALLOCATION_BUFFER,
        max_turnover=MAX_PAPER_TURNOVER,
    )
    target_symbols = set(target_weights[target_weights > 0].index)

    for symbol, target_weight in target_weights.items():
        if target_weight <= 0:
            continue
        price = prices.get(symbol)

        if price is None or price <= 0:
            execution_logs.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "event": "SKIPPED",
                "message": "missing local price data",
            })
            continue

        current_qty = broker.positions.get(symbol, 0.0)
        current_value = current_qty * price
        gross_target_value = portfolio_value * target_weight

        # Portfolio sizing converts model rankings into feasible account
        # weights before order generation. Constraints change realized
        # positions because cash, max position caps, turnover, and transaction
        # costs all affect what can actually be held.
        # Transaction costs are paid from cash. When buying, the simulator
        # slightly reduces the notional order so the account can pay both the
        # shares and the cost without borrowing.
        if gross_target_value > current_value:
            fill_buffer = broker.transaction_cost_rate + broker.slippage_bps / 10_000
            target_value = gross_target_value / (1 + fill_buffer)
        else:
            target_value = gross_target_value
        dollar_delta = target_value - current_value

        # Rebalancing trades the difference between the current position and
        # desired target exposure rather than buying the full target every day.
        if abs(dollar_delta) < MIN_REBALANCE_DOLLARS:
            execution_logs.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "event": "SKIPPED",
                "message": "already close to target weight",
            })
            continue

        side = "BUY" if dollar_delta > 0 else "SELL"
        quantity = abs(dollar_delta) / price
        orders.append(
            SimulatedOrder(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                reason="rebalance to long-only target weight",
            )
        )

    for symbol, quantity in broker.positions.items():
        if symbol in target_symbols:
            continue
        price = prices.get(symbol)
        if price is None or price <= 0:
            execution_logs.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "event": "SKIPPED",
                "message": "missing local price data for existing position",
            })
            continue
        # Existing positions not in the new LONG list are sold down to zero.
        # This does not short; it only closes shares already owned.
        orders.append(
            SimulatedOrder(
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                price=price,
                reason="remove position no longer in latest LONG signal list",
            )
        )

    execution_logs.append({
        "timestamp": timestamp,
        "symbol": "PORTFOLIO",
        "event": "ALLOCATION_BUILT",
        "message": (
            f"target_weight_sum={allocation_diagnostics['final_target_weight_sum']:.4f}, "
            f"expected_turnover={allocation_diagnostics['expected_turnover']:.4f}, "
            f"expected_cash_weight={allocation_diagnostics['expected_cash_weight']:.4f}"
        ),
    })
    return orders, execution_logs, target_weights, allocation_diagnostics


def calculate_daily_performance(
    snapshot: dict,
    history_path: str = SIMULATED_PORTFOLIO_HISTORY_PATH,
) -> dict:
    history_path_obj = Path(history_path)
    previous_value = None
    if history_path_obj.exists():
        history = pd.read_csv(history_path_obj)
        if not history.empty and "portfolio_value" in history.columns:
            previous_value = pd.to_numeric(history["portfolio_value"], errors="coerce").dropna()
            previous_value = previous_value.iloc[-1] if not previous_value.empty else None

    portfolio_value = float(snapshot["portfolio_value"])
    daily_return = (
        portfolio_value / previous_value - 1
        if previous_value and previous_value > 0
        else 0.0
    )
    return {
        "timestamp": snapshot["timestamp"],
        "portfolio_value": portfolio_value,
        "daily_return": daily_return,
        "turnover": snapshot.get("turnover", 0.0),
        "number_of_trades": snapshot.get("trades_filled", 0),
        "number_of_positions": snapshot.get("number_of_positions", 0),
        "cash": snapshot.get("cash", 0.0),
        "gross_exposure": snapshot.get("gross_exposure", 0.0),
        "net_exposure": snapshot.get("net_exposure", 0.0),
        "drawdown": snapshot.get("drawdown", 0.0),
    }


def calculate_portfolio_drawdown(portfolio_value: float) -> float:
    history_path = Path(SIMULATED_PORTFOLIO_HISTORY_PATH)
    if not history_path.exists():
        return 0.0
    history = pd.read_csv(history_path)
    values = pd.to_numeric(history.get("portfolio_value", pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return 0.0
    peak = max(values.max(), portfolio_value)
    return portfolio_value / peak - 1 if peak else 0.0


def top_holding_lines(position_rows: list[dict], top_n: int = 5) -> list[str]:
    frame = pd.DataFrame(position_rows)
    if frame.empty:
        return ["- No open positions."]
    frame = frame.sort_values("weight", key=lambda series: series.abs(), ascending=False)
    lines = []
    for row in frame.head(top_n).to_dict("records"):
        lines.append(
            f"- {row['symbol']}: {row['weight']:.2%} weight, value {row['market_value']:.2f}"
        )
    return lines


def save_paper_trading_status_report(
    snapshot: dict,
    position_rows: list[dict],
    warnings: list[str],
    rejected_rows: list[dict],
) -> None:
    # Paper trading must be validated before live trading because it exposes
    # operational problems: stale data, bad fills, unexpected turnover, and cash
    # accounting errors. This report gives a human operator a daily review page.
    path = Path(PAPER_TRADING_STATUS_REPORT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Paper Trading Status",
        "",
        f"- Latest run date: {snapshot.get('timestamp')}",
        f"- Price date: {snapshot.get('price_date')}",
        f"- Signal date: {snapshot.get('signal_date')}",
        f"- Portfolio value: {float(snapshot.get('portfolio_value', 0.0)):.2f}",
        f"- Cash: {float(snapshot.get('cash', 0.0)):.2f}",
        f"- Number of positions: {int(snapshot.get('number_of_positions', 0))}",
        f"- Turnover: {float(snapshot.get('turnover', 0.0)):.4f}",
        f"- Drawdown: {float(snapshot.get('drawdown', 0.0)):.2%}",
        f"- Trades filled: {int(snapshot.get('trades_filled', 0))}",
        f"- Rejected orders: {int(snapshot.get('orders_rejected', 0))}",
        "",
        "## Top Holdings",
        "",
        *top_holding_lines(position_rows),
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in warnings] or ["- None."])
    lines.extend(["", "## Rejected Orders", ""])
    if rejected_rows:
        for row in rejected_rows:
            lines.append(
                f"- {row.get('symbol')}: {row.get('side')} {row.get('quantity'):.6f} rejected because {row.get('message')}"
            )
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n")


def run_daily_simulation() -> dict:
    configure_logging()
    run_timestamp = datetime.now(UTC).isoformat(timespec="seconds")

    broker = SimulatedBroker.load(SIMULATED_BROKER_STATE_PATH)
    prices, price_date = load_latest_prices()
    targets = load_long_only_targets()
    signal_date = validate_signal_freshness(targets, price_date)
    warnings = []

    starting_value = broker.portfolio_value(prices)
    orders, execution_logs, target_weights, allocation_diagnostics = build_rebalance_orders(
        broker,
        targets,
        prices,
        starting_value,
        run_timestamp,
    )

    trade_rows = []
    rejected_rows = []
    traded_notional = 0.0

    for order in orders:
        try:
            # Order execution updates cash and positions immediately in the
            # simulated account. Invalid orders are rejected and logged.
            trade = broker.process_order(order, run_timestamp, prices=prices)
            trade_rows.append(trade)
            traded_notional += trade["notional"]
        except ValueError as exc:
            rejected_rows.append({
                "timestamp": run_timestamp,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": order.price,
                "status": "REJECTED",
                "message": str(exc),
            })

    ending_value = broker.portfolio_value(prices)
    pnl = ending_value - starting_value
    turnover = traded_notional / starting_value if starting_value else 0.0
    if turnover > MAX_PAPER_TURNOVER:
        # High turnover can indicate a broken signal file or a portfolio flip
        # that would be expensive and operationally risky in real trading.
        warnings.append(
            f"Turnover {turnover:.2%} exceeds max paper turnover {MAX_PAPER_TURNOVER:.2%}."
        )
    reconciliation = broker.reconcile(prices, ending_value)

    # PnL is the change in account value from the start to the end of this
    # simulation run. On a same-price rebalance day, transaction costs will
    # usually make PnL slightly negative.
    snapshot = broker.snapshot(prices, run_timestamp)
    snapshot.update({
        "price_date": price_date,
        "signal_date": signal_date,
        "signal_staleness_days": (
            pd.Timestamp(price_date).normalize() - signal_date.normalize()
        ).days,
        "starting_portfolio_value": starting_value,
        "ending_portfolio_value": ending_value,
        "pnl": pnl,
        "turnover": turnover,
        "trades_filled": len(trade_rows),
        "orders_rejected": len(rejected_rows),
        "target_weight_sum": allocation_diagnostics.get("final_target_weight_sum", 0.0),
        "expected_cash_weight": allocation_diagnostics.get("expected_cash_weight", 0.0),
    })
    snapshot.update(reconciliation)
    snapshot["drawdown"] = calculate_portfolio_drawdown(ending_value)
    execution_logs.append({
        "timestamp": run_timestamp,
        "symbol": "PORTFOLIO",
        "event": "RUN_COMPLETED",
        "message": (
            f"filled={len(trade_rows)}, rejected={len(rejected_rows)}, "
            f"pnl={pnl:.2f}, turnover={turnover:.6f}"
        ),
    })

    broker.save(SIMULATED_BROKER_STATE_PATH)
    append_csv(SIMULATED_TRADE_HISTORY_PATH, trade_rows)
    append_csv(SIMULATED_REJECTED_ORDERS_PATH, rejected_rows)
    append_csv(SIMULATED_EXECUTION_LOG_PATH, execution_logs + rejected_rows)
    position_rows = broker.position_snapshot_rows(prices, run_timestamp)
    daily_performance = calculate_daily_performance(snapshot)
    realized_vs_target = realized_vs_target_frame(
        target_weights,
        broker.positions,
        prices,
        ending_value,
        rejected_rows,
    )
    rejected_weight_impact = realized_vs_target.loc[
        realized_vs_target["rejected_order_symbol"],
        "absolute_weight_gap",
    ].sum()
    allocation_summary = {
        "timestamp": run_timestamp,
        **allocation_diagnostics,
        "starting_portfolio_value": starting_value,
        "ending_portfolio_value": ending_value,
        "cash": broker.cash,
        "cash_utilization": 1 - (broker.cash / ending_value if ending_value else 0.0),
        "realized_gross_exposure": snapshot.get("gross_exposure", 0.0),
        "realized_net_exposure": snapshot.get("net_exposure", 0.0),
        "rejected_orders": len(rejected_rows),
        "rejected_weight_impact": rejected_weight_impact,
        "max_realized_weight": (
            realized_vs_target["realized_weight"].abs().max()
            if not realized_vs_target.empty else 0.0
        ),
        "top_5_realized_concentration": (
            realized_vs_target["realized_weight"].abs().nlargest(5).sum()
            if not realized_vs_target.empty else 0.0
        ),
    }
    append_csv(SIMULATED_PORTFOLIO_HISTORY_PATH, [snapshot])
    append_csv(SIMULATED_DAILY_PERFORMANCE_PATH, [daily_performance])
    append_csv(ALLOCATION_SUMMARY_PATH, [allocation_summary])
    pd.DataFrame([snapshot]).to_csv(SIMULATED_LATEST_SNAPSHOT_PATH, index=False)
    pd.DataFrame(position_rows).to_csv(SIMULATED_POSITION_SNAPSHOT_PATH, index=False)
    realized_vs_target.insert(0, "timestamp", run_timestamp)
    realized_vs_target.to_csv(REALIZED_VS_TARGET_WEIGHTS_PATH, index=False)
    save_paper_trading_status_report(snapshot, position_rows, warnings, rejected_rows)

    logging.info(
        "Simulation completed: value %.2f, pnl %.2f, turnover %.4f, trades %s",
        ending_value,
        pnl,
        turnover,
        len(trade_rows),
    )
    return snapshot


def main() -> None:
    try:
        snapshot = run_daily_simulation()
    except Exception as exc:
        configure_logging()
        logging.error("Local simulated broker run failed: %s", exc)
        run_timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        append_csv(SIMULATED_EXECUTION_LOG_PATH, [{
            "timestamp": run_timestamp,
            "symbol": "PORTFOLIO",
            "event": "RUN_FAILED",
            "message": str(exc),
        }])
        print(f"Local simulated broker run failed: {exc}")
        print(f"Execution log saved to {SIMULATED_EXECUTION_LOG_PATH}")
        raise SystemExit(1) from exc

    print("Local simulated broker daily run completed.")
    print()
    for key, value in snapshot.items():
        print(f"{key}: {value}")
    print()
    print(f"Trade history saved to {SIMULATED_TRADE_HISTORY_PATH}")
    print(f"Portfolio history saved to {SIMULATED_PORTFOLIO_HISTORY_PATH}")
    print(f"Execution log saved to {SIMULATED_EXECUTION_LOG_PATH}")
    print(f"Latest snapshot saved to {SIMULATED_LATEST_SNAPSHOT_PATH}")
    print(f"Latest position snapshot saved to {SIMULATED_POSITION_SNAPSHOT_PATH}")
    print(f"Paper trading status saved to {PAPER_TRADING_STATUS_REPORT_PATH}")


if __name__ == "__main__":
    main()
