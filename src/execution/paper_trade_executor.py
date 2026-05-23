import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.config import (
    EXECUTION_RESULTS_DIR,
    LIVE_SIGNALS_PATH,
    MAX_PAPER_ALLOCATION,
    MAX_PAPER_POSITIONS,
    MIN_REBALANCE_DOLLARS,
    PAPER_EXECUTION_SUMMARY_PATH,
    PAPER_ORDERS_PATH,
)


PAPER_TRADING_BASE_URL = "https://paper-api.alpaca.markets/v2"
MARKET_DATA_BASE_URL = "https://data.alpaca.markets/v2"


@dataclass
class AlpacaCredentials:
    api_key: str
    secret_key: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_credentials() -> AlpacaCredentials:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca paper credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
        )
    return AlpacaCredentials(api_key=api_key, secret_key=secret_key)


def alpaca_request(
    credentials: AlpacaCredentials,
    base_url: str,
    path: str,
    method: str = "GET",
    payload: dict | None = None,
    query: dict | None = None,
) -> dict | list:
    url = f"{base_url}{path}"
    if query:
        url = f"{url}?{urlencode(query)}"

    body = None
    headers = {
        "APCA-API-KEY-ID": credentials.api_key,
        "APCA-API-SECRET-KEY": credentials.secret_key,
        "Content-Type": "application/json",
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body) if response_body else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"Alpaca API error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Unable to reach Alpaca API: {exc.reason}") from exc


def load_target_signals(path: str = LIVE_SIGNALS_PATH) -> pd.DataFrame:
    signals = pd.read_csv(path, parse_dates=["date"])
    longs = signals[signals["side"] == "LONG"].copy()

    # Long-only execution ignores HOLD rows as new targets. The engine only
    # considers the explicit LONG list when sizing desired positions.
    longs = longs.sort_values("signal_score", ascending=False).head(MAX_PAPER_POSITIONS)
    longs["target_weight"] = longs["target_weight"].clip(upper=MAX_PAPER_ALLOCATION)

    if len(longs) > MAX_PAPER_POSITIONS:
        raise RuntimeError("Signal file exceeds the maximum allowed number of positions.")
    if longs["target_weight"].sum() > 1.0:
        raise RuntimeError("Target weights imply leverage; refusing to continue.")
    if (longs["target_weight"] < 0).any():
        raise RuntimeError("Negative target weights are not allowed in long-only mode.")

    return longs


def get_account_equity(credentials: AlpacaCredentials) -> float:
    account = alpaca_request(credentials, PAPER_TRADING_BASE_URL, "/account")
    return float(account["equity"])


def get_current_positions(credentials: AlpacaCredentials) -> dict[str, dict]:
    positions = alpaca_request(credentials, PAPER_TRADING_BASE_URL, "/positions")
    current = {}
    for position in positions:
        qty = float(position["qty"])
        market_value = float(position["market_value"])
        if qty > 0:
            # Position tracking records what the paper account currently owns.
            # Short or zero positions are ignored because this engine is long-only.
            current[position["symbol"]] = {
                "qty": qty,
                "market_value": market_value,
            }
    return current


def get_latest_price(credentials: AlpacaCredentials, symbol: str) -> float | None:
    try:
        trade = alpaca_request(
            credentials,
            MARKET_DATA_BASE_URL,
            f"/stocks/{symbol}/trades/latest",
            query={"feed": "iex"},
        )
        return float(trade["trade"]["p"])
    except Exception as exc:
        logging.warning("Skipping %s because latest price is unavailable: %s", symbol, exc)
        return None


def build_rebalance_orders(
    signals: pd.DataFrame,
    positions: dict[str, dict],
    prices: dict[str, float],
    account_equity: float,
) -> pd.DataFrame:
    target_symbols = set(signals["ticker"])
    target_weights = dict(zip(signals["ticker"], signals["target_weight"]))
    orders = []

    for symbol, target_weight in target_weights.items():
        price = prices.get(symbol)
        if price is None or price <= 0:
            continue

        current_value = positions.get(symbol, {}).get("market_value", 0.0)
        target_value = account_equity * target_weight
        dollar_delta = target_value - current_value

        # Rebalancing means trading only the difference between the current
        # dollar exposure and the desired target dollar exposure.
        if abs(dollar_delta) < MIN_REBALANCE_DOLLARS:
            continue

        side = "buy" if dollar_delta > 0 else "sell"
        qty = abs(dollar_delta) / price
        orders.append({
            "symbol": symbol,
            "side": side,
            "qty": round(qty, 6),
            "latest_price": price,
            "current_value": current_value,
            "target_weight": target_weight,
            "target_value": target_value,
            "dollar_delta": dollar_delta,
            "reason": "rebalance current paper position to long-only target weight",
        })

    for symbol, position in positions.items():
        if symbol in target_symbols:
            continue
        price = prices.get(symbol)
        if price is None:
            continue
        # Existing long positions that are no longer targets are reduced to zero.
        # This is still long-only because it sells owned shares and never shorts.
        orders.append({
            "symbol": symbol,
            "side": "sell",
            "qty": round(position["qty"], 6),
            "latest_price": price,
            "current_value": position["market_value"],
            "target_weight": 0.0,
            "target_value": 0.0,
            "dollar_delta": -position["market_value"],
            "reason": "remove current long position that is not in the latest LONG list",
        })

    return pd.DataFrame(orders)


def submit_paper_orders(credentials: AlpacaCredentials, orders: pd.DataFrame) -> list[dict]:
    submitted = []
    for order in orders.to_dict("records"):
        payload = {
            "symbol": order["symbol"],
            "qty": str(order["qty"]),
            "side": order["side"],
            "type": "market",
            "time_in_force": "day",
        }
        submitted.append(
            alpaca_request(
                credentials,
                PAPER_TRADING_BASE_URL,
                "/orders",
                method="POST",
                payload=payload,
            )
        )
    return submitted


def save_execution_outputs(
    orders: pd.DataFrame,
    summary: str,
) -> None:
    Path(EXECUTION_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    orders.to_csv(PAPER_ORDERS_PATH, index=False)
    Path(PAPER_EXECUTION_SUMMARY_PATH).write_text(summary)


def execute_rebalance(dry_run: bool = True) -> pd.DataFrame:
    configure_logging()
    credentials = load_credentials()
    signals = load_target_signals()
    account_equity = get_account_equity(credentials)
    positions = get_current_positions(credentials)

    symbols_to_price = sorted(set(signals["ticker"]) | set(positions))
    prices = {
        symbol: get_latest_price(credentials, symbol)
        for symbol in symbols_to_price
    }

    orders = build_rebalance_orders(signals, positions, prices, account_equity)
    if orders.empty:
        summary = (
            "No rebalance orders generated. Portfolio is already close to target "
            "or all required prices were unavailable.\n"
        )
    elif dry_run:
        summary = (
            f"DRY RUN: generated {len(orders)} simulated Alpaca paper orders. "
            "No orders were submitted.\n"
        )
    else:
        # Real-money trading is never supported here. This submits only to the
        # Alpaca paper endpoint defined above.
        submitted = submit_paper_orders(credentials, orders)
        summary = f"Submitted {len(submitted)} orders to Alpaca paper trading.\n"

    summary += (
        f"Account equity: {account_equity:.2f}\n"
        f"Current long positions: {len(positions)}\n"
        f"Target long positions: {len(signals)}\n"
        f"Max positions: {MAX_PAPER_POSITIONS}\n"
        f"Max allocation per stock: {MAX_PAPER_ALLOCATION:.2%}\n"
        "No leverage and no shorting are permitted by this engine.\n"
    )
    save_execution_outputs(orders, summary)
    logging.info(summary.strip())
    return orders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpaca paper trading rebalance engine.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit generated orders to Alpaca paper trading. Dry-run is default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        orders = execute_rebalance(dry_run=not args.execute)
        print("Paper trading execution summary")
        print()
        if orders.empty:
            print("No orders generated.")
        else:
            print(orders.to_string(index=False))
        print()
        print(f"Orders saved to {PAPER_ORDERS_PATH}")
        print(f"Summary saved to {PAPER_EXECUTION_SUMMARY_PATH}")
        if not args.execute:
            print("Dry-run mode was used. No Alpaca paper orders were submitted.")
    except Exception as exc:
        configure_logging()
        logging.error("Paper trading executor failed: %s", exc)
        Path(EXECUTION_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(PAPER_EXECUTION_SUMMARY_PATH).write_text(f"Execution failed: {exc}\n")
        print(f"Execution failed: {exc}")
        print(f"Summary saved to {PAPER_EXECUTION_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
