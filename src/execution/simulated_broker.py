import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import MAX_PAPER_ALLOCATION, SIMULATED_INITIAL_CASH, SLIPPAGE_BPS, TRANSACTION_COST


@dataclass
class SimulatedOrder:
    symbol: str
    side: str
    quantity: float
    price: float
    reason: str = ""


@dataclass
class SimulatedBroker:
    cash: float = SIMULATED_INITIAL_CASH
    positions: dict[str, float] = field(default_factory=dict)
    transaction_cost_rate: float = TRANSACTION_COST
    slippage_bps: float = SLIPPAGE_BPS

    def validate_order(self, order: SimulatedOrder, prices: dict[str, float] | None = None) -> None:
        # Execution logs matter because paper trading is a rehearsal for real
        # operations. Every rejection should explain which control protected the
        # account and what needs review before the next run.
        if not math.isfinite(order.quantity):
            raise ValueError(f"Invalid non-finite quantity for {order.symbol}: {order.quantity}")
        if not math.isfinite(order.price):
            raise ValueError(f"Invalid non-finite price for {order.symbol}: {order.price}")
        if order.quantity <= 0:
            raise ValueError(f"Invalid quantity for {order.symbol}: {order.quantity}")
        if order.price <= 0:
            raise ValueError(f"Invalid price for {order.symbol}: {order.price}")
        if prices is not None and (order.symbol not in prices or prices[order.symbol] <= 0):
            raise ValueError(f"Missing executable price for {order.symbol}")
        if order.side not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported order side: {order.side}")

        fill_price = self.estimate_fill_price(order)
        notional = order.quantity * fill_price
        transaction_cost = notional * self.transaction_cost_rate

        if order.side == "BUY" and self.cash < notional + transaction_cost:
            raise ValueError(
                f"Insufficient cash for {order.symbol}: required {notional + transaction_cost:.2f}"
            )

        current_qty = self.positions.get(order.symbol, 0.0)
        if order.side == "SELL" and current_qty + 1e-9 < order.quantity:
            # Long-only mode rejects any sell that would create a negative
            # position. Selling owned shares is fine; shorting is not.
            raise ValueError(
                f"Shorting is not allowed for {order.symbol}: owned {current_qty:.6f}"
            )

    def validate_position_limits_after_order(
        self,
        order: SimulatedOrder,
        prices: dict[str, float],
        max_position_weight: float = MAX_PAPER_ALLOCATION,
    ) -> None:
        # Max position limits keep one mistaken signal, bad price, or oversized
        # fill from dominating a paper account. Institutions validate this
        # before trading because rejecting a bad order is cheaper than unwinding it.
        fill_price = self.estimate_fill_price(order)
        projected_cash = self.cash
        projected_positions = dict(self.positions)
        notional = order.quantity * fill_price
        transaction_cost = notional * self.transaction_cost_rate

        if order.side == "BUY":
            projected_cash -= notional + transaction_cost
            projected_positions[order.symbol] = projected_positions.get(order.symbol, 0.0) + order.quantity
        else:
            projected_cash += notional - transaction_cost
            remaining = projected_positions.get(order.symbol, 0.0) - order.quantity
            if remaining <= 1e-9:
                projected_positions.pop(order.symbol, None)
            else:
                projected_positions[order.symbol] = remaining

        projected_values = {
            symbol: quantity * prices.get(symbol, 0.0)
            for symbol, quantity in projected_positions.items()
        }
        current_values = {
            symbol: quantity * prices.get(symbol, 0.0)
            for symbol, quantity in self.positions.items()
        }
        current_value = self.cash + sum(current_values.values())
        projected_value = projected_cash + sum(projected_values.values())
        if projected_cash < -1e-6:
            raise ValueError(f"Order would create negative cash for {order.symbol}")
        if projected_value <= 0:
            raise ValueError("Order would create non-positive portfolio value")

        projected_symbol_weight = (
            abs(projected_values.get(order.symbol, 0.0)) / projected_value
            if projected_value else 0.0
        )
        current_symbol_weight = (
            abs(current_values.get(order.symbol, 0.0)) / current_value
            if current_value else 0.0
        )
        if (
            projected_symbol_weight > max_position_weight + 1e-9
            and projected_symbol_weight >= current_symbol_weight - 1e-9
        ):
            raise ValueError(
                f"Order would breach max position weight: {projected_symbol_weight:.2%} > {max_position_weight:.2%}"
            )

        # The symbol-level check above is the binding pre-trade control. An
        # unrelated legacy overweight can exist while the allocator gradually
        # repairs it; it should not block buys that remain below the cap.

    def estimate_fill_price(self, order: SimulatedOrder) -> float:
        slippage = self.slippage_bps / 10_000
        if order.side == "BUY":
            return order.price * (1 + slippage)
        return order.price * (1 - slippage)

    def process_order(
        self,
        order: SimulatedOrder,
        timestamp: str,
        prices: dict[str, float] | None = None,
        max_position_weight: float = MAX_PAPER_ALLOCATION,
    ) -> dict:
        # The simulator fills immediately, then applies side-aware slippage to
        # make cash accounting less optimistic than decision-price execution.
        self.validate_order(order, prices)
        if prices is not None:
            self.validate_position_limits_after_order(
                order,
                prices,
                max_position_weight=max_position_weight,
            )

        fill_price = self.estimate_fill_price(order)
        notional = order.quantity * fill_price
        transaction_cost = notional * self.transaction_cost_rate

        if order.side == "BUY":
            self.cash -= notional + transaction_cost
            self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + order.quantity
        else:
            self.cash += notional - transaction_cost
            remaining = self.positions.get(order.symbol, 0.0) - order.quantity
            if remaining <= 1e-9:
                self.positions.pop(order.symbol, None)
            else:
                self.positions[order.symbol] = remaining

        return {
            "timestamp": timestamp,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "decision_price": order.price,
            "fill_price": fill_price,
            "slippage_bps": self.slippage_bps,
            "notional": notional,
            "transaction_cost": transaction_cost,
            "cash_after": self.cash,
            "reason": order.reason,
            "status": "FILLED",
        }

    def position_values(self, prices: dict[str, float]) -> dict[str, float]:
        # Position accounting converts share counts into current dollar values.
        # Missing prices are valued at zero in the snapshot and should be logged
        # by the simulation engine before orders are generated.
        return {
            symbol: quantity * prices.get(symbol, 0.0)
            for symbol, quantity in self.positions.items()
        }

    def position_snapshot_rows(self, prices: dict[str, float], timestamp: str) -> list[dict]:
        portfolio_value = self.portfolio_value(prices)
        rows = []
        for symbol, quantity in sorted(self.positions.items()):
            price = prices.get(symbol, 0.0)
            market_value = quantity * price
            rows.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "market_value": market_value,
                "weight": market_value / portfolio_value if portfolio_value else 0.0,
            })
        return rows

    def portfolio_value(self, prices: dict[str, float]) -> float:
        # Portfolio value is cash plus the market value of all open positions.
        return self.cash + sum(self.position_values(prices).values())

    def reconcile(self, prices: dict[str, float], expected_value: float, tolerance: float = 0.01) -> dict:
        actual_value = self.portfolio_value(prices)
        difference = actual_value - expected_value
        return {
            "expected_portfolio_value": expected_value,
            "actual_portfolio_value": actual_value,
            "reconciliation_difference": difference,
            "reconciled": abs(difference) <= tolerance,
        }

    def snapshot(self, prices: dict[str, float], timestamp: str) -> dict:
        position_values = self.position_values(prices)
        portfolio_value = self.cash + sum(position_values.values())
        gross_exposure = sum(abs(value) for value in position_values.values())
        net_exposure = sum(position_values.values())
        missing_prices = sorted(set(self.positions) - set(prices))
        return {
            "timestamp": timestamp,
            "cash": self.cash,
            "positions_value": sum(position_values.values()),
            "portfolio_value": portfolio_value,
            "cash_weight": self.cash / portfolio_value if portfolio_value else 0.0,
            "gross_exposure": gross_exposure / portfolio_value if portfolio_value else 0.0,
            "net_exposure": net_exposure / portfolio_value if portfolio_value else 0.0,
            "number_of_positions": len(self.positions),
            "missing_price_count": len(missing_prices),
            "missing_price_symbols": ",".join(missing_prices),
        }

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "positions": self.positions,
            "transaction_cost_rate": self.transaction_cost_rate,
            "slippage_bps": self.slippage_bps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimulatedBroker":
        return cls(
            cash=float(data.get("cash", SIMULATED_INITIAL_CASH)),
            positions={k: float(v) for k, v in data.get("positions", {}).items()},
            transaction_cost_rate=float(data.get("transaction_cost_rate", TRANSACTION_COST)),
            slippage_bps=float(data.get("slippage_bps", SLIPPAGE_BPS)),
        )

    @classmethod
    def load(cls, path: str) -> "SimulatedBroker":
        state_path = Path(path)
        if not state_path.exists():
            return cls()
        return cls.from_dict(json.loads(state_path.read_text()))

    def save(self, path: str) -> None:
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def append_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(output_path, index=False)
