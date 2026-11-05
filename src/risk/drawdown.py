"""Drawdown helpers for paper/sim risk controls.

Paper trading already records drawdown for monitoring. These helpers turn the
configured warning threshold into a trading circuit breaker: once drawdown
breaches the limit, paper/sim rebalances are skipped so the book is frozen
(no new risk and no rotation into cash). Local simulated paper-trading only;
never live-money execution.
"""

from __future__ import annotations

from typing import Iterable, TypeVar

OrderT = TypeVar("OrderT")


def portfolio_drawdown(current_value: float, peak_value: float) -> float:
    """Return current drawdown as a fraction of peak (0.0 at the high-water mark)."""
    if peak_value <= 0:
        return 0.0
    return current_value / peak_value - 1.0


def is_drawdown_halted(drawdown: float, threshold: float) -> bool:
    """True when drawdown is at or below the (negative) threshold.

    Example: drawdown=-0.12 and threshold=-0.10 -> halted.
    A non-negative threshold disables the halt.
    """
    if threshold >= 0:
        return False
    return drawdown <= threshold


def filter_orders_for_drawdown_halt(orders: Iterable[OrderT], halted: bool) -> list[OrderT]:
    """Optional filter: when halted, keep only SELL orders.

    Paper/sim paths currently freeze the whole rebalance instead; this helper
    remains for callers that want sells-only de-risking.
    """
    if not halted:
        return list(orders)
    kept: list[OrderT] = []
    for order in orders:
        side = getattr(order, "side", None)
        if side is None and isinstance(order, dict):
            side = order.get("side")
        if str(side).upper() == "SELL":
            kept.append(order)
    return kept
