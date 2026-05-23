import pandas as pd


def current_weights(
    positions: dict[str, float],
    prices: dict[str, float],
    portfolio_value: float,
) -> pd.Series:
    if portfolio_value <= 0:
        return pd.Series(dtype=float)
    return pd.Series({
        symbol: quantity * prices.get(symbol, 0.0) / portfolio_value
        for symbol, quantity in positions.items()
    })


def build_target_weight_series(targets: pd.DataFrame) -> pd.Series:
    weights = pd.Series(
        targets["target_weight"].astype(float).to_numpy(),
        index=targets["ticker"],
        dtype=float,
    )
    return weights.groupby(level=0).sum()


def normalize_long_only_weights(weights: pd.Series, max_position_weight: float) -> pd.Series:
    # Portfolio sizing converts a ranked signal list into investable weights.
    # The allocator caps each name first, then redistributes remaining capital
    # across uncapped names so the book uses cash without weakening position limits.
    clean = weights.clip(lower=0.0, upper=max_position_weight).copy()
    if clean.sum() == 0:
        return clean

    target_total = min(1.0, len(clean[clean > 0]) * max_position_weight)
    for _ in range(20):
        gap = target_total - clean.sum()
        if abs(gap) < 1e-9:
            break
        eligible = clean[(clean > 0) & (clean < max_position_weight - 1e-9)]
        if eligible.empty:
            break
        clean.loc[eligible.index] += gap * (eligible / eligible.sum())
        clean = clean.clip(upper=max_position_weight)
    return clean


def apply_turnover_constraint(
    desired: pd.Series,
    current: pd.Series,
    max_turnover: float,
) -> tuple[pd.Series, float, bool]:
    # Turnover constraints slow down large portfolio changes. This is more
    # realistic than assuming the entire book can flip instantly at no market cost.
    universe = desired.index.union(current.index)
    desired = desired.reindex(universe).astype(float).fillna(0.0)
    current = current.reindex(universe).astype(float).fillna(0.0)
    traded_gross = (desired - current).abs().sum()
    if traded_gross <= max_turnover or traded_gross == 0:
        return desired, traded_gross, False
    blend = max_turnover / traded_gross
    adjusted = current + (desired - current) * blend
    return adjusted, max_turnover, True


def build_paper_target_weights(
    targets: pd.DataFrame,
    positions: dict[str, float],
    prices: dict[str, float],
    portfolio_value: float,
    max_position_weight: float,
    allocation_buffer: float,
    max_turnover: float,
) -> tuple[pd.Series, dict[str, float | bool]]:
    effective_max_weight = max(0.0, max_position_weight - allocation_buffer)
    raw_target = build_target_weight_series(targets)
    constrained_target = normalize_long_only_weights(raw_target, effective_max_weight)
    existing = current_weights(positions, prices, portfolio_value)
    adjusted, expected_turnover, turnover_limited = apply_turnover_constraint(
        constrained_target,
        existing,
        max_turnover,
    )
    adjusted = adjusted.clip(lower=0.0, upper=effective_max_weight)
    adjusted = normalize_long_only_weights(adjusted[adjusted > 0], effective_max_weight)

    diagnostics = {
        "raw_target_weight_sum": raw_target.sum(),
        "constrained_target_weight_sum": constrained_target.sum(),
        "final_target_weight_sum": adjusted.sum(),
        "expected_turnover": expected_turnover,
        "turnover_limited": turnover_limited,
        "expected_cash_weight": max(0.0, 1.0 - adjusted.sum()),
        "max_target_weight": adjusted.max() if not adjusted.empty else 0.0,
        "hard_max_position_weight": max_position_weight,
        "effective_max_position_weight": effective_max_weight,
        "allocation_buffer": allocation_buffer,
        "number_of_target_positions": int((adjusted > 0).sum()),
    }
    return adjusted, diagnostics


def realized_vs_target_frame(
    target_weights: pd.Series,
    positions: dict[str, float],
    prices: dict[str, float],
    portfolio_value: float,
    rejected_rows: list[dict],
) -> pd.DataFrame:
    realized = current_weights(positions, prices, portfolio_value)
    rejected_symbols = {row.get("symbol") for row in rejected_rows}
    universe = target_weights.index.union(realized.index)
    frame = pd.DataFrame(index=universe)
    frame["target_weight"] = target_weights.reindex(universe).fillna(0.0)
    frame["realized_weight"] = realized.reindex(universe).fillna(0.0)
    frame["weight_gap"] = frame["realized_weight"] - frame["target_weight"]
    frame["absolute_weight_gap"] = frame["weight_gap"].abs()
    frame["rejected_order_symbol"] = frame.index.map(lambda symbol: symbol in rejected_symbols)
    frame.index.name = "ticker"
    return frame.reset_index()
