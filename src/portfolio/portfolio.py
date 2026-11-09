from pathlib import Path

import pandas as pd

from src.config import (
    ANNUAL_BORROW_BPS,
    ANNUAL_CASH_FINANCING_BPS,
    ANNUAL_SHORT_REBATE_BPS,
    BETA_NEUTRAL,
    BETA_NEUTRAL_MAX_LEG_RATIO,
    BETA_NEUTRAL_MAX_ROUNDS,
    MAX_SECTOR_NET_EXPOSURE,
    PERIODS_PER_YEAR,
    SECTOR_MAP_PATH,
    SECTOR_NEUTRAL,
)
from src.execution.financing import financing_cost_from_weights
from src.risk.beta import beta_neutralize_with_sector_caps, betas_on_date
from src.risk.volatility import scale_weights, volatility_target_scale


def _beta_neutral_weights(
    weights: pd.Series,
    betas: pd.Series | None,
    *,
    net_cap: float | None,
    sector_map: dict[str, str] | None,
    gross_exposure: float,
    beta_neutral: bool | None = None,
) -> pd.Series:
    """Scale L/S legs to zero ex-ante beta, alternating with sector net caps."""
    use = BETA_NEUTRAL if beta_neutral is None else bool(beta_neutral)
    if not use or betas is None or weights.empty:
        return weights
    caps = None
    if net_cap is not None and net_cap < float("inf"):
        def caps(w: pd.Series) -> pd.Series:
            return apply_sector_net_exposure_limits(
                w,
                max_sector_net_exposure=net_cap,
                sector_map=sector_map,
                target_gross_exposure=gross_exposure,
            )
    return beta_neutralize_with_sector_caps(
        weights,
        betas,
        max_leg_ratio=BETA_NEUTRAL_MAX_LEG_RATIO,
        apply_sector_caps=caps,
        max_rounds=BETA_NEUTRAL_MAX_ROUNDS,
    )


def _borrow_bps_on_date(
    borrow_bps_panel: pd.DataFrame | None,
    date,
    universe: pd.Index,
) -> pd.Series | None:
    """Per-name annual borrow bps for ``date``, or None for flat-rate financing."""
    if borrow_bps_panel is None or borrow_bps_panel.empty:
        return None
    stamp = pd.Timestamp(date)
    if stamp not in borrow_bps_panel.index:
        prior = borrow_bps_panel.index[borrow_bps_panel.index <= stamp]
        if len(prior) == 0:
            return None
        stamp = prior[-1]
    return borrow_bps_panel.loc[stamp].reindex(universe)




def month_end_trading_dates(frame: pd.DataFrame | pd.Series) -> pd.DatetimeIndex:
    """Return the last available observation in each calendar month."""
    index = pd.DatetimeIndex(frame.index).sort_values()
    if index.empty:
        return index
    return (
        pd.Series(index, index=index)
        .groupby(index.to_period("M"))
        .last()
        .pipe(pd.DatetimeIndex)
    )



def month_end_holding_period_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Non-overlapping returns from each month-end to the next month-end.

    Indexed by the rebalance (entry) date. The final month-end has no forward
    holding period and is therefore NaN.
    """
    monthly_prices = prices.loc[month_end_trading_dates(prices)].sort_index()
    return monthly_prices.shift(-1) / monthly_prices - 1


def _eligible_on_date(
    eligibility: pd.DataFrame | None,
    date,
    universe: pd.Index,
) -> pd.Series:
    """Boolean Series aligned to ``universe``; all-True when eligibility is None."""
    if eligibility is None:
        return pd.Series(True, index=universe)
    stamp = pd.Timestamp(date)
    if stamp not in eligibility.index:
        prior = eligibility.index[eligibility.index <= stamp]
        if len(prior) == 0:
            return pd.Series(False, index=universe)
        stamp = prior[-1]
    row = eligibility.loc[stamp].reindex(universe)
    return row.fillna(False).astype(bool)





def load_sector_map(path: str | None = None) -> dict[str, str]:
    """Load ticker→sector map from CSV when present, else ``src.config.SECTOR_MAP``."""
    from src.config import SECTOR_MAP

    map_path = Path(path or SECTOR_MAP_PATH)
    if map_path.exists():
        frame = pd.read_csv(map_path)
        return {
            str(row.ticker).upper().replace(".", "-"): str(row.sector)
            for row in frame.itertuples(index=False)
        }
    return dict(SECTOR_MAP)


def demean_scores_by_sector(
    scores: pd.Series,
    sector_map: dict[str, str] | None = None,
) -> pd.Series:
    """Subtract each name's sector-mean score (cross-sectional sector neutralization)."""
    if scores.empty:
        return scores.copy()
    sectors = sector_map or load_sector_map()
    labels = pd.Index(
        [sectors.get(str(ticker).upper().replace(".", "-"), "Unknown") for ticker in scores.index]
    )
    demeaned = scores.copy().astype(float)
    for sector in pd.unique(labels):
        mask = labels == sector
        group = demeaned.loc[mask]
        demeaned.loc[mask] = group - group.mean()
    return demeaned


def apply_sector_net_exposure_limits(
    weights: pd.Series,
    max_sector_net_exposure: float,
    sector_map: dict[str, str] | None = None,
    target_gross_exposure: float | None = None,
) -> pd.Series:
    """Cap |net| sector exposure via within-sector shifts; restore gross if feasible.

    Excess sector net is removed by a uniform within-sector weight shift so the
    residual |net| equals ``max_sector_net_exposure``. Gross is then scaled up
    toward ``target_gross_exposure`` only as far as the sector-net caps allow —
    a pure one-sector-long / one-sector-short book therefore keeps lower gross
    rather than re-inflating banned sector bets.
    """
    if weights.empty:
        return weights.copy()
    if max_sector_net_exposure < 0:
        raise ValueError("max_sector_net_exposure must be non-negative")

    sectors = sector_map or load_sector_map()
    if target_gross_exposure is None:
        target_gross = float(weights.abs().sum())
    else:
        target_gross = float(target_gross_exposure)

    constrained = weights.astype(float).copy()
    sector_names: dict[str, list] = {}
    for symbol in constrained.index:
        key = sectors.get(str(symbol).upper().replace(".", "-"), "Unknown")
        sector_names.setdefault(key, []).append(symbol)

    for names in sector_names.values():
        net = float(constrained.loc[names].sum())
        if abs(net) <= max_sector_net_exposure + 1e-12:
            continue
        target_net = max_sector_net_exposure if net > 0 else -max_sector_net_exposure
        constrained.loc[names] = constrained.loc[names] - (net - target_net) / len(names)

    gross = float(constrained.abs().sum())
    if gross <= 1e-12 or target_gross <= 0:
        return constrained

    allowed_scale = target_gross / gross
    for names in sector_names.values():
        net = float(constrained.loc[names].sum())
        if abs(net) > 1e-12:
            allowed_scale = min(allowed_scale, max_sector_net_exposure / abs(net))
    if allowed_scale > 0:
        constrained *= allowed_scale
    return constrained




def select_long_short(
    scores: pd.Series,
    long_count: int,
    short_count: int,
) -> tuple[pd.Index, pd.Index]:
    # Portfolio construction is cross-sectional: each rebalance ranks stocks
    # against each other on the same date, then buys the highest scores and
    # shorts the lowest scores with equal weights.
    long_stocks = scores.nlargest(long_count).index
    short_stocks = scores.nsmallest(short_count).index
    return long_stocks, short_stocks


def long_short_return_from_scores(
    scores: pd.Series,
    realized_returns: pd.Series,
    long_count: int,
    short_count: int,
    transaction_cost: float = 0.0,
    *,
    sector_neutral: bool | None = None,
    sector_map: dict[str, str] | None = None,
) -> dict[str, float]:
    use_neutral = SECTOR_NEUTRAL if sector_neutral is None else bool(sector_neutral)
    ranking_scores = (
        demean_scores_by_sector(scores, sector_map or load_sector_map())
        if use_neutral
        else scores
    )
    long_stocks, short_stocks = select_long_short(ranking_scores, long_count, short_count)
    long_return = realized_returns[long_stocks].mean()
    short_return = realized_returns[short_stocks].mean()
    gross_return = long_return - short_return
    portfolio_return = gross_return - transaction_cost * 2

    return {
        "long_return": long_return,
        "short_return": short_return,
        "portfolio_return": portfolio_return,
    }


def equal_weight_long_short_weights(
    scores: pd.Series,
    long_count: int,
    short_count: int,
    gross_exposure: float = 2.0,
    *,
    sector_neutral: bool | None = None,
    max_sector_net_exposure: float | None = None,
    sector_map: dict[str, str] | None = None,
    betas: pd.Series | None = None,
    beta_neutral: bool | None = None,
) -> pd.Series:
    # gross_exposure splits evenly: gross 2.0 means 100% long and 100% short.
    # Optional sector neutralization demeans scores within sector before ranking;
    # optional net-exposure caps shrink concentrated sector bets then restore gross.
    use_neutral = SECTOR_NEUTRAL if sector_neutral is None else bool(sector_neutral)
    net_cap = (
        MAX_SECTOR_NET_EXPOSURE
        if max_sector_net_exposure is None
        else float(max_sector_net_exposure)
    )
    sectors = sector_map or load_sector_map()
    ranking_scores = demean_scores_by_sector(scores, sectors) if use_neutral else scores
    side = gross_exposure / 2
    long_stocks, short_stocks = select_long_short(ranking_scores, long_count, short_count)
    weights = pd.Series(0.0, index=scores.index)
    weights.loc[long_stocks] = side / long_count
    weights.loc[short_stocks] = -side / short_count
    if net_cap is not None and net_cap < float("inf"):
        weights = apply_sector_net_exposure_limits(
            weights,
            max_sector_net_exposure=net_cap,
            sector_map=sectors,
            target_gross_exposure=gross_exposure,
        )
    # Optional beta-neutral legs (betas must be causal, known at the rebalance date).
    weights = _beta_neutral_weights(
        weights,
        betas,
        net_cap=net_cap,
        sector_map=sectors,
        gross_exposure=gross_exposure,
        beta_neutral=beta_neutral,
    )
    return weights


def transaction_cost_from_weight_change(
    current_weights: pd.Series,
    previous_weights: pd.Series | None,
    transaction_cost: float,
) -> tuple[float, float]:
    if previous_weights is None:
        previous_weights = pd.Series(0.0, index=current_weights.index)

    previous_weights = previous_weights.reindex(current_weights.index).fillna(0.0)
    traded_gross_exposure = (current_weights - previous_weights).abs().sum()
    return traded_gross_exposure * transaction_cost, traded_gross_exposure


def build_factor_long_short_returns(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    long_count: int,
    short_count: int,
    min_assets: int,
    transaction_cost: float = 0.0,
    return_column: str = "portfolio_return",
    eligibility: pd.DataFrame | None = None,
    annual_borrow_bps: float = ANNUAL_BORROW_BPS,
    annual_short_rebate_bps: float = ANNUAL_SHORT_REBATE_BPS,
    annual_cash_financing_bps: float = ANNUAL_CASH_FINANCING_BPS,
    periods_per_year: float = PERIODS_PER_YEAR,
    borrow_bps_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    portfolio_returns = []
    monthly_dates = month_end_trading_dates(scores)
    previous_weights = None
    past_portfolio_returns: list[float] = []

    for date in monthly_dates:
        score_row = scores.loc[date]
        future_returns = forward_returns.loc[date]
        eligible = _eligible_on_date(eligibility, date, score_row.index)
        valid = score_row.notna() & future_returns.notna() & eligible

        if valid.sum() >= min_assets:
            weights = equal_weight_long_short_weights(
                score_row[valid],
                long_count,
                short_count,
            )
            vol_scale = volatility_target_scale(past_portfolio_returns)
            weights = scale_weights(weights, vol_scale)
            transaction_cost_amount, traded_gross_exposure = transaction_cost_from_weight_change(
                weights,
                previous_weights,
                transaction_cost,
            )
            aligned_returns = future_returns[valid].reindex(weights.index).fillna(0.0)
            long_mask = weights > 0
            short_mask = weights < 0
            long_w = weights.loc[long_mask]
            short_w = weights.loc[short_mask]
            long_return = (
                float((long_w * aligned_returns.loc[long_mask]).sum() / long_w.sum())
                if float(long_w.sum()) > 0
                else 0.0
            )
            short_return = (
                float(((-short_w) * aligned_returns.loc[short_mask]).sum() / (-short_w).sum())
                if float((-short_w).sum()) > 0
                else 0.0
            )
            gross_return = float((weights * aligned_returns).sum())
            name_borrow = _borrow_bps_on_date(borrow_bps_panel, date, weights.index)
            financing_cost = financing_cost_from_weights(
                weights,
                annual_borrow_bps=annual_borrow_bps,
                annual_short_rebate_bps=annual_short_rebate_bps,
                annual_cash_financing_bps=annual_cash_financing_bps,
                periods_per_year=periods_per_year,
                borrow_bps_by_symbol=name_borrow,
            )
            portfolio_return = gross_return - transaction_cost_amount - financing_cost
            portfolio_returns.append({
                "date": date,
                "long_return": long_return,
                "short_return": short_return,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost_amount,
                "financing_cost": financing_cost,
                "traded_gross_exposure": traded_gross_exposure,
                "vol_scale": vol_scale,
                return_column: portfolio_return,
            })
            previous_weights = weights
            past_portfolio_returns.append(float(portfolio_return))

    results = pd.DataFrame(portfolio_returns)
    results["date"] = pd.to_datetime(results["date"])
    results.set_index("date", inplace=True)
    return results


def build_multi_factor_long_short_weights(
    momentum: pd.DataFrame,
    mean_reversion: pd.DataFrame,
    forward_returns: pd.DataFrame,
    long_count: int,
    short_count: int,
    min_assets: int,
    eligibility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    portfolio_weights = []
    monthly_dates = month_end_trading_dates(momentum)

    for date in monthly_dates:
        mom = momentum.loc[date]
        mr = mean_reversion.loc[date]
        future_returns = forward_returns.loc[date]
        eligible = _eligible_on_date(eligibility, date, mom.index)
        valid = mom.notna() & mr.notna() & future_returns.notna() & eligible

        if valid.sum() >= min_assets:
            mom = mom[valid]
            mr = mr[valid]

            # Use the same composite score as the multi-factor backtest so
            # turnover analytics describe the actual reported strategy.
            mom_z = (mom - mom.mean()) / mom.std()
            mr_z = (mr - mr.mean()) / mr.std()
            composite_score = mom_z - mr_z

            weights = equal_weight_long_short_weights(
                composite_score,
                long_count,
                short_count,
            )
            weights.name = date
            portfolio_weights.append(weights)

    return pd.DataFrame(portfolio_weights).fillna(0.0)


def build_prediction_long_short_returns(
    predictions: pd.DataFrame,
    score_column: str,
    target_column: str,
    long_count: int,
    short_count: int,
    transaction_cost: float = 0.0,
    realized_returns: pd.DataFrame | None = None,
    rebalance_on_month_ends: bool = True,
    gross_exposure: float = 2.0,
    eligibility: pd.DataFrame | None = None,
    annual_borrow_bps: float = ANNUAL_BORROW_BPS,
    annual_short_rebate_bps: float = ANNUAL_SHORT_REBATE_BPS,
    annual_cash_financing_bps: float = ANNUAL_CASH_FINANCING_BPS,
    periods_per_year: float = PERIODS_PER_YEAR,
    borrow_bps_panel: pd.DataFrame | None = None,
    beta_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build long/short portfolio returns from cross-sectional predictions.

    By default portfolios are formed only on month-end trading dates and P&L uses
    non-overlapping holding-period returns (month-end to next month-end) when a
    realized_returns panel is supplied. This avoids treating overlapping daily
    forward-return labels as independent period returns.
    """
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    unique_dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    if rebalance_on_month_ends:
        rebalance_dates = month_end_trading_dates(pd.Series(0.0, index=unique_dates))
    else:
        rebalance_dates = unique_dates

    portfolio_returns = []
    previous_weights = None
    past_portfolio_returns: list[float] = []

    for date in rebalance_dates:
        monthly_data = frame[frame["date"] == date]
        if monthly_data.empty:
            continue
        if eligibility is not None:
            allowed_tickers = _eligible_on_date(
                eligibility,
                date,
                pd.Index(monthly_data["ticker"].unique()),
            )
            monthly_data = monthly_data[
                monthly_data["ticker"].map(allowed_tickers).fillna(False)
            ]
            if monthly_data.empty:
                continue

        if realized_returns is not None:
            if date not in realized_returns.index:
                continue
            realized = realized_returns.loc[date].reindex(monthly_data["ticker"].values)
            monthly_data = monthly_data.assign(_realized_return=realized.values)
            return_column = "_realized_return"
            if monthly_data[return_column].isna().all():
                continue
            monthly_data = monthly_data.dropna(subset=[return_column, score_column])
            if monthly_data.empty:
                continue
        else:
            return_column = target_column
            monthly_data = monthly_data.dropna(subset=[return_column, score_column])
            if monthly_data.empty:
                continue

        ranking = monthly_data.set_index("ticker")[score_column]
        if SECTOR_NEUTRAL:
            ranking = demean_scores_by_sector(ranking)
        ranked = monthly_data.copy()
        ranked["_rank_score"] = ranked["ticker"].map(ranking)
        long_stocks = ranked.nlargest(long_count, "_rank_score")
        short_stocks = ranked.nsmallest(short_count, "_rank_score")

        side = gross_exposure / 2
        weights = pd.Series(0.0, index=monthly_data["ticker"])
        weights.loc[long_stocks["ticker"]] = side / long_count
        weights.loc[short_stocks["ticker"]] = -side / short_count
        if MAX_SECTOR_NET_EXPOSURE < float("inf"):
            weights = apply_sector_net_exposure_limits(
                weights,
                max_sector_net_exposure=MAX_SECTOR_NET_EXPOSURE,
                target_gross_exposure=gross_exposure,
            )
        betas = betas_on_date(beta_panel, date, weights.index)
        weights = _beta_neutral_weights(
            weights,
            betas,
            net_cap=MAX_SECTOR_NET_EXPOSURE,
            sector_map=None,
            gross_exposure=gross_exposure,
        )
        ex_ante_beta = float((weights * betas).sum()) if betas is not None else float("nan")
        vol_scale = volatility_target_scale(past_portfolio_returns)
        weights = scale_weights(weights, vol_scale)
        aligned = monthly_data.set_index("ticker")[return_column].reindex(weights.index).fillna(0.0)
        gross_return = float((weights * aligned).sum())
        transaction_cost_amount, traded_gross_exposure = transaction_cost_from_weight_change(
            weights,
            previous_weights,
            transaction_cost,
        )
        name_borrow = _borrow_bps_on_date(borrow_bps_panel, date, weights.index)
        financing_cost = financing_cost_from_weights(
            weights,
            annual_borrow_bps=annual_borrow_bps,
            annual_short_rebate_bps=annual_short_rebate_bps,
            annual_cash_financing_bps=annual_cash_financing_bps,
            periods_per_year=periods_per_year,
            borrow_bps_by_symbol=name_borrow,
        )
        portfolio_return = gross_return - transaction_cost_amount - financing_cost

        portfolio_returns.append({
            "date": date,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost_amount,
            "financing_cost": financing_cost,
            "traded_gross_exposure": traded_gross_exposure,
            "vol_scale": vol_scale,
            "ex_ante_beta": ex_ante_beta,
            "portfolio_return": portfolio_return,
        })
        previous_weights = weights
        past_portfolio_returns.append(float(portfolio_return))

    results = pd.DataFrame(portfolio_returns)
    if results.empty:
        return results
    results["date"] = pd.to_datetime(results["date"])
    results.set_index("date", inplace=True)
    return results


def build_multi_factor_long_short_returns(
    momentum: pd.DataFrame,
    mean_reversion: pd.DataFrame,
    forward_returns: pd.DataFrame,
    long_count: int,
    short_count: int,
    min_assets: int,
    transaction_cost: float,
    gross_exposure: float = 2.0,
    eligibility: pd.DataFrame | None = None,
    annual_borrow_bps: float = ANNUAL_BORROW_BPS,
    annual_short_rebate_bps: float = ANNUAL_SHORT_REBATE_BPS,
    annual_cash_financing_bps: float = ANNUAL_CASH_FINANCING_BPS,
    periods_per_year: float = PERIODS_PER_YEAR,
    borrow_bps_panel: pd.DataFrame | None = None,
    beta_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    portfolio_returns = []
    monthly_dates = month_end_trading_dates(momentum)
    previous_weights = None
    past_portfolio_returns: list[float] = []

    for date in monthly_dates:
        mom = momentum.loc[date]
        mr = mean_reversion.loc[date]
        future_returns = forward_returns.loc[date]
        eligible = _eligible_on_date(eligibility, date, mom.index)
        valid = mom.notna() & mr.notna() & future_returns.notna() & eligible

        if valid.sum() >= min_assets:
            mom = mom[valid]
            mr = mr[valid]
            future_returns = future_returns[valid]

            # Z-scores put different factors on the same scale. Here, high
            # momentum is good, while high mean-reversion score means the stock
            # recently fell, so the strategy subtracts that reversion component.
            mom_z = (mom - mom.mean()) / mom.std()
            mr_z = (mr - mr.mean()) / mr.std()
            composite_score = mom_z - mr_z

            betas = betas_on_date(beta_panel, date, composite_score.index)
            weights = equal_weight_long_short_weights(
                composite_score,
                long_count,
                short_count,
                gross_exposure=gross_exposure,
                betas=betas,
            )
            ex_ante_beta = float((weights * betas).sum()) if betas is not None else float("nan")
            vol_scale = volatility_target_scale(past_portfolio_returns)
            weights = scale_weights(weights, vol_scale)
            transaction_cost_amount, traded_gross_exposure = transaction_cost_from_weight_change(
                weights,
                previous_weights,
                transaction_cost,
            )
            aligned_returns = future_returns.reindex(weights.index).fillna(0.0)
            gross_return = float((weights * aligned_returns).sum())
            name_borrow = _borrow_bps_on_date(borrow_bps_panel, date, weights.index)
            financing_cost = financing_cost_from_weights(
                weights,
                annual_borrow_bps=annual_borrow_bps,
                annual_short_rebate_bps=annual_short_rebate_bps,
                annual_cash_financing_bps=annual_cash_financing_bps,
                periods_per_year=periods_per_year,
                borrow_bps_by_symbol=name_borrow,
            )
            portfolio_return = gross_return - transaction_cost_amount - financing_cost
            portfolio_returns.append({
                "date": date,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost_amount,
                "financing_cost": financing_cost,
                "traded_gross_exposure": traded_gross_exposure,
                "vol_scale": vol_scale,
                "ex_ante_beta": ex_ante_beta,
                "portfolio_return": portfolio_return,
            })
            previous_weights = weights
            past_portfolio_returns.append(float(portfolio_return))

    results = pd.DataFrame(portfolio_returns)
    results["date"] = pd.to_datetime(results["date"])
    results.set_index("date", inplace=True)
    return results
