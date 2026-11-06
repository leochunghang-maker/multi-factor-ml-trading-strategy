import pandas as pd

from src.config import (
    ANNUAL_BORROW_BPS,
    ANNUAL_CASH_FINANCING_BPS,
    ANNUAL_SHORT_REBATE_BPS,
    PERIODS_PER_YEAR,
)
from src.execution.financing import financing_cost_from_weights


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
) -> dict[str, float]:
    long_stocks, short_stocks = select_long_short(scores, long_count, short_count)
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
) -> pd.Series:
    # gross_exposure splits evenly: gross 2.0 means 100% long and 100% short.
    side = gross_exposure / 2
    long_stocks, short_stocks = select_long_short(scores, long_count, short_count)
    weights = pd.Series(0.0, index=scores.index)
    weights.loc[long_stocks] = side / long_count
    weights.loc[short_stocks] = -side / short_count
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
            transaction_cost_amount, traded_gross_exposure = transaction_cost_from_weight_change(
                weights,
                previous_weights,
                transaction_cost,
            )
            row = long_short_return_from_scores(
                score_row[valid],
                future_returns[valid],
                long_count,
                short_count,
                transaction_cost=0.0,
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
            portfolio_return = row["portfolio_return"] - transaction_cost_amount - financing_cost
            portfolio_returns.append({
                "date": date,
                "long_return": row["long_return"],
                "short_return": row["short_return"],
                "gross_return": row["portfolio_return"],
                "transaction_cost": transaction_cost_amount,
                "financing_cost": financing_cost,
                "traded_gross_exposure": traded_gross_exposure,
                return_column: portfolio_return,
            })
            previous_weights = weights

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

        long_stocks = monthly_data.nlargest(long_count, score_column)
        short_stocks = monthly_data.nsmallest(short_count, score_column)

        long_return = long_stocks[return_column].mean()
        short_return = short_stocks[return_column].mean()
        side = gross_exposure / 2
        gross_return = side * (long_return - short_return)
        weights = pd.Series(0.0, index=monthly_data["ticker"])
        weights.loc[long_stocks["ticker"]] = side / long_count
        weights.loc[short_stocks["ticker"]] = -side / short_count
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
            "portfolio_return": portfolio_return,
        })
        previous_weights = weights

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
) -> pd.DataFrame:
    portfolio_returns = []
    monthly_dates = month_end_trading_dates(momentum)
    previous_weights = None
    side = gross_exposure / 2

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

            weights = equal_weight_long_short_weights(
                composite_score,
                long_count,
                short_count,
                gross_exposure=gross_exposure,
            )
            transaction_cost_amount, traded_gross_exposure = transaction_cost_from_weight_change(
                weights,
                previous_weights,
                transaction_cost,
            )
            row = long_short_return_from_scores(
                composite_score,
                future_returns,
                long_count,
                short_count,
                transaction_cost=0.0,
            )
            gross_return = side * row["portfolio_return"]
            name_borrow = _borrow_bps_on_date(borrow_bps_panel, date, weights.index)
            financing_cost = financing_cost_from_weights(
                weights,
                annual_borrow_bps=annual_borrow_bps,
                annual_short_rebate_bps=annual_short_rebate_bps,
                annual_cash_financing_bps=annual_cash_financing_bps,
                periods_per_year=periods_per_year,
                borrow_bps_by_symbol=name_borrow,
            )
            portfolio_returns.append({
                "date": date,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost_amount,
                "financing_cost": financing_cost,
                "traded_gross_exposure": traded_gross_exposure,
                "portfolio_return": gross_return - transaction_cost_amount - financing_cost,
            })
            previous_weights = weights

    results = pd.DataFrame(portfolio_returns)
    results["date"] = pd.to_datetime(results["date"])
    results.set_index("date", inplace=True)
    return results
