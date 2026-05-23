import pandas as pd


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
) -> pd.Series:
    long_stocks, short_stocks = select_long_short(scores, long_count, short_count)
    weights = pd.Series(0.0, index=scores.index)
    weights.loc[long_stocks] = 1 / long_count
    weights.loc[short_stocks] = -1 / short_count
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
) -> pd.DataFrame:
    portfolio_returns = []
    monthly_dates = month_end_trading_dates(scores)
    previous_weights = None

    for date in monthly_dates:
        score_row = scores.loc[date]
        future_returns = forward_returns.loc[date]
        valid = score_row.notna() & future_returns.notna()

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
            portfolio_return = row["portfolio_return"] - transaction_cost_amount
            portfolio_returns.append({
                "date": date,
                "long_return": row["long_return"],
                "short_return": row["short_return"],
                "gross_return": row["portfolio_return"],
                "transaction_cost": transaction_cost_amount,
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
) -> pd.DataFrame:
    portfolio_weights = []
    monthly_dates = month_end_trading_dates(momentum)

    for date in monthly_dates:
        mom = momentum.loc[date]
        mr = mean_reversion.loc[date]
        future_returns = forward_returns.loc[date]
        valid = mom.notna() & mr.notna() & future_returns.notna()

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
) -> pd.DataFrame:
    portfolio_returns = []
    previous_weights = None

    for date in sorted(predictions["date"].unique()):
        monthly_data = predictions[predictions["date"] == date]
        long_stocks = monthly_data.nlargest(long_count, score_column)
        short_stocks = monthly_data.nsmallest(short_count, score_column)

        long_return = long_stocks[target_column].mean()
        short_return = short_stocks[target_column].mean()
        gross_return = long_return - short_return
        weights = pd.Series(0.0, index=monthly_data["ticker"])
        weights.loc[long_stocks["ticker"]] = 1 / long_count
        weights.loc[short_stocks["ticker"]] = -1 / short_count
        transaction_cost_amount, traded_gross_exposure = transaction_cost_from_weight_change(
            weights,
            previous_weights,
            transaction_cost,
        )
        portfolio_return = gross_return - transaction_cost_amount

        portfolio_returns.append({
            "date": date,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost_amount,
            "traded_gross_exposure": traded_gross_exposure,
            "portfolio_return": portfolio_return,
        })
        previous_weights = weights

    results = pd.DataFrame(portfolio_returns)
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
) -> pd.DataFrame:
    portfolio_returns = []
    monthly_dates = month_end_trading_dates(momentum)
    previous_weights = None

    for date in monthly_dates:
        mom = momentum.loc[date]
        mr = mean_reversion.loc[date]
        future_returns = forward_returns.loc[date]
        valid = mom.notna() & mr.notna() & future_returns.notna()

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
            portfolio_returns.append({
                "date": date,
                "gross_return": row["portfolio_return"],
                "transaction_cost": transaction_cost_amount,
                "traded_gross_exposure": traded_gross_exposure,
                "portfolio_return": row["portfolio_return"] - transaction_cost_amount,
            })
            previous_weights = weights

    results = pd.DataFrame(portfolio_returns)
    results["date"] = pd.to_datetime(results["date"])
    results.set_index("date", inplace=True)
    return results
