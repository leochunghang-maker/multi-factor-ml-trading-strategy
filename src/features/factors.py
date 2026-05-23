import pandas as pd


def calculate_factor_frames(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create the research factor matrixes used by the strategies."""
    # Daily returns are the one-day percentage changes in each stock price.
    daily_returns = prices.pct_change()

    # 12-month momentum asks: how much has the stock moved over about one
    # trading year? Positive values favor persistent medium-term trends.
    momentum_12m = prices / prices.shift(252) - 1

    # Mean reversion is the negative of the last one-month return. A stock that
    # recently fell gets a positive mean-reversion score, and vice versa.
    mean_reversion_1m = -(prices / prices.shift(21) - 1)

    # Forward returns are future returns and are only used as labels/outcomes.
    # They must never be used as live information when forming the portfolio.
    forward_returns_1m = prices.shift(-21) / prices - 1

    return {
        "daily_returns": daily_returns,
        "momentum_12m": momentum_12m,
        "mean_reversion_1m": mean_reversion_1m,
        "forward_returns_1m": forward_returns_1m,
    }


def build_ml_dataset(
    momentum: pd.DataFrame,
    mean_reversion: pd.DataFrame,
    forward_returns: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    volatility = prices.pct_change().rolling(21).std()
    dataset_rows = []

    for date in momentum.index:
        for ticker in momentum.columns:
            mom = momentum.loc[date, ticker]
            mr = mean_reversion.loc[date, ticker]
            vol = volatility.loc[date, ticker]
            target = forward_returns.loc[date, ticker]

            if pd.notna(mom) and pd.notna(mr) and pd.notna(vol) and pd.notna(target):
                dataset_rows.append({
                    "date": date,
                    "ticker": ticker,
                    "momentum_12m": mom,
                    # The ML feature is short-term momentum, so it uses the
                    # positive one-month return rather than the reversion sign.
                    "short_term_momentum": -mr,
                    "volatility_1m": vol,
                    "target_return_1m": target,
                })

    return pd.DataFrame(dataset_rows)
