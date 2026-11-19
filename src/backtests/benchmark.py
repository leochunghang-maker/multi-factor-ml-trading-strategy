from pathlib import Path

import pandas as pd

from src.config import BENCHMARK_PRICES_PATH, PRICE_DATA_PATH
from src.data.io import load_price_data
from src.portfolio import month_end_trading_dates


def _benchmark_prices_from_local_data(symbol: str) -> pd.Series | None:
    for path in (PRICE_DATA_PATH, BENCHMARK_PRICES_PATH):
        if not Path(path).exists():
            continue
        prices = load_price_data(path)
        if symbol in prices.columns:
            return prices[symbol].dropna()
    return None


def _monthly_returns_from_prices(prices: pd.Series) -> pd.Series:
    monthly_prices = prices.loc[month_end_trading_dates(prices)]
    return monthly_prices.pct_change().dropna()


def load_benchmark_returns(symbol: str, start, end) -> pd.Series:
    local_prices = _benchmark_prices_from_local_data(symbol)
    if local_prices is None:
        raise RuntimeError(
            f"Benchmark prices for {symbol} are not in {PRICE_DATA_PATH} or "
            f"{BENCHMARK_PRICES_PATH}. Restore the research snapshot with "
            "`git checkout -- data/benchmark_prices.csv` or refresh with "
            "`python src/data_loader.py`. This function does not download."
        )
    local_prices = local_prices.loc[
        (local_prices.index >= pd.Timestamp(start))
        & (local_prices.index <= pd.Timestamp(end))
    ]
    if local_prices.empty:
        raise RuntimeError(
            f"Benchmark prices for {symbol} have no rows between {start} and {end} "
            f"in {PRICE_DATA_PATH} or {BENCHMARK_PRICES_PATH}."
        )
    return _monthly_returns_from_prices(local_prices)


def load_benchmark_equity_curve(symbol: str, start, end) -> pd.Series:
    benchmark_returns = load_benchmark_returns(symbol, start, end)
    return (1 + benchmark_returns).cumprod()
