import pandas as pd
import yfinance as yf

from src.config import PRICE_DATA_PATH
from src.data.io import load_price_data
from src.portfolio import month_end_trading_dates


def _benchmark_prices_from_local_data(symbol: str) -> pd.Series | None:
    prices = load_price_data(PRICE_DATA_PATH)
    if symbol not in prices.columns:
        return None
    return prices[symbol].dropna()


def _monthly_returns_from_prices(prices: pd.Series) -> pd.Series:
    monthly_prices = prices.loc[month_end_trading_dates(prices)]
    return monthly_prices.pct_change().dropna()


def load_benchmark_returns(symbol: str, start, end) -> pd.Series:
    local_prices = _benchmark_prices_from_local_data(symbol)
    if local_prices is not None:
        local_prices = local_prices.loc[
            (local_prices.index >= pd.Timestamp(start))
            & (local_prices.index <= pd.Timestamp(end))
        ]
        return _monthly_returns_from_prices(local_prices)

    benchmark = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]
    if benchmark.empty:
        raise RuntimeError(
            f"Benchmark data for {symbol} is unavailable locally and could not be downloaded."
        )
    benchmark_returns = _monthly_returns_from_prices(benchmark)
    if isinstance(benchmark_returns, pd.DataFrame):
        benchmark_returns = benchmark_returns.squeeze()
    return benchmark_returns


def load_benchmark_equity_curve(symbol: str, start, end) -> pd.Series:
    benchmark_returns = load_benchmark_returns(symbol, start, end)
    return (1 + benchmark_returns).cumprod()
