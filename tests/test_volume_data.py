from pathlib import Path

import pandas as pd
import pytest

from src.config import MIN_AVERAGE_DAILY_DOLLAR_VOLUME, PRICE_DATA_PATH, VOLUME_DATA_PATH
from src.data_loader import extract_volume_frame
from src.portfolio.constraints import constrain_weights, load_optional_volume_data


def test_extract_volume_frame_matches_price_layout() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="B")
    columns = pd.MultiIndex.from_product([["Close", "Volume"], ["MSFT", "AAPL"]])
    data = pd.DataFrame(1.0, index=index, columns=columns)
    volumes = extract_volume_frame(data, tickers=["MSFT", "AAPL"])
    assert list(volumes.columns) == ["AAPL", "MSFT"]
    assert volumes.index.name == "Date"


def test_liquidity_filter_activates_with_volume_data() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    prices = pd.DataFrame({"LIQ": 100.0, "ILLIQ": 10.0}, index=dates)
    volumes = pd.DataFrame({"LIQ": 1_000_000.0, "ILLIQ": 1_000.0}, index=dates)
    weights = pd.Series({"LIQ": 0.05, "ILLIQ": 0.05})
    _, diag = constrain_weights(weights, prices=prices, volumes=volumes, as_of_date=dates[-1])
    assert diag["liquidity_filter_available"] is True
    assert diag["liquidity_removed_positions"] == 1


@pytest.mark.skipif(not Path(VOLUME_DATA_PATH).exists(), reason="volume data not generated")
def test_repo_volume_data_aligned_with_prices() -> None:
    prices = pd.read_csv(PRICE_DATA_PATH, index_col=0, parse_dates=True)
    volumes = load_optional_volume_data(VOLUME_DATA_PATH)
    assert volumes is not None
    assert list(volumes.columns) == list(prices.columns)
    assert volumes.index.equals(prices.index)
    assert MIN_AVERAGE_DAILY_DOLLAR_VOLUME > 0
