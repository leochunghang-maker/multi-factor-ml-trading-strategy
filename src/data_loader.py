import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yfinance as yf

from src.config import END_DATE, PRICE_DATA_PATH, START_DATE, TICKERS, VOLUME_DATA_PATH
from src.data.universe import drop_empty_price_columns


def _download(tickers=TICKERS, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    return yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
    )


def extract_volume_frame(data: pd.DataFrame, tickers=TICKERS) -> pd.DataFrame:
    """Return a date x ticker share-volume panel (same layout as price_data.csv)."""
    volumes = data["Volume"]
    if isinstance(volumes, pd.Series):
        volumes = volumes.to_frame(name=tickers[0] if len(tickers) == 1 else "Volume")
    volumes = volumes.reindex(columns=sorted(volumes.columns))
    volumes.index.name = "Date"
    return volumes


def save_volume_data(data: pd.DataFrame, path: str = VOLUME_DATA_PATH) -> pd.DataFrame:
    volumes = extract_volume_frame(data)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    volumes.to_csv(path)
    print(f"Saved volume data to {path}")
    return volumes


def download_price_data(include_volume: bool = True):
    Path("data").mkdir(exist_ok=True)

    print("Downloading price data...")
    data = _download()
    prices = drop_empty_price_columns(data["Close"])
    prices.to_csv(PRICE_DATA_PATH)
    print(f"Saved price data to {PRICE_DATA_PATH}")
    print(prices.tail())

    if include_volume:
        save_volume_data(data)


def download_volume_data():
    """Download only share volume, leaving the existing price panel untouched."""
    return save_volume_data(_download())


if __name__ == "__main__":
    if "--volume-only" in sys.argv:
        download_volume_data()
    else:
        download_price_data()
