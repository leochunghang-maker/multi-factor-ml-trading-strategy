import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import yfinance as yf

from src.config import END_DATE, PRICE_DATA_PATH, START_DATE, TICKERS


def download_price_data():
    Path("data").mkdir(exist_ok=True)

    print("Downloading price data...")

    data = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=True,
    )

    prices = data["Close"]

    prices.to_csv(PRICE_DATA_PATH)

    print(f"Saved price data to {PRICE_DATA_PATH}")
    print(prices.tail())

if __name__ == "__main__":
    download_price_data()
