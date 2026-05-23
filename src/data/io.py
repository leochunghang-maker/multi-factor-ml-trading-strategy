from pathlib import Path

import pandas as pd

from src.config import ML_DATASET_PATH, PRICE_DATA_PATH


def ensure_directory(path: str | Path) -> None:
    Path(path).mkdir(exist_ok=True)


def load_price_data(path: str = PRICE_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_factor_frame(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_ml_dataset(path: str = ML_DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_prediction_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_return_series(path: str, column: str) -> pd.Series:
    return pd.read_csv(path, index_col=0, parse_dates=True)[column]
