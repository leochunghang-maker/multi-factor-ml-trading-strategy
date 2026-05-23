import numpy as np
import pandas as pd

from src.config import MAX_LEVERAGE


def apply_volatility_target(
    returns: pd.Series,
    target_volatility: float,
    lookback: int,
    periods_per_year: int = 12,
    max_leverage: float = MAX_LEVERAGE,
) -> pd.Series:
    # Volatility targeting scales exposure after observing recent volatility.
    # The one-period shift avoids using this month's return to size this month.
    rolling_vol = returns.rolling(lookback).std() * np.sqrt(periods_per_year)
    leverage = target_volatility / rolling_vol
    leverage = leverage.clip(upper=max_leverage).fillna(1.0)
    return (returns * leverage.shift(1)).dropna()
