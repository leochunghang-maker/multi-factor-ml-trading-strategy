import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    DAILY_RETURNS_PATH,
    FORWARD_RETURNS_PATH,
    MEAN_REVERSION_PATH,
    MOMENTUM_PATH,
)
from src.data.io import load_price_data
from src.features.factors import calculate_factor_frames


def main() -> None:
    prices = load_price_data()
    factor_frames = calculate_factor_frames(prices)

    factor_frames["daily_returns"].to_csv(DAILY_RETURNS_PATH)
    factor_frames["momentum_12m"].to_csv(MOMENTUM_PATH)
    factor_frames["mean_reversion_1m"].to_csv(MEAN_REVERSION_PATH)
    factor_frames["forward_returns_1m"].to_csv(FORWARD_RETURNS_PATH)

    print("Factor calculations completed.")
    print("Saved momentum_12m, mean_reversion_1m, and forward_returns_1m.")


if __name__ == "__main__":
    main()
