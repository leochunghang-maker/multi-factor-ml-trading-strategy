import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FORWARD_RETURNS_PATH, MEAN_REVERSION_PATH, ML_DATASET_PATH, MOMENTUM_PATH
from src.data.io import load_factor_frame, load_price_data
from src.features.factors import build_ml_dataset


def main() -> None:
    momentum = load_factor_frame(MOMENTUM_PATH)
    mean_reversion = load_factor_frame(MEAN_REVERSION_PATH)
    forward_returns = load_factor_frame(FORWARD_RETURNS_PATH)
    prices = load_price_data()

    ml_dataset = build_ml_dataset(momentum, mean_reversion, forward_returns, prices)

    ml_dataset.to_csv(
        ML_DATASET_PATH,
        index=False,
    )

    print("ML dataset created.")
    print()
    print(ml_dataset.head())
    print()
    print("Dataset shape:", ml_dataset.shape)


if __name__ == "__main__":
    main()
