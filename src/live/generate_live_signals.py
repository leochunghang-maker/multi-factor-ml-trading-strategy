import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import yfinance as yf

from src.config import (
    FEATURES,
    LIVE_LONG_COUNT,
    LIVE_MODEL_PATH,
    LIVE_PRICE_LOOKBACK_PERIOD,
    LIVE_RESULTS_DIR,
    LIVE_SIGNALS_PATH,
    ML_DATASET_PATH,
    TARGET,
    TICKERS,
)
from src.data.io import load_ml_dataset
from src.models.ml import train_random_forest


def download_latest_prices() -> pd.DataFrame:
    # Pull enough recent history to calculate 12-month momentum, 21-day
    # short-term momentum, and 21-day volatility for the latest available date.
    data = yf.download(
        TICKERS,
        period=LIVE_PRICE_LOOKBACK_PERIOD,
        auto_adjust=True,
        progress=False,
    )
    prices = data["Close"]
    prices = prices.dropna(how="all")
    if prices.empty:
        raise RuntimeError(
            "No live prices were downloaded. Check network access or the data vendor response."
        )
    return prices


def calculate_live_features(prices: pd.DataFrame) -> pd.DataFrame:
    # 12-month momentum captures the medium-term trend over roughly one trading
    # year. Positive values mean the stock has appreciated over the lookback.
    momentum_12m = prices / prices.shift(252) - 1

    # Short-term momentum captures the most recent one-month move. This matches
    # the research model feature named short_term_momentum.
    short_term_momentum = prices / prices.shift(21) - 1

    # Volatility is the 21-day standard deviation of daily returns. Higher
    # values mean the stock has been moving around more day to day.
    volatility_1m = prices.pct_change(fill_method=None).rolling(21).std()

    latest_date = prices.dropna(how="all").index.max()
    if pd.isna(latest_date):
        raise RuntimeError("Unable to determine the latest live price date.")

    features = pd.DataFrame({
        "ticker": prices.columns,
        "momentum_12m": momentum_12m.loc[latest_date],
        "short_term_momentum": short_term_momentum.loc[latest_date],
        "volatility_1m": volatility_1m.loc[latest_date],
    })
    features.insert(0, "date", latest_date)
    return features.dropna(subset=FEATURES).reset_index(drop=True)


def train_latest_model():
    # If no saved live model exists, train the same Random Forest architecture
    # used in research on the latest local ML dataset. This produces rankings,
    # not broker orders or guaranteed return forecasts.
    dataset = load_ml_dataset(ML_DATASET_PATH).sort_values("date")
    model = train_random_forest(dataset[FEATURES], dataset[TARGET])
    Path(LIVE_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with Path(LIVE_MODEL_PATH).open("wb") as file:
        pickle.dump(model, file)
    return model


def load_or_train_model():
    model_path = Path(LIVE_MODEL_PATH)
    if model_path.exists():
        with model_path.open("rb") as file:
            return pickle.load(file), "loaded saved Random Forest model"
    return train_latest_model(), "trained Random Forest model from local research dataset"


def composite_factor_score(features: pd.DataFrame) -> pd.Series:
    # Composite fallback score uses the same style of cross-sectional z-scoring
    # as the factor research: put features on comparable scales, reward strong
    # momentum, and penalize higher volatility.
    momentum_z = (
        features["momentum_12m"] - features["momentum_12m"].mean()
    ) / features["momentum_12m"].std()
    short_term_z = (
        features["short_term_momentum"] - features["short_term_momentum"].mean()
    ) / features["short_term_momentum"].std()
    volatility_z = (
        features["volatility_1m"] - features["volatility_1m"].mean()
    ) / features["volatility_1m"].std()
    return momentum_z + short_term_z - volatility_z


def build_live_signals(features: pd.DataFrame, model, model_reason: str) -> pd.DataFrame:
    signals = features.copy()

    if model is not None:
        # Predicted return is the model's estimate for next-month return. It is
        # used only for ranking stocks into a paper signal list.
        signals["predicted_return"] = model.predict(signals[FEATURES])
        signals["signal_score"] = signals["predicted_return"]
        score_reason = model_reason
    else:
        signals["predicted_return"] = pd.NA
        signals["signal_score"] = composite_factor_score(signals)
        score_reason = "used composite factor score because model was unavailable"

    signals = signals.sort_values("signal_score", ascending=False).reset_index(drop=True)
    signals["side"] = "HOLD"
    signals.loc[: LIVE_LONG_COUNT - 1, "side"] = "LONG"

    # Long-only mode is safer for live signal review: top names receive equal
    # positive weights, and every other stock is held at zero target weight.
    target_weight = 1 / LIVE_LONG_COUNT
    signals["target_weight"] = 0.0
    signals.loc[signals["side"] == "LONG", "target_weight"] = target_weight

    signals["reason"] = signals.apply(
        lambda row: (
            f"Top {LIVE_LONG_COUNT} long-only candidate ranked by {score_reason}"
            if row["side"] == "LONG"
            else f"Outside top {LIVE_LONG_COUNT}; hold at zero target weight"
        ),
        axis=1,
    )

    return signals[
        [
            "date",
            "ticker",
            "signal_score",
            "predicted_return",
            "side",
            "target_weight",
            "reason",
        ]
    ]


def generate_live_signals() -> pd.DataFrame:
    prices = download_latest_prices()
    features = calculate_live_features(prices)
    model, model_reason = load_or_train_model()
    signals = build_live_signals(features, model, model_reason)

    Path(LIVE_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    signals.to_csv(LIVE_SIGNALS_PATH, index=False)
    return signals


def main() -> None:
    signals = generate_live_signals()
    print("Live signal generation completed.")
    print()
    print(signals.head(LIVE_LONG_COUNT).to_string(index=False))
    print()
    print(f"Signals saved to {LIVE_SIGNALS_PATH}")
    print("No broker connection was used and no trades were placed.")


if __name__ == "__main__":
    main()
