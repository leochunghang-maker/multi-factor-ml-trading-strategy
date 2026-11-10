import hashlib
import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import sklearn
import yfinance as yf

from src.config import (
    FEATURES,
    LIVE_LONG_COUNT,
    LIVE_MODEL_META_PATH,
    LIVE_MODEL_PATH,
    LIVE_PRICE_LOOKBACK_PERIOD,
    LIVE_RESULTS_DIR,
    LIVE_SIGNALS_PATH,
    ML_DATASET_PATH,
    RANDOM_FOREST_PARAMS,
    TARGET,
    TICKERS,
)
from src.data.io import load_ml_dataset
from src.data.universe import build_eligibility_mask, eligible_tickers_on_date
from src.models.ml import train_random_forest
from src.data.required import run_cli


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


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(dataset_path: str = ML_DATASET_PATH) -> dict:
    # Everything that determines the fitted model. A saved model is reused only
    # while all of these match; otherwise it is stale and gets retrained.
    return {
        "dataset_sha256": _file_sha256(dataset_path),
        "features": list(FEATURES),
        "target": TARGET,
        "random_forest_params": dict(RANDOM_FOREST_PARAMS),
        "sklearn_version": sklearn.__version__,
    }


def _stale_reason(saved: dict, current: dict) -> str | None:
    for key, label in (
        ("dataset_sha256", "ML dataset changed"),
        ("features", "feature list changed"),
        ("target", "target changed"),
        ("random_forest_params", "model parameters changed"),
        ("sklearn_version", "scikit-learn version changed"),
    ):
        if saved.get(key) != current[key]:
            return label
    return None


def train_latest_model(dataset_path: str = ML_DATASET_PATH):
    # Train the same Random Forest architecture used in research on the latest
    # local ML dataset. This produces rankings, not broker orders or guaranteed
    # return forecasts. Rows without a realized label are not trainable.
    dataset = load_ml_dataset(dataset_path).sort_values("date")
    dataset = dataset.dropna(subset=[*FEATURES, TARGET])
    if dataset.empty:
        raise RuntimeError(f"No labelled rows to train the live model in {dataset_path}.")
    model = train_random_forest(dataset[FEATURES], dataset[TARGET])
    Path(LIVE_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with Path(LIVE_MODEL_PATH).open("wb") as file:
        pickle.dump(model, file)
    metadata = {
        **model_fingerprint(dataset_path),
        "dataset_path": str(dataset_path),
        "training_rows": int(len(dataset)),
        "last_label_date": str(pd.Timestamp(dataset["date"].max()).date()),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    Path(LIVE_MODEL_META_PATH).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return model


def load_or_train_model(dataset_path: str = ML_DATASET_PATH):
    """Reuse the saved live model only if it was trained on this exact dataset.

    The daily paper run rebuilds the ML dataset before generating signals, so a
    saved model without matching metadata (dataset hash, features, target, model
    parameters, scikit-learn version) is stale and is retrained.
    """
    model_path = Path(LIVE_MODEL_PATH)
    meta_path = Path(LIVE_MODEL_META_PATH)
    current = model_fingerprint(dataset_path)
    if not model_path.exists():
        reason = "no saved model"
    elif not meta_path.exists():
        reason = "saved model has no training metadata"
    else:
        try:
            saved = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            saved = {}
        reason = _stale_reason(saved, current)
        if reason is None:
            try:
                with model_path.open("rb") as file:
                    model = pickle.load(file)
            except Exception:
                reason = "saved model could not be loaded"
            else:
                return model, (
                    "loaded saved Random Forest model trained on "
                    f"{saved.get('dataset_path', dataset_path)} "
                    f"(labels through {saved.get('last_label_date', 'unknown')})"
                )
    model = train_latest_model(dataset_path)
    return model, f"trained Random Forest model from {dataset_path} ({reason})"


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


def generate_live_signals(
    prices: pd.DataFrame | None = None, dataset_path: str = ML_DATASET_PATH
) -> pd.DataFrame:
    """Build the paper-trading signal file.

    ``prices`` (e.g. the local ``data/price_data.csv`` that passed the paper data
    gate) makes the run offline and consistent with the gate; when omitted the
    standalone script downloads recent prices as before.
    """
    if prices is None:
        prices = download_latest_prices()
    eligibility = build_eligibility_mask(prices)
    latest_date = prices.dropna(how="all").index.max()
    allowed = set(eligible_tickers_on_date(eligibility, latest_date))
    features = calculate_live_features(prices)
    if allowed:
        features = features[features["ticker"].isin(allowed)].reset_index(drop=True)
    if features.empty:
        raise RuntimeError(
            "No eligible live tickers after tradability/index-membership filters."
        )
    model, model_reason = load_or_train_model(dataset_path)
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
    run_cli(main)
