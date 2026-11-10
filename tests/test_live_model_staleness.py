import json

import numpy as np
import pandas as pd
import pytest

import src.live.generate_live_signals as live


def _write_dataset(path, rows: int = 60, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-31", periods=rows // 3, freq="ME").repeat(3)
    pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA", "BBB", "CCC"] * (rows // 3),
            "momentum_12m": rng.normal(size=rows),
            "short_term_momentum": rng.normal(size=rows),
            "volatility_1m": rng.uniform(0.01, 0.05, size=rows),
            "target_return_1m": rng.normal(scale=0.05, size=rows),
        }
    ).to_csv(path, index=False)


@pytest.fixture
def live_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE_RESULTS_DIR", str(tmp_path / "live"))
    monkeypatch.setattr(live, "LIVE_MODEL_PATH", str(tmp_path / "live" / "model.pkl"))
    monkeypatch.setattr(live, "LIVE_MODEL_META_PATH", str(tmp_path / "live" / "model.meta.json"))
    return tmp_path


def test_first_run_trains_and_writes_metadata(live_dir) -> None:
    dataset = live_dir / "ml_dataset.csv"
    _write_dataset(dataset)

    _, reason = live.load_or_train_model(str(dataset))

    assert "trained" in reason and "no saved model" in reason
    meta = json.loads((live_dir / "live" / "model.meta.json").read_text())
    assert meta["dataset_sha256"] == live.model_fingerprint(str(dataset))["dataset_sha256"]
    assert meta["features"] == list(live.FEATURES)
    assert meta["training_rows"] == 60


def test_unchanged_dataset_reuses_saved_model(live_dir) -> None:
    dataset = live_dir / "ml_dataset.csv"
    _write_dataset(dataset)
    live.load_or_train_model(str(dataset))

    _, reason = live.load_or_train_model(str(dataset))

    assert reason.startswith("loaded saved Random Forest model")


def test_refreshed_dataset_retrains_instead_of_reusing_stale_model(live_dir) -> None:
    dataset = live_dir / "ml_dataset.csv"
    _write_dataset(dataset, seed=0)
    live.load_or_train_model(str(dataset))
    first_hash = json.loads((live_dir / "live" / "model.meta.json").read_text())["dataset_sha256"]

    _write_dataset(dataset, rows=63, seed=1)  # the daily paper run appended data
    _, reason = live.load_or_train_model(str(dataset))

    assert "ML dataset changed" in reason
    meta = json.loads((live_dir / "live" / "model.meta.json").read_text())
    assert meta["dataset_sha256"] != first_hash
    assert meta["training_rows"] == 63


def test_legacy_model_without_metadata_is_retrained(live_dir) -> None:
    dataset = live_dir / "ml_dataset.csv"
    _write_dataset(dataset)
    (live_dir / "live").mkdir()
    (live_dir / "live" / "model.pkl").write_bytes(b"not a model trained on this dataset")

    _, reason = live.load_or_train_model(str(dataset))

    assert "saved model has no training metadata" in reason
    assert (live_dir / "live" / "model.meta.json").exists()


def test_changed_model_parameters_force_retrain(live_dir, monkeypatch) -> None:
    dataset = live_dir / "ml_dataset.csv"
    _write_dataset(dataset)
    live.load_or_train_model(str(dataset))

    monkeypatch.setattr(live, "RANDOM_FOREST_PARAMS", {**live.RANDOM_FOREST_PARAMS, "max_depth": 3})
    _, reason = live.load_or_train_model(str(dataset))

    assert "model parameters changed" in reason


def test_unlabelled_rows_are_not_trained_on(live_dir) -> None:
    dataset = live_dir / "ml_dataset.csv"
    _write_dataset(dataset)
    frame = pd.read_csv(dataset)
    frame.loc[frame.index[-3:], "target_return_1m"] = np.nan
    frame.to_csv(dataset, index=False)

    live.load_or_train_model(str(dataset))

    meta = json.loads((live_dir / "live" / "model.meta.json").read_text())
    assert meta["training_rows"] == 57
