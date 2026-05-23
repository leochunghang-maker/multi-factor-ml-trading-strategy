import pandas as pd
import pytest

from src.execution.run_daily_simulation import load_long_only_targets, validate_signal_freshness


def test_validate_signal_freshness_rejects_stale_signals() -> None:
    targets = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "ticker": ["AAA"],
            "side": ["LONG"],
            "target_weight": [1.0],
            "signal_score": [1.0],
        }
    )

    with pytest.raises(RuntimeError, match="stale"):
        validate_signal_freshness(
            targets,
            price_date=pd.Timestamp("2024-01-10"),
            max_staleness_days=2,
        )


def test_validate_signal_freshness_rejects_future_signals() -> None:
    targets = pd.DataFrame({"date": [pd.Timestamp("2024-01-10")]})

    with pytest.raises(RuntimeError, match="newer than the latest local price"):
        validate_signal_freshness(
            targets,
            price_date=pd.Timestamp("2024-01-05"),
            max_staleness_days=2,
        )


def test_load_long_only_targets_rejects_negative_weights(tmp_path) -> None:
    signal_file = tmp_path / "signals.csv"
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-05")],
            "ticker": ["AAA"],
            "side": ["LONG"],
            "target_weight": [-0.10],
            "signal_score": [1.0],
            "reason": ["test"],
        }
    ).to_csv(signal_file, index=False)

    with pytest.raises(RuntimeError, match="Negative target weights"):
        load_long_only_targets(str(signal_file))
