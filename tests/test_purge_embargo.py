"""Tests for purged/embargoed walk-forward sample construction."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.validation import (
    assert_no_label_overlap,
    purged_embargo_year_splits,
    training_rows_without_label_overlap,
)


def test_purged_embargo_year_splits_insert_gaps() -> None:
    dates = pd.to_datetime(
        ["2023-11-15", "2023-12-15", "2024-01-15", "2024-06-15", "2024-12-15"]
    )
    splits = purged_embargo_year_splits(pd.Series(dates), test_years=[2024], purge_months=1, embargo_months=1)
    assert len(splits) == 1
    split = splits[0]
    assert split["test_start"] == pd.Timestamp("2024-01-15")
    assert split["test_end"] == pd.Timestamp("2024-12-15")
    assert split["train_end_before_purge"] == pd.Timestamp("2023-12-15")
    assert split["embargo_lift_date"] == pd.Timestamp("2025-01-15")


def test_training_rows_have_no_label_overlap_with_test() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-11-01",
                    "2023-12-20",  # label end 2024-01-10 overlaps Jan test
                    "2023-12-01",  # label end 2023-12-22 before test
                    "2024-01-10",
                ]
            ),
            "x": [1, 2, 3, 4],
        }
    )
    train = training_rows_without_label_overlap(
        frame,
        date_column="date",
        train_end_before_purge=pd.Timestamp("2023-12-10"),
        test_start=pd.Timestamp("2024-01-01"),
        label_horizon_days=21,
    )
    # 2023-12-01 is kept: within purge cutoff and label ends 2023-12-22 (< test_start).
    # 2023-12-20 is dropped: after purge cutoff and label overlaps January.
    assert list(train["date"].dt.strftime("%Y-%m-%d")) == ["2023-11-01", "2023-12-01"]
    assert_no_label_overlap(train["date"], test_start=pd.Timestamp("2024-01-01"), label_horizon_days=21)


def test_assert_no_label_overlap_raises_on_leakage() -> None:
    train_dates = pd.to_datetime(["2023-12-20"])
    with pytest.raises(AssertionError, match="overlaps test period"):
        assert_no_label_overlap(
            train_dates,
            test_start=pd.Timestamp("2024-01-01"),
            label_horizon_days=21,
        )
