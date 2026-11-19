"""Benchmark returns must come from the local snapshot, never a live download."""

from pathlib import Path

import pandas as pd
import pytest

import src.backtests.benchmark as benchmark


def test_load_benchmark_returns_uses_local_spy(tmp_path, monkeypatch) -> None:
    prices = tmp_path / "benchmark_prices.csv"
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-28"])
    pd.DataFrame({"SPY": [100.0, 110.0, 121.0]}, index=dates).to_csv(prices)
    monkeypatch.setattr(benchmark, "PRICE_DATA_PATH", str(tmp_path / "missing_prices.csv"))
    monkeypatch.setattr(benchmark, "BENCHMARK_PRICES_PATH", str(prices))

    returns = benchmark.load_benchmark_returns("SPY", "2024-01-01", "2024-04-30")
    assert list(returns.round(10)) == [0.1, 0.1]


def test_load_benchmark_returns_errors_without_downloading(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "PRICE_DATA_PATH", str(tmp_path / "price_data.csv"))
    monkeypatch.setattr(benchmark, "BENCHMARK_PRICES_PATH", str(tmp_path / "benchmark_prices.csv"))

    with pytest.raises(RuntimeError, match="git checkout -- data/benchmark_prices.csv"):
        benchmark.load_benchmark_returns("SPY", "2024-01-01", "2024-12-31")


def test_benchmark_module_does_not_import_yfinance() -> None:
    assert "yf" not in benchmark.__dict__
    assert "yfinance" not in benchmark.__dict__
