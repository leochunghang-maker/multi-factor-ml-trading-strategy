"""Standalone signal generation must use local, already-closed session prices.

``make signals`` / ``python src/live/generate_live_signals.py`` used to call
yfinance itself and could score a partial intraday bar. That path is gone:
prices come from ``data/paper/`` or the research snapshot, and any row after
the last closed NYSE session is rejected.
"""

from pathlib import Path

import pandas as pd
import pytest

import src.live.generate_live_signals as live
from src.data.paths import DataPaths
from src.data.required import MissingInputError


NY = "America/New_York"


def _ny(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=NY)


def _write_prices(path: Path, end: str, symbol: str = "AAA") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(end=end, periods=5)
    pd.DataFrame({symbol: range(100, 100 + len(dates))}, index=dates).to_csv(path)


def _use_roots(monkeypatch, paper: Path, research: Path) -> None:
    monkeypatch.setattr(
        live.DataPaths, "paper", classmethod(lambda cls, *args, **kwargs: DataPaths(str(paper), "paper"))
    )
    monkeypatch.setattr(
        live.DataPaths, "research", classmethod(lambda cls: DataPaths(str(research), "research"))
    )


def test_standalone_paths_prefer_paper_prices_when_present(tmp_path, monkeypatch) -> None:
    paper, research = tmp_path / "paper", tmp_path / "research"
    _write_prices(paper / "price_data.csv", "2026-10-30")
    _write_prices(research / "price_data.csv", "2026-09-30")
    _use_roots(monkeypatch, paper, research)

    paths = live.resolve_standalone_data_paths()
    assert paths.kind == "paper"
    assert Path(paths.price_data) == paper / "price_data.csv"
    assert Path(paths.ml_dataset) == paper / "ml_dataset.csv"


def test_standalone_paths_fall_back_to_the_research_snapshot(tmp_path, monkeypatch) -> None:
    paper, research = tmp_path / "paper", tmp_path / "research"
    _write_prices(research / "price_data.csv", "2026-10-30")
    _use_roots(monkeypatch, paper, research)

    paths = live.resolve_standalone_data_paths()
    assert paths.kind == "research"
    assert Path(paths.price_data) == research / "price_data.csv"


def test_standalone_paths_name_the_paper_loader_when_nothing_is_local(tmp_path, monkeypatch) -> None:
    paper, research = tmp_path / "paper", tmp_path / "research"
    _use_roots(monkeypatch, paper, research)

    with pytest.raises(MissingInputError, match="data_loader.py --paper"):
        live.resolve_standalone_data_paths()


def test_incomplete_session_prices_are_rejected() -> None:
    prices = pd.DataFrame(
        {"AAA": [1.0, 2.0]},
        index=pd.to_datetime(["2026-11-06", "2026-11-09"]),
    )
    with pytest.raises(RuntimeError, match="partial intraday bar"):
        live.reject_incomplete_session_prices(prices, now=_ny("2026-11-09 12:10"))

    live.reject_incomplete_session_prices(prices, now=_ny("2026-11-09 17:00"))


def test_generate_live_signals_without_prices_does_not_download(tmp_path, monkeypatch) -> None:
    paper, research = tmp_path / "paper", tmp_path / "research"
    _write_prices(paper / "price_data.csv", "2026-10-30")
    _use_roots(monkeypatch, paper, research)

    def _boom(*args, **kwargs):
        raise AssertionError("standalone signal generation must not download")

    monkeypatch.setattr(live, "load_price_data", lambda path: pd.DataFrame(
        {"AAA": [1.0]}, index=pd.to_datetime(["2026-10-30"])
    ))
    monkeypatch.setattr(live, "reject_incomplete_session_prices", lambda *args, **kwargs: None)
    monkeypatch.setattr(live, "build_eligibility_mask", lambda prices: pd.DataFrame(True, index=prices.index, columns=prices.columns))
    monkeypatch.setattr(live, "eligible_tickers_on_date", lambda mask, date: list(mask.columns))
    features = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-10-30")],
            "ticker": ["AAA"],
            "momentum_12m": [0.1],
            "short_term_momentum": [0.02],
            "volatility_1m": [0.03],
        }
    )
    monkeypatch.setattr(live, "calculate_live_features", lambda prices: features)
    monkeypatch.setattr(live, "load_or_train_model", lambda dataset_path: (object(), "test model"))
    monkeypatch.setattr(
        live,
        "build_live_signals",
        lambda feats, model, reason: pd.DataFrame(
            {
                "date": feats["date"],
                "ticker": feats["ticker"],
                "signal_score": [1.0],
                "predicted_return": [0.0],
                "side": ["LONG"],
                "target_weight": [1.0],
                "reason": [reason],
            }
        ),
    )
    monkeypatch.setattr(live, "LIVE_RESULTS_DIR", str(tmp_path / "live"))
    monkeypatch.setattr(live, "LIVE_SIGNALS_PATH", str(tmp_path / "live" / "latest_signals.csv"))

    # If a download helper were reintroduced, this sentinel would be called only
    # if generate_live_signals still had a network fallback. Confirm it is gone.
    assert not hasattr(live, "download_latest_prices")
    assert "yfinance" not in getattr(live, "__dict__", {})

    signals = live.generate_live_signals(now=_ny("2026-10-30 17:00"))
    assert list(signals["ticker"]) == ["AAA"]
    assert Path(live.LIVE_SIGNALS_PATH).is_file()


def test_module_does_not_expose_a_download_helper() -> None:
    assert not hasattr(live, "download_latest_prices")
    assert "yf" not in live.__dict__
