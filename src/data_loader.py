import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from src.config import (
    BENCHMARK_PRICES_PATH,
    BENCHMARK_SOURCE_PATH,
    BENCHMARK_SYMBOL,
    END_DATE,
    PRICE_DATA_PATH,
    START_DATE,
    TICKERS,
    VOLUME_DATA_PATH,
)
from src.data.universe import drop_empty_price_columns


def _download(tickers=TICKERS, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    return yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
    )


def extract_volume_frame(data: pd.DataFrame, tickers=TICKERS) -> pd.DataFrame:
    """Return a date x ticker share-volume panel (same layout as price_data.csv)."""
    volumes = data["Volume"]
    if isinstance(volumes, pd.Series):
        volumes = volumes.to_frame(name=tickers[0] if len(tickers) == 1 else "Volume")
    volumes = volumes.reindex(columns=sorted(volumes.columns))
    volumes.index.name = "Date"
    return volumes


def save_volume_data(
    data: pd.DataFrame,
    path: str = VOLUME_DATA_PATH,
    excluded: dict[str, str] | None = None,
) -> pd.DataFrame:
    from src.data.quality import blank_excluded_symbols

    volumes = extract_volume_frame(data)
    volumes = blank_excluded_symbols(volumes, _excluded_symbols() if excluded is None else excluded)
    volumes.index.name = "Date"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    volumes.to_csv(path)
    print(f"Saved volume data to {path}")
    return volumes


def _excluded_symbols() -> dict[str, str]:
    from src.data.quality import data_check_settings

    return dict(data_check_settings().get("excluded_symbols") or {})


def download_price_data(include_volume: bool = True, end: str = END_DATE):
    Path("data").mkdir(exist_ok=True)
    from src.data.quality import blank_excluded_symbols

    print(f"Downloading price data ({START_DATE} to {end}, end exclusive)...")
    data = _download(end=end)
    excluded = _excluded_symbols()
    # Excluded symbols (recorded decisions, e.g. BK) stay all-NaN so a refresh
    # never changes the universe; other empty columns are dropped as before.
    prices = drop_empty_price_columns(data["Close"].drop(columns=list(excluded), errors="ignore"))
    prices = blank_excluded_symbols(prices, excluded)
    prices.to_csv(PRICE_DATA_PATH)
    print(f"Saved price data to {PRICE_DATA_PATH}")
    print(prices.tail())

    if include_volume:
        save_volume_data(data, excluded=excluded)


def download_volume_data(end: str = END_DATE):
    """Download only share volume, leaving the existing price panel untouched."""
    return save_volume_data(_download(end=end))


def download_benchmark_data(symbol: str = BENCHMARK_SYMBOL, end: str = END_DATE) -> pd.DataFrame:
    """Download the benchmark's adjusted close into ``data/benchmark_prices.csv``.

    Uses the same yfinance call (auto_adjust=True, START_DATE..END_DATE) as the
    equity panel and records provenance next to it, so benchmark-relative tests
    are reproducible offline from checked-in data.
    """
    data = _download(tickers=[symbol], end=end)
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbol)
    close = close.reindex(columns=[symbol]).dropna(how="all")
    close.index.name = "Date"
    Path(BENCHMARK_PRICES_PATH).parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(BENCHMARK_PRICES_PATH)
    source = {
        "symbol": symbol,
        "source": "Yahoo Finance via yfinance.download(auto_adjust=True)",
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "start": START_DATE,
        "end_exclusive": end,
        "first_date": str(close.index.min().date()),
        "last_date": str(close.index.max().date()),
        "rows": int(len(close)),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Total-return-style adjusted close (dividends/splits). Yahoo may revise "
                "adjusted history; re-download changes values slightly.",
    }
    Path(BENCHMARK_SOURCE_PATH).write_text(json.dumps(source, indent=2) + "\n")
    print(f"Saved benchmark prices to {BENCHMARK_PRICES_PATH}")
    return close


def report_changes_vs_manifest() -> None:
    """After a download, flag files whose content differs from data/data_manifest.json.

    Read-only: the manifest is only updated by ``python src/check_data.py --update-manifest``
    after the new data has been reviewed.
    """
    from src.data.quality import check_manifest, data_check_settings

    settings = data_check_settings()
    findings = check_manifest(settings["manifest_path"], list(settings["manifest_files"]), settings)
    if not findings:
        print("Data manifest: downloaded files match the recorded hashes.")
        return
    for finding in findings:
        print(f"{finding.severity} {finding.check} {finding.file}: {finding.message}")
    print("Review with `python src/check_data.py --write-report` before `--update-manifest`.")


def resolve_end_date(value: str | None) -> str:
    """``None`` → config END_DATE; ``today`` → tomorrow (yfinance end is exclusive)."""
    if value is None:
        return END_DATE
    if value.lower() == "today":
        return (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Download prices, volume and the SPY benchmark (Yahoo).")
    parser.add_argument("--benchmark-only", action="store_true", help="Only data/benchmark_prices.csv.")
    parser.add_argument("--volume-only", action="store_true", help="Only data/volume_data.csv.")
    parser.add_argument("--with-benchmark", action="store_true",
                        help="Also refresh the benchmark (keeps price/volume/SPY date ranges aligned).")
    parser.add_argument("--end", default=None,
                        help=f"End date, exclusive (default {END_DATE}; 'today' for a paper refresh). "
                             "Changing it changes the research data snapshot.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    end = resolve_end_date(args.end)
    if args.benchmark_only:
        download_benchmark_data(end=end)
    elif args.volume_only:
        download_volume_data(end=end)
    else:
        download_price_data(end=end)
        if args.with_benchmark:
            download_benchmark_data(end=end)
    report_changes_vs_manifest()
