import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import shutil
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from src.config import (
    BENCHMARK_SYMBOL,
    END_DATE,
    START_DATE,
    TICKERS,
    VOLUME_DATA_PATH,
)
from src.data.paths import DataPaths
from src.data.universe import drop_empty_price_columns


def _download(tickers=TICKERS, start=START_DATE, end=END_DATE, progress: bool = True) -> pd.DataFrame:
    return yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=progress,
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


def _paths(paths: DataPaths | None) -> DataPaths:
    return paths or DataPaths.research()


def download_price_data(
    include_volume: bool = True,
    end: str = END_DATE,
    paths: DataPaths | None = None,
    progress: bool = True,
    skip_excluded: bool = False,
):
    """Download prices (and volume) into ``paths`` (default: the research ``data/`` snapshot).

    ``progress=False`` silences the yfinance progress bar (paper refreshes);
    ``skip_excluded`` does not request excluded symbols at all (they are blanked
    either way, so the saved files are the same, without the failed-download noise).
    """
    paths = _paths(paths)
    Path(paths.price_data).parent.mkdir(parents=True, exist_ok=True)
    from src.data.quality import blank_excluded_symbols

    print(f"Downloading price data ({START_DATE} to {end}, end exclusive) into {paths.root}/ ...")
    excluded = _excluded_symbols()
    tickers = [t for t in TICKERS if t not in excluded] if skip_excluded else TICKERS
    data = _download(tickers=tickers, end=end, progress=progress)
    # Excluded symbols (recorded decisions, e.g. BK) stay all-NaN so a refresh
    # never changes the universe; other empty columns are dropped as before.
    prices = drop_empty_price_columns(data["Close"].drop(columns=list(excluded), errors="ignore"))
    prices = blank_excluded_symbols(prices, excluded)
    prices.to_csv(paths.price_data)
    print(f"Saved price data to {paths.price_data}")
    if progress:
        print(prices.tail())
    else:
        print(f"  {prices.shape[0]} rows x {prices.shape[1]} symbols, last date {prices.index.max().date()}")

    if include_volume:
        save_volume_data(data, path=paths.volume_data, excluded=excluded)


def download_volume_data(end: str = END_DATE, paths: DataPaths | None = None):
    """Download only share volume, leaving the existing price panel untouched."""
    return save_volume_data(_download(end=end), path=_paths(paths).volume_data)


def download_benchmark_data(
    symbol: str = BENCHMARK_SYMBOL, end: str = END_DATE, paths: DataPaths | None = None, progress: bool = True
) -> pd.DataFrame:
    """Download the benchmark's adjusted close into ``<data dir>/benchmark_prices.csv``.

    Uses the same yfinance call (auto_adjust=True, START_DATE..END_DATE) as the
    equity panel and records provenance next to it, so benchmark-relative tests
    are reproducible offline from checked-in data.
    """
    paths = _paths(paths)
    data = _download(tickers=[symbol], end=end, progress=progress)
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbol)
    close = close.reindex(columns=[symbol]).dropna(how="all")
    close.index.name = "Date"
    Path(paths.benchmark_prices).parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(paths.benchmark_prices)
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
    Path(paths.benchmark_source).write_text(json.dumps(source, indent=2) + "\n")
    print(f"Saved benchmark prices to {paths.benchmark_prices}")
    return close


def write_data_manifest(paths: DataPaths, note: str) -> dict:
    """Record SHA-256/shape of a non-research data dir's dated inputs (paper provenance)."""
    from src.data.quality import build_manifest, data_check_settings

    if paths.kind == "research":
        raise ValueError("the research manifest is only updated by `python src/check_data.py --update-manifest`")
    manifest = build_manifest(
        list(paths.dated_inputs().values()),
        decisions=data_check_settings().get("excluded_symbols") or None,
        description=f"Paper data provenance ({note}); rewritten after every paper refresh.",
    )
    Path(paths.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(paths.manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {paths.manifest} ({len(manifest['files'])} files)")
    return manifest


def refresh_paper_data(end: str | None = "today", paths: DataPaths | None = None) -> DataPaths:
    """Full aligned refresh (prices, volume, SPY) into the paper dir; never touches data/."""
    paths = paths or DataPaths.paper()
    end = resolve_end_date(end)
    # Quiet: no yfinance progress bar in paper logs, and excluded symbols are not requested.
    download_price_data(end=end, paths=paths, progress=False, skip_excluded=True)
    download_benchmark_data(end=end, paths=paths, progress=False)
    write_data_manifest(paths, f"yfinance download, end exclusive {end}")
    return paths


def seed_paper_data_from_research(paths: DataPaths | None = None) -> DataPaths:
    """Copy the checked-in research inputs into the paper dir (offline demo; data/ is only read)."""
    paths = paths or DataPaths.paper()
    research = DataPaths.research()
    Path(paths.root).mkdir(parents=True, exist_ok=True)
    for name in ("price_data", "volume_data", "benchmark_prices", "benchmark_source"):
        src, dst = getattr(research, name), getattr(paths, name)
        if Path(src).exists():
            shutil.copyfile(src, dst)
            print(f"Copied {src} -> {dst}")
    write_data_manifest(paths, "seeded from the checked-in research snapshot")
    return paths


def report_changes_vs_manifest(paths: DataPaths | None = None) -> None:
    """After a download, flag files whose content differs from the data manifest.

    Read-only for the research manifest, which is only updated by
    ``python src/check_data.py --update-manifest`` after the new data has been reviewed.
    """
    from src.data.quality import check_manifest, data_check_settings, paper_manifest_settings

    paths = _paths(paths)
    settings = paper_manifest_settings(data_check_settings(), paths)
    findings = check_manifest(settings["manifest_path"], list(settings["manifest_files"]), settings)
    if not findings:
        print(f"Data manifest ({settings['manifest_path']}): files match the recorded hashes.")
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

    parser = argparse.ArgumentParser(
        description="Download prices, volume and the SPY benchmark (Yahoo). Without --paper this "
                    "rewrites the checked-in research snapshot under data/."
    )
    parser.add_argument("--paper", action="store_true",
                        help="Write to the paper data dir (data/paper/ or $PAPER_DATA_DIR) instead of "
                             "data/; always refreshes prices, volume and SPY together (default --end today).")
    parser.add_argument("--paper-dir", default=None, help="Paper data dir override (implies --paper).")
    parser.add_argument("--seed-from-research", action="store_true",
                        help="With --paper: copy the checked-in research inputs instead of downloading "
                             "(offline demo).")
    parser.add_argument("--benchmark-only", action="store_true", help="Only data/benchmark_prices.csv.")
    parser.add_argument("--volume-only", action="store_true", help="Only data/volume_data.csv.")
    parser.add_argument("--with-benchmark", action="store_true",
                        help="Also refresh the benchmark (keeps price/volume/SPY date ranges aligned).")
    parser.add_argument("--end", default=None,
                        help=f"End date, exclusive (research default {END_DATE}; paper default 'today'). "
                             "Without --paper, changing it changes the research data snapshot.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.paper or args.paper_dir:
        paths = DataPaths.paper(args.paper_dir)
        if args.seed_from_research:
            seed_paper_data_from_research(paths)
        else:
            refresh_paper_data(end=args.end or "today", paths=paths)
        print(f"Paper data ready in {paths.root}/ (research data/ untouched).")
        return 0
    if args.seed_from_research:
        raise SystemExit("--seed-from-research requires --paper")
    end = resolve_end_date(args.end)
    if args.end is not None:
        print(f"NOTE: writing the research snapshot under data/ with end {end}. "
              "For paper trading use --paper instead.")
    if args.benchmark_only:
        download_benchmark_data(end=end)
    elif args.volume_only:
        download_volume_data(end=end)
    else:
        download_price_data(end=end)
        if args.with_benchmark:
            download_benchmark_data(end=end)
    report_changes_vs_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
