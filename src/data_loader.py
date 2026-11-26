import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import os
import shutil
import uuid
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

# Staging / rollback folders for paper refreshes. They live under the paper dir
# and are removed after each attempt (including leftover dirs from a crashed run).
_PAPER_REFRESH_PREFIX = ".refresh-"
_PAPER_INPUT_ATTRS = ("price_data", "volume_data", "benchmark_prices", "benchmark_source")
_PAPER_SNAPSHOT_ATTRS = _PAPER_INPUT_ATTRS + ("manifest",)


def _download(tickers=TICKERS, start=START_DATE, end=END_DATE, progress: bool = True) -> pd.DataFrame:
    return yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=progress,
    )


def _require_download(data: pd.DataFrame, label: str, end: str) -> pd.DataFrame:
    if data is None or len(data) == 0:
        raise ValueError(f"yfinance download returned no {label} ({START_DATE} to {end})")
    return data


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
    data = _require_download(_download(tickers=tickers, end=end, progress=progress), "price data", end)
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
    data = _require_download(_download(end=end), "volume data", end)
    return save_volume_data(data, path=_paths(paths).volume_data)


def download_benchmark_data(
    symbol: str = BENCHMARK_SYMBOL, end: str = END_DATE, paths: DataPaths | None = None, progress: bool = True
) -> pd.DataFrame:
    """Download the benchmark's adjusted close into ``<data dir>/benchmark_prices.csv``.

    Uses the same yfinance call (auto_adjust=True, START_DATE..END_DATE) as the
    equity panel and records provenance next to it, so benchmark-relative tests
    are reproducible offline from checked-in data.
    """
    paths = _paths(paths)
    data = _require_download(_download(tickers=[symbol], end=end, progress=progress), f"{symbol} data", end)
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbol)
    close = close.reindex(columns=[symbol]).dropna(how="all")
    if close.empty:
        raise ValueError(f"yfinance download returned no {symbol} data ({START_DATE} to {end})")
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


def _cleanup_paper_refresh_dirs(root: Path) -> None:
    root = Path(root)
    if not root.is_dir():
        return
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith(_PAPER_REFRESH_PREFIX):
            shutil.rmtree(path, ignore_errors=True)


def _load_dated_panel(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    return frame


def _validate_paper_refresh(staging: DataPaths, previous: DataPaths, end: str) -> None:
    """Reject a staged download that is empty, misaligned, or drops already-held history.

    Failures leave the previous paper snapshot in place (the caller installs only
    after this returns). Already-held sessions strictly before the exclusive end
    date must still be present — a truncated Yahoo response must not replace a
    longer good snapshot.
    """
    kept = "the previous paper snapshot was left unchanged"
    needed = [Path(getattr(staging, name)) for name in _PAPER_INPUT_ATTRS]
    missing = [str(path) for path in needed if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"paper refresh produced missing or empty files: {', '.join(missing)}; {kept}")

    prices = _load_dated_panel(staging.price_data)
    volumes = _load_dated_panel(staging.volume_data)
    benchmark = _load_dated_panel(staging.benchmark_prices)
    if prices.empty or volumes.empty or benchmark.empty:
        raise ValueError(f"paper refresh produced an empty price, volume or benchmark file; {kept}")
    if not prices.index.equals(volumes.index):
        raise ValueError(f"paper refresh price and volume dates do not match; {kept}")
    if list(prices.columns) != list(volumes.columns):
        raise ValueError(f"paper refresh price and volume columns do not match; {kept}")

    from src.data.market_calendar import previous_trading_day

    expected = pd.Timestamp(previous_trading_day(end)).normalize()
    last = pd.Timestamp(prices.index.max()).normalize()
    if last != expected:
        raise ValueError(
            f"paper refresh last date is {last.date()}, expected the last NYSE session "
            f"before exclusive end {end} ({expected.date()}); {kept}"
        )
    bench_last = pd.Timestamp(benchmark.index.max()).normalize()
    if bench_last != last:
        raise ValueError(
            f"paper refresh benchmark last date is {bench_last.date()}, expected {last.date()}; {kept}"
        )

    previous_prices = Path(previous.price_data)
    if previous_prices.exists() and previous_prices.stat().st_size > 0:
        old = _load_dated_panel(previous_prices)
        if not old.empty:
            held = old.index[old.index < pd.Timestamp(end).normalize()]
            missing_dates = held.difference(prices.index)
            if len(missing_dates):
                first = pd.Timestamp(missing_dates.min()).date()
                raise ValueError(
                    f"paper refresh dropped {len(missing_dates)} already-held session(s) "
                    f"(first missing {first}); {kept}"
                )
            held_columns = [column for column in old.columns if old[column].notna().any()]
            missing_columns = [column for column in held_columns if column not in prices.columns]
            if missing_columns:
                raise ValueError(
                    f"paper refresh dropped already-held symbols: {', '.join(missing_columns)}; {kept}"
                )


def _backup_paper_snapshot(paths: DataPaths, backup_root: Path) -> None:
    backup_root.mkdir(parents=True, exist_ok=True)
    index: dict[str, str | None] = {}
    for src in [Path(getattr(paths, name)) for name in _PAPER_SNAPSHOT_ATTRS]:
        if src.exists():
            shutil.copy2(src, backup_root / src.name)
            index[str(src)] = src.name
        else:
            index[str(src)] = None
    (backup_root / "index.json").write_text(json.dumps(index))


def _restore_paper_snapshot(backup_root: Path) -> None:
    index = json.loads((backup_root / "index.json").read_text())
    for original, saved in index.items():
        target = Path(original)
        if saved is None:
            if target.exists():
                target.unlink()
            continue
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.restore")
        shutil.copy2(backup_root / saved, tmp)
        os.replace(tmp, target)


def _install_paper_inputs(staging: DataPaths, dest: DataPaths) -> None:
    Path(dest.root).mkdir(parents=True, exist_ok=True)
    for name in _PAPER_INPUT_ATTRS:
        src, target = Path(getattr(staging, name)), Path(getattr(dest, name))
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.install")
        shutil.copy2(src, tmp)
        os.replace(tmp, target)


def refresh_paper_data(end: str | None = "today", paths: DataPaths | None = None) -> DataPaths:
    """Full aligned refresh (prices, volume, SPY) into the paper dir; never touches data/.

    Downloads into a staging folder under the paper dir and replaces the current
    snapshot only after the files validate (non-empty, price/volume aligned, last
    date is the last NYSE session before the exclusive end, already-held history
    retained). A failed or truncated download leaves the previous snapshot in
    place. Derived paper files (features, ML dataset) are not touched.
    """
    paths = paths or DataPaths.paper()
    end = resolve_end_date(end)
    root = Path(paths.root)
    _cleanup_paper_refresh_dirs(root)
    token = uuid.uuid4().hex
    staging_root = root / f"{_PAPER_REFRESH_PREFIX}{token}"
    backup_root = root / f"{_PAPER_REFRESH_PREFIX}prev-{token}"
    staging = DataPaths(str(staging_root), "paper")
    try:
        # Quiet: no yfinance progress bar in paper logs, and excluded symbols are not requested.
        download_price_data(end=end, paths=staging, progress=False, skip_excluded=True)
        download_benchmark_data(end=end, paths=staging, progress=False)
        _validate_paper_refresh(staging, paths, end)
        _backup_paper_snapshot(paths, backup_root)
        try:
            _install_paper_inputs(staging, paths)
            write_data_manifest(paths, f"yfinance download, end exclusive {end}")
        except Exception:
            _restore_paper_snapshot(backup_root)
            raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)
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


def resolve_end_date(value: str | None, now=None) -> str:
    """``None`` → config END_DATE; ``today`` → the day after the last *closed* NYSE session.

    yfinance's end is exclusive. "today" means the latest session whose New York close
    plus ``data_checks.session_settle_minutes`` has passed, so a refresh during US market
    hours (e.g. a morning run in Hong Kong is still the previous US evening, an evening
    run is mid-session) never downloads a partial intraday bar as a close.
    """
    if value is None:
        return END_DATE
    if value.lower() == "today":
        from src.data.market_calendar import last_completed_session
        from src.data.quality import data_check_settings

        settings = data_check_settings()
        last = last_completed_session(now, str(settings.get("calendar", "NYSE")),
                                      int(settings.get("session_settle_minutes", 30)))
        return (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
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
