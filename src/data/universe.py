"""Point-in-time universe eligibility helpers.

The research ticker list in ``src.config.TICKERS`` is a modern S&P 100-style
snapshot. Using that static list for all of history introduces survivorship bias.
This module reduces (does not eliminate) that bias by combining:

1. **Tradability** from the local price panel — require a valid price on the
   evaluation date, a minimum history window, and treat trailing all-NaN stretches
   after the last valid print as delisted / dead.
2. **Index membership overlay** — optional month-end S&P 500 membership flags for
   the same tickers (``data/universe_sp500_membership.csv``). S&P 100 is a subset
   of S&P 500, so excluding non-members is a conservative PIT filter when a free
   official S&P 100 history is unavailable.

Residual limitation: names that left the index and are absent from today's
``TICKERS`` list are still missing from the price panel.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    MIN_UNIVERSE_HISTORY_DAYS,
    REQUIRE_INDEX_MEMBERSHIP,
    UNIVERSE_MEMBERSHIP_PATH,
)

# Yahoo / Wikipedia ticker renames that should count as continuous membership.
TICKER_MEMBERSHIP_ALIASES: dict[str, tuple[str, ...]] = {
    "META": ("META", "FB"),
    "RTX": ("RTX", "UTX"),
    "BRK-B": ("BRK-B", "BRK.B", "BRK/B"),
}


def normalize_ticker(ticker: str) -> str:
    """Map common share-class spellings onto the repo's canonical tickers."""
    text = str(ticker).strip().upper()
    return text.replace(".", "-").replace("/", "-")


def drop_empty_price_columns(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop tickers with no valid prices (e.g. failed Yahoo downloads)."""
    if prices.empty:
        return prices
    keep = prices.columns[prices.notna().any(axis=0)]
    return prices.loc[:, keep].copy()


def detect_delist_dates(prices: pd.DataFrame) -> pd.Series:
    """Return the first calendar day a name is treated as delisted.

    A name is delisted on the trading day after its last valid price when the
    panel continues with trailing NaNs through the sample end. Names that are
    still printing on the final panel date return NaT.
    """
    if prices.empty:
        return pd.Series(dtype="datetime64[ns]")

    last_valid = prices.apply(lambda column: column.last_valid_index())
    panel_end = prices.index.max()
    delisted = {}
    for ticker, last in last_valid.items():
        if last is None or pd.isna(last):
            # Never traded in-panel: treat as delisted from the first date.
            delisted[ticker] = prices.index.min()
            continue
        if last < panel_end and prices.loc[last:, ticker].iloc[1:].isna().all():
            # Next index label after last valid print.
            position = prices.index.get_loc(last)
            if isinstance(position, slice):
                position = position.start
            if position + 1 < len(prices.index):
                delisted[ticker] = prices.index[position + 1]
            else:
                delisted[ticker] = pd.NaT
        else:
            delisted[ticker] = pd.NaT
    return pd.Series(delisted)


def build_tradability_mask(
    prices: pd.DataFrame,
    min_history_days: int = MIN_UNIVERSE_HISTORY_DAYS,
) -> pd.DataFrame:
    """Boolean date x ticker mask: tradable for research / paper construction."""
    if prices.empty:
        return prices.astype(bool)

    valid = prices.notna()
    history = valid.cumsum()
    enough_history = history >= int(min_history_days)
    delist_dates = detect_delist_dates(prices)
    still_listed = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    for ticker, delist_date in delist_dates.items():
        if pd.isna(delist_date):
            continue
        still_listed.loc[prices.index >= delist_date, ticker] = False
    return valid & enough_history & still_listed


def load_membership_table(path: str | Path = UNIVERSE_MEMBERSHIP_PATH) -> pd.DataFrame:
    """Load month-end membership rows with columns ``date``, ``ticker``."""
    membership_path = Path(path)
    if not membership_path.exists():
        return pd.DataFrame(columns=["date", "ticker"])
    frame = pd.read_csv(membership_path, parse_dates=["date"])
    if frame.empty:
        return frame
    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_membership_mask(
    index: pd.DatetimeIndex,
    tickers: list[str] | pd.Index,
    membership: pd.DataFrame | None = None,
    path: str | Path = UNIVERSE_MEMBERSHIP_PATH,
) -> pd.DataFrame:
    """As-of month-end membership mask aligned to ``index`` x ``tickers``."""
    tickers = [normalize_ticker(t) for t in tickers]
    mask = pd.DataFrame(False, index=pd.DatetimeIndex(index), columns=tickers)
    if membership is None:
        membership = load_membership_table(path)
    if membership.empty or mask.empty:
        return mask

    # Snapshot sets keyed by membership date; each row takes the latest
    # snapshot dated on or before it (as-of lookup, vectorized).
    grouped = {
        stamp: set(group["ticker"].map(normalize_ticker))
        for stamp, group in membership.groupby("date")
    }
    snapshot_dates = pd.DatetimeIndex(sorted(grouped))
    snapshot_matrix = np.array(
        [[ticker in grouped[stamp] for ticker in tickers] for stamp in snapshot_dates],
        dtype=bool,
    ).reshape(len(snapshot_dates), len(tickers))
    position = snapshot_dates.searchsorted(mask.index, side="right") - 1
    values = np.zeros(mask.shape, dtype=bool)
    has_snapshot = position >= 0
    values[has_snapshot] = snapshot_matrix[position[has_snapshot]]
    return pd.DataFrame(values, index=mask.index, columns=mask.columns)


def build_eligibility_mask(
    prices: pd.DataFrame,
    membership: pd.DataFrame | None = None,
    min_history_days: int = MIN_UNIVERSE_HISTORY_DAYS,
    require_index_membership: bool = REQUIRE_INDEX_MEMBERSHIP,
    membership_path: str | Path = UNIVERSE_MEMBERSHIP_PATH,
) -> pd.DataFrame:
    """Combine tradability and optional index-membership filters."""
    clean = drop_empty_price_columns(prices)
    tradable = build_tradability_mask(clean, min_history_days=min_history_days)
    if not require_index_membership:
        # Reindex to original columns so callers can align easily; empty cols False.
        return tradable.reindex(index=prices.index, columns=prices.columns, fill_value=False)

    members = build_membership_mask(
        tradable.index,
        tradable.columns,
        membership=membership,
        path=membership_path,
    )
    eligible = tradable & members.reindex_like(tradable).fillna(False)
    return eligible.reindex(index=prices.index, columns=prices.columns, fill_value=False)


def eligible_tickers_on_date(
    eligibility: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.Index:
    """Tickers marked eligible on ``date`` (exact index label required)."""
    stamp = pd.Timestamp(date)
    if stamp not in eligibility.index:
        prior = eligibility.index[eligibility.index <= stamp]
        if len(prior) == 0:
            return pd.Index([])
        stamp = prior[-1]
    row = eligibility.loc[stamp]
    return row.index[row.fillna(False).astype(bool)]


def load_default_eligibility_mask(
    prices: pd.DataFrame | None = None,
    membership_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build the repo-default eligibility mask from local prices + membership CSV."""
    from src.data.io import load_price_data

    panel = prices if prices is not None else load_price_data()
    return build_eligibility_mask(panel, membership_path=membership_path or UNIVERSE_MEMBERSHIP_PATH)

def resolve_membership_path(data_paths: Any | None = None, root: str | Path = ".") -> str:
    """Paper-local membership file when present, otherwise the checked-in research file."""
    root = Path(root)
    if data_paths is not None:
        candidate = Path(data_paths.universe_membership)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            return str(candidate)
    research = root / UNIVERSE_MEMBERSHIP_PATH
    return str(research if research.is_file() else Path(UNIVERSE_MEMBERSHIP_PATH))


def required_membership_month_end(as_of) -> pd.Timestamp:
    """Last calendar month-end on or before as_of.

    Mid-month dates do not yet require that month's snapshot: as-of 2026-10-30
    the required snapshot is 2026-09-30. A 2026-11-01 snapshot covers 2026-10-31
    but not 2026-11-30.
    """
    as_of = pd.Timestamp(as_of).tz_localize(None).normalize()
    return (as_of.replace(day=1) - pd.Timedelta(days=1)).normalize()


def month_ends_after(after, through) -> list[pd.Timestamp]:
    """Calendar month-ends strictly after after and on or before through."""
    after = pd.Timestamp(after).tz_localize(None).normalize()
    through = pd.Timestamp(through).tz_localize(None).normalize()
    cursor = (after + pd.offsets.MonthEnd(0)).normalize()
    if cursor <= after:
        cursor = (cursor + pd.offsets.MonthEnd(1)).normalize()
    ends: list[pd.Timestamp] = []
    while cursor <= through:
        ends.append(pd.Timestamp(cursor).normalize())
        cursor = (cursor + pd.offsets.MonthEnd(1)).normalize()
    return ends


def parse_historical_components(
    source: str | Path | pd.DataFrame,
    tickers,
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Filter an upstream historical-constituents table down to tickers.

    Accepts the chinobing sp_500_historical_components.csv layout
    (date,tickers with a comma-separated list) or a long date,ticker
    table. Alias spellings are mapped onto the repo's canonical tickers.
    """
    aliases = aliases or TICKER_MEMBERSHIP_ALIASES
    wanted: dict[str, str] = {}
    for ticker in tickers:
        canonical = normalize_ticker(ticker)
        wanted[canonical] = canonical
        for name in aliases.get(canonical, ()):
            wanted[normalize_ticker(name)] = canonical

    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        path_like = isinstance(source, Path) or (
            isinstance(source, str) and chr(10) not in source and Path(source).exists()
        )
        text = Path(source).read_text() if path_like else str(source)
        raw = pd.read_csv(io.StringIO(text))

    raw.columns = [str(column).strip().lower() for column in raw.columns]
    rows: list[tuple[pd.Timestamp, str]] = []
    if "tickers" in raw.columns:
        for rec in raw.itertuples(index=False):
            stamp = pd.Timestamp(getattr(rec, "date")).tz_localize(None).normalize()
            packed = getattr(rec, "tickers")
            if pd.isna(packed):
                continue
            for token in str(packed).split(","):
                token = normalize_ticker(token)
                if token in wanted:
                    rows.append((stamp, wanted[token]))
    elif "ticker" in raw.columns:
        for rec in raw.itertuples(index=False):
            token = normalize_ticker(getattr(rec, "ticker"))
            if token in wanted:
                stamp = pd.Timestamp(getattr(rec, "date")).tz_localize(None).normalize()
                rows.append((stamp, wanted[token]))
    else:
        raise ValueError("historical components need a tickers or ticker column")

    if not rows:
        return pd.DataFrame(columns=["date", "ticker"])
    frame = pd.DataFrame(rows, columns=["date", "ticker"])
    return frame.drop_duplicates(["date", "ticker"]).sort_values(["date", "ticker"]).reset_index(drop=True)


def append_month_end_membership(
    base: pd.DataFrame,
    historical: pd.DataFrame,
    through,
) -> pd.DataFrame:
    """Keep base rows and append month-end snapshots after its last date.

    Each new month-end uses the latest historical row dated on or before it.
    Snapshot dates are calendar month-ends, matching the research file.
    """
    frames: list[pd.DataFrame] = []
    if base is not None and not base.empty:
        kept = base[["date", "ticker"]].copy()
        kept["date"] = pd.to_datetime(kept["date"]).dt.tz_localize(None).dt.normalize()
        kept["ticker"] = kept["ticker"].map(normalize_ticker)
        frames.append(kept)
        last = pd.Timestamp(kept["date"].max()).normalize()
    else:
        last = pd.Timestamp("1677-01-01")

    if historical is None or historical.empty:
        hist = pd.DataFrame(columns=["date", "ticker"])
        hist_dates = pd.DatetimeIndex([])
    else:
        hist = historical[["date", "ticker"]].copy()
        hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None).dt.normalize()
        hist["ticker"] = hist["ticker"].map(normalize_ticker)
        hist_dates = pd.DatetimeIndex(sorted(hist["date"].unique()))

    for end in month_ends_after(last, through):
        available = hist_dates[hist_dates <= end]
        if len(available) == 0:
            continue
        snap = hist.loc[hist["date"] == available[-1], ["ticker"]].copy()
        snap.insert(0, "date", end)
        frames.append(snap)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["date", "ticker"]).sort_values(["date", "ticker"]).reset_index(drop=True)


def write_membership_table(path: str | Path, frame: pd.DataFrame) -> None:
    """Write date,ticker membership rows (ISO dates, canonical tickers)."""
    out = frame[["date", "ticker"]].copy() if not frame.empty else pd.DataFrame(columns=["date", "ticker"])
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
        out["ticker"] = out["ticker"].map(normalize_ticker)
        out = out.drop_duplicates().sort_values(["date", "ticker"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

