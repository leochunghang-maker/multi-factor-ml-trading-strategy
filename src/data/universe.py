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

from pathlib import Path

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


def load_default_eligibility_mask(prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the repo-default eligibility mask from local prices + membership CSV."""
    from src.data.io import load_price_data

    panel = prices if prices is not None else load_price_data()
    return build_eligibility_mask(panel)
