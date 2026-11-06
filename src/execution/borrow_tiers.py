"""Name-level hard-to-borrow (HTB) fee tiers via a transparent liquidity proxy.

This is **not** a live locate / prime-broker stock-loan feed. Large-cap S&P 100
names are usually general collateral (GC); we still differentiate fees using:

1. Optional checked-in overrides (``data/borrow_tier_overrides.csv``) for demo /
   known special names.
2. An ADV$ + realized-volatility proxy: weaker liquidity and higher vol map to
   elevated / HTB tiers with higher annualized borrow bps.

Document the proxy honestly in README; production desks should replace this with
vendor borrow/HTB data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    ANNUAL_BORROW_BPS,
    BORROW_TIER_LOOKBACK_DAYS,
    BORROW_TIER_OVERRIDES_PATH,
    BORROW_TIER_RATES_BPS,
    BORROW_TIERING_ENABLED,
)
from src.portfolio.constraints import average_daily_dollar_volume

DEFAULT_TIER_RATES_BPS: dict[str, float] = {
    "gc": 50.0,
    "elevated": 300.0,
    "htb": 1000.0,
    "special": 3000.0,
}


def load_borrow_tier_overrides(
    path: str | Path = BORROW_TIER_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Load optional ticker → tier overrides (columns: ticker, tier)."""
    overrides_path = Path(path)
    if not overrides_path.exists():
        return pd.DataFrame(columns=["ticker", "tier"])
    frame = pd.read_csv(overrides_path)
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "tier"])
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    frame["tier"] = frame["tier"].astype(str).str.lower().str.strip()
    return frame[["ticker", "tier"]]


def tier_rates_bps(config_rates: dict[str, Any] | None = None) -> dict[str, float]:
    rates = dict(DEFAULT_TIER_RATES_BPS)
    # Keep GC aligned with the flat book rate unless explicitly overridden.
    rates["gc"] = float(ANNUAL_BORROW_BPS)
    if config_rates:
        for key, value in config_rates.items():
            rates[str(key).lower()] = float(value)
    return rates


def classify_borrow_tier(
    adv_dollars: float | None,
    volatility: float | None,
    *,
    adv_p20: float,
    adv_p10: float,
    vol_p80: float,
    vol_p90: float,
    override_tier: str | None = None,
) -> str:
    """Map liquidity/vol proxies (and optional override) onto a borrow tier."""
    if override_tier:
        return str(override_tier).lower()
    adv = float(adv_dollars) if adv_dollars is not None and np.isfinite(adv_dollars) else float("nan")
    vol = float(volatility) if volatility is not None and np.isfinite(volatility) else float("nan")

    weak_liq = np.isfinite(adv) and adv <= adv_p20
    very_weak_liq = np.isfinite(adv) and adv <= adv_p10
    high_vol = np.isfinite(vol) and vol >= vol_p80
    very_high_vol = np.isfinite(vol) and vol >= vol_p90

    if (very_weak_liq and high_vol) or (weak_liq and very_high_vol):
        return "htb"
    if weak_liq or very_high_vol:
        return "elevated"
    if not np.isfinite(adv):
        # Missing ADV → conservative elevated rather than optimistic GC.
        return "elevated"
    return "gc"


def assign_borrow_bps(
    adv: pd.Series,
    volatility: pd.Series | None = None,
    *,
    overrides: pd.DataFrame | None = None,
    rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return a frame indexed by ticker with columns tier, annual_borrow_bps, adv_dollars."""
    rates = tier_rates_bps(rates)
    adv = adv.astype(float)
    vol = (
        volatility.reindex(adv.index).astype(float)
        if volatility is not None
        else pd.Series(np.nan, index=adv.index)
    )
    finite_adv = adv.replace([np.inf, -np.inf], np.nan).dropna()
    finite_vol = vol.replace([np.inf, -np.inf], np.nan).dropna()
    adv_p20 = float(finite_adv.quantile(0.20)) if len(finite_adv) else 0.0
    adv_p10 = float(finite_adv.quantile(0.10)) if len(finite_adv) else 0.0
    vol_p80 = float(finite_vol.quantile(0.80)) if len(finite_vol) else float("inf")
    vol_p90 = float(finite_vol.quantile(0.90)) if len(finite_vol) else float("inf")

    override_map: dict[str, str] = {}
    if overrides is not None and not overrides.empty:
        override_map = {
            str(row.ticker).upper(): str(row.tier).lower()
            for row in overrides.itertuples(index=False)
        }

    rows = []
    for ticker in adv.index:
        key = str(ticker).upper()
        tier = classify_borrow_tier(
            adv.get(ticker),
            vol.get(ticker),
            adv_p20=adv_p20,
            adv_p10=adv_p10,
            vol_p80=vol_p80,
            vol_p90=vol_p90,
            override_tier=override_map.get(key),
        )
        rows.append({
            "ticker": key,
            "tier": tier,
            "annual_borrow_bps": float(rates.get(tier, rates["gc"])),
            "adv_dollars": float(adv.get(ticker)) if np.isfinite(adv.get(ticker, np.nan)) else np.nan,
            "volatility": float(vol.get(ticker)) if np.isfinite(vol.get(ticker, np.nan)) else np.nan,
        })
    return pd.DataFrame(rows).set_index("ticker")


def borrow_bps_series_as_of(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None,
    as_of_date: pd.Timestamp,
    *,
    lookback: int | None = None,
    overrides: pd.DataFrame | None = None,
    rates: dict[str, float] | None = None,
) -> pd.Series:
    """Point-in-time annual borrow bps per ticker using ADV + vol proxy."""
    lookback = int(lookback or BORROW_TIER_LOOKBACK_DAYS or LIQUIDITY_LOOKBACK_DAYS)
    as_of = pd.Timestamp(as_of_date)
    adv = average_daily_dollar_volume(prices, volumes, as_of, lookback)
    if adv.empty:
        # No volume panel → flat GC.
        return pd.Series(float(tier_rates_bps(rates)["gc"]), index=prices.columns, dtype=float)

    hist = prices.loc[prices.index <= as_of].tail(lookback)
    vol = hist.pct_change(fill_method=None).std()
    # Align to price columns so every name gets a rate.
    adv = adv.reindex(prices.columns)
    assigned = assign_borrow_bps(adv, vol, overrides=overrides, rates=rates)
    return assigned["annual_borrow_bps"].reindex(prices.columns).fillna(tier_rates_bps(rates)["gc"])


def build_borrow_bps_panel(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None,
    dates: pd.DatetimeIndex | list,
    *,
    lookback: int | None = None,
    overrides: pd.DataFrame | None = None,
    rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Date × ticker annual borrow bps for research rebalance dates."""
    if overrides is None:
        overrides = load_borrow_tier_overrides()
    rows = []
    index = []
    for date in pd.DatetimeIndex(dates):
        series = borrow_bps_series_as_of(
            prices,
            volumes,
            date,
            lookback=lookback,
            overrides=overrides,
            rates=rates,
        )
        rows.append(series)
        index.append(date)
    if not rows:
        return pd.DataFrame(columns=prices.columns)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


def snapshot_borrow_tiers(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None,
    as_of_date: pd.Timestamp | None = None,
    *,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Write a human-readable latest tier map for docs / debugging."""
    as_of = pd.Timestamp(as_of_date or prices.index.max())
    overrides = load_borrow_tier_overrides()
    adv = average_daily_dollar_volume(
        prices, volumes, as_of, int(BORROW_TIER_LOOKBACK_DAYS)
    ).reindex(prices.columns)
    hist = prices.loc[prices.index <= as_of].tail(int(BORROW_TIER_LOOKBACK_DAYS))
    vol = hist.pct_change(fill_method=None).std()
    assigned = assign_borrow_bps(adv, vol, overrides=overrides)
    assigned.insert(0, "as_of_date", as_of.strftime("%Y-%m-%d"))
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        assigned.reset_index().to_csv(out, index=False)
    return assigned



def research_borrow_bps_panel(
    prices: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame | None:
    """Build a month-end borrow bps panel when tiering is enabled; else None."""
    if not BORROW_TIERING_ENABLED:
        return None
    from src.data.io import load_price_data
    from src.portfolio.constraints import load_optional_volume_data
    from src.portfolio.portfolio import month_end_trading_dates
    from src.config import PRICE_DATA_PATH, VOLUME_DATA_PATH

    prices = prices if prices is not None else load_price_data(PRICE_DATA_PATH)
    if volumes is None:
        volumes = load_optional_volume_data(VOLUME_DATA_PATH)
    if dates is None:
        dates = month_end_trading_dates(prices)
    return build_borrow_bps_panel(prices, volumes, dates)
