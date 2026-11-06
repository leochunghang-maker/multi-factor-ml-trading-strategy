"""Borrow and cash-financing costs for research and paper/sim paths.

Research long/short books pay annualized **stock-borrow** on short exposure.
When name-level HTB tiering is enabled, each short name can carry its own
annualized borrow bps (GC / elevated / HTB / special). Optional short-proceeds
rebate and cash financing still apply at the book level.

Paper/sim is long-only today, so short borrow is normally zero; daily accrual
still supports per-name short rates if shorts exist.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def period_rate_from_annual_bps(annual_bps: float, periods_per_year: float) -> float:
    """Convert annualized basis points into a per-period fractional rate."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    return (float(annual_bps) / 10_000.0) / float(periods_per_year)


def holding_period_financing_cost(
    short_gross: float,
    long_gross: float = 0.0,
    *,
    annual_borrow_bps: float,
    annual_short_rebate_bps: float = 0.0,
    annual_cash_financing_bps: float = 0.0,
    periods_per_year: float = 12.0,
) -> float:
    """Fractional NAV drag for one holding period (positive = cost).

    Flat-rate helper: ``short_gross`` / ``long_gross`` are absolute side
    exposures as fractions of NAV.
    """
    short = max(0.0, float(short_gross))
    long = max(0.0, float(long_gross))
    borrow = short * period_rate_from_annual_bps(annual_borrow_bps, periods_per_year)
    rebate = short * period_rate_from_annual_bps(annual_short_rebate_bps, periods_per_year)
    financed = max(0.0, long - short)
    cash_financing = financed * period_rate_from_annual_bps(
        annual_cash_financing_bps, periods_per_year
    )
    return borrow + cash_financing - rebate


def financing_cost_from_weights(
    weights,
    *,
    annual_borrow_bps: float,
    annual_short_rebate_bps: float = 0.0,
    annual_cash_financing_bps: float = 0.0,
    periods_per_year: float = 12.0,
    borrow_bps_by_symbol: Mapping[str, float] | pd.Series | None = None,
) -> float:
    """Holding-period financing drag implied by a signed weight vector.

    If ``borrow_bps_by_symbol`` is provided, short borrow is the exposure-weighted
    sum of each name's annualized borrow bps. Otherwise the flat
    ``annual_borrow_bps`` rate is applied to total short gross.
    """
    weights = pd.Series(weights, dtype=float)
    short_weights = (-weights.clip(upper=0.0))
    long_gross = float(weights.clip(lower=0.0).sum())
    short_gross = float(short_weights.sum())

    if borrow_bps_by_symbol is None or short_gross <= 0:
        borrow = short_gross * period_rate_from_annual_bps(
            annual_borrow_bps, periods_per_year
        )
    else:
        rates = pd.Series(borrow_bps_by_symbol, dtype=float).reindex(short_weights.index)
        rates = rates.fillna(annual_borrow_bps)
        borrow = float(
            (short_weights * (rates / 10_000.0 / float(periods_per_year))).sum()
        )

    rebate = short_gross * period_rate_from_annual_bps(
        annual_short_rebate_bps, periods_per_year
    )
    financed = max(0.0, long_gross - short_gross)
    cash_financing = financed * period_rate_from_annual_bps(
        annual_cash_financing_bps, periods_per_year
    )
    return borrow + cash_financing - rebate


def daily_financing_dollars(
    *,
    short_market_value: float = 0.0,
    margin_debit: float = 0.0,
    annual_borrow_bps: float,
    annual_short_rebate_bps: float = 0.0,
    annual_cash_financing_bps: float = 0.0,
    trading_days_per_year: float = 252.0,
    short_market_values_by_symbol: Mapping[str, float] | None = None,
    borrow_bps_by_symbol: Mapping[str, float] | None = None,
) -> float:
    """Dollar financing accrual for one trading day (positive = cash outflow).

    Prefer ``short_market_values_by_symbol`` + ``borrow_bps_by_symbol`` for
    name-level HTB; otherwise fall back to flat ``short_market_value``.
    """
    if short_market_values_by_symbol and borrow_bps_by_symbol is not None:
        borrow = 0.0
        short_total = 0.0
        for symbol, value in short_market_values_by_symbol.items():
            mv = max(0.0, float(value))
            if mv <= 0:
                continue
            short_total += mv
            bps = float(borrow_bps_by_symbol.get(symbol, annual_borrow_bps))
            borrow += mv * period_rate_from_annual_bps(bps, trading_days_per_year)
        short = short_total
    else:
        short = max(0.0, float(short_market_value))
        borrow = short * period_rate_from_annual_bps(annual_borrow_bps, trading_days_per_year)

    debit = max(0.0, float(margin_debit))
    rebate = short * period_rate_from_annual_bps(annual_short_rebate_bps, trading_days_per_year)
    cash_financing = debit * period_rate_from_annual_bps(
        annual_cash_financing_bps, trading_days_per_year
    )
    return borrow + cash_financing - rebate
