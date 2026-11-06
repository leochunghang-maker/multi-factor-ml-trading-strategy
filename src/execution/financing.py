"""Borrow and cash-financing costs for research and paper/sim paths.

Research long/short books pay an annualized **stock-borrow** fee on short gross
exposure each holding period. Optional **short-proceeds rebate** reduces that
drag; optional **cash financing** charges any net long not covered by short
proceeds (margin-style debit on the research weight book).

Paper/sim is long-only and cash-constrained today, so short borrow is normally
zero and cash financing only applies if cash goes negative (margin debit). The
same helpers still accrue daily if shorts ever exist. Defaults are conservative
(borrow charged, rebate off, cash financing off unless configured).
"""

from __future__ import annotations


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

    ``short_gross`` / ``long_gross`` are absolute side exposures as fractions of
    NAV (e.g. 1.0 + 1.0 for research gross 2.0).
    """
    short = max(0.0, float(short_gross))
    long = max(0.0, float(long_gross))
    borrow = short * period_rate_from_annual_bps(annual_borrow_bps, periods_per_year)
    rebate = short * period_rate_from_annual_bps(annual_short_rebate_bps, periods_per_year)
    # Dollar-neutral books fund longs with short proceeds; only excess long is
    # treated as a cash/margin debit on the research weight book.
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
) -> float:
    """Holding-period financing drag implied by a signed weight vector."""
    short_gross = float((-weights.clip(upper=0.0)).sum())
    long_gross = float(weights.clip(lower=0.0).sum())
    return holding_period_financing_cost(
        short_gross,
        long_gross,
        annual_borrow_bps=annual_borrow_bps,
        annual_short_rebate_bps=annual_short_rebate_bps,
        annual_cash_financing_bps=annual_cash_financing_bps,
        periods_per_year=periods_per_year,
    )


def daily_financing_dollars(
    *,
    short_market_value: float = 0.0,
    margin_debit: float = 0.0,
    annual_borrow_bps: float,
    annual_short_rebate_bps: float = 0.0,
    annual_cash_financing_bps: float = 0.0,
    trading_days_per_year: float = 252.0,
) -> float:
    """Dollar financing accrual for one trading day (positive = cash outflow).

    ``short_market_value`` is the absolute dollar short book (>= 0).
    ``margin_debit`` is ``max(0, -cash)`` for the cash/margin account.
    """
    short = max(0.0, float(short_market_value))
    debit = max(0.0, float(margin_debit))
    borrow = short * period_rate_from_annual_bps(annual_borrow_bps, trading_days_per_year)
    rebate = short * period_rate_from_annual_bps(annual_short_rebate_bps, trading_days_per_year)
    cash_financing = debit * period_rate_from_annual_bps(
        annual_cash_financing_bps, trading_days_per_year
    )
    return borrow + cash_financing - rebate
