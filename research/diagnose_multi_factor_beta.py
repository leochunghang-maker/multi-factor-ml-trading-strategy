"""Diagnose why the beta-neutral multi-factor book has negative realized beta.

Descriptive only, run on the fe8b173 construction (252-day OLS betas). It does not
choose any parameter: the EWMA estimator (trial 43) was fixed before this ran.
Look-ahead quantities (``ex_post_*``) are labelled and used only for diagnosis.

    python research/diagnose_multi_factor_beta.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    BENCHMARK_PRICES_PATH,
    BETA_BENCHMARK,
    BETA_LOOKBACK_DAYS,
    BETA_MIN_OBS,
    BETA_SHRINKAGE,
    FORWARD_RETURNS_PATH,
    MEAN_REVERSION_PATH,
    MOMENTUM_PATH,
    MULTI_FACTOR_LONG_COUNT,
    MULTI_FACTOR_MIN_ASSETS,
    MULTI_FACTOR_SHORT_COUNT,
    RESEARCH_LONG_SHORT_GROSS_EXPOSURE,
)
from src.data.io import load_factor_frame, load_price_data  # noqa: E402
from src.data.universe import load_default_eligibility_mask  # noqa: E402
from src.portfolio.portfolio import (  # noqa: E402
    _eligible_on_date,
    equal_weight_long_short_weights,
    month_end_trading_dates,
)
from src.risk.beta import build_beta_panel, betas_on_date, load_benchmark_close  # noqa: E402

OUT = ROOT / "results" / "multi_factor_beta_diagnosis.csv"


def _ols_beta(y: pd.Series, x: pd.Series) -> float:
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 12:
        return float("nan")
    return float(np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1] / df.iloc[:, 1].var())


def main() -> None:
    prices = load_price_data()
    momentum = load_factor_frame(MOMENTUM_PATH)
    mean_reversion = load_factor_frame(MEAN_REVERSION_PATH)
    forward = load_factor_frame(FORWARD_RETURNS_PATH)
    eligibility = load_default_eligibility_mask()
    spy = load_benchmark_close(BENCHMARK_PRICES_PATH, BETA_BENCHMARK)
    dates = month_end_trading_dates(momentum)
    common = dict(
        lookback=BETA_LOOKBACK_DAYS, min_obs=BETA_MIN_OBS, shrinkage=BETA_SHRINKAGE, estimator="ols",
    )
    beta_ante = build_beta_panel(prices, spy, dates, **common)
    # Look-ahead (diagnostic only): betas over the 252 days AFTER each rebalance,
    # i.e. what the names' betas turned out to be while (and after) they were held.
    fwd_dates = []
    idx = prices.index
    for d in dates:
        pos = idx.searchsorted(d)
        fwd_dates.append(idx[min(pos + BETA_LOOKBACK_DAYS, len(idx) - 1)])
    beta_post = build_beta_panel(prices, spy, fwd_dates, **common)
    beta_post.index = dates
    daily = prices.pct_change(fill_method=None)
    vol63 = daily.rolling(63).std() * np.sqrt(252)

    spy_m = spy.reindex(dates, method="ffill")
    spy_fwd = spy_m.shift(-1) / spy_m - 1

    rows = []
    for date in dates:
        mom, mr, fut = momentum.loc[date], mean_reversion.loc[date], forward.loc[date]
        eligible = _eligible_on_date(eligibility, date, mom.index)
        valid = mom.notna() & mr.notna() & fut.notna() & eligible
        if valid.sum() < MULTI_FACTOR_MIN_ASSETS:
            continue
        mom_z = (mom[valid] - mom[valid].mean()) / mom[valid].std()
        mr_z = (mr[valid] - mr[valid].mean()) / mr[valid].std()
        score = mom_z - mr_z
        b = betas_on_date(beta_ante, date, score.index)
        w = equal_weight_long_short_weights(
            score, MULTI_FACTOR_LONG_COUNT, MULTI_FACTOR_SHORT_COUNT,
            gross_exposure=RESEARCH_LONG_SHORT_GROSS_EXPOSURE, betas=b,
        )
        bp = beta_post.loc[date].reindex(w.index).fillna(1.0)
        r = fut.reindex(w.index).fillna(0.0)
        lo, sh = w > 0, w < 0
        v = vol63.loc[date].reindex(w.index)
        rows.append({
            "date": date,
            "long_gross": float(w[lo].sum()),
            "short_gross": float(-w[sh].sum()),
            "ex_ante_beta": float((w * b).sum()),
            "ex_ante_long_beta": float((w[lo] * b[lo]).sum() / w[lo].sum()),
            "ex_ante_short_beta": float((-w[sh] * b[sh]).sum() / -w[sh].sum()),
            "ex_post_beta_lookahead": float((w * bp).sum()),
            "ex_post_long_beta_lookahead": float((w[lo] * bp[lo]).sum() / w[lo].sum()),
            "ex_post_short_beta_lookahead": float((-w[sh] * bp[sh]).sum() / -w[sh].sum()),
            "long_vol63": float(v[lo].mean()),
            "short_vol63": float(v[sh].mean()),
            "universe_vol63": float(v.mean()),
            "long_leg_return": float((w[lo] * r[lo]).sum() / w[lo].sum()),
            "short_leg_return": float((-w[sh] * r[sh]).sum() / -w[sh].sum()),
            "book_return_unscaled": float((w * r).sum()),
            "spy_next_month": float(spy_fwd.loc[date]),
        })
    diag = pd.DataFrame(rows).set_index("date")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(OUT)

    reported = pd.read_csv(ROOT / "results" / "multi_factor_results.csv", index_col=0, parse_dates=True)
    x = diag["spy_next_month"]
    summary = {
        "months": len(diag),
        "mean_ex_ante_beta": diag["ex_ante_beta"].mean(),
        "mean_ex_post_beta_lookahead": diag["ex_post_beta_lookahead"].mean(),
        "mean_ex_ante_long_beta": diag["ex_ante_long_beta"].mean(),
        "mean_ex_ante_short_beta": diag["ex_ante_short_beta"].mean(),
        "mean_ex_post_long_beta_lookahead": diag["ex_post_long_beta_lookahead"].mean(),
        "mean_ex_post_short_beta_lookahead": diag["ex_post_short_beta_lookahead"].mean(),
        "mean_long_gross": diag["long_gross"].mean(),
        "mean_short_gross": diag["short_gross"].mean(),
        "mean_long_vol63": diag["long_vol63"].mean(),
        "mean_short_vol63": diag["short_vol63"].mean(),
        "mean_universe_vol63": diag["universe_vol63"].mean(),
        "realized_beta_long_leg_monthly": _ols_beta(diag["long_leg_return"], x),
        "realized_beta_short_leg_monthly": _ols_beta(diag["short_leg_return"], x),
        "realized_beta_book_unscaled_monthly": _ols_beta(diag["book_return_unscaled"], x),
        "realized_beta_book_reported_monthly": _ols_beta(reported["portfolio_return"], x),
        "realized_beta_unscaled_up_months": _ols_beta(diag.loc[x > 0, "book_return_unscaled"], x[x > 0]),
        "realized_beta_unscaled_down_months": _ols_beta(diag.loc[x <= 0, "book_return_unscaled"], x[x <= 0]),
        "corr_vol_scale_vs_spy_next": float(reported["vol_scale"].corr(x)),
    }
    for key, value in summary.items():
        print(f"{key:40s} {value:.3f}" if isinstance(value, float) else f"{key:40s} {value}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
