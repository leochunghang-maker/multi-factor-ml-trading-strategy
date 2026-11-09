"""How much of the research Sharpe could be luck? Bootstrap CI, PSR and DSR.

Usage:
    python src/run_significance_report.py              # reruns walk-forward (~25s)
    python src/run_significance_report.py --cached-walk-forward

Writes ``results/significance_report.csv``. Research/simulation analytics only.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import (
    MULTI_DAY_SIMULATION_DIR,
    MULTI_FACTOR_RESULTS_PATH,
    SIGNIFICANCE_BLOCK_LENGTH_DAILY,
    SIGNIFICANCE_BLOCK_LENGTH_MONTHLY,
    SIGNIFICANCE_BOOTSTRAP_METHOD,
    SIGNIFICANCE_BOOTSTRAP_N_RESAMPLES,
    SIGNIFICANCE_CONFIDENCE,
    SIGNIFICANCE_N_TRIALS,
    SIGNIFICANCE_N_TRIALS_SENSITIVITY,
    SIGNIFICANCE_REPORT_PATH,
    SIGNIFICANCE_SEED,
    SIGNIFICANCE_TRIAL_SHARPES,
    WALK_FORWARD_PERIOD_RETURNS_PATH,
)
from src.reporting.significance import (
    deflated_sharpe_ratio,
    significance_summary,
    trial_sharpe_variance,
)


def load_walk_forward_returns(cached: bool) -> pd.Series:
    path = Path(WALK_FORWARD_PERIOD_RETURNS_PATH)
    if cached and path.exists():
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        return frame["portfolio_return"]
    from src.backtests.ml_backtests import run_walk_forward_ml_backtest

    results, _, _ = run_walk_forward_ml_backtest()
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path)
    return results["portfolio_return"]


def load_multi_factor_returns() -> pd.Series:
    frame = pd.read_csv(MULTI_FACTOR_RESULTS_PATH, parse_dates=["date"]).set_index("date")
    return frame["portfolio_return"]


def load_multi_day_sim_returns() -> pd.Series | None:
    path = Path(MULTI_DAY_SIMULATION_DIR) / "portfolio_nav_history.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    returns = frame["daily_return"].astype(float)
    # Drop the all-cash warm-up before the first fill so T is not inflated by zeros.
    active = returns.ne(0.0)
    if not active.any():
        return None
    return returns.loc[active.idxmax():]


def load_equal_weight_universe_returns(start, end) -> pd.Series:
    """Daily equal-weight buy-and-hold of the (survivorship-biased) price panel.

    Context only for the long-only simulation: shows how much Sharpe plain
    equity beta earned over the same window. Not a trial and not deflated.
    """
    from src.data.io import load_price_data

    prices = load_price_data()
    daily = prices.pct_change(fill_method=None)
    daily = daily.loc[(daily.index >= pd.Timestamp(start)) & (daily.index <= pd.Timestamp(end))]
    return daily.mean(axis=1, skipna=True).dropna()


def build_report(cached_walk_forward: bool = False) -> pd.DataFrame:
    series = {
        "walk_forward_ml": (load_walk_forward_returns(cached_walk_forward), 12, SIGNIFICANCE_BLOCK_LENGTH_MONTHLY),
        "multi_factor": (load_multi_factor_returns(), 12, SIGNIFICANCE_BLOCK_LENGTH_MONTHLY),
    }
    sim = load_multi_day_sim_returns()
    if sim is not None:
        series["multi_day_sim"] = (sim, 252, SIGNIFICANCE_BLOCK_LENGTH_DAILY)

    trial_counts = sorted(set([SIGNIFICANCE_N_TRIALS, *SIGNIFICANCE_N_TRIALS_SENSITIVITY]))
    pooled_trials = [x for values in SIGNIFICANCE_TRIAL_SHARPES.values() for x in values]
    rows = []
    for name, (returns, ppy, block) in series.items():
        variance = trial_sharpe_variance(SIGNIFICANCE_TRIAL_SHARPES.get(name, []), ppy)
        pooled_variance = trial_sharpe_variance(pooled_trials, ppy)
        for n_trials in trial_counts:
            summary = significance_summary(
                returns,
                periods_per_year=ppy,
                n_trials=n_trials,
                sharpe_variance=variance,
                block_length=block,
                n_resamples=SIGNIFICANCE_BOOTSTRAP_N_RESAMPLES,
                confidence=SIGNIFICANCE_CONFIDENCE,
                method=SIGNIFICANCE_BOOTSTRAP_METHOD,
                seed=SIGNIFICANCE_SEED,
            )
            dsr_pooled, sr0_pooled = deflated_sharpe_ratio(
                summary["sharpe_per_period"],
                summary["n_obs"],
                summary["skewness"],
                summary["kurtosis"],
                n_trials=n_trials,
                sharpe_variance=pooled_variance,
            )
            summary["sr0_annualized_pooled_var"] = float(sr0_pooled * ppy ** 0.5)
            summary["deflated_sharpe_ratio_pooled_var"] = dsr_pooled
            rows.append({
                "strategy": name,
                "primary": n_trials == SIGNIFICANCE_N_TRIALS,
                "start": str(returns.index.min().date()),
                "end": str(returns.index.max().date()),
                "periods_per_year": ppy,
                "bootstrap_method": SIGNIFICANCE_BOOTSTRAP_METHOD,
                "confidence": SIGNIFICANCE_CONFIDENCE,
                **summary,
            })
    if sim is not None:
        reference = load_equal_weight_universe_returns(sim.index.min(), sim.index.max())
        summary = significance_summary(
            reference,
            periods_per_year=252,
            n_trials=1,
            sharpe_variance=0.0,
            block_length=SIGNIFICANCE_BLOCK_LENGTH_DAILY,
            n_resamples=SIGNIFICANCE_BOOTSTRAP_N_RESAMPLES,
            confidence=SIGNIFICANCE_CONFIDENCE,
            method=SIGNIFICANCE_BOOTSTRAP_METHOD,
            seed=SIGNIFICANCE_SEED,
        )
        rows.append({
            "strategy": "reference_equal_weight_universe",
            "primary": True,
            "start": str(reference.index.min().date()),
            "end": str(reference.index.max().date()),
            "periods_per_year": 252,
            "bootstrap_method": SIGNIFICANCE_BOOTSTRAP_METHOD,
            "confidence": SIGNIFICANCE_CONFIDENCE,
            **summary,
        })
    report = pd.DataFrame(rows)
    Path(SIGNIFICANCE_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(SIGNIFICANCE_REPORT_PATH, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-walk-forward", action="store_true",
                        help="Use results/walk_forward_period_returns.csv instead of rerunning.")
    args = parser.parse_args()
    report = build_report(cached_walk_forward=args.cached_walk_forward)
    cols = ["strategy", "n_trials", "n_obs", "sharpe_annualized", "ci_lower", "ci_upper",
            "prob_sharpe_le_zero", "psr_vs_zero", "sr0_annualized", "deflated_sharpe_ratio",
            "deflated_sharpe_ratio_pooled_var"]
    with pd.option_context("display.width", 200):
        print(report[cols].round(4).to_string(index=False))
    print(f"\nSaved {SIGNIFICANCE_REPORT_PATH}")


if __name__ == "__main__":
    main()
