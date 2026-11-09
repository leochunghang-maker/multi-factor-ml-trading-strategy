"""How much of the research Sharpe could be luck? Bootstrap CI, PSR, DSR, benchmark tests.

Usage:
    python src/run_significance_report.py              # reruns walk-forward (~25s)
    python src/run_significance_report.py --cached-walk-forward

Writes ``results/significance_report.csv`` (absolute Sharpe) and
``results/benchmark_relative_report.csv`` (alpha/beta/IR vs equal-weight universe
and SPY). ``n_trials`` comes from ``research/trial_log.csv`` + documented padding.
Research/simulation analytics only.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import (
    BENCHMARK_PRICES_PATH,
    BENCHMARK_RELATIVE_REPORT_PATH,
    BENCHMARK_SYMBOL,
    MULTI_DAY_SIMULATION_DIR,
    MULTI_FACTOR_RESULTS_PATH,
    SIGNIFICANCE_BLOCK_LENGTH_DAILY,
    SIGNIFICANCE_BLOCK_LENGTH_MONTHLY,
    SIGNIFICANCE_BOOTSTRAP_METHOD,
    SIGNIFICANCE_BOOTSTRAP_N_RESAMPLES,
    SIGNIFICANCE_CAUSAL_BETA_WINDOW_DAILY,
    SIGNIFICANCE_CAUSAL_BETA_WINDOW_MONTHLY,
    SIGNIFICANCE_CONFIDENCE,
    SIGNIFICANCE_HAC_LAGS,
    SIGNIFICANCE_IR_VARIANCE_SOURCE,
    SIGNIFICANCE_N_TRIALS_SENSITIVITY,
    SIGNIFICANCE_REPORT_PATH,
    SIGNIFICANCE_SEED,
    SIGNIFICANCE_TRIAL_LOG_PATH,
    SIGNIFICANCE_TRIAL_PADDING,
    WALK_FORWARD_PERIOD_RETURNS_PATH,
)
from src.data.io import load_price_data
from src.data.required import require_file, run_cli
from src.portfolio.portfolio import month_end_holding_period_returns
from src.reporting.significance import (
    benchmark_relative_summary,
    deflated_sharpe_ratio,
    significance_summary,
    trial_sharpe_variance,
)
from src.reporting.trial_log import count_trials, family_sharpes, load_trial_log, pooled_sharpes


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
    frame = pd.read_csv(require_file(MULTI_FACTOR_RESULTS_PATH), parse_dates=["date"]).set_index("date")
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


def benchmark_daily_returns() -> dict[str, pd.Series]:
    """Daily benchmark returns from checked-in data only (reproducible offline).

    ``equal_weight_universe``: daily-rebalanced equal weight of every name in
    ``data/price_data.csv`` with a return that day (today's ticker list, so it
    shares the strategies' survivorship bias). ``SPY``: adjusted close from
    ``data/benchmark_prices.csv`` (see ``data/benchmark_source.json``).
    """
    prices = load_price_data()
    out = {"equal_weight_universe": prices.pct_change(fill_method=None).mean(axis=1, skipna=True).dropna()}
    path = Path(BENCHMARK_PRICES_PATH)
    if path.exists():
        bench = load_price_data(str(path))
        if BENCHMARK_SYMBOL in bench.columns:
            out[BENCHMARK_SYMBOL] = bench[BENCHMARK_SYMBOL].pct_change(fill_method=None).dropna()
    return out


def monthly_forward_returns(daily: pd.Series, price_index: pd.DatetimeIndex) -> pd.Series:
    """Month-end → next month-end benchmark return, indexed at the entry date
    (same convention as the research holding-period returns)."""
    wealth = (1.0 + daily.reindex(price_index).fillna(0.0)).cumprod().to_frame("w")
    return month_end_holding_period_returns(wealth)["w"].dropna()


def build_report(cached_walk_forward: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = load_trial_log(SIGNIFICANCE_TRIAL_LOG_PATH)
    n_primary = count_trials(log, SIGNIFICANCE_TRIAL_PADDING)
    print(f"Trial log: {int(log['trial_increment'].sum())} logged + "
          f"{SIGNIFICANCE_TRIAL_PADDING} padding = n_trials {n_primary}")
    series = {
        "walk_forward_ml": (load_walk_forward_returns(cached_walk_forward), 12,
                            SIGNIFICANCE_BLOCK_LENGTH_MONTHLY, SIGNIFICANCE_CAUSAL_BETA_WINDOW_MONTHLY),
        "multi_factor": (load_multi_factor_returns(), 12,
                         SIGNIFICANCE_BLOCK_LENGTH_MONTHLY, SIGNIFICANCE_CAUSAL_BETA_WINDOW_MONTHLY),
    }
    sim = load_multi_day_sim_returns()
    if sim is not None:
        series["multi_day_sim"] = (sim, 252, SIGNIFICANCE_BLOCK_LENGTH_DAILY,
                                   SIGNIFICANCE_CAUSAL_BETA_WINDOW_DAILY)

    trial_counts = sorted(set([n_primary, *SIGNIFICANCE_N_TRIALS_SENSITIVITY]))
    pooled = pooled_sharpes(log)
    common = dict(
        n_resamples=SIGNIFICANCE_BOOTSTRAP_N_RESAMPLES,
        confidence=SIGNIFICANCE_CONFIDENCE,
        method=SIGNIFICANCE_BOOTSTRAP_METHOD,
        seed=SIGNIFICANCE_SEED,
    )

    rows = []
    for name, (returns, ppy, block, _) in series.items():
        variance = trial_sharpe_variance(family_sharpes(log, name), ppy)
        pooled_variance = trial_sharpe_variance(pooled, ppy)
        for n_trials in trial_counts:
            summary = significance_summary(
                returns, periods_per_year=ppy, n_trials=n_trials,
                sharpe_variance=variance, block_length=block, **common,
            )
            dsr_pooled, sr0_pooled = deflated_sharpe_ratio(
                summary["sharpe_per_period"], summary["n_obs"], summary["skewness"],
                summary["kurtosis"], n_trials=n_trials, sharpe_variance=pooled_variance,
            )
            summary["sr0_annualized_pooled_var"] = float(sr0_pooled * ppy ** 0.5)
            summary["deflated_sharpe_ratio_pooled_var"] = dsr_pooled
            rows.append({
                "strategy": name,
                "primary": n_trials == n_primary,
                "start": str(returns.index.min().date()),
                "end": str(returns.index.max().date()),
                "periods_per_year": ppy,
                "bootstrap_method": SIGNIFICANCE_BOOTSTRAP_METHOD,
                "confidence": SIGNIFICANCE_CONFIDENCE,
                **summary,
            })

    daily_benchmarks = benchmark_daily_returns()
    if sim is not None:
        reference = daily_benchmarks["equal_weight_universe"]
        reference = reference.loc[(reference.index >= sim.index.min()) & (reference.index <= sim.index.max())]
        summary = significance_summary(
            reference, periods_per_year=252, n_trials=1, sharpe_variance=0.0,
            block_length=SIGNIFICANCE_BLOCK_LENGTH_DAILY, **common,
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

    # Benchmark-relative tests (IR DSR uses pooled V[SR]: there is no IR trial history).
    price_index = load_price_data().index
    rel_rows = []
    for name, (returns, ppy, block, beta_window) in series.items():
        ir_source = pooled if SIGNIFICANCE_IR_VARIANCE_SOURCE == "pooled" else family_sharpes(log, name)
        ir_variance = trial_sharpe_variance(ir_source, ppy)
        for bench_name, daily in daily_benchmarks.items():
            bench = daily if ppy == 252 else monthly_forward_returns(daily, price_index)
            rel = benchmark_relative_summary(
                returns, bench, periods_per_year=ppy, n_trials=n_primary,
                ir_sharpe_variance=ir_variance, hac_lags=SIGNIFICANCE_HAC_LAGS,
                causal_beta_window=beta_window, block_length=block, **common,
            )
            rel_rows.append({"strategy": name, "benchmark": bench_name,
                             "periods_per_year": ppy, **rel})
    relative = pd.DataFrame(rel_rows)
    relative.to_csv(BENCHMARK_RELATIVE_REPORT_PATH, index=False)
    return report, relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-walk-forward", action="store_true",
                        help="Use results/walk_forward_period_returns.csv instead of rerunning.")
    args = parser.parse_args()
    report, relative = build_report(cached_walk_forward=args.cached_walk_forward)
    cols = ["strategy", "n_trials", "n_obs", "sharpe_annualized", "ci_lower", "ci_upper",
            "prob_sharpe_le_zero", "psr_vs_zero", "sr0_annualized", "deflated_sharpe_ratio",
            "deflated_sharpe_ratio_pooled_var"]
    rel_cols = ["strategy", "benchmark", "n_obs", "alpha_annualized", "alpha_t_hac", "beta",
                "information_ratio", "ir_ci_lower", "ir_ci_upper", "ir_psr_vs_zero", "ir_deflated",
                "hedged_sharpe_in_sample", "hedged_sharpe_causal", "hedged_causal_deflated"]
    with pd.option_context("display.width", 250):
        print(report[cols].round(4).to_string(index=False))
        print()
        print(relative[rel_cols].round(4).to_string(index=False))
    print(f"\nSaved {SIGNIFICANCE_REPORT_PATH} and {BENCHMARK_RELATIVE_REPORT_PATH}")


if __name__ == "__main__":
    run_cli(main)
