from pathlib import Path

import numpy as np
import pandas as pd

from src.backtests.benchmark import load_benchmark_returns
from src.config import (
    BENCHMARK_SYMBOL,
    CONSTRAINT_BINDING_CSV_PATH,
    CONSTRAINT_BINDING_REPORT_PATH,
    DAILY_RETURNS_PATH,
    FEATURES,
    FEATURE_STABILITY_REPORT_PATH,
    MEAN_REVERSION_PATH,
    MAX_SECTOR_GROSS_EXPOSURE,
    MAX_SINGLE_NAME_WEIGHT,
    ML_PREDICTIONS_PATH,
    MOMENTUM_PATH,
    MULTI_FACTOR_RESULTS_PATH,
    PORTFOLIO_EXPOSURE_REPORT_PATH,
    PREDICTION_COLUMN,
    PRICE_DATA_PATH,
    REPORT_EXPOSURE_CSV_PATH,
    REPORT_ROLLING_RISK_CSV_PATH,
    REPORT_SECTOR_EXPOSURE_CSV_PATH,
    REPORT_TURNOVER_CSV_PATH,
    REPORTS_DIR,
    RISK_CONSTRAINT_SUMMARY_PATH,
    SECTOR_MAP,
    SUMMARY_REPORT_CSV_PATH,
    SUMMARY_REPORT_MARKDOWN_PATH,
    TARGET,
    TURNOVER_REPORT_PATH,
    VOLUME_DATA_PATH,
    WALK_FORWARD_DECILE_REPORT_PATH,
)
from src.data.io import (
    load_factor_frame,
    load_ml_dataset,
    load_prediction_dataset,
    load_price_data,
    load_return_series,
)
from src.models.validation import (
    decile_spread_analysis,
    feature_stability_report,
    prediction_stability_report,
)
from src.backtests.factor_backtests import build_reported_multi_factor_book
from src.portfolio import constrain_weight_history
from src.reporting.constraint_binding import save_binding_report
from src.portfolio.constraints import load_optional_volume_data
from src.reporting.analytics import (
    concentration_limit_report,
    concentration_summary,
    cumulative_return,
    exposure_summary,
    rolling_beta,
    rolling_risk_report,
    sector_exposure,
    turnover_series,
    turnover_statistics,
)
from src.reporting.charts import save_risk_report_charts
from src.reporting.metrics import calculate_performance_metrics
from src.risk.exposure import (
    exposure_report,
    factor_exposure_summary,
    portfolio_beta_exposure,
    sector_exposures,
)


def build_multi_factor_turnover() -> pd.Series:
    weights = build_multi_factor_weights()
    return turnover_series(weights)


def build_multi_factor_weights() -> pd.DataFrame:
    """Held weights of the reported multi-factor book (the one in multi_factor_results.csv).

    Rebuilt with the same construction as ``src/multi_factor_backtest.py``
    (PIT eligibility, beta-neutral legs, vol scaling). Warns if the rebuilt
    returns differ from the committed result file (stale artifact).
    """
    results, weights = build_reported_multi_factor_book()
    committed_path = Path(MULTI_FACTOR_RESULTS_PATH)
    if committed_path.exists():
        committed = load_return_series(MULTI_FACTOR_RESULTS_PATH, "portfolio_return")
        rebuilt = results["portfolio_return"].reindex(committed.index)
        if not np.allclose(rebuilt.to_numpy(), committed.to_numpy(), atol=1e-10, equal_nan=False):
            print(
                f"WARNING: rebuilt multi-factor book differs from {MULTI_FACTOR_RESULTS_PATH}; "
                "regenerate it with python src/multi_factor_backtest.py (or make research)."
            )
    return weights


def save_portfolio_constraint_reports(weights: pd.DataFrame) -> pd.DataFrame:
    prices = load_price_data(PRICE_DATA_PATH)
    daily_returns = load_factor_frame(DAILY_RETURNS_PATH)
    volumes = load_optional_volume_data(VOLUME_DATA_PATH)

    constrained_weights, diagnostics = constrain_weight_history(
        weights,
        daily_returns=daily_returns,
        prices=prices,
        volumes=volumes,
    )

    raw_exposure = exposure_report(weights).add_prefix("proposed_")
    constrained_exposure = exposure_report(constrained_weights).add_prefix("constrained_")
    factor_summary = factor_exposure_summary(
        constrained_weights,
        {
            "momentum_12m": load_factor_frame(MOMENTUM_PATH),
            "mean_reversion_1m": load_factor_frame(MEAN_REVERSION_PATH),
        },
    )

    portfolio_report = pd.concat(
        [raw_exposure, constrained_exposure, factor_summary],
        axis=1,
    )

    try:
        benchmark_returns = load_benchmark_returns(
            BENCHMARK_SYMBOL,
            start=weights.index.min(),
            end=weights.index.max(),
        )
        monthly_asset_returns = prices.resample("ME").last().pct_change().dropna(how="all")
        portfolio_report["constrained_beta_exposure"] = portfolio_beta_exposure(
            constrained_weights,
            monthly_asset_returns,
            benchmark_returns,
            window=36,
        )
    except Exception as exc:
        print(f"Portfolio beta exposure unavailable: {exc}")
        portfolio_report["constrained_beta_exposure"] = np.nan

    portfolio_report.to_csv(PORTFOLIO_EXPOSURE_REPORT_PATH)

    turnover_report = pd.DataFrame(index=weights.index)
    turnover_report["proposed_one_way_turnover"] = turnover_series(weights)
    turnover_report["constrained_one_way_turnover"] = turnover_series(constrained_weights)
    turnover_report = turnover_report.join(
        diagnostics[
            [
                "traded_gross_exposure",
                "estimated_transaction_cost",
                "transaction_cost_bps",
                "slippage_bps",
                "turnover_limited",
            ]
        ],
        how="left",
    )
    turnover_report.to_csv(TURNOVER_REPORT_PATH)

    sector_history = sector_exposures(constrained_weights)
    max_sector = sector_history.abs().max(axis=1).rename("constrained_max_abs_sector_exposure")
    risk_summary = diagnostics.join(max_sector, how="left")
    risk_summary["proposed_position_limit_breach"] = raw_exposure[
        "proposed_position_limit_breach"
    ]
    risk_summary["proposed_sector_limit_breach"] = raw_exposure[
        "proposed_sector_limit_breach"
    ]
    risk_summary["proposed_gross_limit_breach"] = raw_exposure[
        "proposed_gross_limit_breach"
    ]
    risk_summary["constrained_position_limit_breach"] = constrained_exposure[
        "constrained_position_limit_breach"
    ]
    risk_summary["constrained_sector_limit_breach"] = constrained_exposure[
        "constrained_sector_limit_breach"
    ]
    risk_summary["constrained_gross_limit_breach"] = constrained_exposure[
        "constrained_gross_limit_breach"
    ]
    risk_summary.to_csv(RISK_CONSTRAINT_SUMMARY_PATH)
    # Descriptive counterfactual: the same guardrails without per-name vol sizing.
    without_vol_sizing = constrain_weight_history(
        weights,
        daily_returns=None,
        prices=prices,
        volumes=volumes,
    )
    save_binding_report(
        diagnostics,
        weights,
        constrained_weights,
        CONSTRAINT_BINDING_CSV_PATH,
        CONSTRAINT_BINDING_REPORT_PATH,
        without_vol_sizing=without_vol_sizing,
    )

    # Preserve earlier report filenames for compatibility with the existing
    # project while adding the explicit institutional report names requested.
    portfolio_report.to_csv(REPORT_EXPOSURE_CSV_PATH)
    turnover_report.to_csv(REPORT_TURNOVER_CSV_PATH)
    sector_history.to_csv(REPORT_SECTOR_EXPOSURE_CSV_PATH)
    return risk_summary


def build_summary_metrics(returns: pd.Series) -> dict[str, float]:
    performance = calculate_performance_metrics(returns)
    weights = build_multi_factor_weights()
    latest_weights = weights.iloc[-1] if not weights.empty else pd.Series(dtype=float)
    turnover = turnover_series(weights)
    turnover_stats = turnover_statistics(turnover)
    exposures = exposure_summary(latest_weights)
    concentration = concentration_summary(latest_weights)
    limits = concentration_limit_report(
        latest_weights,
        SECTOR_MAP,
        max_single_name_weight=MAX_SINGLE_NAME_WEIGHT,
        max_sector_gross_exposure=MAX_SECTOR_GROSS_EXPOSURE,
    )

    try:
        benchmark_returns = load_benchmark_returns(
            BENCHMARK_SYMBOL,
            start=returns.index.min(),
            end=returns.index.max(),
        )
        beta_series = rolling_beta(returns, benchmark_returns)
        beta = beta_series.dropna().iloc[-1] if not beta_series.dropna().empty else np.nan
        rolling_risk_report(returns, benchmark_returns).to_csv(REPORT_ROLLING_RISK_CSV_PATH)
    except Exception as exc:
        print(f"Benchmark beta unavailable: {exc}")
        beta = np.nan
        rolling_risk_report(returns).to_csv(REPORT_ROLLING_RISK_CSV_PATH)

    cumulative = cumulative_return(returns)
    sector_exposure(latest_weights, SECTOR_MAP).to_csv(REPORT_SECTOR_EXPOSURE_CSV_PATH)
    risk_summary = save_portfolio_constraint_reports(weights)
    latest_constraint = risk_summary.iloc[-1] if not risk_summary.empty else pd.Series(dtype=float)

    return {
        "annual_return": performance["annual_return"],
        "annual_volatility": performance["annual_volatility"],
        "sharpe_ratio": performance["sharpe_ratio"],
        "max_drawdown": performance["max_drawdown"],
        "average_one_way_turnover": turnover_stats["average_turnover"],
        "latest_turnover": turnover_stats["latest_turnover"],
        "beta": beta,
        "cumulative_return": cumulative.iloc[-1],
        "long_exposure": exposures["long_exposure"],
        "short_exposure": exposures["short_exposure"],
        "gross_exposure": exposures["gross_exposure"],
        "net_exposure": exposures["net_exposure"],
        "largest_name_weight": concentration["largest_name_weight"],
        "top_5_gross_weight": concentration["top_5_gross_weight"],
        "herfindahl_index": concentration["herfindahl_index"],
        "max_sector_gross_exposure": limits["max_sector_gross_exposure"],
        "single_name_limit_breached": float(limits["single_name_limit_breached"]),
        "sector_limit_breached": float(limits["sector_limit_breached"]),
        "constrained_gross_exposure": latest_constraint.get("final_gross_exposure", np.nan),
        "constrained_net_exposure": latest_constraint.get("final_net_exposure", np.nan),
        "constraint_estimated_transaction_cost": latest_constraint.get(
            "estimated_transaction_cost",
            np.nan,
        ),
        "liquidity_filter_available": float(
            latest_constraint.get("liquidity_filter_available", False)
        ),
        "capacity_filter_available": float(
            latest_constraint.get("capacity_filter_available", False)
        ),
        "capacity_clipped_positions": float(
            latest_constraint.get("capacity_clipped_positions", 0)
        ),
    }


def save_ml_validation_reports() -> None:
    dataset = load_ml_dataset()
    feature_stability_report(dataset, "date", FEATURES).to_csv(
        FEATURE_STABILITY_REPORT_PATH,
        index=False,
    )

    predictions_path = Path(ML_PREDICTIONS_PATH)
    if not predictions_path.exists():
        return

    predictions = load_prediction_dataset(ML_PREDICTIONS_PATH)
    deciles = decile_spread_analysis(
        predictions,
        date_column="date",
        score_column=PREDICTION_COLUMN,
        target_column=TARGET,
    )
    deciles.to_csv(WALK_FORWARD_DECILE_REPORT_PATH, index=False)
    prediction_stability_report(
        predictions,
        date_column="date",
        symbol_column="ticker",
        score_column=PREDICTION_COLUMN,
    ).to_csv(Path(REPORTS_DIR) / "prediction_stability.csv", index=False)


def save_summary_report(metrics: dict[str, float]) -> pd.DataFrame:
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    )
    report.to_csv(SUMMARY_REPORT_CSV_PATH, index=False)

    lines = ["# Strategy Risk Summary", ""]
    for metric, value in metrics.items():
        lines.append(f"- **{metric.replace('_', ' ').title()}**: {value:.6f}")
    Path(SUMMARY_REPORT_MARKDOWN_PATH).write_text("\n".join(lines) + "\n")
    return report


def generate_summary_report() -> pd.DataFrame:
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    returns = load_return_series(MULTI_FACTOR_RESULTS_PATH, "portfolio_return")
    metrics = build_summary_metrics(returns)
    save_risk_report_charts(returns)
    save_ml_validation_reports()
    return save_summary_report(metrics)
