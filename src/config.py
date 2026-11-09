import json
from copy import deepcopy
from pathlib import Path
from typing import Any

TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMAT", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DIS", "DUK", "EMR", "FDX", "GD", "GE", "GEV", "GILD",
    "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG",
    "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "LRCX", "MA", "MCD",
    "MDLZ", "MDT", "META", "MMM", "MO", "MRK", "MS", "MSFT", "MU", "NEE",
    "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG", "PLTR", "PM",
    "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TMO", "TMUS", "TSLA",
    "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLATFORM_CONFIG_PATH = "config/platform_config.json"
LOCAL_PLATFORM_CONFIG_PATH = "config/platform_config.local.json"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_platform_settings() -> dict[str, Any]:
    """Load risk/execution/sim knobs from platform_config.json (optional local overlay).

    `config/platform_config.json` is the single source of truth for deployment-facing
    risk, execution, and simulation settings. `src/config.py` mirrors those values so
    research and paper-trading code stay aligned without duplicating knobs.
    """
    default_path = _PROJECT_ROOT / PLATFORM_CONFIG_PATH
    settings = json.loads(default_path.read_text())
    local_path = _PROJECT_ROOT / LOCAL_PLATFORM_CONFIG_PATH
    if local_path.exists():
        settings = _deep_update(settings, json.loads(local_path.read_text()))
    return settings


_PLATFORM = _load_platform_settings()
_STRATEGY = _PLATFORM.get("strategy", {})
_EXECUTION = _PLATFORM.get("execution", {})
_RISK = _PLATFORM.get("risk_limits", {})
_SIMULATION = _PLATFORM.get("simulation", {})
_MONITORING = _PLATFORM.get("monitoring", {})
_RESEARCH = _PLATFORM.get("research", {})

RANDOM_SEED = int(_STRATEGY.get("model_seed", 42))
OPERATIONS_RESULTS_DIR = "results/operations"
STRUCTURED_LOG_PATH = "results/operations/platform_events.jsonl"
RUN_METADATA_DIR = "results/operations/run_metadata"
CONFIG_SNAPSHOT_DIR = "results/operations/config_snapshots"
SYSTEM_STATUS_REPORT_PATH = "reports/system_status.md"
SYSTEM_STATUS_JSON_PATH = "results/operations/latest_system_status.json"

START_DATE = "2018-01-01"
END_DATE = "2026-11-01"

PRICE_DATA_PATH = "data/price_data.csv"
VOLUME_DATA_PATH = "data/volume_data.csv"
DAILY_RETURNS_PATH = "data/daily_returns.csv"
MOMENTUM_PATH = "data/momentum_12m.csv"
MEAN_REVERSION_PATH = "data/mean_reversion_1m.csv"
FORWARD_RETURNS_PATH = "data/forward_returns_1m.csv"
ML_DATASET_PATH = "data/ml_dataset.csv"
UNIVERSE_MEMBERSHIP_PATH = "data/universe_sp500_membership.csv"
UNIVERSE_MEMBERSHIP_SOURCE_PATH = "data/universe_membership_source.json"

MOMENTUM_BACKTEST_PATH = "results/momentum_backtest.csv"
MOMENTUM_EQUITY_CURVE_PATH = "results/equity_curve.png"
MOMENTUM_IC_RESULTS_PATH = "results/momentum_ic_results.csv"
MULTI_FACTOR_RESULTS_PATH = "results/multi_factor_results.csv"
MULTI_FACTOR_EQUITY_CURVE_PATH = "results/multi_factor_equity_curve.csv"
STRATEGY_VS_BENCHMARK_PATH = "results/strategy_vs_benchmark.png"
ML_PREDICTIONS_PATH = "results/ml_predictions.csv"
XGBOOST_PREDICTIONS_PATH = "results/xgboost_predictions.csv"
ML_EQUITY_CURVE_PATH = "results/ml_strategy_equity_curve.png"
WALK_FORWARD_CHART_PATH = "results/walk_forward_ml_strategy.png"
WALK_FORWARD_DECILE_REPORT_PATH = "results/reports/walk_forward_decile_spread.csv"
FEATURE_STABILITY_REPORT_PATH = "results/reports/feature_stability.csv"
RISK_MANAGED_EQUITY_CURVE_PATH = "results/risk_managed_equity_curve.png"
REPORTS_DIR = "results/reports"
SUMMARY_REPORT_CSV_PATH = "results/reports/summary_report.csv"
SUMMARY_REPORT_MARKDOWN_PATH = "results/reports/summary_report.md"
REPORT_EQUITY_CURVE_PATH = "results/reports/equity_curve.png"
REPORT_DRAWDOWN_PATH = "results/reports/drawdown.png"
REPORT_ROLLING_SHARPE_PATH = "results/reports/rolling_sharpe.png"
REPORT_ROLLING_VOLATILITY_PATH = "results/reports/rolling_volatility.png"
REPORT_ROLLING_RISK_CSV_PATH = "results/reports/rolling_risk_report.csv"
REPORT_EXPOSURE_CSV_PATH = "results/reports/portfolio_exposures.csv"
REPORT_SECTOR_EXPOSURE_CSV_PATH = "results/reports/sector_exposures.csv"
REPORT_TURNOVER_CSV_PATH = "results/reports/turnover.csv"
PORTFOLIO_EXPOSURE_REPORT_PATH = "results/reports/portfolio_exposure_report.csv"
TURNOVER_REPORT_PATH = "results/reports/turnover_report.csv"
RISK_CONSTRAINT_SUMMARY_PATH = "results/reports/risk_constraint_summary.csv"
REPORT_MONTHLY_RETURN_HEATMAP_PATH = "results/reports/monthly_return_heatmap.png"
ML_VALIDATION_REPORT_PATH = "results/reports/ml_validation_report.csv"
ML_DECILE_SPREAD_PATH = "results/reports/ml_decile_spread.csv"
ML_FEATURE_STABILITY_PATH = "results/reports/ml_feature_stability.csv"
ML_BASELINE_REPORT_PATH = "results/reports/ml_baseline_comparison.csv"
LIVE_RESULTS_DIR = "results/live"
LIVE_SIGNALS_PATH = "results/live/latest_signals.csv"
LIVE_MODEL_PATH = "results/live/latest_model.pkl"
EXECUTION_RESULTS_DIR = "results/execution"
PAPER_ORDERS_PATH = "results/execution/latest_paper_orders.csv"
PAPER_EXECUTION_SUMMARY_PATH = "results/execution/latest_execution_summary.txt"
SIMULATION_RESULTS_DIR = "results/simulation"
SIMULATED_BROKER_STATE_PATH = "results/simulation/broker_state.json"
SIMULATED_TRADE_HISTORY_PATH = "results/simulation/trade_history.csv"
SIMULATED_REJECTED_ORDERS_PATH = "results/simulation/rejected_orders.csv"
SIMULATED_PORTFOLIO_HISTORY_PATH = "results/simulation/portfolio_history.csv"
SIMULATED_DAILY_PERFORMANCE_PATH = "results/simulation/daily_performance.csv"
SIMULATED_EXECUTION_LOG_PATH = "results/simulation/execution_log.csv"
SIMULATED_LATEST_SNAPSHOT_PATH = "results/simulation/latest_account_snapshot.csv"
SIMULATED_POSITION_SNAPSHOT_PATH = "results/simulation/latest_position_snapshot.csv"
ALLOCATION_SUMMARY_PATH = "results/simulation/allocation_summary.csv"
REALIZED_VS_TARGET_WEIGHTS_PATH = "results/simulation/realized_vs_target_weights.csv"
PAPER_TRADING_STATUS_REPORT_PATH = "reports/paper_trading_status.md"
MULTI_DAY_SIMULATION_DIR = "results/simulation/multi_day"
MULTI_DAY_NAV_PATH = "results/simulation/multi_day/portfolio_nav_history.csv"
MULTI_DAY_EXPOSURE_PATH = "results/simulation/multi_day/exposure_history.csv"
MULTI_DAY_TURNOVER_PATH = "results/simulation/multi_day/turnover_history.csv"
MULTI_DAY_TRADE_HISTORY_PATH = "results/simulation/multi_day/trade_history.csv"
MULTI_DAY_WARNING_LOG_PATH = "results/simulation/multi_day/warnings.csv"
MULTI_DAY_MONTHLY_RETURNS_PATH = "results/simulation/multi_day/monthly_returns.csv"
MULTI_DAY_STABILITY_METRICS_PATH = "results/simulation/multi_day/portfolio_stability_metrics.csv"
MULTI_DAY_DRAWDOWN_CHART_PATH = "results/simulation/multi_day/rolling_drawdown.png"
MULTI_DAY_TURNOVER_CHART_PATH = "results/simulation/multi_day/turnover_trends.png"
MULTI_DAY_EXPOSURE_CHART_PATH = "results/simulation/multi_day/exposure_trends.png"
MULTI_DAY_SIMULATION_REPORT_PATH = "reports/multi_day_simulation_report.md"


STRUCTURED_LOG_PATH = _MONITORING.get("structured_log_path", STRUCTURED_LOG_PATH)
RUN_METADATA_DIR = _MONITORING.get("metadata_dir", RUN_METADATA_DIR)
SYSTEM_STATUS_REPORT_PATH = _MONITORING.get("status_report_path", SYSTEM_STATUS_REPORT_PATH)

FEATURES = list(_STRATEGY.get("features", [
    "momentum_12m",
    "short_term_momentum",
    "volatility_1m",
]))
TARGET = "target_return_1m"
PREDICTION_COLUMN = "predicted_return"
MODEL_SPLIT_DATE = "2024-01-01"
WALK_FORWARD_TEST_YEARS = [2022, 2023, 2024, 2025, 2026]

MOMENTUM_LONG_COUNT = 5
MOMENTUM_SHORT_COUNT = 5
MOMENTUM_MIN_ASSETS = 10
MULTI_FACTOR_LONG_COUNT = 10
MULTI_FACTOR_SHORT_COUNT = 10
MULTI_FACTOR_MIN_ASSETS = 20
ML_LONG_COUNT = 10
ML_SHORT_COUNT = 10
RESEARCH_MIN_ASSETS = 10
LIVE_LONG_COUNT = int(_STRATEGY.get("live_long_count", 10))
LIVE_PRICE_LOOKBACK_PERIOD = "2y"
MAX_PAPER_POSITIONS = int(_RISK.get("max_positions", 10))
MAX_PAPER_ALLOCATION = float(_RISK.get("max_position_weight", 0.10))
PAPER_ALLOCATION_BUFFER = float(_RISK.get("paper_allocation_buffer", 0.001))
MIN_REBALANCE_DOLLARS = float(_EXECUTION.get("min_rebalance_dollars", 25.0))
SIMULATED_INITIAL_CASH = float(_SIMULATION.get("starting_capital", 100000.0))
MULTI_DAY_REBALANCE_FREQUENCY = str(_SIMULATION.get("rebalance_frequency", "monthly"))
MULTI_DAY_STARTING_CAPITAL = float(_SIMULATION.get("starting_capital", 100000.0))
MULTI_DAY_ROLLING_WINDOW = int(_SIMULATION.get("rolling_window", 63))

TRANSACTION_COST = float(_EXECUTION.get("transaction_cost", 0.001))
BENCHMARK_SYMBOL = "SPY"
TARGET_VOLATILITY = float(_RESEARCH.get("target_volatility", _RISK.get("target_volatility", 0.15)))
VOLATILITY_LOOKBACK = int(_RESEARCH.get("vol_lookback_periods", 6))
MAX_LEVERAGE = float(_RESEARCH.get("vol_max_scale", 1.5))
VOL_TARGETING_ENABLED = bool(_RESEARCH.get("vol_targeting_enabled", True))
VOL_MIN_SCALE = float(_RESEARCH.get("vol_min_scale", 0.5))
VOL_MAX_SCALE = float(_RESEARCH.get("vol_max_scale", 1.5))
VOL_PERIODS_PER_YEAR = float(_RESEARCH.get("vol_periods_per_year", 12))
VOL_ESTIMATOR = str(_RESEARCH.get("vol_estimator", "ewma")).strip().lower()
VOL_EWMA_HALF_LIFE = float(_RESEARCH.get("vol_ewma_half_life", 3.0))
PAPER_VOL_TARGETING_ENABLED = bool(_RISK.get("vol_targeting_enabled", True))
PAPER_VOL_LOOKBACK_DAYS = int(_RISK.get("vol_lookback_days", 63))
PAPER_VOL_MIN_SCALE = float(_RISK.get("vol_min_scale", 0.5))
PAPER_VOL_MAX_SCALE = float(_RISK.get("vol_max_scale", 1.0))
PAPER_VOL_ESTIMATOR = str(_RISK.get("vol_estimator", "ewma")).strip().lower()
PAPER_VOL_EWMA_HALF_LIFE = float(_RISK.get("vol_ewma_half_life", 21.0))
RISK_ANALYTICS_WINDOW = 12
PERIODS_PER_YEAR = 12
MAX_SINGLE_NAME_WEIGHT = float(_RISK.get("max_position_weight", 0.10))
MAX_SECTOR_GROSS_EXPOSURE = 0.35
MAX_POSITION_WEIGHT = MAX_SINGLE_NAME_WEIGHT
MAX_SECTOR_WEIGHT = MAX_SECTOR_GROSS_EXPOSURE
# platform_config.json is authoritative; keep research long/short scalable via this cap.
MAX_GROSS_EXPOSURE = float(_RISK.get("max_gross_exposure", 1.0))
SIGNAL_STALENESS_DAYS = int(_RISK.get("max_data_staleness_days", 5))
SLIPPAGE_BPS = float(_EXECUTION.get("slippage_bps", 5.0))
# Simulated execution realism (paper/sim only): half the quoted spread is paid on
# every fill, plus square-root market impact: impact_bps = coef * sqrt(notional / ADV$).
HALF_SPREAD_BPS = float(_EXECUTION.get("half_spread_bps", 2.0))
IMPACT_COEFFICIENT_BPS = float(_EXECUTION.get("impact_coefficient_bps", 100.0))
IMPACT_ADV_LOOKBACK_DAYS = int(_EXECUTION.get("impact_adv_lookback_days", 63))
# Stock-loan / financing (research L/S + paper/sim accrual). Borrow is charged on
# short gross; rebate defaults to 0; cash financing defaults to 0 (paper is cash
# long-only). Units are annualized basis points.
ANNUAL_BORROW_BPS = float(_EXECUTION.get("annual_borrow_bps", 100.0))
ANNUAL_SHORT_REBATE_BPS = float(_EXECUTION.get("annual_short_rebate_bps", 0.0))
ANNUAL_CASH_FINANCING_BPS = float(_EXECUTION.get("annual_cash_financing_bps", 0.0))
BORROW_TIERING_ENABLED = bool(_EXECUTION.get("borrow_tiering_enabled", True))
BORROW_TIER_LOOKBACK_DAYS = int(_EXECUTION.get("borrow_tier_lookback_days", 63))
BORROW_TIER_OVERRIDES_PATH = str(
    _EXECUTION.get("borrow_tier_overrides_path", "data/borrow_tier_overrides.csv")
)
BORROW_TIER_MAP_PATH = str(
    _EXECUTION.get("borrow_tier_map_path", "data/borrow_tier_map.csv")
)
BORROW_TIER_RATES_BPS = {
    str(k).lower(): float(v)
    for k, v in dict(_EXECUTION.get("borrow_tier_rates_bps", {
        "gc": ANNUAL_BORROW_BPS,
        "elevated": 300.0,
        "htb": 1000.0,
        "special": 3000.0,
    })).items()
}
MAX_TURNOVER = float(_RISK.get("max_turnover", 1.0))
MAX_PAPER_TURNOVER = float(_RISK.get("max_turnover", 1.0))
# Paper/sim circuit breaker: when portfolio drawdown falls to or below this
# (negative) fraction, new buys are halted; sells may still reduce risk.
MAX_DRAWDOWN_WARNING = float(_RISK.get("max_drawdown_warning", -0.1))
MIN_AVERAGE_DAILY_DOLLAR_VOLUME = 25_000_000.0
LIQUIDITY_LOOKBACK_DAYS = 63
# Cap |Δw| * NAV at this fraction of 63-day ADV$ per name (capacity / participation).
MAX_ADV_PARTICIPATION = float(_RISK.get("max_adv_participation", 0.05))
PORTFOLIO_TRANSACTION_COST_BPS = 10.0
PORTFOLIO_SLIPPAGE_BPS = 5.0
ML_EMBARGO_DAYS = 21
# Research long/short backtests run 100% long + 100% short (gross 2.0). This is a
# research setting only; paper trading is capped by MAX_GROSS_EXPOSURE (1.0).
RESEARCH_LONG_SHORT_GROSS_EXPOSURE = float(_RESEARCH.get("long_short_gross_exposure", 2.0))
# Survivorship mitigation: require S&P500 membership (S&P100 subset proxy) and
# a minimum non-NaN price history before a name can enter research portfolios.
REQUIRE_INDEX_MEMBERSHIP = bool(_RESEARCH.get("require_index_membership", True))
MIN_UNIVERSE_HISTORY_DAYS = int(_RESEARCH.get("min_universe_history_days", 63))
UNIVERSE_MEMBERSHIP_PATH = str(
    _RESEARCH.get("universe_membership_path", UNIVERSE_MEMBERSHIP_PATH)
)
NESTED_WALK_FORWARD = bool(_RESEARCH.get("nested_walk_forward", True))
NESTED_MIN_TRAIN_YEARS = int(_RESEARCH.get("nested_min_train_years", 2))
NESTED_USE_MONTH_END_SUBSET = bool(_RESEARCH.get("nested_use_month_end_subset", True))
NESTED_INNER_SCORE = str(_RESEARCH.get("nested_inner_score", "cost_aware_sharpe"))
NESTED_INNER_VOL_TARGETING = bool(_RESEARCH.get("nested_inner_vol_targeting", True))
NESTED_SELECTION_RULE = str(_RESEARCH.get("nested_selection_rule", "one_se_fallback")).strip().lower()
NESTED_DEFAULT_MODEL = str(_RESEARCH.get("nested_default_model", "rf_default"))
NESTED_SELECTION_SE_THRESHOLD = float(_RESEARCH.get("nested_selection_se_threshold", 1.0))
NESTED_INNER_NAME_LEVEL_BORROW = bool(_RESEARCH.get("nested_inner_name_level_borrow", True))
NESTED_SELECTION_REPORT_PATH = "results/reports/nested_model_selection.csv"
SECTOR_NEUTRAL = bool(_RESEARCH.get("sector_neutral", True))
MAX_SECTOR_NET_EXPOSURE = float(_RESEARCH.get("max_sector_net_exposure", 0.15))
SECTOR_MAP_PATH = str(_RESEARCH.get("sector_map_path", "data/sector_map.csv"))

SECTOR_MAP = {
    "AAPL": "Information Technology",
    "ABBV": "Health Care",
    "ABT": "Health Care",
    "ACN": "Information Technology",
    "ADBE": "Information Technology",
    "AMAT": "Information Technology",
    "AMD": "Information Technology",
    "AMGN": "Health Care",
    "AMT": "Real Estate",
    "AMZN": "Consumer Discretionary",
    "AVGO": "Information Technology",
    "AXP": "Financials",
    "BA": "Industrials",
    "BAC": "Financials",
    "BK": "Financials",
    "BKNG": "Consumer Discretionary",
    "BLK": "Financials",
    "BMY": "Health Care",
    "BRK-B": "Financials",
    "C": "Financials",
    "CAT": "Industrials",
    "CL": "Consumer Staples",
    "CMCSA": "Communication Services",
    "COF": "Financials",
    "COP": "Energy",
    "COST": "Consumer Staples",
    "CRM": "Information Technology",
    "CSCO": "Information Technology",
    "CVS": "Health Care",
    "CVX": "Energy",
    "DE": "Industrials",
    "DHR": "Health Care",
    "DIS": "Communication Services",
    "DUK": "Utilities",
    "EMR": "Industrials",
    "FDX": "Industrials",
    "GD": "Industrials",
    "GE": "Industrials",
    "GEV": "Industrials",
    "GILD": "Health Care",
    "GM": "Consumer Discretionary",
    "GOOG": "Communication Services",
    "GOOGL": "Communication Services",
    "GS": "Financials",
    "HD": "Consumer Discretionary",
    "HON": "Industrials",
    "IBM": "Information Technology",
    "INTC": "Information Technology",
    "INTU": "Information Technology",
    "ISRG": "Health Care",
    "JNJ": "Health Care",
    "JPM": "Financials",
    "KO": "Consumer Staples",
    "LIN": "Materials",
    "LLY": "Health Care",
    "LMT": "Industrials",
    "LOW": "Consumer Discretionary",
    "LRCX": "Information Technology",
    "MA": "Financials",
    "MCD": "Consumer Discretionary",
    "MDLZ": "Consumer Staples",
    "MDT": "Health Care",
    "META": "Communication Services",
    "MMM": "Industrials",
    "MO": "Consumer Staples",
    "MRK": "Health Care",
    "MS": "Financials",
    "MSFT": "Information Technology",
    "MU": "Information Technology",
    "NEE": "Utilities",
    "NFLX": "Communication Services",
    "NKE": "Consumer Discretionary",
    "NOW": "Information Technology",
    "NVDA": "Information Technology",
    "ORCL": "Information Technology",
    "PEP": "Consumer Staples",
    "PFE": "Health Care",
    "PG": "Consumer Staples",
    "PLTR": "Information Technology",
    "PM": "Consumer Staples",
    "QCOM": "Information Technology",
    "RTX": "Industrials",
    "SBUX": "Consumer Discretionary",
    "SCHW": "Financials",
    "SO": "Utilities",
    "SPG": "Real Estate",
    "T": "Communication Services",
    "TMO": "Health Care",
    "TMUS": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "TXN": "Information Technology",
    "UBER": "Industrials",
    "UNH": "Health Care",
    "UNP": "Industrials",
    "UPS": "Industrials",
    "USB": "Financials",
    "V": "Financials",
    "VZ": "Communication Services",
    "WFC": "Financials",
    "WMT": "Consumer Staples",
    "XOM": "Energy",
}


RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": RANDOM_SEED,
}

# Significance / "how much could be luck" settings (see README).
_SIGNIFICANCE = _PLATFORM.get("significance", {})
SIGNIFICANCE_N_TRIALS = int(_SIGNIFICANCE.get("n_trials", 50))
SIGNIFICANCE_N_TRIALS_SENSITIVITY = [
    int(x) for x in _SIGNIFICANCE.get("n_trials_sensitivity", [15, 50, 100, 200])
]
SIGNIFICANCE_BOOTSTRAP_METHOD = str(_SIGNIFICANCE.get("bootstrap_method", "stationary")).strip().lower()
SIGNIFICANCE_BOOTSTRAP_N_RESAMPLES = int(_SIGNIFICANCE.get("bootstrap_n_resamples", 5000))
SIGNIFICANCE_BLOCK_LENGTH_MONTHLY = float(_SIGNIFICANCE.get("bootstrap_block_length_monthly", 4))
SIGNIFICANCE_BLOCK_LENGTH_DAILY = float(_SIGNIFICANCE.get("bootstrap_block_length_daily", 13))
SIGNIFICANCE_CONFIDENCE = float(_SIGNIFICANCE.get("bootstrap_confidence", 0.95))
SIGNIFICANCE_SEED = int(_SIGNIFICANCE.get("bootstrap_seed", 42))
SIGNIFICANCE_TRIAL_SHARPES = {
    "walk_forward_ml": [float(x) for x in _SIGNIFICANCE.get("trial_sharpes_walk_forward", [])],
    "multi_factor": [float(x) for x in _SIGNIFICANCE.get("trial_sharpes_multi_factor", [])],
    "multi_day_sim": [float(x) for x in _SIGNIFICANCE.get("trial_sharpes_multi_day_sim", [])],
}
SIGNIFICANCE_REPORT_PATH = "results/significance_report.csv"
WALK_FORWARD_PERIOD_RETURNS_PATH = "results/walk_forward_period_returns.csv"
