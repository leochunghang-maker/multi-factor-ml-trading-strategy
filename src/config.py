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

RANDOM_SEED = 42
PLATFORM_CONFIG_PATH = "config/platform_config.json"
LOCAL_PLATFORM_CONFIG_PATH = "config/platform_config.local.json"
OPERATIONS_RESULTS_DIR = "results/operations"
STRUCTURED_LOG_PATH = "results/operations/platform_events.jsonl"
RUN_METADATA_DIR = "results/operations/run_metadata"
CONFIG_SNAPSHOT_DIR = "results/operations/config_snapshots"
SYSTEM_STATUS_REPORT_PATH = "reports/system_status.md"
SYSTEM_STATUS_JSON_PATH = "results/operations/latest_system_status.json"

START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

PRICE_DATA_PATH = "data/price_data.csv"
VOLUME_DATA_PATH = "data/volume_data.csv"
DAILY_RETURNS_PATH = "data/daily_returns.csv"
MOMENTUM_PATH = "data/momentum_12m.csv"
MEAN_REVERSION_PATH = "data/mean_reversion_1m.csv"
FORWARD_RETURNS_PATH = "data/forward_returns_1m.csv"
ML_DATASET_PATH = "data/ml_dataset.csv"

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

FEATURES = [
    "momentum_12m",
    "short_term_momentum",
    "volatility_1m",
]
TARGET = "target_return_1m"
PREDICTION_COLUMN = "predicted_return"
MODEL_SPLIT_DATE = "2024-01-01"
WALK_FORWARD_TEST_YEARS = [2022, 2023, 2024, 2025]

MOMENTUM_LONG_COUNT = 5
MOMENTUM_SHORT_COUNT = 5
MOMENTUM_MIN_ASSETS = 10
MULTI_FACTOR_LONG_COUNT = 10
MULTI_FACTOR_SHORT_COUNT = 10
MULTI_FACTOR_MIN_ASSETS = 20
ML_LONG_COUNT = 10
ML_SHORT_COUNT = 10
RESEARCH_MIN_ASSETS = 10
LIVE_LONG_COUNT = 10
LIVE_PRICE_LOOKBACK_PERIOD = "2y"
MAX_PAPER_POSITIONS = 10
MAX_PAPER_ALLOCATION = 0.10
PAPER_ALLOCATION_BUFFER = 0.001
MIN_REBALANCE_DOLLARS = 25.0
SIMULATED_INITIAL_CASH = 100000.0
MULTI_DAY_REBALANCE_FREQUENCY = "monthly"
MULTI_DAY_STARTING_CAPITAL = 100000.0
MULTI_DAY_ROLLING_WINDOW = 63

TRANSACTION_COST = 0.001
BENCHMARK_SYMBOL = "SPY"
TARGET_VOLATILITY = 0.15
VOLATILITY_LOOKBACK = 6
MAX_LEVERAGE = 2.0
RISK_ANALYTICS_WINDOW = 12
PERIODS_PER_YEAR = 12
MAX_POSITION_WEIGHT = 0.10
MAX_SECTOR_WEIGHT = 0.35
MAX_GROSS_EXPOSURE = 2.0
SIGNAL_STALENESS_DAYS = 5
SLIPPAGE_BPS = 5.0
MAX_TURNOVER = 1.0
MAX_PAPER_TURNOVER = 1.0
MIN_AVERAGE_DAILY_DOLLAR_VOLUME = 25_000_000.0
LIQUIDITY_LOOKBACK_DAYS = 63
MAX_ADV_PARTICIPATION = 0.05
PORTFOLIO_TRANSACTION_COST_BPS = 10.0
PORTFOLIO_SLIPPAGE_BPS = 5.0

MAX_SINGLE_NAME_WEIGHT = 0.10
MAX_SECTOR_GROSS_EXPOSURE = 0.35
MAX_POSITION_WEIGHT = MAX_SINGLE_NAME_WEIGHT
MAX_SECTOR_WEIGHT = MAX_SECTOR_GROSS_EXPOSURE
MAX_GROSS_EXPOSURE = 2.0
ML_EMBARGO_DAYS = 21

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
    "random_state": 42,
    "n_jobs": -1,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
}
