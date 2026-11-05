"""Ensure src.config mirrors config/platform_config.json risk/execution/sim knobs."""

from src import config
from src.operations.configuration import load_platform_config


def test_config_values_agree_with_platform_config() -> None:
    platform = load_platform_config()
    risk = platform["risk_limits"]
    execution = platform["execution"]
    simulation = platform["simulation"]
    strategy = platform["strategy"]

    assert config.MAX_GROSS_EXPOSURE == risk["max_gross_exposure"] == 1.0
    assert config.MAX_POSITION_WEIGHT == risk["max_position_weight"]
    assert config.MAX_SINGLE_NAME_WEIGHT == risk["max_position_weight"]
    assert config.MAX_PAPER_ALLOCATION == risk["max_position_weight"]
    assert config.MAX_PAPER_POSITIONS == risk["max_positions"]
    assert config.PAPER_ALLOCATION_BUFFER == risk["paper_allocation_buffer"]
    assert config.MAX_TURNOVER == risk["max_turnover"]
    assert config.MAX_PAPER_TURNOVER == risk["max_turnover"]
    assert config.SIGNAL_STALENESS_DAYS == risk["max_data_staleness_days"]
    assert config.MAX_DRAWDOWN_WARNING == risk["max_drawdown_warning"]

    assert config.TRANSACTION_COST == execution["transaction_cost"]
    assert config.SLIPPAGE_BPS == execution["slippage_bps"]
    assert config.MIN_REBALANCE_DOLLARS == execution["min_rebalance_dollars"]

    assert config.SIMULATED_INITIAL_CASH == simulation["starting_capital"]
    assert config.MULTI_DAY_STARTING_CAPITAL == simulation["starting_capital"]
    assert config.MULTI_DAY_REBALANCE_FREQUENCY == simulation["rebalance_frequency"]
    assert config.MULTI_DAY_ROLLING_WINDOW == simulation["rolling_window"]

    assert config.LIVE_LONG_COUNT == strategy["live_long_count"]
    assert list(config.FEATURES) == list(strategy["features"])
    assert config.RANDOM_SEED == strategy["model_seed"]
    assert config.RANDOM_FOREST_PARAMS["random_state"] == strategy["model_seed"]
    assert config.XGBOOST_PARAMS["random_state"] == strategy["model_seed"]
