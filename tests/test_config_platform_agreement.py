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
    assert config.HALF_SPREAD_BPS == execution["half_spread_bps"]
    assert config.IMPACT_COEFFICIENT_BPS == execution["impact_coefficient_bps"]
    assert config.ANNUAL_BORROW_BPS == execution["annual_borrow_bps"]
    assert config.ANNUAL_SHORT_REBATE_BPS == execution["annual_short_rebate_bps"]
    assert config.ANNUAL_CASH_FINANCING_BPS == execution["annual_cash_financing_bps"]

    assert config.SIMULATED_INITIAL_CASH == simulation["starting_capital"]
    assert config.MULTI_DAY_STARTING_CAPITAL == simulation["starting_capital"]
    assert config.MULTI_DAY_REBALANCE_FREQUENCY == simulation["rebalance_frequency"]
    assert config.MULTI_DAY_ROLLING_WINDOW == simulation["rolling_window"]

    assert config.LIVE_LONG_COUNT == strategy["live_long_count"]
    assert list(config.FEATURES) == list(strategy["features"])
    assert config.RANDOM_SEED == strategy["model_seed"]
    assert config.RANDOM_FOREST_PARAMS["random_state"] == strategy["model_seed"]
    assert config.XGBOOST_PARAMS["random_state"] == strategy["model_seed"]

    research = platform["research"]
    assert config.RESEARCH_LONG_SHORT_GROSS_EXPOSURE == research["long_short_gross_exposure"]
    assert config.REQUIRE_INDEX_MEMBERSHIP == research["require_index_membership"]
    assert config.MIN_UNIVERSE_HISTORY_DAYS == research["min_universe_history_days"]
    assert config.UNIVERSE_MEMBERSHIP_PATH == research["universe_membership_path"]
    assert config.NESTED_WALK_FORWARD == research["nested_walk_forward"]
    assert config.NESTED_MIN_TRAIN_YEARS == research["nested_min_train_years"]
    assert config.NESTED_USE_MONTH_END_SUBSET == research["nested_use_month_end_subset"]
