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
    assert config.MAX_ADV_PARTICIPATION == risk["max_adv_participation"]

    assert config.TRANSACTION_COST == execution["transaction_cost"]
    assert config.SLIPPAGE_BPS == execution["slippage_bps"]
    assert config.MIN_REBALANCE_DOLLARS == execution["min_rebalance_dollars"]
    assert config.HALF_SPREAD_BPS == execution["half_spread_bps"]
    assert config.IMPACT_COEFFICIENT_BPS == execution["impact_coefficient_bps"]
    assert config.ANNUAL_BORROW_BPS == execution["annual_borrow_bps"]
    assert config.ANNUAL_SHORT_REBATE_BPS == execution["annual_short_rebate_bps"]
    assert config.ANNUAL_CASH_FINANCING_BPS == execution["annual_cash_financing_bps"]
    assert config.BORROW_TIERING_ENABLED == execution["borrow_tiering_enabled"]
    assert config.BORROW_TIER_LOOKBACK_DAYS == execution["borrow_tier_lookback_days"]
    assert config.BORROW_TIER_RATES_BPS["htb"] == execution["borrow_tier_rates_bps"]["htb"]

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
    assert config.NESTED_INNER_SCORE == research["nested_inner_score"]
    assert config.NESTED_INNER_VOL_TARGETING == research["nested_inner_vol_targeting"]
    assert config.NESTED_SELECTION_RULE == research["nested_selection_rule"]
    assert config.NESTED_DEFAULT_MODEL == research["nested_default_model"]
    assert config.NESTED_SELECTION_SE_THRESHOLD == research["nested_selection_se_threshold"]
    assert config.NESTED_INNER_NAME_LEVEL_BORROW == research["nested_inner_name_level_borrow"]
    significance = platform["significance"]
    assert config.SIGNIFICANCE_TRIAL_LOG_PATH == significance["trial_log_path"]
    assert config.SIGNIFICANCE_TRIAL_PADDING == significance["trial_padding"]
    assert config.SIGNIFICANCE_SEED == significance["bootstrap_seed"]
    assert config.SIGNIFICANCE_IR_VARIANCE_SOURCE == significance["ir_variance_source"]
    assert config.BETA_NEUTRAL == research["beta_neutral"]
    assert config.BETA_LOOKBACK_DAYS == research["beta_lookback_days"]
    assert config.BETA_SHRINKAGE == research["beta_shrinkage"]
    assert config.BETA_ESTIMATOR == research["beta_estimator"]
    assert config.BETA_EWMA_HALF_LIFE_DAYS == research["beta_ewma_half_life_days"]
    assert config.BETA_NEUTRAL_MAX_LEG_RATIO == research["beta_neutral_max_leg_ratio"]
    assert config.SECTOR_NEUTRAL == research["sector_neutral"]
    assert config.MAX_SECTOR_NET_EXPOSURE == research["max_sector_net_exposure"]
    assert config.SECTOR_MAP_PATH == research["sector_map_path"]
    assert config.VOL_TARGETING_ENABLED == research["vol_targeting_enabled"]
    assert config.TARGET_VOLATILITY == research["target_volatility"]
    assert config.VOLATILITY_LOOKBACK == research["vol_lookback_periods"]
    assert config.VOL_MIN_SCALE == research["vol_min_scale"]
    assert config.VOL_MAX_SCALE == research["vol_max_scale"]
    assert config.VOL_ESTIMATOR == research["vol_estimator"]
    assert config.VOL_EWMA_HALF_LIFE == research["vol_ewma_half_life"]
    assert config.PAPER_VOL_ESTIMATOR == platform["risk_limits"]["vol_estimator"]
    assert config.PAPER_VOL_EWMA_HALF_LIFE == platform["risk_limits"]["vol_ewma_half_life"]
    assert config.PAPER_VOL_TARGETING_ENABLED == platform["risk_limits"]["vol_targeting_enabled"]
    assert config.PAPER_VOL_LOOKBACK_DAYS == platform["risk_limits"]["vol_lookback_days"]
    assert config.PAPER_VOL_MAX_SCALE == platform["risk_limits"]["vol_max_scale"]


def test_research_snapshot_end_matches_config_end_date() -> None:
    """The data check's frozen-snapshot reference is the downloader's research END_DATE."""
    assert load_platform_config()["data_checks"]["research_snapshot_end"] == config.END_DATE
