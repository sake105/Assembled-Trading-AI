"""QA and health check modules for the trading pipeline.

This package handles:
- Health checks for pipeline outputs
- Backtest validation
- Data quality assurance
- Performance metrics validation
- Factor analysis and IC computation (Phase C1)

Current modules:
- health: aggregate_qa_status, check_prices, check_orders, check_portfolio
- factor_analysis: compute_ic, compute_rank_ic, summarize_ic_series, compute_rolling_ic
"""

from src.assembled_core.qa.factor_analysis import (
    add_forward_returns,
    compute_factor_ic,
    compute_factor_rank_ic,
    summarize_factor_ic,
    run_factor_report,
    # New Phase C1 functions
    compute_ic,
    compute_rank_ic,
    summarize_ic_series,
    compute_rolling_ic,
    example_factor_analysis_workflow,
    # Phase C2 functions
    build_factor_portfolio_returns,
    build_long_short_portfolio_returns,
    summarize_factor_portfolios,
    compute_deflated_sharpe_ratio,
)

from src.assembled_core.qa.event_study import (
    build_event_window_prices,
    compute_event_returns,
    aggregate_event_study,
)

from src.assembled_core.qa.factor_ranking import build_factor_ranking

from src.assembled_core.qa.metrics import (
    deflated_sharpe_ratio,  # B4: New primary API in metrics.py
    deflated_sharpe_ratio_from_returns,  # B4: Convenience function
)

from src.assembled_core.qa.point_in_time_checks import (
    PointInTimeViolationError,
    check_altdata_events_pit_safe,
    check_features_pit_safe,
    validate_feature_builder_pit_safe,
)

from src.assembled_core.qa.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
    WalkForwardWindowResult,
    generate_walk_forward_splits,
    make_engine_backtest_fn,
    run_walk_forward_backtest,
)

# Legacy alias for backward compatibility
compute_rank_ic_legacy = compute_factor_rank_ic

__all__ = [
    # Legacy functions (for backward compatibility)
    "add_forward_returns",
    "compute_factor_ic",
    "compute_factor_rank_ic",
    "compute_rank_ic_legacy",
    "summarize_factor_ic",
    "run_factor_report",
    # Phase C1 functions
    "compute_ic",
    "compute_rank_ic",
    "summarize_ic_series",
    "compute_rolling_ic",
    "example_factor_analysis_workflow",
    # Phase C2 functions
    "build_factor_portfolio_returns",
    "build_long_short_portfolio_returns",
    "summarize_factor_portfolios",
    "compute_deflated_sharpe_ratio",  # Legacy (from factor_analysis.py)
    "deflated_sharpe_ratio",  # B4: New primary API (from metrics.py)
    "deflated_sharpe_ratio_from_returns",  # B4: Convenience function
    # Phase C3 functions (Event Study)
    "build_event_window_prices",
    "compute_event_returns",
    "aggregate_event_study",
    # Factor Ranking
    "build_factor_ranking",
    # B2 Point-in-Time Safety
    "PointInTimeViolationError",
    "check_features_pit_safe",
    "check_altdata_events_pit_safe",
    "validate_feature_builder_pit_safe",
    # B3 Walk-Forward Analysis
    "WalkForwardConfig",
    "WalkForwardWindow",
    "WalkForwardWindowResult",
    "WalkForwardResult",
    "generate_walk_forward_splits",
    "run_walk_forward_backtest",
    "make_engine_backtest_fn",
]
