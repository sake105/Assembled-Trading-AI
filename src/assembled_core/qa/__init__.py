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
]
