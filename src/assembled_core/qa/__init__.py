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

# Wired 2026-04-22: previously orphan QA modules
from src.assembled_core.qa.ab_testing import (
    ABTestResult,
    MultipleTestResult,
    paired_ab_test,
    run_multiple_ab_tests,
)
from src.assembled_core.qa.e2e_integration import (
    E2ESuiteResult,
    E2ETestResult,
    test_data_to_features_pipeline,
    test_portfolio_construction,
    test_signal_generation,
)
from src.assembled_core.qa.regime_aware_wf import (
    RegimeWalkForwardResult,
    run_regime_aware_walk_forward,
    tag_regime_for_window,
)
from src.assembled_core.qa.reverse_stress import (
    ReverseStressResult,
    get_all_scenario_names,
    get_scenario,
    reverse_stress_test,
    run_multiple_reverse_stress,
    stress_test_portfolio_against_scenarios,
)
from src.assembled_core.qa.tca_arrival import (
    compute_implementation_shortfall,
    summarize_implementation_shortfall,
)
from src.assembled_core.qa.adversarial_testing import (  # noqa: F401
    AdversarialReport,
    PerturbationResult,
    detect_out_of_bounds,
    detect_stale_features,
    detect_sudden_jumps,
    fgsm_perturbation,
    run_adversarial_audit,
)
from src.assembled_core.qa.backtest_comparison import (  # noqa: F401
    BacktestComparisonReport,
    PairwiseComparison,
    StrategyMetrics,
    compare_backtests,
    rank_strategies,
)
from src.assembled_core.qa.backtest_overfit import (  # noqa: F401
    PBOResult,
    compute_pbo,
    performance_degradation,
)
from src.assembled_core.qa.benchmark_metrics import (  # noqa: F401
    BenchmarkMetrics,
    BrinsonAttribution,
    brinson_fachler_attribution,
    compute_benchmark_metrics,
)
from src.assembled_core.qa.candidate_gate import (  # noqa: F401
    check_candidate_allowed,
    read_reconciliation_ok_from_manifest,
    read_robustness_ok_from_manifest,
)
from src.assembled_core.qa.capacity import (  # noqa: F401
    CapacityEstimate,
    estimate_strategy_capacity,
)
from src.assembled_core.qa.deflated_sharpe import (  # noqa: F401
    DSRResult,
    deflated_sharpe,
    sharpe_std_error,
    sharpe_threshold,
)
from src.assembled_core.qa.ml_evaluation import (  # noqa: F401
    evaluate_meta_model,
    plot_calibration_curve,
)
from src.assembled_core.qa.multiple_testing import (  # noqa: F401
    MultipleTestingResult,
    benjamini_hochberg_fdr,
    holm_bonferroni_fwer,
    screen_factors_with_fdr,
)
from src.assembled_core.qa.parallel_grid import GridPoint, run_grid_parallel  # noqa: F401
from src.assembled_core.qa.scenario_simulator import (  # noqa: F401
    ScenarioResult,
    StressTestReport,
    run_stress_test,
    simulate_correlation_breakdown_scenario,
    simulate_crash_scenario,
    simulate_vol_spike_scenario,
)
from src.assembled_core.qa.signal_decay import (  # noqa: F401
    SignalDecayProfile,
    analyze_all_signals,
    analyze_signal_decay,
    compute_forward_return_half_life,
    compute_ic_half_life,
    compute_ic_series,
    compute_rank_stability,
    compute_signal_autocorrelation,
)
from src.assembled_core.qa.validation import (  # noqa: F401
    ModelValidationResult,
    compute_backtest_realism_score,
    run_full_model_validation,
    validate_data_quality,
    validate_overfitting,
    validate_performance,
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
    # A/B testing
    "ABTestResult",
    "MultipleTestResult",
    "paired_ab_test",
    "run_multiple_ab_tests",
    # E2E integration
    "E2ESuiteResult",
    "E2ETestResult",
    "test_data_to_features_pipeline",
    "test_portfolio_construction",
    "test_signal_generation",
    # Regime-aware walk-forward
    "RegimeWalkForwardResult",
    "run_regime_aware_walk_forward",
    "tag_regime_for_window",
    # Reverse stress
    "ReverseStressResult",
    "get_all_scenario_names",
    "get_scenario",
    "reverse_stress_test",
    "run_multiple_reverse_stress",
    "stress_test_portfolio_against_scenarios",
    # TCA
    "compute_implementation_shortfall",
    "summarize_implementation_shortfall",
]
