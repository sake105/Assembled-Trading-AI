"""QA and health check modules for the trading pipeline."""

from src.assembled_core.qa.backtest_comparison import (  # noqa: F401
    BacktestComparisonReport,
    PairwiseComparison,
    StrategyMetrics,
    compare_backtests,
    rank_strategies,
)
from src.assembled_core.qa.benchmark_metrics import (  # noqa: F401
    BenchmarkMetrics,
    BrinsonAttribution,
    brinson_fachler_attribution,
    compute_benchmark_metrics,
)
from src.assembled_core.qa.deflated_sharpe import (  # noqa: F401
    DSRResult,
    deflated_sharpe,
    sharpe_std_error,
    sharpe_threshold,
)
from src.assembled_core.qa.event_study import (
    aggregate_event_study,
    build_event_window_prices,
    compute_event_returns,
)
from src.assembled_core.qa.factor_analysis import (
    add_forward_returns,
    build_factor_portfolio_returns,
    build_long_short_portfolio_returns,
    compute_deflated_sharpe_ratio,
    compute_factor_ic,
    compute_factor_rank_ic,
    compute_ic,
    compute_rank_ic,
    compute_rolling_ic,
    example_factor_analysis_workflow,
    run_factor_report,
    summarize_factor_ic,
    summarize_factor_portfolios,
    summarize_ic_series,
)
from src.assembled_core.qa.factor_ranking import build_factor_ranking
from src.assembled_core.qa.metrics import (
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
)
from src.assembled_core.qa.ml_evaluation import (  # noqa: F401
    evaluate_meta_model,
    plot_calibration_curve,
)
from src.assembled_core.qa.parallel_grid import (  # noqa: F401
    GridPoint,
    run_grid_parallel,
)
from src.assembled_core.qa.point_in_time_checks import (
    PointInTimeViolationError,
    check_altdata_events_pit_safe,
    check_features_pit_safe,
    validate_feature_builder_pit_safe,
)
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
from src.assembled_core.qa.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
    WalkForwardWindowResult,
    generate_walk_forward_splits,
    make_engine_backtest_fn,
    run_walk_forward_backtest,
)

compute_rank_ic_legacy = compute_factor_rank_ic

__all__ = [
    "add_forward_returns",
    "compute_factor_ic",
    "compute_factor_rank_ic",
    "compute_rank_ic_legacy",
    "summarize_factor_ic",
    "run_factor_report",
    "compute_ic",
    "compute_rank_ic",
    "summarize_ic_series",
    "compute_rolling_ic",
    "example_factor_analysis_workflow",
    "build_factor_portfolio_returns",
    "build_long_short_portfolio_returns",
    "summarize_factor_portfolios",
    "compute_deflated_sharpe_ratio",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_returns",
    "build_event_window_prices",
    "compute_event_returns",
    "aggregate_event_study",
    "build_factor_ranking",
    "PointInTimeViolationError",
    "check_features_pit_safe",
    "check_altdata_events_pit_safe",
    "validate_feature_builder_pit_safe",
    "WalkForwardConfig",
    "WalkForwardWindow",
    "WalkForwardWindowResult",
    "WalkForwardResult",
    "generate_walk_forward_splits",
    "run_walk_forward_backtest",
    "make_engine_backtest_fn",
]
