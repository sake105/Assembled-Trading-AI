"""Machine Learning modules for the Assembled-Trading-AI system.

This package exposes a public API aggregating factor models, explainability,
advanced learners (Bayesian NN, Gaussian Process, GNN, MAML, RL),
causal inference, conformal prediction, factor timing, symbolic regression,
topological data analysis (TDA) and temporal attention models.

The imports below wire every orphan module into the public ML surface so
they are discoverable via ``from src.assembled_core import ml``.
"""

from __future__ import annotations

from src.assembled_core.ml.factor_models import (
    MLExperimentConfig,
    MLModelConfig,
    evaluate_ml_predictions,
    prepare_ml_dataset,
    run_time_series_cv,
)
from src.assembled_core.ml.explainability import (
    compute_model_feature_importance,
    compute_permutation_importance,
    summarize_feature_importance_global,
)

from src.assembled_core.ml.automl import (
    AutoMLResult,
    ModelCandidate,
    run_automl,
    select_features_mi,
)
from src.assembled_core.ml.bayesian_nn import BNNPrediction, MCDropoutMLP
from src.assembled_core.ml.causal_inference import (
    CausalEffectResult,
    GrangerResult,
    difference_in_differences,
    estimate_propensity_score,
    granger_causality_test,
    iv_two_stage_least_squares,
    propensity_score_matching,
    screen_factors_causal,
)
from src.assembled_core.ml.conformal_prediction import (
    AdaptiveConformal,
    ConformalResult,
    SplitConformal,
    evaluate_coverage,
)
from src.assembled_core.ml.factor_timing import (
    FactorTimingConfig,
    FactorTimingResult,
    adjust_factor_weights,
    compute_factor_crowding,
    compute_factor_mean_reversion,
    compute_factor_momentum,
    compute_macro_conditional_timing,
    compute_value_spread,
)
from src.assembled_core.ml.gaussian_process import (
    FactorGPR,
    GPRResult,
    build_gpr_position_sizing_signal,
)
from src.assembled_core.ml.gnn_stocks import (
    GNNConfig,
    GNNEmbedder,
    StockGraph,
    build_stock_graph,
    compute_gnn_alpha_signals,
)
from src.assembled_core.ml.graph_models import (
    GraphEdge,
    GraphNode,
    GraphSignal,
    build_correlation_graph,
    compute_pagerank,
    detect_lead_lag,
    generate_graph_signals,
    propagate_signals,
)
from src.assembled_core.ml.maml import MAMLConfig, MAMLPredictor, MAMLResult
from src.assembled_core.ml.regime_weight_trainer import (
    compute_per_regime_ic,
    train_regime_weights,
    validate_regime_weights_wf,
)
from src.assembled_core.ml.rl_execution import (
    ExecutionAction,
    ExecutionState,
    QLearningExecutionAgent,
    compute_execution_reward,
    simulate_execution_episode,
    train_execution_agent,
)
from src.assembled_core.ml.rl_portfolio import (
    PortfolioEnv,
    RLPortfolioConfig,
    RLPortfolioOptimizer,
)
from src.assembled_core.ml.symbolic_regression import (
    DiscoveredFormula,
    SymbolicSearchResult,
    discover_formulas,
    discover_formulas_brute_force,
    discover_formulas_gplearn,
)
from src.assembled_core.ml.tda_regime import (
    TDAFeatures,
    compute_persistence_features,
    extract_tda_features,
    rolling_tda_features,
)
from src.assembled_core.ml.temporal_attention import (
    AttentionResult,
    TemporalAttentionConfig,
    TemporalAttentionModel,
)

from src.assembled_core.ml.calibration import (  # noqa: F401
    CalibrationResult,
    IsotonicCalibrator,
    TemperatureScaler,
    compute_calibration_error,
)
from src.assembled_core.ml.calibration_monitor import (  # noqa: F401
    CalibrationReport,
    PlattCalibrator,
    compute_calibration,
)
from src.assembled_core.ml.combined_regime import (  # noqa: F401
    CombinedRegimeClassifier,
    CombinedRegimeOutput,
)
from src.assembled_core.ml.experiment_tracking import (  # noqa: F401
    ExperimentRun,
    ExperimentTracker,
    ModelVersion,
)
from src.assembled_core.ml.feature_importance_tracker import (  # noqa: F401
    FeatureImportanceTracker,
    ImportanceSnapshot,
    PruningDecision,
)
from src.assembled_core.ml.feature_selection import (  # noqa: F401
    FeatureSelectionResult,
    check_factor_diversification,
    collinearity_filter,
    conditional_mutual_information,
    ic_prescreen,
    mutual_information_ranking,
    run_feature_selection,
    stability_filter,
)
from src.assembled_core.ml.meta_labeling import MetaLabeler, MetaLabelRecord  # noqa: F401
from src.assembled_core.ml.nested_meta_labeling import (  # noqa: F401
    NestedMetaLabeler,
    NestedPrediction,
)
from src.assembled_core.ml.online_gradient_boosting import OnlineAdaptiveLearner  # noqa: F401
from src.assembled_core.ml.online_hmm_regime import (  # noqa: F401
    OnlineHMMRegimeDetector,
    RegimeState,
)
from src.assembled_core.ml.regime_model_router import (  # noqa: F401
    RegimeModelRouter,
    RegimeRouterConfig,
    RegimeRouterResult,
)
from src.assembled_core.ml.triple_barrier import (  # noqa: F401
    apply_triple_barrier,
    build_triple_barrier_labels,
    compute_daily_volatility,
)
from src.assembled_core.ml.adversarial_validation import (  # noqa: F401
    AdversarialResult,
    run_adversarial_validation,
    sample_weight_from_adversarial,
)
from src.assembled_core.ml.bayesian_ensemble import (  # noqa: F401
    BMAResult,
    compute_bma_weights,
    run_bayesian_ensemble,
)
from src.assembled_core.ml.feature_clustering import (  # noqa: F401
    FeatureClusterResult,
    cluster_features_by_correlation,
    clustered_mda,
    select_features_by_cluster_ic,
)
from src.assembled_core.ml.model_registry import (  # noqa: F401
    ModelRecord,
    ModelRegistry,
)

__all__ = [
    "MLModelConfig",
    "MLExperimentConfig",
    "prepare_ml_dataset",
    "run_time_series_cv",
    "evaluate_ml_predictions",
    "compute_model_feature_importance",
    "compute_permutation_importance",
    "summarize_feature_importance_global",
    "AutoMLResult",
    "ModelCandidate",
    "run_automl",
    "select_features_mi",
    "BNNPrediction",
    "MCDropoutMLP",
    "CausalEffectResult",
    "GrangerResult",
    "difference_in_differences",
    "estimate_propensity_score",
    "granger_causality_test",
    "iv_two_stage_least_squares",
    "propensity_score_matching",
    "screen_factors_causal",
    "AdaptiveConformal",
    "ConformalResult",
    "SplitConformal",
    "evaluate_coverage",
    "FactorTimingConfig",
    "FactorTimingResult",
    "adjust_factor_weights",
    "compute_factor_crowding",
    "compute_factor_mean_reversion",
    "compute_factor_momentum",
    "compute_macro_conditional_timing",
    "compute_value_spread",
    "FactorGPR",
    "GPRResult",
    "build_gpr_position_sizing_signal",
    "GNNConfig",
    "GNNEmbedder",
    "StockGraph",
    "build_stock_graph",
    "compute_gnn_alpha_signals",
    "GraphEdge",
    "GraphNode",
    "GraphSignal",
    "build_correlation_graph",
    "compute_pagerank",
    "detect_lead_lag",
    "generate_graph_signals",
    "propagate_signals",
    "MAMLConfig",
    "MAMLPredictor",
    "MAMLResult",
    "compute_per_regime_ic",
    "train_regime_weights",
    "validate_regime_weights_wf",
    "ExecutionAction",
    "ExecutionState",
    "QLearningExecutionAgent",
    "compute_execution_reward",
    "simulate_execution_episode",
    "train_execution_agent",
    "PortfolioEnv",
    "RLPortfolioConfig",
    "RLPortfolioOptimizer",
    "DiscoveredFormula",
    "SymbolicSearchResult",
    "discover_formulas",
    "discover_formulas_brute_force",
    "discover_formulas_gplearn",
    "TDAFeatures",
    "compute_persistence_features",
    "extract_tda_features",
    "rolling_tda_features",
    "AttentionResult",
    "TemporalAttentionConfig",
    "TemporalAttentionModel",
    "AdversarialResult",
    "run_adversarial_validation",
    "sample_weight_from_adversarial",
    "BMAResult",
    "compute_bma_weights",
    "run_bayesian_ensemble",
    "FeatureClusterResult",
    "cluster_features_by_correlation",
    "clustered_mda",
    "select_features_by_cluster_ic",
    "ModelRecord",
    "ModelRegistry",
]
