"""Machine Learning Models for Factor-Based Return Prediction.

This module provides ML models for predicting forward returns from factor panels.
It extends the factor analysis framework (Phase C) with machine learning capabilities.

Key features:
- Time-series cross-validation (expanding/rolling window)
- Multiple model types (Linear, Ridge, Lasso, Random Forest)
- Factor-specific metrics (IC, Rank-IC) in addition to classical ML metrics
- Portfolio performance evaluation from predictions

Usage:
    from src.assembled_core.ml import (
        MLModelConfig,
        MLExperimentConfig,
        prepare_ml_dataset,
        run_time_series_cv,
        evaluate_ml_predictions,
    )
    
    # Configure experiment
    experiment = MLExperimentConfig(
        label_col="fwd_return_20d",
        n_splits=5,
    )
    
    # Configure model
    model_cfg = MLModelConfig(
        name="ridge_20d",
        model_type="ridge",
        params={"alpha": 0.1},
    )
    
    # Run CV
    result = run_time_series_cv(factor_panel_df, experiment, model_cfg)
"""
from __future__ import annotations

from src.assembled_core.ml.factor_models import (
    MLExperimentConfig,
    MLModelConfig,
    evaluate_ml_predictions,
    prepare_ml_dataset,
    run_time_series_cv,
)

# Import explainability functions (E2 - may raise NotImplementedError if not yet implemented)
from src.assembled_core.ml.explainability import (
    compute_model_feature_importance,
    compute_permutation_importance,
    summarize_feature_importance_global,
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
]

