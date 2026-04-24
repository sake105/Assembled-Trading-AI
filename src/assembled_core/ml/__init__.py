"""Machine Learning modules for the Assembled-Trading-AI system."""

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
    "FeatureSelectionResult",
    "check_factor_diversification",
    "collinearity_filter",
    "conditional_mutual_information",
    "ic_prescreen",
    "mutual_information_ranking",
    "run_feature_selection",
    "stability_filter",
    "MetaLabeler",
    "MetaLabelRecord",
    "AdversarialResult",
    "run_adversarial_validation",
    "sample_weight_from_adversarial",
    "FeatureClusterResult",
    "cluster_features_by_correlation",
    "clustered_mda",
    "select_features_by_cluster_ic",
    "ModelRecord",
    "ModelRegistry",
]
