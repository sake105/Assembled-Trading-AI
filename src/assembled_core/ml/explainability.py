"""Model Explainability & Feature Importance for ML Factor Models.

This module provides explainability tools for ML models trained on factor panels.
It implements Phase E2 (Model Explainability & Feature Importance) from the Advanced Analytics roadmap.

DESIGN OVERVIEW:
===============

Goal:
-----
Provide tools to understand which factors (features) are most important for ML model
predictions. This enables:
- Feature importance ranking across models
- Identification of redundant or irrelevant factors
- Comparison with classical factor rankings (IC/IR)
- Model interpretability for regulatory compliance

Data Contracts:
---------------

Input:
- X (pd.DataFrame): Feature DataFrame with columns = feature names
- y (pd.Series): Target values (forward returns)
- Trained sklearn models (LinearRegression, Ridge, Lasso, RandomForestRegressor, etc.)
- OR: Predictions DataFrame from run_time_series_cv() containing y_pred and model context

Output:
- feature_importance_df (pd.DataFrame): Feature importance scores
  - Columns: feature_name, importance_score, importance_type (e.g., "coefficient", "feature_importance", "permutation")
  - May include: mean, std (for permutation importance with multiple repeats)

- Optional: local_explanations_df (pd.DataFrame, future E2.2)
  - Per-sample explanations (e.g., SHAP values)
  - Columns: sample_index, feature_name, contribution_to_prediction

Limitations:
-----------
- Only local data: No API calls, works only with pre-computed factor panels and trained models
- Primarily sklearn-based: Uses sklearn.inspection.permutation_importance and model-specific
  feature importance methods (coef_, feature_importances_)
- No SHAP yet: SHAP integration planned for E2.2/E2.3 (requires shap library)

Integration:
------------
- Builds on E1 (ML Validation): Uses trained models from run_time_series_cv()
- Can be integrated with Model Zoo: Compare feature importance across multiple models
- Future: Integration with Factor Ranking (C1/C2) to compare ML feature importance
  with classical IC/IR rankings

Future Enhancements (E2.2/E2.3):
--------------------------------
- SHAP values for model-agnostic explainability
- Partial dependence plots (PDP) and ICE plots
- Feature interaction analysis
- Local vs. global explanations
- Time-series specific explainability (temporal feature importance)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try importing sklearn inspection tools
try:
    from sklearn.inspection import permutation_importance

    SKLEARN_INSPECTION_AVAILABLE = True
except ImportError:
    SKLEARN_INSPECTION_AVAILABLE = False
    permutation_importance = None  # type: ignore

# Optional SHAP import
try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None  # type: ignore


def compute_model_feature_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute feature importance from a trained sklearn model.

    Supports:
    - Linear models (LinearRegression, Ridge, Lasso): Uses coefficients (abs values)
    - Tree models (RandomForestRegressor): Uses feature_importances_

    Args:
        model: Trained sklearn model (must have coef_ or feature_importances_ attribute)
        feature_names: List of feature names (must match order of features in model)

    Returns:
        DataFrame with columns:
        - feature: Name of the feature
        - importance: Feature importance (absolute coefficient or feature_importances_)
        - raw_value: Original coefficient/feature_importance value
        - direction: Direction of effect (+1 for positive, -1 for negative, None for tree models)

    Raises:
        ValueError: If model does not have coef_ or feature_importances_ attribute
        ValueError: If len(feature_names) does not match model dimensions

    Example:
        >>> from sklearn.linear_model import Ridge
        >>> model = Ridge().fit(X_train, y_train)
        >>> importance_df = compute_model_feature_importance(model, feature_names=["factor_mom", "factor_value"])
        >>> print(importance_df)
           feature  importance  raw_value  direction
        0     factor_mom        0.65       0.65          1
        1   factor_value        0.35       0.35          1
    """
    # Check if model has coef_ attribute (linear models)
    if hasattr(model, "coef_"):
        coef = model.coef_
        # Handle multi-output models (e.g., MultiOutputRegressor)
        if coef.ndim > 1:
            if coef.shape[0] == 1:
                coef = coef[0]
            else:
                # For multi-output, use mean absolute coefficient across outputs
                coef = np.mean(np.abs(coef), axis=0)
                logger.warning(
                    "Multi-output model detected, using mean absolute coefficient across outputs"
                )

        raw_values = np.array(coef).flatten()
        importance_scores = np.abs(raw_values)
        directions = np.sign(raw_values).astype(int)

    # Check if model has feature_importances_ attribute (tree models)
    elif hasattr(model, "feature_importances_"):
        raw_values = model.feature_importances_
        importance_scores = raw_values.copy()
        directions = np.full(
            len(raw_values), None, dtype=object
        )  # Tree models have no direction

    else:
        raise ValueError(
            f"Model {type(model).__name__} does not have coef_ or feature_importances_ attribute. "
            f"Supported models: LinearRegression, Ridge, Lasso, RandomForestRegressor, and similar sklearn models."
        )

    # Validate dimensions
    if len(feature_names) != len(importance_scores):
        raise ValueError(
            f"len(feature_names)={len(feature_names)} does not match model dimensions={len(importance_scores)}. "
            f"Ensure feature_names matches the order of features used for training."
        )

    # Create DataFrame
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_scores,
            "raw_value": raw_values,
            "direction": directions,
        }
    )

    # Sort by importance descending
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    return df


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "r2",
    n_repeats: int = 10,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Compute permutation importance for a trained model.

    Permutation importance measures how much model performance decreases when
    a feature is randomly shuffled. This provides a model-agnostic measure of
    feature importance that works for any sklearn-compatible model.

    Args:
        model: Trained sklearn model (must implement fit() and predict() or score())
        X: Feature DataFrame (columns must match feature order used for training)
        y: Target Series
        scoring: Scoring metric (default: "r2"). Can be "r2", "neg_mean_squared_error", etc.
        n_repeats: Number of times to permute each feature (default: 10)
        random_state: Random seed for reproducibility (default: None)

    Returns:
        DataFrame with columns:
        - feature: Name of the feature
        - importance_mean: Mean importance score across n_repeats
        - importance_std: Standard deviation of importance scores
        - importance_median: Median importance score across n_repeats
        - importance_type: Always "permutation"

    Raises:
        ImportError: If sklearn.inspection.permutation_importance is not available
        ValueError: If X is empty or has no columns

    Note:
        Permutation importance can be computationally expensive for large datasets
        or many features. Consider subsampling X if performance is an issue.

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor().fit(X_train, y_train)
        >>> perm_importance = compute_permutation_importance(model, X_test, y_test, n_repeats=10)
        >>> print(perm_importance)
           feature  importance_mean  importance_std  importance_median importance_type
        0     factor_mom            0.15            0.02              0.14      permutation
        1   factor_value            0.08            0.01              0.08      permutation
    """
    if not SKLEARN_INSPECTION_AVAILABLE:
        raise ImportError(
            "sklearn.inspection.permutation_importance is not available. "
            "Please install scikit-learn >= 0.22: pip install scikit-learn"
        )

    if X.empty or len(X.columns) == 0:
        raise ValueError("X DataFrame is empty or has no columns")

    if len(y) != len(X):
        raise ValueError(f"Length mismatch: X has {len(X)} rows, y has {len(y)} rows")

    # Compute permutation importance
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,  # Use all CPUs
    )

    # Extract feature names and importance statistics
    feature_names = X.columns.tolist()
    importance_means = result.importances_mean
    importance_stds = result.importances_std
    importance_medians = np.median(
        result.importances, axis=1
    )  # Compute median from raw importances

    # Create DataFrame
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importance_means,
            "importance_std": importance_stds,
            "importance_median": importance_medians,
        }
    )

    # Sort by importance_mean descending
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return df


def summarize_feature_importance_global(
    feature_importance_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Aggregate feature importance across multiple models.

    Combines feature importance from different models (e.g., from Model Zoo)
    into a single aggregated summary DataFrame. Useful for identifying factors that are
    consistently important across different model types.

    Args:
        feature_importance_dfs: Dictionary mapping model_name -> feature_importance DataFrame
            Each DataFrame should have a column named "feature" and a column named "importance"
            (or "importance_mean" for permutation importance results)

    Returns:
        Aggregated DataFrame with columns:
        - feature: Name of the feature
        - importance_mean: Mean importance across all models
        - importance_median: Median importance across all models
        - importance_max: Maximum importance across all models
        - n_models: Number of models that included this feature

    Raises:
        ValueError: If feature_importance_dfs is empty
        ValueError: If a DataFrame is missing required columns

    Example:
        >>> ridge_importance = compute_model_feature_importance(ridge_model, feature_names)
        >>> rf_importance = compute_model_feature_importance(rf_model, feature_names)
        >>> summary = summarize_feature_importance_global({
        ...     "ridge": ridge_importance,
        ...     "random_forest": rf_importance,
        ... })
        >>> print(summary)
           feature  importance_mean  importance_median  importance_max  n_models
        0     factor_mom            0.58              0.58           0.65         2
        1   factor_value            0.42              0.42           0.49         2
    """
    if not feature_importance_dfs:
        return pd.DataFrame(
            columns=[
                "feature",
                "importance_mean",
                "importance_median",
                "importance_max",
                "n_models",
            ]
        )

    # Validate and prepare DataFrames
    prepared_dfs = []
    for model_name, df in feature_importance_dfs.items():
        if df.empty:
            logger.warning(f"DataFrame for model '{model_name}' is empty, skipping...")
            continue

        # Check required columns
        if "feature" not in df.columns:
            raise ValueError(
                f"DataFrame for model '{model_name}' missing 'feature' column. "
                f"Available columns: {list(df.columns)}"
            )

        # Use importance_mean if available (from permutation importance), otherwise importance
        if "importance_mean" in df.columns:
            importance_col = "importance_mean"
        elif "importance" in df.columns:
            importance_col = "importance"
        else:
            raise ValueError(
                f"DataFrame for model '{model_name}' missing 'importance' or 'importance_mean' column. "
                f"Available columns: {list(df.columns)}"
            )

        # Extract feature and importance columns
        df_prepared = df[["feature", importance_col]].copy()
        df_prepared["model_name"] = model_name
        df_prepared = df_prepared.rename(columns={importance_col: "importance"})

        prepared_dfs.append(df_prepared)

    if not prepared_dfs:
        logger.warning("No valid DataFrames to aggregate")
        return pd.DataFrame(
            columns=[
                "feature",
                "importance_mean",
                "importance_median",
                "importance_max",
                "n_models",
            ]
        )

    # Concatenate all DataFrames
    combined_df = pd.concat(prepared_dfs, ignore_index=True)

    # Aggregate across models
    aggregation = (
        combined_df.groupby("feature")
        .agg(
            {
                "importance": ["mean", "median", "max", "count"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    aggregation.columns = [
        "feature",
        "importance_mean",
        "importance_median",
        "importance_max",
        "n_models",
    ]

    # Sort by importance_mean descending
    aggregation = aggregation.sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)

    return aggregation


# ---------------------------------------------------------------------------
# SHAP Integration (E2.3)
# ---------------------------------------------------------------------------


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    model_type: str = "auto",
    max_samples: int = 500,
) -> pd.DataFrame:
    """Compute SHAP feature attributions for a trained model.

    Args:
        model: Trained model (sklearn, xgboost, lightgbm, catboost compatible)
        X: Feature DataFrame (columns = feature names)
        model_type: "tree", "linear", or "auto" (auto-detects from model class name)
        max_samples: Cap samples for background dataset to limit compute (default: 500)

    Returns:
        DataFrame with columns: feature, mean_abs_shap, shap_std
        Sorted by mean_abs_shap descending.

    Raises:
        ImportError: If shap is not installed.
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "shap is not installed. Run: pip install 'shap>=0.44.0'"
        )

    if X.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "shap_std"])

    X_bg = X.iloc[:max_samples] if len(X) > max_samples else X

    # Auto-detect explainer type
    if model_type == "auto":
        cls_name = type(model).__name__.lower()
        if any(k in cls_name for k in ("forest", "xgb", "lgbm", "catboost", "tree", "gradient")):
            model_type = "tree"
        elif any(k in cls_name for k in ("ridge", "lasso", "linear", "logistic")):
            model_type = "linear"
        else:
            model_type = "kernel"

    try:
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_bg)
        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, X_bg)
            shap_values = explainer.shap_values(X_bg)
        else:
            # Fallback: model-agnostic kernel explainer (slow)
            bg = shap.sample(X_bg, min(100, len(X_bg)))
            explainer = shap.KernelExplainer(model.predict, bg)
            shap_values = explainer.shap_values(X_bg, nsamples=100)
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "shap_std"])

    shap_arr = np.abs(shap_values)
    mean_abs = shap_arr.mean(axis=0)
    std_abs = shap_arr.std(axis=0)

    result = pd.DataFrame(
        {
            "feature": list(X.columns),
            "mean_abs_shap": mean_abs,
            "shap_std": std_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return result


def compute_shap_temporal_drift(
    model: Any,
    panel_df: pd.DataFrame,
    feature_cols: list[str],
    timestamp_col: str = "timestamp",
    model_type: str = "auto",
    freq: str = "ME",
) -> pd.DataFrame:
    """Compute SHAP importance per time period to detect feature drift.

    Computes mean absolute SHAP per calendar month (or other freq) and
    returns a long-form DataFrame showing how feature importance changes
    over time.

    Args:
        model: Trained model
        panel_df: Factor panel with timestamp column
        feature_cols: Feature columns to use
        timestamp_col: Name of timestamp column (default: "timestamp")
        model_type: Passed to compute_shap_values (default: "auto")
        freq: Pandas resample frequency (default: "ME" = month-end)

    Returns:
        DataFrame with columns: period, feature, mean_abs_shap
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is not installed. Run: pip install 'shap>=0.44.0'")

    if timestamp_col not in panel_df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in panel_df")

    df = panel_df[[timestamp_col] + feature_cols].dropna()
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)

    records = []
    for period, group in df.resample(freq):
        if len(group) < 10:
            continue
        X_period = group[feature_cols]
        shap_df = compute_shap_values(model, X_period, model_type=model_type)
        for _, row in shap_df.iterrows():
            records.append(
                {
                    "period": period,
                    "feature": row["feature"],
                    "mean_abs_shap": row["mean_abs_shap"],
                }
            )

    if not records:
        return pd.DataFrame(columns=["period", "feature", "mean_abs_shap"])
    return pd.DataFrame(records)


# Implementation Roadmap:
# =======================
#
# Phase E2.1 (Current - Skeleton):
# - ✅ Module structure and design documentation
# - ✅ Function signatures with Type Hints
# - ✅ TODO comments for implementation
#
# Phase E2.2 (Core Implementation):
# - Implement compute_model_feature_importance() for linear and tree models
# - Implement compute_permutation_importance() using sklearn.inspection
# - Implement summarize_feature_importance_global() for model comparison
# - Add CLI script for feature importance analysis (e.g., scripts/analyze_ml_feature_importance.py)
# - Integration with Model Zoo: Automatically compute feature importance for all models
# - Add tests (tests/test_ml_explainability.py)
#
# Phase E2.3 (Advanced Explainability):
# - SHAP values integration (requires shap library)
#   - Model-agnostic SHAP (KernelExplainer, PermutationExplainer)
#   - Tree-specific SHAP (TreeExplainer for RandomForest)
#   - Local explanations DataFrame
# - Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) plots
# - Feature interaction analysis
# - Time-series specific explainability (temporal importance analysis)
#
# Phase E2.4 (Integration & Reporting):
# - Feature importance reports (Markdown + visualization if matplotlib available)
# - Comparison with classical factor rankings (IC/IR from Phase C1/C2)
# - Feature importance stability analysis across CV splits
# - Integration with Model Zoo summary reports
