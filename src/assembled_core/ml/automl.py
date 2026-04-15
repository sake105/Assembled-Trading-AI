"""AutoML — Automated Model Selection and Feature Engineering (M32).

Implements automated model selection for factor-based alpha prediction:
  1. Model zoo: linear, ridge, lasso, random forest, gradient boosting
  2. Automated feature selection via mutual information + forward selection
  3. Time-series cross-validation with purging
  4. Automatic hyperparameter search (grid-based, no external deps)
  5. Model ranking by information coefficient (IC)

The goal: remove human bias from model selection. Let the data choose
the best model family and feature subset for each prediction horizon.

Reference:
    Hutter, F. et al. (2019). "Automated Machine Learning."
    de Prado, M.L. (2018). "Advances in Financial Machine Learning."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import Ridge, Lasso  # type: ignore[import]
    from sklearn.ensemble import (  # type: ignore[import]
        RandomForestRegressor,
        GradientBoostingRegressor,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ModelCandidate:
    """A candidate model with performance metrics.

    Attributes:
        model_type: Model family name.
        params: Hyperparameters used.
        ic_mean: Mean information coefficient across folds.
        ic_std: Standard deviation of IC across folds.
        ic_ir: IC information ratio (ic_mean / ic_std).
        n_features: Number of features selected.
        feature_names: Names of selected features.
        rank: Rank among candidates (1 = best).
    """

    model_type: str
    params: dict
    ic_mean: float
    ic_std: float
    ic_ir: float
    n_features: int
    feature_names: list[str]
    rank: int = 0


@dataclass
class AutoMLResult:
    """Result of AutoML model selection.

    Attributes:
        best_model: Best performing ModelCandidate.
        all_candidates: All evaluated candidates ranked by IC-IR.
        selected_features: Final feature set.
        n_models_evaluated: Total models evaluated.
        cv_folds: Number of CV folds used.
    """

    best_model: ModelCandidate
    all_candidates: list[ModelCandidate]
    selected_features: list[str]
    n_models_evaluated: int
    cv_folds: int


def compute_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Compute rank information coefficient (Spearman correlation).

    Args:
        predictions: Model predictions.
        actuals: Actual forward returns.

    Returns:
        IC (Spearman rank correlation).
    """
    p = np.asarray(predictions, dtype=float).ravel()
    a = np.asarray(actuals, dtype=float).ravel()
    mask = np.isfinite(p) & np.isfinite(a)
    p, a = p[mask], a[mask]

    if len(p) < 5:
        return 0.0

    # Rank
    def _rank(x):
        order = x.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1)
        return ranks

    rp = _rank(p)
    ra = _rank(a)
    n = len(rp)

    d = rp - ra
    rho = 1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1))
    return float(rho)


def select_features_mi(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 15,
    min_mi_threshold: float = 0.01,
) -> list[str]:
    """Select features using mutual information proxy.

    Uses absolute Spearman correlation as a computationally cheap
    proxy for mutual information (no sklearn dependency required).

    Args:
        X: Feature matrix.
        y: Target variable.
        max_features: Maximum number of features to select.
        min_mi_threshold: Minimum MI proxy score to include.

    Returns:
        List of selected feature names.
    """
    scores = {}
    y_vals = y.values

    for col in X.columns:
        x_vals = X[col].values
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if mask.sum() < 20:
            scores[col] = 0.0
            continue
        ic = abs(compute_ic(x_vals[mask], y_vals[mask]))
        scores[col] = ic

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [
        name for name, score in ranked[:max_features]
        if score >= min_mi_threshold
    ]

    logger.info(
        "[AutoML] Selected %d/%d features (top MI: %.3f)",
        len(selected), len(X.columns),
        ranked[0][1] if ranked else 0.0,
    )

    return selected


def _build_model(model_type: str, params: dict):
    """Build a model from type and parameters."""
    if not SKLEARN_AVAILABLE:
        return None

    if model_type == "ridge":
        return Ridge(alpha=params.get("alpha", 1.0))
    elif model_type == "lasso":
        return Lasso(alpha=params.get("alpha", 0.01), max_iter=2000)
    elif model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            min_samples_leaf=params.get("min_samples_leaf", 10),
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            min_samples_leaf=params.get("min_samples_leaf", 10),
            random_state=42,
        )
    return None


# Default hyperparameter grids
MODEL_ZOO: dict[str, list[dict]] = {
    "ridge": [
        {"alpha": 0.1},
        {"alpha": 1.0},
        {"alpha": 10.0},
    ],
    "lasso": [
        {"alpha": 0.001},
        {"alpha": 0.01},
        {"alpha": 0.1},
    ],
    "random_forest": [
        {"n_estimators": 50, "max_depth": 3, "min_samples_leaf": 20},
        {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 10},
        {"n_estimators": 200, "max_depth": 7, "min_samples_leaf": 5},
    ],
    "gradient_boosting": [
        {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.01},
    ],
}


def time_series_cv_split(
    n_samples: int,
    n_folds: int = 5,
    purge_gap: int = 5,
    min_train_size: int = 60,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate time-series cross-validation splits with purging.

    Args:
        n_samples: Total number of samples.
        n_folds: Number of folds.
        purge_gap: Number of samples to skip between train and test.
        min_train_size: Minimum training set size.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    fold_size = max(1, (n_samples - min_train_size) // (n_folds + 1))
    splits = []

    for i in range(n_folds):
        test_start = min_train_size + i * fold_size
        test_end = min(test_start + fold_size, n_samples)
        train_end = max(0, test_start - purge_gap)

        if train_end < min_train_size // 2 or test_start >= n_samples:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) >= min_train_size // 2 and len(test_idx) >= 5:
            splits.append((train_idx, test_idx))

    return splits


def run_automl(
    features_df: pd.DataFrame,
    target: pd.Series,
    model_types: list[str] | None = None,
    max_features: int = 15,
    n_folds: int = 5,
) -> AutoMLResult:
    """Run automated model selection.

    Evaluates all model types with their hyperparameter grids,
    using time-series CV with purging, and ranks by IC-IR.

    Args:
        features_df: Feature matrix (dates x features).
        target: Forward returns (aligned with features).
        model_types: Model types to evaluate. Default: all in MODEL_ZOO.
        max_features: Maximum features to select.
        n_folds: Number of CV folds.

    Returns:
        AutoMLResult with best model and all candidates.
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("[AutoML] sklearn not available, returning dummy result")
        dummy = ModelCandidate(
            model_type="none", params={}, ic_mean=0.0, ic_std=1.0,
            ic_ir=0.0, n_features=0, feature_names=[], rank=1,
        )
        return AutoMLResult(
            best_model=dummy, all_candidates=[dummy],
            selected_features=[], n_models_evaluated=0, cv_folds=0,
        )

    if model_types is None:
        model_types = list(MODEL_ZOO.keys())

    # Align features and target
    common_idx = features_df.index.intersection(target.index)
    X = features_df.loc[common_idx].copy()
    y = target.loc[common_idx].copy()

    # Drop NaN rows
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if len(X) < 60:
        logger.warning("[AutoML] Insufficient data (%d rows), need >= 60", len(X))
        dummy = ModelCandidate(
            model_type="none", params={}, ic_mean=0.0, ic_std=1.0,
            ic_ir=0.0, n_features=0, feature_names=[], rank=1,
        )
        return AutoMLResult(
            best_model=dummy, all_candidates=[dummy],
            selected_features=[], n_models_evaluated=0, cv_folds=0,
        )

    # Feature selection
    selected_features = select_features_mi(X, y, max_features)
    if not selected_features:
        selected_features = list(X.columns[:max_features])
    X_sel = X[selected_features]

    # CV splits
    splits = time_series_cv_split(len(X_sel), n_folds)
    if not splits:
        splits = [(np.arange(len(X_sel) // 2), np.arange(len(X_sel) // 2, len(X_sel)))]

    # Evaluate all candidates
    candidates = []
    X_arr = X_sel.values
    y_arr = y.values

    for model_type in model_types:
        param_grid = MODEL_ZOO.get(model_type, [{}])
        for params in param_grid:
            fold_ics = []
            for train_idx, test_idx in splits:
                model = _build_model(model_type, params)
                if model is None:
                    continue

                X_train, y_train = X_arr[train_idx], y_arr[train_idx]
                X_test, y_test = X_arr[test_idx], y_arr[test_idx]

                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    ic = compute_ic(preds, y_test)
                    fold_ics.append(ic)
                except Exception:
                    continue

            if fold_ics:
                ic_mean = float(np.mean(fold_ics))
                ic_std = float(np.std(fold_ics)) + 1e-6
                candidates.append(ModelCandidate(
                    model_type=model_type,
                    params=dict(params),
                    ic_mean=round(ic_mean, 4),
                    ic_std=round(ic_std, 4),
                    ic_ir=round(ic_mean / ic_std, 4),
                    n_features=len(selected_features),
                    feature_names=list(selected_features),
                ))

    # Rank by IC-IR
    candidates.sort(key=lambda c: c.ic_ir, reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    best = candidates[0] if candidates else ModelCandidate(
        model_type="none", params={}, ic_mean=0.0, ic_std=1.0,
        ic_ir=0.0, n_features=0, feature_names=[], rank=1,
    )

    logger.info(
        "[AutoML] Evaluated %d models, best: %s (IC-IR=%.3f, IC=%.4f)",
        len(candidates), best.model_type, best.ic_ir, best.ic_mean,
    )

    return AutoMLResult(
        best_model=best,
        all_candidates=candidates,
        selected_features=selected_features,
        n_models_evaluated=len(candidates),
        cv_folds=len(splits),
    )


__all__ = [
    "ModelCandidate",
    "AutoMLResult",
    "compute_ic",
    "select_features_mi",
    "time_series_cv_split",
    "run_automl",
    "MODEL_ZOO",
]
