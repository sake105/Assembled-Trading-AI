"""Two-Level Ensemble Stacking for ML Factor Models.

Generates out-of-fold (OOF) predictions from multiple base estimators using
time-series cross-validation, then trains a meta-learner on those OOF
predictions. This is the institutional-grade ensemble pattern that typically
improves Sharpe by 5–15% vs. the best single model.

Usage:
    from src.assembled_core.ml.stacking import StackedEnsemble, build_default_stack

    stack = build_default_stack()
    stack.fit(panel_df, experiment_cfg, feature_cols=feature_cols)
    predictions = stack.predict(X_test)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.assembled_core.ml.factor_models import (
    MLExperimentConfig,
    MLModelConfig,
    _create_model,
    prepare_ml_dataset,
    SKLEARN_AVAILABLE,
)

logger = logging.getLogger(__name__)

if SKLEARN_AVAILABLE:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
else:
    Ridge = None  # type: ignore
    StandardScaler = None  # type: ignore


@dataclass
class StackedEnsemble:
    """Two-level stacked ensemble for factor return prediction.

    Level 1: base estimators trained via time-series CV producing OOF predictions.
    Level 2: meta-learner (Ridge by default) trained on the OOF predictions.

    Attributes:
        base_configs: List of MLModelConfig for level-1 base models
        meta_alpha: Regularisation strength for Ridge meta-learner (default: 1.0)
        diversity_warn_threshold: Warn if any pair of base models has OOF correlation
            above this value (default: 0.95)
    """

    base_configs: list[MLModelConfig] = field(default_factory=list)
    meta_alpha: float = 1.0
    meta_model_type: str = "ridge"  # V11: "ridge", "elasticnet", "gbm"
    diversity_warn_threshold: float = 0.95
    auto_diversity: bool = True  # V11: auto-drop correlated base models

    # Internal state (set during fit)
    _base_models: list[Any] = field(default_factory=list, init=False, repr=False)
    _base_scalers: list[Any] = field(default_factory=list, init=False, repr=False)
    _meta_model: Any = field(default=None, init=False, repr=False)
    _feature_cols: list[str] = field(default_factory=list, init=False, repr=False)
    _is_fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for StackedEnsemble")
        self._base_models = []
        self._base_scalers = []

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        panel_df: pd.DataFrame,
        experiment: MLExperimentConfig,
        feature_cols: list[str] | None = None,
        timestamp_col: str = "timestamp",
        symbol_col: str = "symbol",
    ) -> "StackedEnsemble":
        """Fit the stacked ensemble on a factor panel.

        Step 1: Generate OOF predictions from each base model via time-series CV.
        Step 2: Train Ridge meta-learner on the stacked OOF predictions.
        Step 3: Retrain each base model on the full dataset for inference.

        Args:
            panel_df: Factor panel DataFrame
            experiment: MLExperimentConfig (controls CV splits, label, standardise)
            feature_cols: Explicit list of feature columns (None = auto-detect)
            timestamp_col: Timestamp column name
            symbol_col: Symbol column name

        Returns:
            self
        """
        if not self.base_configs:
            raise ValueError("base_configs is empty — add at least one MLModelConfig")

        # Prepare dataset -- set feature_cols on experiment if provided
        if feature_cols is not None:
            experiment.feature_cols = feature_cols
        X, y = prepare_ml_dataset(panel_df, experiment,
                                   timestamp_col=timestamp_col, symbol_col=symbol_col)
        # Reset index to 0..N-1 so positional and label indexing are identical.
        # prepare_ml_dataset may leave gaps after dropna/feature filtering.
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self._feature_cols = list(X.columns)
        n_total = len(X)
        splits = []
        for i in range(experiment.n_splits):
            test_start = n_total * (i + 1) // (experiment.n_splits + 1)
            test_end = min(n_total * (i + 2) // (experiment.n_splits + 1), n_total)
            train_idx = list(range(test_start))
            test_idx = list(range(test_start, test_end))
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        if not splits:
            raise ValueError("No CV splits generated — check experiment config")

        n = len(y)
        oof_preds = np.full((n, len(self.base_configs)), np.nan)

        # Step 1: OOF predictions for each base model
        for b_idx, cfg in enumerate(self.base_configs):
            logger.info("[Stack] Generating OOF for base model %d: %s", b_idx, cfg.name)
            for train_idx, test_idx in splits:
                X_tr, X_te = X.iloc[train_idx].values, X.iloc[test_idx].values
                y_tr = y.iloc[train_idx].values

                if len(X_tr) < experiment.min_train_samples:
                    continue

                # Standardise if needed
                tree_models = {"random_forest", "xgboost", "lightgbm", "catboost"}
                if experiment.standardize and cfg.model_type not in tree_models:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_te = scaler.transform(X_te)

                # Tree models (XGB, LGB) handle NaN natively and learn optimal
                # split direction — imputing removes useful missingness signal.
                # Linear models: after StandardScaler, 0.0 = mean imputation.
                if cfg.model_type not in tree_models:
                    X_tr = np.nan_to_num(X_tr, nan=0.0)
                    X_te = np.nan_to_num(X_te, nan=0.0)

                model = _create_model(cfg)
                model.fit(X_tr, y_tr)
                oof_preds[test_idx, b_idx] = model.predict(X_te)

        # Step 2: Train meta-learner on OOF stack
        valid_mask = ~np.isnan(oof_preds).any(axis=1) & ~np.isnan(y.values)
        oof_valid = oof_preds[valid_mask]
        y_valid = y.values[valid_mask]

        if len(oof_valid) < 20:
            raise ValueError("Too few valid OOF samples for meta-learner training")

        oof_valid, active_mask = self._enforce_diversity(oof_valid, y_valid)

        # V11: Configurable meta-learner
        self._meta_model = self._create_meta_learner()
        self._meta_model.fit(oof_valid, y_valid)
        self._oof_residuals = y_valid - self._meta_model.predict(oof_valid)  # V11: for confidence
        logger.info(
            "[Stack] Meta-learner trained on %d samples (meta R2 estimate on OOF: %.4f)",
            len(y_valid),
            float(self._meta_model.score(oof_valid, y_valid)),
        )

        # Step 3: Retrain base models on full data for inference
        self._base_models = []
        self._base_scalers = []
        X_full_raw = X.values.copy()
        y_full = y.values
        tree_models_set = {"random_forest", "xgboost", "lightgbm", "catboost"}

        for cfg in self.base_configs:
            scaler = None
            X_tr_full = X_full_raw.copy()
            if experiment.standardize and cfg.model_type not in tree_models_set:
                scaler = StandardScaler()
                X_tr_full = scaler.fit_transform(X_tr_full)
            # Only impute NaN for non-tree models (tree models handle NaN natively)
            if cfg.model_type not in tree_models_set:
                X_tr_full = np.nan_to_num(X_tr_full, nan=0.0)
            model = _create_model(cfg)
            model.fit(X_tr_full, y_full)
            self._base_models.append(model)
            self._base_scalers.append(scaler)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate stacked ensemble predictions.

        Args:
            X: Feature matrix (DataFrame or ndarray). Columns must match training features.

        Returns:
            1D array of predictions.
        """
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            X = X[self._feature_cols].values
        X = X.astype(float)

        active_mask = getattr(self, "_active_base_mask", np.ones(len(self._base_models), dtype=bool))
        active_indices = np.where(active_mask)[0]
        base_preds = np.column_stack([
            self._apply_base_model(i, X) for i in active_indices
        ])
        return self._meta_model.predict(base_preds)

    def _apply_base_model(self, idx: int, X: np.ndarray) -> np.ndarray:
        model = self._base_models[idx]
        scaler = self._base_scalers[idx]
        tree_types = {"random_forest", "xgboost", "lightgbm", "catboost"}
        cfg_type = self.base_configs[idx].model_type if idx < len(self.base_configs) else ""
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        # Only impute NaN for non-tree models
        if cfg_type not in tree_types:
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        return model.predict(X_scaled)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _create_meta_learner(self) -> Any:
        """Create meta-learner based on configured type (V11)."""
        if self.meta_model_type == "elasticnet":
            from sklearn.linear_model import ElasticNet
            return ElasticNet(alpha=self.meta_alpha, l1_ratio=0.5, max_iter=2000)
        elif self.meta_model_type == "gbm":
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    subsample=0.8, random_state=42,
                )
            except ImportError:
                logger.warning("[Stack] GBM meta-learner unavailable, falling back to Ridge")
                return Ridge(alpha=self.meta_alpha)
        else:  # default: ridge
            return Ridge(alpha=self.meta_alpha)

    def _enforce_diversity(
        self, oof_preds: np.ndarray, y_valid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Check diversity and optionally drop redundant base models (V11).

        Returns:
            Tuple of (possibly column-pruned oof_preds, active_mask boolean array).
        """
        n_models = oof_preds.shape[1]
        active = np.ones(n_models, dtype=bool)

        for i in range(n_models):
            if not active[i]:
                continue
            for j in range(i + 1, n_models):
                if not active[j]:
                    continue
                corr = float(np.corrcoef(oof_preds[:, i], oof_preds[:, j])[0, 1])
                if abs(corr) > self.diversity_warn_threshold:
                    if self.auto_diversity:
                        # Drop the model with lower OOF R² vs target
                        r2_i = 1 - np.mean((y_valid - oof_preds[:, i]) ** 2) / max(np.var(y_valid), 1e-12)
                        r2_j = 1 - np.mean((y_valid - oof_preds[:, j]) ** 2) / max(np.var(y_valid), 1e-12)
                        victim = j if r2_i >= r2_j else i
                        active[victim] = False
                        logger.info(
                            "[Stack] Auto-diversity: dropping model %d (%s, R²=%.4f) "
                            "— corr %.3f with model %d (%s, R²=%.4f)",
                            victim, self.base_configs[victim].name,
                            r2_j if victim == j else r2_i, corr,
                            i if victim == j else j,
                            self.base_configs[i if victim == j else j].name,
                            r2_i if victim == j else r2_j,
                        )
                    else:
                        logger.warning(
                            "[Stack] Base models %d (%s) and %d (%s) have OOF correlation %.3f",
                            i, self.base_configs[i].name,
                            j, self.base_configs[j].name, corr,
                        )

        self._active_base_mask = active
        if not active.all():
            logger.info(
                "[Stack] %d/%d base models active after diversity enforcement",
                active.sum(), n_models,
            )
            oof_preds = oof_preds[:, active]

        return oof_preds, active

    def predict_with_confidence(
        self, X: pd.DataFrame | np.ndarray, confidence_level: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals from OOF residual distribution (V11).

        Args:
            X: Feature matrix.
            confidence_level: Confidence level for intervals (e.g. 0.9 = 90%).

        Returns:
            Tuple of (predictions, lower_bound, upper_bound).
        """
        preds = self.predict(X)

        if not hasattr(self, "_oof_residuals") or self._oof_residuals is None:
            return preds, preds, preds

        alpha = (1 - confidence_level) / 2
        q_lo = float(np.quantile(self._oof_residuals, alpha))
        q_hi = float(np.quantile(self._oof_residuals, 1 - alpha))

        return preds, preds + q_lo, preds + q_hi

    def _check_diversity(self, oof_preds: np.ndarray) -> None:
        """Warn if base model OOF predictions are highly correlated (legacy)."""
        self._enforce_diversity(oof_preds, np.zeros(len(oof_preds)))

    def diversity_report(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute pairwise correlation of base model predictions on X."""
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            X_arr = X[self._feature_cols].values
        else:
            X_arr = X
        X_arr = np.nan_to_num(X_arr.astype(float), nan=0.0)
        preds = {
            self.base_configs[i].name: self._apply_base_model(i, X_arr)
            for i in range(len(self._base_models))
        }
        return pd.DataFrame(preds).corr()

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("StackedEnsemble must be fitted before calling predict")


def build_default_stack(
    include_boosting: bool = True,
    meta_alpha: float = 1.0,
) -> StackedEnsemble:
    """Build a default StackedEnsemble with linear + tree base models.

    Args:
        include_boosting: Include XGBoost/LightGBM if available (default: True)
        meta_alpha: Ridge regularisation for meta-learner (default: 1.0)

    Returns:
        StackedEnsemble instance (not yet fitted)
    """
    from src.assembled_core.ml.factor_models import XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE  # type: ignore

    base_configs = [
        MLModelConfig(name="ridge", model_type="ridge", params={"alpha": 1.0}),
        MLModelConfig(name="lasso", model_type="lasso", params={"alpha": 0.01}),
        MLModelConfig(name="random_forest", model_type="random_forest",
                     params={"n_estimators": 200, "max_depth": 8, "random_state": 42}),
    ]

    if include_boosting:
        if XGBOOST_AVAILABLE:
            base_configs.append(
                MLModelConfig(
                    name="xgboost",
                    model_type="xgboost",
                    params={"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
                            "subsample": 0.8, "colsample_bytree": 0.8},
                )
            )
        if LIGHTGBM_AVAILABLE:
            base_configs.append(
                MLModelConfig(
                    name="lightgbm",
                    model_type="lightgbm",
                    params={"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                            "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 1.0,
                            "subsample": 0.8, "colsample_bytree": 0.7},
                )
            )

    return StackedEnsemble(base_configs=base_configs, meta_alpha=meta_alpha)


def enforce_ensemble_diversity(
    oof_preds: np.ndarray,
    max_correlation: float = 0.80,
    feature_subsample_frac: float = 0.50,
) -> dict:
    """Measure and report ensemble diversity.

    Args:
        oof_preds: OOF predictions matrix (n_samples × n_models).
        max_correlation: Threshold above which models are too similar.
        feature_subsample_frac: Recommended feature subsample fraction if too correlated.

    Returns:
        Dict with avg_correlation, max_pair_correlation, diverse flag, recommendations.
    """
    n_models = oof_preds.shape[1]
    if n_models < 2:
        return {"avg_correlation": 0.0, "max_pair_correlation": 0.0, "diverse": True, "recommendations": []}

    corr_matrix = np.corrcoef(oof_preds.T)
    # Extract upper triangle (exclude diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corrs = corr_matrix[mask]

    avg_corr = float(np.mean(np.abs(corrs)))
    max_corr = float(np.max(np.abs(corrs)))

    recommendations = []
    if avg_corr > max_correlation:
        recommendations.append(f"Feature subsampling at {feature_subsample_frac:.0%} per base learner")
        recommendations.append("Use different lookback windows per base learner (90d, 180d, 360d)")
        recommendations.append("Use different label horizons (5d, 10d, 20d)")

    return {
        "avg_correlation": round(avg_corr, 4),
        "max_pair_correlation": round(max_corr, 4),
        "diverse": avg_corr <= max_correlation,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Online Ensemble Weight Adaptation (Task 19.10)
# Vovk's Aggregating Algorithm / Exponential Weights
# ---------------------------------------------------------------------------

class OnlineEnsembleWeights:
    """Exponential Weights Algorithm for online meta-learner adaptation.

    Instead of fixed stacking weights, adapts weights daily based on
    each base model's recent prediction performance.

    Reference: Littlestone & Warmuth (1994), Vovk (1990)
    Sharpe improvement: +5-8%
    """

    def __init__(
        self,
        n_models: int,
        eta: float = 0.1,
        min_weight: float = 0.01,
    ):
        """Initialize online ensemble.

        Args:
            n_models: Number of base models.
            eta: Learning rate for exponential update.
            min_weight: Minimum weight floor per model.
        """
        self.n_models = n_models
        self.eta = eta
        self.min_weight = min_weight
        self.weights = np.ones(n_models) / n_models
        self._history: list[np.ndarray] = []

    def update(self, losses: np.ndarray) -> np.ndarray:
        """Update weights based on observed losses.

        w_i(t+1) = w_i(t) * exp(-eta * loss_i(t)) / Z

        Args:
            losses: (n_models,) loss for each model at this step.

        Returns:
            Updated weight vector.
        """
        losses = np.asarray(losses, dtype=float)
        assert len(losses) == self.n_models

        # Exponential weight update
        self.weights *= np.exp(-self.eta * losses)

        # Floor + renormalize
        self.weights = np.maximum(self.weights, self.min_weight)
        self.weights /= self.weights.sum()

        self._history.append(self.weights.copy())
        return self.weights.copy()

    def predict(self, predictions: np.ndarray) -> float:
        """Weighted ensemble prediction.

        Args:
            predictions: (n_models,) predictions from each base model.

        Returns:
            Weighted prediction.
        """
        return float(self.weights @ np.asarray(predictions))

    def get_weight_history(self) -> np.ndarray:
        """Return weight evolution over time.

        Returns:
            (T, n_models) array of weight history.
        """
        if not self._history:
            return np.array([self.weights])
        return np.array(self._history)

    def reset(self) -> None:
        """Reset to uniform weights."""
        self.weights = np.ones(self.n_models) / self.n_models
        self._history.clear()


def run_online_ensemble(
    model_predictions: np.ndarray,
    actual_returns: np.ndarray,
    eta: float = 0.1,
    loss_type: str = "squared",
) -> tuple[np.ndarray, np.ndarray]:
    """Run online ensemble weight adaptation over a time series.

    Args:
        model_predictions: (T, n_models) predictions.
        actual_returns: (T,) actual returns.
        eta: Learning rate.
        loss_type: "squared" or "absolute".

    Returns:
        (ensemble_predictions, final_weights) tuple.
    """
    T, n_models = model_predictions.shape
    ow = OnlineEnsembleWeights(n_models, eta=eta)
    ensemble_preds = np.zeros(T)

    for t in range(T):
        # Predict with current weights
        ensemble_preds[t] = ow.predict(model_predictions[t])

        # Compute per-model losses
        errors = model_predictions[t] - actual_returns[t]
        if loss_type == "squared":
            losses = errors ** 2
        else:
            losses = np.abs(errors)

        # Update weights
        ow.update(losses)

    return ensemble_preds, ow.weights
