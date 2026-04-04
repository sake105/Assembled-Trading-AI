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
    _split_time_series,
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
    diversity_warn_threshold: float = 0.95

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

        # Prepare dataset
        X, y = prepare_ml_dataset(panel_df, experiment, feature_cols=feature_cols,
                                   timestamp_col=timestamp_col, symbol_col=symbol_col)
        self._feature_cols = list(X.columns)
        splits = list(_split_time_series(y.index, experiment))
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
                if experiment.standardize and cfg.model_type not in {"random_forest", "xgboost", "lightgbm", "catboost"}:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_te = scaler.transform(X_te)

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

        self._check_diversity(oof_valid)

        self._meta_model = Ridge(alpha=self.meta_alpha)
        self._meta_model.fit(oof_valid, y_valid)
        logger.info(
            "[Stack] Meta-learner trained on %d samples (meta R2 estimate on OOF: %.4f)",
            len(y_valid),
            float(self._meta_model.score(oof_valid, y_valid)),
        )

        # Step 3: Retrain base models on full data for inference
        self._base_models = []
        self._base_scalers = []
        X_full = np.nan_to_num(X.values, nan=0.0)
        y_full = y.values

        for cfg in self.base_configs:
            scaler = None
            X_tr_full = X_full.copy()
            if experiment.standardize and cfg.model_type not in {"random_forest", "xgboost", "lightgbm", "catboost"}:
                scaler = StandardScaler()
                X_tr_full = scaler.fit_transform(X_tr_full)
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
        X = np.nan_to_num(X.astype(float), nan=0.0)

        base_preds = np.column_stack([
            self._apply_base_model(i, X) for i in range(len(self._base_models))
        ])
        return self._meta_model.predict(base_preds)

    def _apply_base_model(self, idx: int, X: np.ndarray) -> np.ndarray:
        model = self._base_models[idx]
        scaler = self._base_scalers[idx]
        X_scaled = scaler.transform(X) if scaler is not None else X
        return model.predict(X_scaled)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _check_diversity(self, oof_preds: np.ndarray) -> None:
        """Warn if base model OOF predictions are highly correlated."""
        n_models = oof_preds.shape[1]
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr = float(np.corrcoef(oof_preds[:, i], oof_preds[:, j])[0, 1])
                if abs(corr) > self.diversity_warn_threshold:
                    logger.warning(
                        "[Stack] Base models %d (%s) and %d (%s) have OOF correlation %.3f "
                        "— ensemble may provide limited benefit",
                        i, self.base_configs[i].name,
                        j, self.base_configs[j].name,
                        corr,
                    )

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
                    params={"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63,
                            "subsample": 0.8, "colsample_bytree": 0.8},
                )
            )

    return StackedEnsemble(base_configs=base_configs, meta_alpha=meta_alpha)
