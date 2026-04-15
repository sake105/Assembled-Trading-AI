"""Gaussian Process Regression for Factor Return Prediction (M19b).

Provides uncertainty-aware return prediction using Gaussian Process Regression.
Unlike point-estimate models, GPR naturally produces:
  - Mean prediction (expected return)
  - Predictive variance (uncertainty)
  - Full predictive distribution

This is valuable for:
  - Position sizing proportional to confidence
  - Detecting regime changes (high uncertainty = uncertain regime)
  - Combining with conformal prediction for calibrated intervals

Uses sklearn's GaussianProcessRegressor when available, with a lightweight
RBF-kernel fallback for environments without sklearn.

Reference:
    Rasmussen, C.E. & Williams, C.K.I. (2006).
    "Gaussian Processes for Machine Learning."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        WhiteKernel,
    )
    SKLEARN_GP_AVAILABLE = True
except ImportError:
    SKLEARN_GP_AVAILABLE = False


@dataclass
class GPRResult:
    """Result of Gaussian Process Regression prediction.

    Attributes:
        mean: Mean predictions (expected returns).
        std: Standard deviation of predictions (uncertainty).
        lower_95: Lower 95% credible interval bound.
        upper_95: Upper 95% credible interval bound.
        confidence: 1 / std — higher = more confident predictions.
    """

    mean: np.ndarray
    std: np.ndarray
    lower_95: np.ndarray
    upper_95: np.ndarray
    confidence: np.ndarray


class FactorGPR:
    """Gaussian Process Regression for factor-based return prediction.

    Fits a GP model to factor exposures -> forward returns, producing
    both point predictions and uncertainty estimates.

    The kernel is RBF (smooth factor interactions) + WhiteKernel (noise):
        k(x, x') = sigma_f^2 * exp(-||x-x'||^2 / (2*l^2)) + sigma_n^2

    Attributes:
        length_scale: RBF kernel length scale (controls smoothness).
        noise_level: Expected observation noise level.
        n_restarts: Number of optimizer restarts for hyperparameter tuning.
        max_train_samples: Maximum training samples (GPR is O(n^3)).
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        noise_level: float = 0.1,
        n_restarts: int = 3,
        max_train_samples: int = 500,
    ):
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.n_restarts = n_restarts
        self.max_train_samples = max_train_samples
        self._model = None
        self._X_mean: np.ndarray | None = None
        self._X_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._fitted = False

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> FactorGPR:
        """Fit the Gaussian Process to training data.

        Args:
            X: Feature matrix (n_samples, n_features). Typically factor exposures.
            y: Target vector (forward returns).

        Returns:
            self for chaining.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).flatten()

        # Subsample if too large (GPR is O(n^3))
        if len(X_arr) > self.max_train_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_arr), self.max_train_samples, replace=False)
            X_arr = X_arr[idx]
            y_arr = y_arr[idx]
            logger.info(
                "[GPR] Subsampled to %d training points (was %d)",
                self.max_train_samples, len(X),
            )

        # Standardize features
        self._X_mean = X_arr.mean(axis=0)
        self._X_std = X_arr.std(axis=0)
        self._X_std[self._X_std < 1e-10] = 1.0
        X_scaled = (X_arr - self._X_mean) / self._X_std

        # Standardize target
        self._y_mean = float(y_arr.mean())
        self._y_std = float(y_arr.std())
        if self._y_std < 1e-10:
            self._y_std = 1.0
        y_scaled = (y_arr - self._y_mean) / self._y_std

        if SKLEARN_GP_AVAILABLE:
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * RBF(self.length_scale, (1e-2, 1e2))
                + WhiteKernel(self.noise_level, (1e-5, 1e1))
            )
            self._model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts,
                normalize_y=False,
                alpha=1e-6,
            )
            self._model.fit(X_scaled, y_scaled)
            logger.info(
                "[GPR] Fitted sklearn GP on %d samples, %d features. "
                "Kernel: %s",
                len(X_scaled), X_scaled.shape[1],
                self._model.kernel_,
            )
        else:
            # Lightweight fallback: store training data for kernel regression
            self._X_train = X_scaled
            self._y_train = y_scaled
            logger.info(
                "[GPR] sklearn not available, using kernel regression fallback "
                "(%d samples, %d features)",
                len(X_scaled), X_scaled.shape[1],
            )

        self._fitted = True
        return self

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> GPRResult:
        """Predict with uncertainty estimates.

        Args:
            X: Test feature matrix.

        Returns:
            GPRResult with mean, std, and credible intervals.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        X_arr = np.asarray(X, dtype=float)
        X_scaled = (X_arr - self._X_mean) / self._X_std

        if SKLEARN_GP_AVAILABLE and self._model is not None:
            mean_scaled, std_scaled = self._model.predict(X_scaled, return_std=True)
        else:
            mean_scaled, std_scaled = self._kernel_regression(X_scaled)

        # Unscale predictions
        mean = mean_scaled * self._y_std + self._y_mean
        std = std_scaled * self._y_std

        # Ensure non-negative std
        std = np.maximum(std, 1e-10)

        lower_95 = mean - 1.96 * std
        upper_95 = mean + 1.96 * std
        confidence = 1.0 / std

        return GPRResult(
            mean=mean,
            std=std,
            lower_95=lower_95,
            upper_95=upper_95,
            confidence=confidence,
        )

    def _kernel_regression(
        self,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lightweight kernel regression fallback when sklearn unavailable.

        Uses RBF kernel weights for prediction with leave-one-out variance
        estimation for uncertainty.
        """
        X_train = self._X_train
        y_train = self._y_train
        ls = self.length_scale

        means = []
        stds = []

        for x_new in X_test:
            # RBF kernel distances
            dists = np.sum((X_train - x_new) ** 2, axis=1)
            weights = np.exp(-dists / (2 * ls**2))
            w_sum = weights.sum()

            if w_sum < 1e-10:
                means.append(0.0)
                stds.append(1.0)
                continue

            weights /= w_sum
            mean = float(np.dot(weights, y_train))
            # Weighted variance
            var = float(np.dot(weights, (y_train - mean) ** 2))
            stds.append(max(np.sqrt(var + self.noise_level**2), 1e-10))
            means.append(mean)

        return np.array(means), np.array(stds)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


def build_gpr_position_sizing_signal(
    gpr_result: GPRResult,
    base_score: np.ndarray | None = None,
    confidence_scaling: float = 1.0,
) -> np.ndarray:
    """Convert GPR predictions + uncertainty into position sizing multipliers.

    High confidence predictions get larger positions, uncertain predictions
    get smaller positions.

    Args:
        gpr_result: Prediction result from FactorGPR.
        base_score: Optional base signal to scale. If None, uses GPR mean.
        confidence_scaling: How aggressively to scale by confidence (0-2).
            0 = ignore confidence, 1 = linear, 2 = quadratic.

    Returns:
        Array of position sizing multipliers.
    """
    score = base_score if base_score is not None else gpr_result.mean

    # Normalize confidence to [0, 1] range
    conf = gpr_result.confidence
    if conf.max() > conf.min():
        conf_norm = (conf - conf.min()) / (conf.max() - conf.min())
    else:
        conf_norm = np.ones_like(conf)

    # Apply confidence scaling
    conf_weight = conf_norm ** confidence_scaling

    return score * conf_weight


__all__ = [
    "GPRResult",
    "FactorGPR",
    "build_gpr_position_sizing_signal",
    "SKLEARN_GP_AVAILABLE",
]
