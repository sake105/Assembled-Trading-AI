"""Online learning and incremental model updates (Plan 2.7).

Provides:
- Exponentially Weighted Recursive Least Squares (EWRLS) for linear
  factor models with automatic regime adaptation.
- Performance-based retraining triggers (not calendar-based).
- Model age tracking for meta-model confidence.

EWRLS forgets old data exponentially, making the model self-adaptive
to regime changes without explicit regime detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EWRLSModel:
    """Exponentially Weighted Recursive Least Squares model.

    Maintains a running estimate of linear regression coefficients
    ``y = X @ beta`` that exponentially downweights old observations.

    Attributes:
        n_features: Number of input features.
        forgetting_factor: Lambda in [0.95, 1.0]. Lower = forget faster.
        beta: Current coefficient estimates.
        P: Inverse covariance matrix (precision).
        n_updates: Number of updates performed.
    """

    n_features: int
    forgetting_factor: float = 0.98
    beta: np.ndarray = field(default=None, repr=False)
    P: np.ndarray = field(default=None, repr=False)
    n_updates: int = 0
    last_update: str = ""

    def __post_init__(self):
        if self.beta is None:
            self.beta = np.zeros(self.n_features)
        if self.P is None:
            self.P = np.eye(self.n_features) * 100.0  # large initial uncertainty

    def update(self, x: np.ndarray, y: float) -> float:
        """Update model with a single observation.

        Args:
            x: Feature vector (n_features,).
            y: Target scalar.

        Returns:
            Prediction error (residual) before update.
        """
        lam = self.forgetting_factor

        # Prediction
        y_hat = float(x @ self.beta)
        error = y - y_hat

        # Kalman gain
        Px = self.P @ x
        denom = lam + float(x @ Px)
        K = Px / denom

        # Update beta
        self.beta = self.beta + K * error

        # Update P (inverse covariance)
        self.P = (self.P - np.outer(K, x @ self.P)) / lam

        self.n_updates += 1
        return error

    def predict(self, x: np.ndarray) -> float:
        """Predict target for feature vector."""
        return float(x @ self.beta)

    def batch_update(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Update model with multiple observations sequentially.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).

        Returns:
            Array of prediction errors.
        """
        errors = np.empty(len(y))
        for i in range(len(y)):
            errors[i] = self.update(X[i], y[i])
        return errors

    @property
    def model_age_days(self) -> int:
        """Days since last update (requires last_update to be set)."""
        if not self.last_update:
            return 0
        try:
            last = datetime.fromisoformat(self.last_update)
            return (datetime.now() - last).days
        except Exception:
            return 0


@dataclass
class RetrainingTrigger:
    """Performance-based retraining trigger.

    Instead of retraining every N days (calendar-based), triggers
    retraining when model performance degrades:
    - Rolling OOS IC drops below threshold for N consecutive days
    - Prediction error exceeds historical norms

    Attributes:
        ic_threshold: Minimum acceptable rolling IC.
        ic_window: Rolling window for IC evaluation.
        consecutive_bad_days: Required bad days before trigger.
        error_zscore_threshold: Z-score of error that triggers retraining.
    """

    ic_threshold: float = 0.0
    ic_window: int = 20
    consecutive_bad_days: int = 10
    error_zscore_threshold: float = 3.0

    _bad_day_count: int = 0
    _error_history: list = field(default_factory=list)

    def check(self, ic_value: float, prediction_error: float) -> bool:
        """Check if retraining should be triggered.

        Args:
            ic_value: Current rolling IC value.
            prediction_error: Current prediction error.

        Returns:
            True if retraining is recommended.
        """
        # IC-based trigger
        if ic_value < self.ic_threshold:
            self._bad_day_count += 1
        else:
            self._bad_day_count = 0

        if self._bad_day_count >= self.consecutive_bad_days:
            logger.info("[Retrain] IC below %.2f for %d consecutive days — triggering",
                       self.ic_threshold, self._bad_day_count)
            self._bad_day_count = 0
            return True

        # Error-based trigger
        self._error_history.append(abs(prediction_error))
        if len(self._error_history) > 100:
            self._error_history = self._error_history[-100:]

        if len(self._error_history) >= 30:
            mean_err = np.mean(self._error_history[:-1])
            std_err = np.std(self._error_history[:-1])
            if std_err > 0:
                z = (abs(prediction_error) - mean_err) / std_err
                if z > self.error_zscore_threshold:
                    logger.info("[Retrain] Error z-score %.1f exceeds threshold — triggering", z)
                    return True

        return False

    def reset(self):
        """Reset trigger state after retraining."""
        self._bad_day_count = 0
        self._error_history.clear()


def compute_model_age_confidence(
    days_since_refit: int,
    half_life_days: int = 30,
) -> float:
    """Compute confidence multiplier based on model freshness.

    Fresh models (just retrained) get confidence=1.0.
    Old models get exponentially decaying confidence.

    Args:
        days_since_refit: Days since last model refit.
        half_life_days: Half-life of confidence decay.

    Returns:
        Confidence multiplier in (0, 1].
    """
    if days_since_refit <= 0:
        return 1.0
    return float(np.exp(-np.log(2) * days_since_refit / half_life_days))


__all__ = [
    "EWRLSModel",
    "RetrainingTrigger",
    "compute_model_age_confidence",
]
