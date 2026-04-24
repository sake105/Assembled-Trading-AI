"""Conformal Prediction for Valid Prediction Intervals (M19a).

Implements split conformal prediction that wraps any point predictor to
produce prediction intervals with finite-sample coverage guarantees.

Unlike quantile regression, conformal prediction provides:
  - Distribution-free coverage guarantees (1-alpha probability of containing
    the true value, regardless of the underlying distribution)
  - Works with any black-box model (no retraining needed)
  - Adaptive interval widths based on local difficulty

Methods:
  - SplitConformal: Standard split conformal with calibration set
  - AdaptiveConformal: Locally-weighted conformal with adaptive widths

Reference:
    Vovk, V., Gammerman, A., & Shafer, G. (2005).
    "Algorithmic Learning in a Random World."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConformalResult:
    """Result of conformal prediction.

    Attributes:
        predictions: Point predictions from the base model.
        lower: Lower bound of prediction interval.
        upper: Upper bound of prediction interval.
        interval_width: Width of prediction interval at each point.
        coverage_target: Target coverage level (1-alpha).
        calibration_score: Quantile of nonconformity scores used.
    """

    predictions: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    interval_width: np.ndarray
    coverage_target: float
    calibration_score: float


class SplitConformal:
    """Split Conformal Prediction with coverage guarantees.

    Uses a held-out calibration set to compute nonconformity scores
    (|y - y_hat|), then constructs intervals using the (1-alpha) quantile
    of these scores.

    Coverage guarantee: P(y_new in interval) >= 1-alpha for exchangeable data.

    Usage:
        model = train_any_model(X_train, y_train)
        cp = SplitConformal(alpha=0.10)
        cp.calibrate(model.predict, X_cal, y_cal)
        result = cp.predict(X_test)
        # result.lower, result.upper have 90% coverage guarantee
    """

    def __init__(self, alpha: float = 0.10):
        """Initialize conformal predictor.

        Args:
            alpha: Miscoverage rate. Intervals will cover (1-alpha)
                of future observations. Default 0.10 = 90% coverage.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self._scores: np.ndarray | None = None
        self._quantile: float = 0.0
        self._predictor = None

    def calibrate(
        self,
        predictor: callable,
        X_cal: np.ndarray | pd.DataFrame,
        y_cal: np.ndarray | pd.Series,
    ) -> float:
        """Calibrate using held-out calibration data.

        Args:
            predictor: Function that takes X and returns predictions.
            X_cal: Calibration features.
            y_cal: Calibration true values.

        Returns:
            The calibration quantile (nonconformity score threshold).
        """
        self._predictor = predictor
        y_pred = np.asarray(predictor(X_cal)).flatten()
        y_true = np.asarray(y_cal).flatten()

        # Nonconformity scores: absolute residuals
        self._scores = np.abs(y_true - y_pred)

        # Quantile with finite-sample correction
        n = len(self._scores)
        q_level = min(1.0, (1 - self.alpha) * (1 + 1 / n))
        self._quantile = float(np.quantile(self._scores, q_level))

        logger.info(
            "[Conformal] Calibrated with %d points: q(%.2f)=%.4f, "
            "median_score=%.4f, max_score=%.4f",
            n, q_level, self._quantile,
            float(np.median(self._scores)),
            float(np.max(self._scores)),
        )
        return self._quantile

    def predict(
        self,
        X_test: np.ndarray | pd.DataFrame,
    ) -> ConformalResult:
        """Generate prediction intervals for new data.

        Args:
            X_test: Test features.

        Returns:
            ConformalResult with predictions, lower, upper bounds.
        """
        if self._predictor is None or self._scores is None:
            raise RuntimeError("Must call calibrate() before predict()")

        y_pred = np.asarray(self._predictor(X_test)).flatten()
        lower = y_pred - self._quantile
        upper = y_pred + self._quantile
        width = np.full_like(y_pred, 2 * self._quantile)

        return ConformalResult(
            predictions=y_pred,
            lower=lower,
            upper=upper,
            interval_width=width,
            coverage_target=1 - self.alpha,
            calibration_score=self._quantile,
        )

    @property
    def is_calibrated(self) -> bool:
        return self._scores is not None


class AdaptiveConformal:
    """Adaptive Conformal Prediction with locally-weighted intervals.

    Instead of a fixed-width interval, this variant uses normalized
    nonconformity scores that adapt to local prediction difficulty.
    Harder-to-predict regions get wider intervals automatically.

    The normalization uses the model's own uncertainty estimate (e.g.,
    ensemble variance, distance to training data, or residual MAD).

    Usage:
        cp = AdaptiveConformal(alpha=0.10)
        cp.calibrate(predictor, difficulty_fn, X_cal, y_cal)
        result = cp.predict(X_test)
    """

    def __init__(self, alpha: float = 0.10):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self._scores: np.ndarray | None = None
        self._quantile: float = 0.0
        self._predictor = None
        self._difficulty_fn = None

    def calibrate(
        self,
        predictor: callable,
        difficulty_fn: callable,
        X_cal: np.ndarray | pd.DataFrame,
        y_cal: np.ndarray | pd.Series,
    ) -> float:
        """Calibrate with adaptive normalization.

        Args:
            predictor: Point prediction function.
            difficulty_fn: Function X -> positive values estimating
                local prediction difficulty (higher = harder).
            X_cal: Calibration features.
            y_cal: Calibration targets.

        Returns:
            Calibration quantile.
        """
        self._predictor = predictor
        self._difficulty_fn = difficulty_fn

        y_pred = np.asarray(predictor(X_cal)).flatten()
        y_true = np.asarray(y_cal).flatten()
        difficulty = np.asarray(difficulty_fn(X_cal)).flatten()

        # Prevent division by zero
        difficulty = np.maximum(difficulty, 1e-10)

        # Normalized nonconformity scores
        self._scores = np.abs(y_true - y_pred) / difficulty

        n = len(self._scores)
        q_level = min(1.0, (1 - self.alpha) * (1 + 1 / n))
        self._quantile = float(np.quantile(self._scores, q_level))

        logger.info(
            "[AdaptiveConformal] Calibrated with %d points: quantile=%.4f",
            n, self._quantile,
        )
        return self._quantile

    def predict(
        self,
        X_test: np.ndarray | pd.DataFrame,
    ) -> ConformalResult:
        """Generate adaptive prediction intervals.

        Args:
            X_test: Test features.

        Returns:
            ConformalResult with varying-width intervals.
        """
        if self._predictor is None or self._difficulty_fn is None:
            raise RuntimeError("Must call calibrate() before predict()")

        y_pred = np.asarray(self._predictor(X_test)).flatten()
        difficulty = np.asarray(self._difficulty_fn(X_test)).flatten()
        difficulty = np.maximum(difficulty, 1e-10)

        margin = self._quantile * difficulty
        lower = y_pred - margin
        upper = y_pred + margin
        width = 2 * margin

        return ConformalResult(
            predictions=y_pred,
            lower=lower,
            upper=upper,
            interval_width=width,
            coverage_target=1 - self.alpha,
            calibration_score=self._quantile,
        )


def evaluate_coverage(
    result: ConformalResult,
    y_true: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Evaluate actual coverage and interval quality.

    Args:
        result: ConformalResult from predict().
        y_true: True target values.

    Returns:
        Dict with coverage, avg_width, median_width, coverage_gap.
    """
    y = np.asarray(y_true).flatten()
    covered = (y >= result.lower) & (y <= result.upper)
    actual_coverage = float(covered.mean())

    return {
        "actual_coverage": round(actual_coverage, 4),
        "target_coverage": result.coverage_target,
        "coverage_gap": round(actual_coverage - result.coverage_target, 4),
        "avg_interval_width": round(float(result.interval_width.mean()), 4),
        "median_interval_width": round(float(np.median(result.interval_width)), 4),
        "pct_covered": round(actual_coverage * 100, 1),
    }


__all__ = [
    "ConformalResult",
    "SplitConformal",
    "AdaptiveConformal",
    "evaluate_coverage",
]
