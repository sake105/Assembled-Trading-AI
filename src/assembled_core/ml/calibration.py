"""Probability Calibration for ML Predictions.

Converts raw model scores into well-calibrated probabilities using:
    - Platt scaling (logistic calibration)
    - Isotonic regression
    - Temperature scaling

Calibrated probabilities enable better position sizing:
    size ∝ calibrated_probability × confidence_level

References:
    Platt (2000), Niculescu-Mizil & Caruana (2005)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.calibration import CalibratedClassifierCV  # type: ignore
    from sklearn.isotonic import IsotonicRegression  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class CalibrationResult:
    """Result of probability calibration evaluation."""
    method: str
    brier_score: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    n_bins: int


def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        y_true: Binary true labels (0 or 1).
        y_prob: Predicted probabilities (0 to 1).
        n_bins: Number of bins for calibration.

    Returns:
        CalibrationResult with ECE, MCE, and Brier score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Brier score
    brier = float(np.mean((y_prob - y_true) ** 2))

    # Bin-based calibration
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (y_prob == bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_pred = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        gap = abs(avg_pred - avg_true)
        ece += (count / n) * gap
        mce = max(mce, gap)

    return CalibrationResult(
        method="raw",
        brier_score=round(brier, 6),
        ece=round(ece, 6),
        mce=round(mce, 6),
        n_bins=n_bins,
    )


class IsotonicCalibrator:
    """Isotonic regression calibrator for probability calibration.

    Non-parametric, monotonic — stronger than Platt scaling for
    non-sigmoidal calibration curves.
    """

    def __init__(self) -> None:
        self._fitted = False
        self._model = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "IsotonicCalibrator":
        """Fit isotonic regression on calibration data.

        Args:
            y_true: True labels (binary or continuous).
            y_prob: Predicted scores/probabilities.

        Returns:
            self
        """
        if SKLEARN_AVAILABLE:
            self._model = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip",
            )
            self._model.fit(y_prob, y_true)
        else:
            # Fallback: store sorted pairs for interpolation
            sorted_idx = np.argsort(y_prob)
            self._x_sorted = np.asarray(y_prob)[sorted_idx]
            self._y_sorted = np.asarray(y_true)[sorted_idx]

        self._fitted = True
        logger.info("[Calibration] Isotonic calibrator fitted on %d samples", len(y_true))
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration to new predictions.

        Args:
            y_prob: Raw predicted probabilities.

        Returns:
            Calibrated probabilities.
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted.")

        y_prob = np.asarray(y_prob, dtype=float)

        if SKLEARN_AVAILABLE and self._model is not None:
            return self._model.transform(y_prob)

        # Fallback: linear interpolation
        return np.interp(y_prob, self._x_sorted, self._y_sorted)


class TemperatureScaler:
    """Temperature scaling for neural network calibration.

    Divides logits by a learned temperature T before softmax.
    Simpler than isotonic but preserves ordering.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self._fitted = False

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        lr: float = 0.01,
        n_iter: int = 100,
    ) -> "TemperatureScaler":
        """Learn optimal temperature via NLL minimization.

        Args:
            logits: Raw model logits (pre-sigmoid).
            y_true: True binary labels.
            lr: Learning rate.
            n_iter: Number of gradient descent steps.

        Returns:
            self
        """
        logits = np.asarray(logits, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        T = 1.0

        for _ in range(n_iter):
            scaled = logits / T
            probs = 1 / (1 + np.exp(-scaled))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            # NLL gradient w.r.t. T
            nll_grad = np.mean(
                -(y_true * scaled / T - np.log(1 + np.exp(scaled)) * (-logits / T ** 2))
            )
            T -= lr * nll_grad
            T = max(0.01, min(T, 10.0))

        self.temperature = T
        self._fitted = True
        logger.info("[Calibration] Temperature scaling: T=%.4f", T)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling.

        Args:
            logits: Raw logits.

        Returns:
            Calibrated probabilities.
        """
        if not self._fitted:
            raise RuntimeError("TemperatureScaler not fitted.")
        scaled = np.asarray(logits, dtype=float) / self.temperature
        return 1 / (1 + np.exp(-scaled))
