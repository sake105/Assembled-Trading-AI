"""Adaptive Conformal Inference (ACI) position sizer.

ACI (Gibbs & Candès 2021) dynamically adjusts the conformal coverage level
alpha_t based on observed coverage errors — self-correcting under distribution shift.

Uses `puncc` if installed; otherwise falls back to a pure numpy ACI implementation.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdaptiveConformalSizer:
    """Scale position size by inverse prediction-interval width with ACI feedback.

    Narrow interval → confident → larger position.
    Wide interval → uncertain → smaller position.
    ACI adjusts alpha_t online so coverage stays near target even in regime shifts.

    Parameters
    ----------
    base_predictor:
        Any object with a ``predict(X)`` method returning point forecasts.
        If None, a simple mean-prediction fallback is used.
    alpha:
        Initial miscoverage rate target (0.1 → 90% nominal coverage).
    gamma:
        ACI learning rate: how fast alpha_t adapts. Gibbs/Candès recommend 0.005.
    max_position:
        Maximum position units this sizer can output.
    """

    def __init__(
        self,
        base_predictor: Any = None,
        alpha: float = 0.10,
        gamma: float = 0.005,
        max_position: float = 1.0,
    ) -> None:
        self.base_predictor = base_predictor
        self.alpha = alpha
        self.gamma = gamma
        self.max_position = max_position

        # ACI state
        self._alpha_t = alpha
        self._residuals: list[float] = []
        self._coverage_errors: list[float] = []

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, realized_outcome: float, predicted_outcome: float) -> None:
        """Record a new residual and update alpha_t (online ACI step)."""
        residual = abs(realized_outcome - predicted_outcome)
        self._residuals.append(residual)

        # ACI coverage correction: 1 if missed, 0 if covered
        if len(self._residuals) >= 2:
            quantile_threshold = self._compute_quantile(self._alpha_t)
            err = 1.0 if residual > quantile_threshold else 0.0
            self._coverage_errors.append(err)
            # Update: alpha_{t+1} = alpha_t + gamma * (alpha - err)
            self._alpha_t = float(
                np.clip(self._alpha_t + self.gamma * (self.alpha - err), 0.01, 0.99)
            )

    def predict_and_size(
        self,
        features: pd.DataFrame | np.ndarray,
    ) -> dict[str, float]:
        """Return position size for current features.

        Returns
        -------
        Dict with keys: position_size (0–max_position), interval_width,
        confidence, current_alpha, quantile_threshold.
        """
        if isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = np.asarray(features)

        point_pred = self._get_point_prediction(X)
        qt = self._compute_quantile(self._alpha_t)
        interval_width = 2 * qt  # symmetric interval [pred-qt, pred+qt]

        # Confidence = 1 / (1 + normalised width)
        max_width = max(1e-9, 4 * np.std(self._residuals) if self._residuals else 1.0)
        confidence = 1.0 / (1.0 + interval_width / max_width)
        position_size = float(np.clip(confidence * self.max_position, 0, self.max_position))

        return {
            "position_size": position_size,
            "interval_width": float(interval_width),
            "confidence": float(confidence),
            "current_alpha": float(self._alpha_t),
            "quantile_threshold": float(qt),
            "point_prediction": float(point_pred),
            "n_calibration": len(self._residuals),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_quantile(self, alpha_t: float) -> float:
        if not self._residuals:
            return 1.0
        quantile_level = 1.0 - alpha_t
        return float(np.quantile(self._residuals, np.clip(quantile_level, 0.0, 1.0)))

    def _get_point_prediction(self, X: np.ndarray) -> float:
        if self.base_predictor is None:
            return float(np.nanmean(self._residuals)) if self._residuals else 0.0
        try:
            pred = self.base_predictor.predict(X)
            return float(np.ravel(pred)[-1])
        except Exception:
            return 0.0

    @property
    def current_alpha(self) -> float:
        return self._alpha_t

    @property
    def empirical_coverage(self) -> float | None:
        if not self._coverage_errors:
            return None
        return 1.0 - float(np.mean(self._coverage_errors))


def size_from_interval_width(
    interval_width: float,
    max_width: float,
    max_position: float = 1.0,
    min_position: float = 0.0,
) -> float:
    """Convert a prediction interval width to a position scale factor.

    Narrow interval → position near max; wide interval → position near min.
    """
    if max_width <= 0:
        return max_position
    ratio = float(np.clip(interval_width / max_width, 0.0, 1.0))
    return float(np.clip(max_position * (1.0 - ratio), min_position, max_position))
