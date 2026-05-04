"""Conformal Prediction for position sizing via MAPIE.

From 11_FREE_MODELLE.md §11.7 and 00_MASTER_PLAN.md.
Uses EnbPI (Xu/Xie 2021) for time-series — removes iid assumption.

Integration:
  - Fit on return forecast residuals
  - Prediction-interval width as confidence proxy
  - Narrow band → full size, wide band → reduce size

Install: pip install mapie==0.9.2

Design:
  size_factor = 1 - (interval_width / max_observed_width).clip(0, 1)
  Combined with vol-targeting: final_size = base * vol_factor * conf_factor
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _try_mapie():
    try:
        from mapie.regression import MapieTimeSeriesRegressor

        return MapieTimeSeriesRegressor
    except ImportError:
        logger.warning("mapie not installed — pip install mapie==0.9.2")
        return None


class ConformalPositionSizer:
    """Wrap a return-forecast model with conformal prediction intervals.

    Produces position-size discounts based on forecast uncertainty.

    Usage:
        sizer = ConformalPositionSizer(base_model=lgb_model, alpha=0.1)
        sizer.fit(X_train, y_train)
        sizes = sizer.predict_size(X_test)  # float array in [0, 1]
    """

    def __init__(
        self,
        base_model: Any,
        alpha: float = 0.1,
        cv: str = "enbpi",
        max_width_quantile: float = 0.95,
    ):
        """
        Args:
            base_model: sklearn-compatible regressor (e.g. LightGBM)
            alpha: Miscoverage level (0.1 = 90% coverage intervals)
            cv: MAPIE cross-validation strategy ('enbpi' for time series)
            max_width_quantile: Quantile for normalizing interval widths
        """
        self.base_model = base_model
        self.alpha = alpha
        self.cv = cv
        self.max_width_quantile = max_width_quantile
        self._mapie = None
        self._max_width: float = 1.0

    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> "ConformalPositionSizer":
        """Fit the conformal predictor on training data."""
        MapieTimeSeriesRegressor = _try_mapie()
        if MapieTimeSeriesRegressor is None:
            logger.warning(
                "ConformalPositionSizer: MAPIE unavailable — no fitting done"
            )
            return self

        self._mapie = MapieTimeSeriesRegressor(
            estimator=self.base_model,
            cv=self.cv,
            alpha=self.alpha,
        )
        X_arr = np.asarray(X) if isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(y) if isinstance(y, pd.Series) else y
        self._mapie.fit(X_arr, y_arr)

        # Calibrate max width from training intervals
        _, pis = self._mapie.predict(X_arr, alpha=self.alpha)
        widths = pis[:, 1, 0] - pis[:, 0, 0]
        self._max_width = float(np.nanquantile(widths, self.max_width_quantile))
        if self._max_width <= 0:
            self._max_width = 1.0
        logger.info(
            "ConformalPositionSizer fitted. Max width (p%.0f): %.4f",
            self.max_width_quantile * 100,
            self._max_width,
        )
        return self

    def predict_intervals(
        self, X: np.ndarray | pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (y_pred, lower_bound, upper_bound) arrays."""
        if self._mapie is None:
            n = len(X)
            nans = np.full(n, np.nan)
            return nans, nans, nans

        X_arr = np.asarray(X) if isinstance(X, pd.DataFrame) else X
        y_pred, pis = self._mapie.predict(X_arr, alpha=self.alpha)
        lower = pis[:, 0, 0]
        upper = pis[:, 1, 0]
        return y_pred, lower, upper

    def predict_size(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return position-size factors in [0, 1] based on interval width.

        Narrow interval (high confidence) → size ≈ 1.0
        Wide interval (high uncertainty) → size ≈ 0.0
        """
        if self._mapie is None:
            return np.ones(len(X))

        _, lower, upper = self.predict_intervals(X)
        widths = np.clip(upper - lower, 0, None)
        normalized = np.clip(widths / self._max_width, 0, 1)
        size_factors = 1.0 - normalized
        return np.clip(size_factors, 0, 1)

    @property
    def is_fitted(self) -> bool:
        return self._mapie is not None


def conformal_size_factor(
    interval_width: float,
    max_width: float,
    min_factor: float = 0.1,
) -> float:
    """Scalar version of conformal size factor for single-sample use.

    Args:
        interval_width: Width of the prediction interval
        max_width: Reference maximum width (95th-percentile from training)
        min_factor: Floor on position size (default 0.1 = never fully zero)

    Returns:
        Size factor in [min_factor, 1.0].
    """
    if max_width <= 0:
        return 1.0
    normalized = float(np.clip(interval_width / max_width, 0, 1))
    return float(max(min_factor, 1.0 - normalized))


__all__ = [
    "ConformalPositionSizer",
    "conformal_size_factor",
]
