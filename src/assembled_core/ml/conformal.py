"""Conformal Prediction für kalibrierte Unsicherheitsintervalle.

Standard ML gibt Punkt-Predictions. Conformal Prediction gibt Intervalle mit
strikter Coverage-Garantie (z.B. 90% aller echten Werte liegen im Intervall).

Anwendung für Position-Sizing:
- Schmales Intervall → hohe Konfidenz → größere Position
- Breites Intervall → unsicher → kleinere oder keine Position

Algorithmus — Split-Conformal Prediction:
1. Train Model auf Training-Set
2. Compute Residuals auf Calibration-Set (separat von Training)
3. α-Quantile der absoluten Residuals = Half-Width des Intervalls
4. Bei Inferenz: prediction ± quantile_half_width

PIT-Invariante:
- Calibration-Set muss zeitlich NACH Training-Set liegen (time-series split)
- Oder mindestens disjunkt sein
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConformalResult:
    """Conformal-Prediction-Ausgabe."""

    point_predictions: pd.Series
    lower_bounds: pd.Series
    upper_bounds: pd.Series
    half_width: float
    alpha: float
    """Miscoverage-Level (0.1 = 90%-Intervall)."""

    def interval_width(self) -> pd.Series:
        return self.upper_bounds - self.lower_bounds

    def confidence(self) -> pd.Series:
        """Normalisierte Konfidenz in [0, 1]: schmales Intervall → hoch."""
        widths = self.interval_width()
        max_w = widths.max()
        if max_w < 1e-9:
            return pd.Series(np.ones(len(widths)), index=widths.index)
        return 1.0 - (widths / max_w)


class SplitConformalPredictor:
    """Split-Conformal Prediction für beliebige sklearn-kompatible Modelle.

    Usage:
        predictor = SplitConformalPredictor(model=LGBMRegressor(), alpha=0.1)
        predictor.fit(X_train, y_train, X_calib, y_calib)
        result = predictor.predict(X_test)
        # result.lower_bounds / upper_bounds gelten mit 90% Coverage
    """

    def __init__(
        self,
        model: object,
        alpha: float = 0.1,
    ) -> None:
        """Args:
            model: sklearn-kompatibles Regressionsmodell (mit .fit/.predict)
            alpha: Miscoverage-Level (0.1 → 90%-Intervall, 0.05 → 95%)
        """
        self.model = model
        self.alpha = alpha
        self._residual_quantile: float | None = None

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_calib: np.ndarray | pd.DataFrame,
        y_calib: np.ndarray | pd.Series,
    ) -> "SplitConformalPredictor":
        """Trainiert Modell auf Train-Set und kalibriert auf Calib-Set."""
        X_train_vals = _to_numpy(X_train)
        y_train_vals = _to_numpy(y_train)
        X_calib_vals = _to_numpy(X_calib)
        y_calib_vals = _to_numpy(y_calib)

        self.model.fit(X_train_vals, y_train_vals)  # type: ignore[attr-defined]

        calib_preds = self.model.predict(X_calib_vals)  # type: ignore[attr-defined]
        residuals = np.abs(y_calib_vals - calib_preds)

        # (1 - alpha) Quantile mit Konformität-Korrektur: ceil((n+1)(1-alpha))/n
        n = len(residuals)
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self._residual_quantile = float(np.quantile(residuals, q_level))

        logger.info(
            "[Conformal] Kalibriert: α=%.2f → %.0f%%-Intervall, "
            "Half-Width=%.4f (n_calib=%d)",
            self.alpha, (1 - self.alpha) * 100, self._residual_quantile, n,
        )
        return self

    def predict(
        self,
        X_test: np.ndarray | pd.DataFrame,
        return_index: pd.Index | None = None,
    ) -> ConformalResult:
        """Predictions + Intervalle berechnen."""
        if self._residual_quantile is None:
            raise RuntimeError("Model not calibrated — call fit() first")

        X_test_vals = _to_numpy(X_test)
        preds = self.model.predict(X_test_vals)  # type: ignore[attr-defined]

        idx = return_index if return_index is not None else (
            X_test.index if hasattr(X_test, "index") else pd.RangeIndex(len(preds))
        )

        q = self._residual_quantile
        return ConformalResult(
            point_predictions=pd.Series(preds, index=idx, name="prediction"),
            lower_bounds=pd.Series(preds - q, index=idx, name="lower"),
            upper_bounds=pd.Series(preds + q, index=idx, name="upper"),
            half_width=q,
            alpha=self.alpha,
        )


def conformal_position_size(
    conformal_result: ConformalResult,
    max_position: float = 1.0,
    min_width_for_full_size: float | None = None,
) -> pd.Series:
    """Position-Sizing basierend auf Conformal-Intervall-Breite.

    Schmales Intervall → volle Position.
    Breites Intervall → kleinere Position.

    Args:
        conformal_result: Output aus SplitConformalPredictor.predict()
        max_position: Maximale absolute Position
        min_width_for_full_size: Intervall-Breite unter der volle Position gesetzt wird.
                                  None → Median der Breiten als Referenz.

    Returns:
        pd.Series mit Positions-Skalierung in [-max_position, max_position].
        Vorzeichen = sign(point_prediction).
    """
    widths = conformal_result.interval_width()
    preds = conformal_result.point_predictions

    ref_width = min_width_for_full_size if min_width_for_full_size is not None else float(widths.median())
    if ref_width < 1e-9:
        ref_width = 1e-9

    # Kleine Breiten → Scale nahe 1.0; große Breiten → Scale nahe 0
    scale = np.clip(ref_width / np.maximum(widths.values, 1e-9), 0.0, 1.0)

    positions = np.sign(preds.values) * scale * max_position
    return pd.Series(positions, index=preds.index, name="position_size")


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.values
    return np.asarray(x)


__all__ = [
    "ConformalResult",
    "SplitConformalPredictor",
    "conformal_position_size",
]
