"""Probability/Prediction Calibration Monitoring.

Testet, ob Modell-Predictions KALIBRIERT sind: "wenn Modell 70% Confidence
vorhersagt, passiert der Event tatsächlich in 70% der Fälle?"

Standardtests:
- Reliability Diagram (Binning)
- Expected Calibration Error (ECE)
- Brier Score
- Platt Scaling / Isotonic Regression (Recalibration)

Anwendung:
- Meta-Labeler Calibration: wenn Meta-Confidence ≠ tatsächlich Hit-Rate →
  Position-Sizing ist systematisch falsch
- Monthly Check: Calibration-Drift aufdecken
- Recalibration: Wrapper um Primary-Modell der Predictions korrigiert

PIT-Invariante: Calibration wird auf historischen closed_at Records
gemessen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CalibrationReport:
    """Calibration-Metriken."""

    ece: float
    """Expected Calibration Error in [0, 1]. Niedriger ist besser. <0.05 = gut."""

    brier_score: float
    """Mean squared error. Niedriger ist besser."""

    n_bins: int
    bin_stats: list[dict] = field(default_factory=list)
    """Pro Bin: {confidence_mean, accuracy, count}"""

    n_samples: int = 0

    def is_well_calibrated(self, ece_threshold: float = 0.05) -> bool:
        return self.ece < ece_threshold


def compute_calibration(
    predictions: np.ndarray | pd.Series,
    actuals: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> CalibrationReport:
    """Berechnet ECE + Brier Score + Reliability-Diagramm-Daten.

    Args:
        predictions: Probabilities in [0, 1] oder Predictions in reeller Skala.
        actuals: Binäre Labels (0/1) oder kontinuierliche Werte.
        n_bins: Anzahl Konfidenz-Bins.

    Returns:
        CalibrationReport
    """
    pred = np.asarray(predictions, dtype=float)
    act = np.asarray(actuals, dtype=float)

    if len(pred) != len(act):
        raise ValueError(f"Länge mismatch: predictions={len(pred)}, actuals={len(act)}")

    # Auf [0, 1] clippen falls außerhalb
    pred_clipped = np.clip(pred, 0.0, 1.0)

    # Brier Score = mean (pred - actual)^2
    brier = float(np.mean((pred_clipped - act) ** 2))

    # ECE via Binning
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(pred_clipped)
    bin_stats: list[dict] = []

    for i in range(n_bins):
        mask = (pred_clipped >= bin_edges[i]) & (pred_clipped < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (pred_clipped >= bin_edges[i]) & (pred_clipped <= bin_edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            bin_stats.append({
                "bin_low": float(bin_edges[i]),
                "bin_high": float(bin_edges[i + 1]),
                "confidence_mean": 0.0,
                "accuracy": 0.0,
                "count": 0,
            })
            continue
        conf_mean = float(pred_clipped[mask].mean())
        acc = float(act[mask].mean())
        bin_stats.append({
            "bin_low": float(bin_edges[i]),
            "bin_high": float(bin_edges[i + 1]),
            "confidence_mean": conf_mean,
            "accuracy": acc,
            "count": count,
        })
        ece += (count / n) * abs(conf_mean - acc)

    return CalibrationReport(
        ece=float(ece),
        brier_score=brier,
        n_bins=n_bins,
        bin_stats=bin_stats,
        n_samples=n,
    )


class PlattCalibrator:
    """Platt Scaling: Logistic Regression als Recalibration-Schicht.

    Trainiert auf (raw_predictions, true_labels) und gibt kalibrierte
    Predictions zurück.
    """

    def __init__(self) -> None:
        self._model: object | None = None

    def fit(
        self,
        raw_predictions: np.ndarray | pd.Series,
        actuals: np.ndarray | pd.Series,
    ) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        pred = np.asarray(raw_predictions, dtype=float).reshape(-1, 1)
        act = np.asarray(actuals, dtype=int)

        if len(np.unique(act)) < 2:
            logger.warning("[PlattCal] Nur 1 Klasse — Calibration identität")
            return self

        self._model = LogisticRegression(max_iter=200)
        self._model.fit(pred, act)  # type: ignore[attr-defined]
        return self

    def transform(self, raw_predictions: np.ndarray | pd.Series) -> np.ndarray:
        """Wendet Platt-Scaling auf rohe Predictions an."""
        pred = np.asarray(raw_predictions, dtype=float).reshape(-1, 1)
        if self._model is None:
            return pred.ravel()
        return self._model.predict_proba(pred)[:, 1]  # type: ignore[attr-defined]


class IsotonicCalibrator:
    """Isotonic Regression: flexibler als Platt, aber braucht mehr Daten."""

    def __init__(self) -> None:
        self._model: object | None = None

    def fit(
        self,
        raw_predictions: np.ndarray | pd.Series,
        actuals: np.ndarray | pd.Series,
    ) -> "IsotonicCalibrator":
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            logger.warning("[IsoCal] sklearn fehlt — no-op")
            return self

        pred = np.asarray(raw_predictions, dtype=float)
        act = np.asarray(actuals, dtype=float)
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._model.fit(pred, act)  # type: ignore[attr-defined]
        return self

    def transform(self, raw_predictions: np.ndarray | pd.Series) -> np.ndarray:
        pred = np.asarray(raw_predictions, dtype=float)
        if self._model is None:
            return pred
        return self._model.transform(pred)  # type: ignore[attr-defined]


__all__ = [
    "CalibrationReport",
    "compute_calibration",
    "PlattCalibrator",
    "IsotonicCalibrator",
]
