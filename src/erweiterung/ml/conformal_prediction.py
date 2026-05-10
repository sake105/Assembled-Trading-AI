"""Conformal Prediction — distribution-free uncertainty quantification.

Theorie
-------
Vovk et al. (2005), Lei et al. (2018, *JASA*): Conformal Prediction liefert
**Vorhersageintervalle mit garantierter Coverage** unter sehr schwachen Annahmen
(nur exchangeability, keine Verteilungsannahmen).

Variants
--------
1. **Split-Conformal** (Inductive Conformal): Train/Calib/Test split.
   Calib-Residuen → Quantile-Schätzung → Konfidenz-Intervall.
2. **Mondrian-Conformal**: Per-Group-Calibration (z. B. per Volatilitäts-Bucket).
3. **Adaptive-Conformal-Inference (ACI)**: Online-Update von α, robust gegen
   Distribution-Shift (Gibbs/Candès 2021).

Anwendung in Trading
--------------------
- Predictive-Intervalls statt Punkt-Vorhersagen.
- Confidence-Width als Sizing-Input (kleiner Width = volle Größe, breiter = klein).
- Conformal als Filter: Trade nur, wenn Confidence > X.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConformalIntervals:
    """Output eines Conformal-Predictors."""

    point_estimates: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    alpha: float

    def coverage_check(self, y_true: np.ndarray) -> float:
        """Empirische Coverage (sollte ~ 1 - alpha sein)."""
        in_interval = (y_true >= self.lower) & (y_true <= self.upper)
        return float(in_interval.mean())


def split_conformal_regression(
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
    calib_frac: float = 0.3,
    random_state: int = 42,
) -> ConformalIntervals:
    """Split-Conformal-Regression.

    Args:
        fit_predict: Callable ``f(X_train, y_train, X_eval) -> y_pred``.
            Repräsentiert beliebigen Regressor (Linear, GBM, NN, ...).
        X, y: Trainingsdaten.
        X_test: Testdaten.
        alpha: Miscoverage rate (typisch 0.1 für 90 %-Intervall).
        calib_frac: Fraction für Calibration-Set.
        random_state: Seed.

    Returns:
        ``ConformalIntervals`` mit lower/upper/Punkt für X_test.

    Garantie
    --------
    Bei i.i.d.-Daten: P(y_test ∈ [lower, upper]) >= 1 - alpha.
    """
    if len(X) < 30:
        raise ValueError("need >= 30 training samples")
    rng = np.random.default_rng(random_state)
    n = len(X)
    n_calib = max(int(n * calib_frac), 10)
    perm = rng.permutation(n)
    calib_idx = perm[:n_calib]
    train_idx = perm[n_calib:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_calib, y_calib = X[calib_idx], y[calib_idx]

    # Fit on training, predict on calib + test
    y_calib_pred = fit_predict(X_train, y_train, X_calib)
    y_test_pred = fit_predict(X_train, y_train, X_test)

    # Nonconformity scores: |y_calib - y_calib_pred|
    scores = np.abs(y_calib - y_calib_pred)
    # Quantile (1-alpha)*(n_calib+1)/n_calib for finite-sample correction
    q_level = np.ceil((1 - alpha) * (n_calib + 1)) / n_calib
    q_level = min(q_level, 1.0)
    q = float(np.quantile(scores, q_level, method="higher"))

    lower = y_test_pred - q
    upper = y_test_pred + q
    return ConformalIntervals(
        point_estimates=y_test_pred, lower=lower, upper=upper, alpha=alpha
    )


def adaptive_conformal_inference(
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    X_test_stream: list[np.ndarray],
    y_test_stream: list[float],
    target_alpha: float = 0.1,
    eta: float = 0.05,
) -> tuple[list[ConformalIntervals], np.ndarray]:
    """Adaptive Conformal Inference (Gibbs/Candès 2021).

    Online-update von alpha, sodass empirische Coverage = (1 - target_alpha)
    auch unter Distribution-Shift.

    Args:
        fit_predict: callable model fitter.
        X, y: initial training data.
        X_test_stream: list of (1, p) test arrays sequential.
        y_test_stream: actual y revealed after prediction.
        target_alpha: Ziel-Miscoverage.
        eta: lernrate für alpha-Update.

    Returns:
        (intervals_list, alpha_track).
    """
    alpha = target_alpha
    intervals_list: list[ConformalIntervals] = []
    alpha_track = []
    X_running = X.copy()
    y_running = y.copy()

    for t, X_t in enumerate(X_test_stream):
        ci = split_conformal_regression(
            fit_predict, X_running, y_running, X_t, alpha=alpha
        )
        intervals_list.append(ci)
        alpha_track.append(alpha)
        # update alpha based on hit/miss
        y_t = y_test_stream[t]
        in_interval = ci.lower[0] <= y_t <= ci.upper[0]
        err = (1 if not in_interval else 0) - target_alpha
        alpha = float(np.clip(alpha - eta * err, 0.001, 0.5))

        # Append to running training set
        X_running = np.vstack([X_running, X_t])
        y_running = np.append(y_running, y_t)

    return intervals_list, np.array(alpha_track)


def conformal_to_signal(
    intervals: ConformalIntervals, threshold_quantile: float = 0.5
) -> np.ndarray:
    """Wandle Intervalle in long/flat/short Signal um.

    Logik
    -----
    - Wenn ``lower > 0`` => starkes long-Signal (gesamtes Intervall positiv).
    - Wenn ``upper < 0`` => starkes short-Signal.
    - Sonst: 0.

    Optional: Skalierung mit Width (engere Intervalle = stärker).

    Returns:
        Array mit Werten in {-1, 0, +1}.
    """
    sig = np.zeros(len(intervals.point_estimates))
    sig[intervals.lower > 0] = +1
    sig[intervals.upper < 0] = -1
    return sig


__all__ = [
    "ConformalIntervals",
    "split_conformal_regression",
    "adaptive_conformal_inference",
    "conformal_to_signal",
]
