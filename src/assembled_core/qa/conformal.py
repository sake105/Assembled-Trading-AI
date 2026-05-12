# src/assembled_core/qa/conformal.py
"""Inductive Conformal Prediction (Vovk-Gammerman 2005) wrapper (audit C2-030).

ICP gives **distribution-free** prediction intervals with marginal
coverage guarantee 1 - alpha:

    P( y_true ∈ [pred - q, pred + q] ) >= 1 - alpha

The recipe is:

1. Split training data into ``proper_train`` + ``calibration``.
2. Fit the model on proper_train.
3. On calibration, compute per-sample non-conformity scores
   ``s_i = | y_i - y_hat_i |``.
4. The conformal quantile ``q`` is the ceil((n+1)(1-alpha)) / n quantile
   of the calibration scores.

For each test point we return the point prediction plus a symmetric
half-width ``q``. The audit (C2-030 / C2-034) wants this number wired
into position-sizing: ``size ∝ edge / half_width``.

The implementation is **model-agnostic**: pass any object with
``.fit(X, y) -> None`` and ``.predict(X) -> np.ndarray``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class _Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class ConformalIntervals:
    predictions: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    half_width: float
    alpha: float
    n_calibration: int


def fit_conformal(
    model: _Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    calibration_frac: float = 0.2,
    alpha: float = 0.1,
    seed: int = 42,
) -> tuple[_Model, np.ndarray]:
    """Split-train + fit + compute non-conformity scores.

    Returns the fitted ``model`` and a sorted array of calibration
    non-conformity scores. Pass both to ``predict_with_intervals`` for
    inference.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if calibration_frac <= 0 or calibration_frac >= 1:
        raise ValueError(f"calibration_frac must be in (0, 1), got {calibration_frac}")

    rng = np.random.default_rng(seed)
    n = len(X_train)
    n_cal = max(20, int(np.ceil(n * calibration_frac)))
    if n_cal >= n:
        raise ValueError(
            f"calibration_frac={calibration_frac} leaves no proper-train rows"
        )
    idx = np.arange(n)
    rng.shuffle(idx)
    cal_idx, train_idx = idx[:n_cal], idx[n_cal:]

    model.fit(X_train[train_idx], y_train[train_idx])
    cal_preds = np.asarray(model.predict(X_train[cal_idx]))
    cal_scores = np.abs(y_train[cal_idx] - cal_preds)
    cal_scores.sort()
    return model, cal_scores


def predict_with_intervals(
    model: _Model,
    cal_scores: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float = 0.1,
) -> ConformalIntervals:
    """Apply the calibrated model to ``X_test`` and return prediction intervals."""
    preds = np.asarray(model.predict(X_test))
    n = len(cal_scores)
    # The ceil((n+1)(1-alpha))/n quantile is the conformal recipe; use
    # numpy quantile with the canonical ``higher`` mode to avoid
    # interpolation between the two adjacent calibration scores.
    q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    q = float(np.quantile(cal_scores, q_level, method="higher"))
    return ConformalIntervals(
        predictions=preds,
        lower=preds - q,
        upper=preds + q,
        half_width=q,
        alpha=alpha,
        n_calibration=n,
    )


def conformal_size_factor(
    edge: float,
    half_width: float,
    *,
    floor: float = 0.0,
    cap: float = 1.0,
) -> float:
    """Size signal-to-noise ratio (audit C2-034).

    ``size = clip(edge / half_width, floor, cap)``. When ``half_width``
    is zero the signal-to-noise is undefined; the helper returns 0 in
    that case (audit's "pause when SNR<1" rule generalises to "pause
    whenever you cannot estimate SNR").
    """
    if not np.isfinite(half_width) or half_width <= 0:
        return 0.0
    snr = edge / half_width
    return float(np.clip(snr, floor, cap))


__all__ = [
    "ConformalIntervals",
    "fit_conformal",
    "predict_with_intervals",
    "conformal_size_factor",
]
