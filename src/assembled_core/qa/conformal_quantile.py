"""Conformalized Quantile Regression — CQR (audit C2-032).

Romano, Patterson & Candès (2019), *Conformalized Quantile
Regression*, NeurIPS. Standard ICP (qa/conformal.py) gives symmetric
intervals (``pred ± q``) which is wasteful when the underlying error
distribution is heteroscedastic or skewed. CQR uses a *quantile*
regressor to produce asymmetric raw bands :math:`[q_{lo}, q_{hi}]`
and a conformity score of the form

.. math::

    s_i = \\max( q_{lo,i} - y_i,\\ y_i - q_{hi,i} )

The calibrated interval at level :math:`1 - \\alpha` is

.. math::

    C_\\alpha(X) = [\\,\\hat q_{lo}(X) - Q,\\ \\hat q_{hi}(X) + Q\\,]

where :math:`Q` is the ceil-(n+1)(1-alpha)/n quantile of the
calibration scores. The interval inherits the asymmetry of the
quantile regressor while keeping ICP's distribution-free coverage
guarantee.

This module is model-agnostic: the caller passes a quantile-regressor
that exposes ``.predict_quantiles(X) -> (q_lo, q_hi)`` arrays.
``sklearn.ensemble.GradientBoostingRegressor(loss="quantile")`` fits
this shape after a trivial two-model wrapper; we provide a tiny
helper for that pairing but do not import sklearn at module load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class _QuantileRegressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...

    def predict_quantiles(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class CQRIntervals:
    q_lo: np.ndarray
    q_hi: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    Q: float
    alpha: float
    n_calibration: int


def fit_cqr(
    model: _QuantileRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    calibration_frac: float = 0.2,
    seed: int = 42,
) -> tuple[_QuantileRegressor, np.ndarray]:
    """Split-train + fit + compute CQR non-conformity scores.

    Returns the fitted ``model`` and the sorted calibration scores.
    Pass both to :func:`predict_with_intervals` for inference.
    """
    if calibration_frac <= 0 or calibration_frac >= 1:
        raise ValueError(f"calibration_frac must be in (0, 1), got {calibration_frac}")

    n = len(X_train)
    n_cal = max(20, int(np.ceil(n * calibration_frac)))
    if n_cal >= n:
        raise ValueError(
            f"calibration_frac={calibration_frac} leaves no proper-train rows"
        )
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cal_idx, train_idx = idx[:n_cal], idx[n_cal:]

    model.fit(X_train[train_idx], y_train[train_idx])
    q_lo_cal, q_hi_cal = model.predict_quantiles(X_train[cal_idx])
    q_lo_cal = np.asarray(q_lo_cal, dtype=float)
    q_hi_cal = np.asarray(q_hi_cal, dtype=float)
    y_cal = np.asarray(y_train[cal_idx], dtype=float)
    cal_scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    cal_scores.sort()
    return model, cal_scores


def predict_with_intervals(
    model: _QuantileRegressor,
    cal_scores: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float = 0.1,
) -> CQRIntervals:
    """Apply the calibrated CQR model to ``X_test``."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    q_lo, q_hi = model.predict_quantiles(X_test)
    q_lo = np.asarray(q_lo, dtype=float)
    q_hi = np.asarray(q_hi, dtype=float)
    n = len(cal_scores)
    q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    Q = float(np.quantile(cal_scores, q_level, method="higher"))
    return CQRIntervals(
        q_lo=q_lo,
        q_hi=q_hi,
        lower=q_lo - Q,
        upper=q_hi + Q,
        Q=Q,
        alpha=alpha,
        n_calibration=n,
    )


__all__ = ["CQRIntervals", "fit_cqr", "predict_with_intervals"]
