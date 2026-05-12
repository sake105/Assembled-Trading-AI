"""Cross-Conformal Prediction — K-fold variant (audit C2-033).

Standard split-conformal (:mod:`qa.conformal`) needs to reserve a
calibration set that is never seen by the model fit. With small T this
is wasteful — the model under-fits because it cannot train on the
calibration rows.

Vovk (2015), *Cross-conformal predictors*, AISTATS, generalizes the
recipe to K folds: each fold is held out once as calibration, and the
final conformity scores are the **pooled** scores across all K folds.
The marginal coverage guarantee is preserved (under exchangeability),
and the bias toward shorter intervals from data starvation is much
smaller — useful when only a few hundred observations are available.

Recipe:

1. Partition the training data into K equal folds.
2. For each fold ``k`` in 1..K:
   - Fit a clone of the model on the other K-1 folds.
   - Compute non-conformity scores on fold ``k``.
3. Concatenate all K fold-score arrays into one calibration set.
4. Fit a *final* model on ALL training data (used for predictions).
5. Apply the standard quantile-of-(n+1)(1-α) over the pooled scores.

We implement the lightweight ``KFoldCV`` variant (no purging) and a
purged-CPCV variant that respects a horizon argument — handy for
financial series where samples are temporally dependent.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class _Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class CrossConformalIntervals:
    """Output of :func:`predict_with_intervals_cross`."""

    predictions: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    half_width: float
    alpha: float
    n_calibration: int
    n_folds: int


def fit_cross_conformal(
    model: _Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_folds: int = 5,
    purge_horizon: int = 0,
    seed: int = 42,
) -> tuple[_Model, np.ndarray]:
    """Fit K models, pool calibration scores, return final model + scores.

    Args:
        model: any object with ``.fit(X, y)`` and ``.predict(X)``. A
            deep-copy is taken for each fold's intermediate fit; the
            **final** model returned is fit on all training data.
        X_train, y_train: training arrays of shape (n, p) and (n,).
        n_folds: number of CV folds, K. Larger K = less data starvation
            but K× more fits.
        purge_horizon: if > 0, drop this many rows from each side of
            the held-out fold from the training data used to fit that
            fold's model. Useful for financial series with overlapping
            labels (Lopez de Prado 2018, Ch. 7).
        seed: RNG seed for fold shuffling.

    Returns:
        A 2-tuple of ``(final_fitted_model, calibration_scores)``,
        where ``calibration_scores`` is a sorted 1-D array.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if purge_horizon < 0:
        raise ValueError(f"purge_horizon must be >= 0, got {purge_horizon}")

    n = len(X_train)
    if n < n_folds:
        raise ValueError(f"n_train ({n}) must be >= n_folds ({n_folds})")

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    # When purge_horizon > 0 we keep contiguous time-ordered folds —
    # shuffling would scatter the calibration block across the series
    # and the purge would consume nearly every training row.
    if purge_horizon == 0:
        rng.shuffle(indices)
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[: n % n_folds] += 1
    fold_bounds = np.concatenate(([0], np.cumsum(fold_sizes)))

    all_scores: list[np.ndarray] = []
    for k in range(n_folds):
        cal_mask = np.zeros(n, dtype=bool)
        cal_mask[indices[fold_bounds[k] : fold_bounds[k + 1]]] = True
        train_mask = ~cal_mask
        if purge_horizon > 0:
            # Drop the `purge_horizon` rows adjacent to the calibration
            # block to break label-overlap leakage. We assume the input
            # arrays are in *time order* — caller's responsibility.
            cal_idx_sorted = np.where(cal_mask)[0]
            if cal_idx_sorted.size:
                lo = max(0, int(cal_idx_sorted.min()) - purge_horizon)
                hi = min(n, int(cal_idx_sorted.max()) + purge_horizon + 1)
                train_mask[lo:hi] = False
        if not train_mask.any():
            raise ValueError("purge_horizon left no training rows in a fold")

        fold_model: _Model = deepcopy(model)
        fold_model.fit(X_train[train_mask], y_train[train_mask])
        preds = np.asarray(fold_model.predict(X_train[cal_mask]))
        scores = np.abs(y_train[cal_mask] - preds)
        all_scores.append(scores)

    # Final model: fit on ALL training data.
    final_model: _Model = deepcopy(model)
    final_model.fit(X_train, y_train)

    pooled = np.concatenate(all_scores)
    pooled.sort()
    return final_model, pooled


def predict_with_intervals_cross(
    model: _Model,
    cal_scores: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float = 0.1,
    n_folds: int = 5,
) -> CrossConformalIntervals:
    """Apply pooled cross-conformal calibration to ``X_test`` predictions."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    preds = np.asarray(model.predict(X_test))
    n = len(cal_scores)
    q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    q = float(np.quantile(cal_scores, q_level, method="higher"))
    return CrossConformalIntervals(
        predictions=preds,
        lower=preds - q,
        upper=preds + q,
        half_width=q,
        alpha=alpha,
        n_calibration=n,
        n_folds=n_folds,
    )


__all__ = [
    "CrossConformalIntervals",
    "fit_cross_conformal",
    "predict_with_intervals_cross",
]
