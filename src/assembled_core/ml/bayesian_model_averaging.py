"""Bayesian Model Averaging — robust ensemble weights (audit C2-058).

Stacking-Ensemble produces a single point-estimate weight per base model
via a second-stage regression. The audit (C2-058) flags this as fragile
under regime shifts: when one base model's training-error distribution
is non-stationary, the stacking weights are over-confident.

Bayesian Model Averaging (Hoeting, Madigan, Raftery, Volinsky 1999)
computes the posterior probability of each model being "correct" given
the data, and produces predictions as the **expectation across models**
weighted by these posteriors. The result is naturally robust: if one
model has a very strong recent track record relative to its
calibration-set performance, its weight rises smoothly rather than
discontinuously.

This module implements BMA over a collection of base models whose
out-of-sample predictions and realized targets are already available
(typical workflow: K base models cross-validated, the caller passes
the K OOS-prediction arrays and the matching y array).

Two scoring rules are supported:

1. **Gaussian likelihood** with shared σ̂² (estimated per model):

   .. math::

       w_m \\propto \\pi_m \\cdot \\prod_t \\mathcal{N}(y_t \\mid \\hat y_{m,t}, \\hat\\sigma_m^2)

2. **MSE-derived BIC approximation** (when likelihood is hard to
   compute directly):

   .. math::

       \\text{BIC}_m = T \\log(\\text{MSE}_m) + k_m \\log T
       \\quad w_m \\propto \\pi_m e^{-\\text{BIC}_m/2}

with priors :math:`\\pi_m` uniform by default. The output weights
sum to 1 and can be fed straight into a weighted-mean predictor.

Why BMA over stacking here? The audit's complaint about stacking is
the discontinuous winner-takes-most behavior when one base model
fits the calibration set marginally better. BMA's posterior is
smooth in calibration-set log-likelihood, so the same input data
yields more stable weight vectors across CV folds.

References
----------
- Hoeting et al. (1999), *Bayesian Model Averaging: A Tutorial*,
  Statistical Science 14(4), 382–417.
- Raftery, Madigan, Hoeting (1997), *BMA for Linear Regression Models*,
  JASA 92(437), 179–191.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


ScoringRule = Literal["gaussian", "bic"]


@dataclass(frozen=True)
class BMAResult:
    """Posterior model weights + diagnostic per-model log-evidence."""

    weights: np.ndarray
    log_evidence: np.ndarray
    scoring_rule: ScoringRule
    effective_n_models: float


def _gaussian_log_likelihood(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float]:
    """Plug-in Gaussian log-likelihood with estimated σ²."""
    resid = y - y_hat
    sigma2 = float(np.var(resid, ddof=1)) if resid.size > 1 else float(np.var(resid))
    sigma2 = max(sigma2, 1e-12)
    n = resid.size
    ll = (
        -0.5 * n * np.log(2.0 * np.pi * sigma2) - 0.5 * float(np.sum(resid**2)) / sigma2
    )
    return ll, sigma2


def bma_weights(
    oos_predictions: Sequence[np.ndarray],
    y_true: np.ndarray,
    *,
    priors: Sequence[float] | None = None,
    scoring_rule: ScoringRule = "gaussian",
    model_complexities: Sequence[int] | None = None,
) -> BMAResult:
    """Compute BMA posterior weights from a panel of base-model OOS predictions.

    Args:
        oos_predictions: sequence of length M, each entry an array of
            shape (T,) with the m-th model's out-of-sample predictions
            on the SAME T evaluation points.
        y_true: shape (T,) array of realized targets.
        priors: optional length-M prior probabilities π_m; defaults to
            uniform 1/M.
        scoring_rule: ``"gaussian"`` (plug-in Gaussian likelihood) or
            ``"bic"`` (BIC approximation — needs ``model_complexities``).
        model_complexities: required for ``"bic"`` — number of free
            parameters per model. Ignored for ``"gaussian"``.

    Returns:
        :class:`BMAResult` with normalized ``weights`` (sum to 1),
        the raw ``log_evidence`` per model, and an
        ``effective_n_models`` diagnostic = ``1 / Σ w_m²`` (1 = single
        model dominates, M = uniform weights).

    Raises:
        ValueError: malformed inputs.
    """
    if scoring_rule not in ("gaussian", "bic"):
        raise ValueError(f"unknown scoring_rule={scoring_rule!r}")
    M = len(oos_predictions)
    if M < 2:
        raise ValueError(f"need >= 2 base models, got {M}")
    y_arr = np.asarray(y_true, dtype=float).ravel()
    T = y_arr.size
    if T < 2:
        raise ValueError(f"need >= 2 evaluation points, got {T}")
    if any(np.asarray(p).ravel().size != T for p in oos_predictions):
        raise ValueError("each prediction array must have the same length as y_true")
    if priors is None:
        log_pi = np.full(M, -np.log(M), dtype=float)
    else:
        prior_arr = np.asarray(priors, dtype=float).ravel()
        if prior_arr.size != M or (prior_arr <= 0).any():
            raise ValueError("priors must be a positive M-length array")
        log_pi = np.log(prior_arr / prior_arr.sum())

    log_ev = np.empty(M, dtype=float)
    if scoring_rule == "gaussian":
        for m, p in enumerate(oos_predictions):
            ll, _ = _gaussian_log_likelihood(y_arr, np.asarray(p, dtype=float).ravel())
            log_ev[m] = ll + log_pi[m]
    else:  # bic
        if model_complexities is None:
            raise ValueError("scoring_rule='bic' requires model_complexities")
        k_arr = np.asarray(model_complexities, dtype=float).ravel()
        if k_arr.size != M or (k_arr < 0).any():
            raise ValueError("model_complexities must be a non-negative M-length array")
        for m, p in enumerate(oos_predictions):
            resid = y_arr - np.asarray(p, dtype=float).ravel()
            mse = float(np.mean(resid**2))
            mse = max(mse, 1e-12)
            bic_m = T * np.log(mse) + k_arr[m] * np.log(T)
            log_ev[m] = -0.5 * bic_m + log_pi[m]

    # Numerically-stable softmax for normalization.
    log_ev_shifted = log_ev - log_ev.max()
    w = np.exp(log_ev_shifted)
    w /= w.sum()

    eff_n = float(1.0 / float(np.sum(w**2)))

    return BMAResult(
        weights=w,
        log_evidence=log_ev,
        scoring_rule=scoring_rule,
        effective_n_models=eff_n,
    )


def bma_predict(
    weights: np.ndarray,
    model_predictions: Sequence[np.ndarray],
) -> np.ndarray:
    """Combine M base-model predictions into one BMA-weighted prediction."""
    w = np.asarray(weights, dtype=float).ravel()
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must sum to 1 (got sum=%.6f)" % w.sum())
    P = np.stack(
        [np.asarray(p, dtype=float).ravel() for p in model_predictions], axis=1
    )
    if P.shape[1] != w.size:
        raise ValueError("len(model_predictions) must match len(weights)")
    return P @ w


__all__ = ["BMAResult", "bma_weights", "bma_predict", "ScoringRule"]
