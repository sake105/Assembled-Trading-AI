"""Adaptive Conformal Inference — online alpha update (audit C2-031).

Standard split-conformal (qa/conformal.py) gives a **marginal**
coverage guarantee :math:`P(y \\in C(X)) \\geq 1 - \\alpha`, but the
guarantee only holds if the data are exchangeable. In live trading
the joint distribution drifts (regime changes, vol clustering), and
the empirical coverage of a frozen conformal interval drops below
the nominal level.

Gibbs & Candès (2021), *Adaptive Conformal Inference Under
Distribution Shift*, NeurIPS, fix this with a tiny online update:

.. math::

    \\alpha_{t+1} = \\alpha_t + \\gamma \\cdot (\\alpha - \\mathbf{1}\\{y_t \\notin C_t(X_t)\\})

If the most recent observation **missed** the interval, indicator=1,
so we *decrease* :math:`\\alpha_t` (wider future intervals). If we
covered it, indicator=0 and we *increase* :math:`\\alpha_t`
(narrower future intervals). Across long horizons the average
coverage converges to the target :math:`1 - \\alpha` regardless of
distribution drift, with only a single tunable step-size
:math:`\\gamma`.

This module implements the recursion and a thin wrapper that
maintains the running calibration-quantile so callers get prediction
intervals back without re-fitting anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_TARGET_ALPHA = 0.1
DEFAULT_GAMMA = 0.005
MIN_ALPHA = 1e-4
MAX_ALPHA = 1.0 - 1e-4


@dataclass
class ACIState:
    """Running state of an Adaptive Conformal Inference stream."""

    target_alpha: float = DEFAULT_TARGET_ALPHA
    gamma: float = DEFAULT_GAMMA
    current_alpha: float = DEFAULT_TARGET_ALPHA
    miss_count: int = 0
    total_count: int = 0
    cal_scores: np.ndarray = field(default_factory=lambda: np.array([]))

    def empirical_miss_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.miss_count / self.total_count


def init_aci(
    cal_scores: np.ndarray,
    *,
    target_alpha: float = DEFAULT_TARGET_ALPHA,
    gamma: float = DEFAULT_GAMMA,
) -> ACIState:
    """Initialize ACI state from a sorted-or-unsorted calibration array."""
    if not (0.0 < target_alpha < 1.0):
        raise ValueError(f"target_alpha must be in (0, 1), got {target_alpha}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    arr = np.asarray(cal_scores, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 1:
        raise ValueError("need at least 1 finite calibration score")
    arr.sort()
    return ACIState(
        target_alpha=target_alpha,
        gamma=gamma,
        current_alpha=target_alpha,
        cal_scores=arr,
    )


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The same ceil-(n+1)(1-alpha)/n quantile used in qa/conformal.py."""
    n = scores.size
    q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    return float(np.quantile(scores, q_level, method="higher"))


def current_half_width(state: ACIState) -> float:
    """Half-width corresponding to the current alpha."""
    return _conformal_quantile(state.cal_scores, state.current_alpha)


def update_aci(state: ACIState, *, y_true: float, y_pred: float) -> ACIState:
    """Apply the Gibbs-Candès online update for one new observation.

    Args:
        state: current ACI state.
        y_true: realized observation.
        y_pred: model's point prediction for this observation.

    Returns:
        The state object, mutated in place AND returned (for chaining).
    """
    score = float(abs(y_true - y_pred))
    q = _conformal_quantile(state.cal_scores, state.current_alpha)
    missed = score > q
    state.total_count += 1
    if missed:
        state.miss_count += 1
    indicator = 1.0 if missed else 0.0
    new_alpha = state.current_alpha + state.gamma * (state.target_alpha - indicator)
    state.current_alpha = float(np.clip(new_alpha, MIN_ALPHA, MAX_ALPHA))
    return state


__all__ = [
    "ACIState",
    "init_aci",
    "update_aci",
    "current_half_width",
    "DEFAULT_TARGET_ALPHA",
    "DEFAULT_GAMMA",
]
