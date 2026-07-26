# src/assembled_core/qa/adversarial_perturbation.py
"""Adversarial feature perturbation (audit C2-020).

For a given prediction function ``f(X) -> y_pred``, find the smallest
L-infinity perturbation ``delta`` that flips the SIGN of the prediction
at a sample point. If ``||delta||_inf < 0.5 * sigma`` of the relevant
feature, the model is considered ADVERSARIALLY UNSTABLE — a tiny
plausibly-noise-sized input shift completely changes the trading
decision.

This is binary-search on perturbation magnitude (no gradients required
— works for any black-box model including tree ensembles).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def min_perturbation_to_flip(
    predict_fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    *,
    feature_sigmas: np.ndarray | None = None,
    n_features_to_perturb: int | None = None,
    max_eps: float = 5.0,
    tol: float = 1e-3,
    max_iter: int = 40,
    seed: int = 42,
) -> dict[str, object]:
    """Find the smallest L-inf perturbation that flips the sign of f(x).

    Args:
        predict_fn: black-box model evaluator.
        x: input sample, shape (n_features,).
        feature_sigmas: per-feature standard deviations (for measuring
            "how plausibly noise-sized" the perturbation is). Defaults
            to a unit-sigma vector if not provided.
        n_features_to_perturb: top-K features to allow perturbing
            (greedy by gradient-free Bernoulli mask). None = all features.
        max_eps: upper bound on the search (in L-inf units of x).
        tol: binary-search tolerance.
        max_iter: bisection iteration cap.
        seed: RNG seed for the perturbation direction.

    Returns:
        Dict with keys:
            ``flipped``     — whether a flip was achieved within max_eps,
            ``eps``         — the L-inf magnitude of the smallest flipping
                              perturbation (NaN if not flipped),
            ``eps_in_sigma``— eps measured in feature-sigma units (per
                              the feature that moved most),
            ``unstable``    — True iff eps_in_sigma < 0.5 (audit C2-020 rule).
    """
    base = float(predict_fn(x))
    base_sign = np.sign(base)
    if base_sign == 0:
        return {
            "flipped": False,
            "eps": float("nan"),
            "eps_in_sigma": float("nan"),
            "unstable": False,
            "reason": "zero baseline prediction — no sign to flip",
        }

    n = len(x)
    sigmas = np.ones(n) if feature_sigmas is None else np.asarray(feature_sigmas)
    _ = np.random.default_rng(seed)  # reserved for future randomised-direction probes

    # Generate the perturbation direction. The audit asks for L-inf, so
    # the direction lives on the {-1, +1}^n hypercube. We pick the
    # direction that maximally moves the prediction (estimated via
    # one-shot finite difference at eps = max_eps / 100). Restricting to
    # the top-K features makes the test cheaper on wide inputs.
    probe = max_eps / 100.0
    grad_est = np.zeros(n)
    for i in range(n):
        x_perturbed = x.copy()
        x_perturbed[i] += probe
        grad_est[i] = predict_fn(x_perturbed) - base

    direction = -np.sign(
        grad_est * base_sign
    )  # push the prediction toward the opposite sign
    direction[direction == 0] = 1.0  # avoid all-zero direction
    if n_features_to_perturb is not None and n_features_to_perturb < n:
        # Keep only the top-K features by |gradient|
        k_idx = np.argsort(np.abs(grad_est))[-n_features_to_perturb:]
        mask = np.zeros(n, dtype=bool)
        mask[k_idx] = True
        direction = np.where(mask, direction, 0.0)

    def flipped_at(eps: float) -> bool:
        # predict_fn returns a scalar prediction; result is already used in
        # boolean contexts only — bool() is a no-op wrapper for mypy.
        return bool(np.sign(predict_fn(x + eps * direction)) != base_sign)

    # First check whether a flip is even possible inside max_eps.
    if not flipped_at(max_eps):
        return {
            "flipped": False,
            "eps": float("nan"),
            "eps_in_sigma": float("nan"),
            "unstable": False,
            "reason": f"no sign flip within eps={max_eps}",
        }

    # Bisection search.
    lo, hi = 0.0, max_eps
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if flipped_at(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    eps_min = hi
    moved = np.abs(eps_min * direction)
    # Convert to sigma units of the feature that moved the most.
    moved_in_sigma = moved / np.maximum(sigmas, 1e-12)
    eps_in_sigma = float(moved_in_sigma.max())

    return {
        "flipped": True,
        "eps": float(eps_min),
        "eps_in_sigma": eps_in_sigma,
        "unstable": eps_in_sigma < 0.5,
        "reason": "ok",
    }


__all__ = ["min_perturbation_to_flip"]
