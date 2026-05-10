"""Bayesian Hyperparameter Optimization (Gaussian-Process based).

Theorie
-------
Statt Grid- oder Random-Search nutzt BO eine Surrogate-Function (GP) der
black-box-Loss-Funktion und wählt nächsten Test-Punkt via Acquisition (z.B.
Expected Improvement). Sample-efficient für teure Evaluations.

Reference
---------
- Snoek, J., Larochelle, H. & Adams, R. (2012). Practical Bayesian
  Optimization of Machine Learning Algorithms. NeurIPS 2012.

Implementation
--------------
Minimaler GP-Regressor mit Squared-Exponential-Kernel + EI-Acquisition.
NumPy-only — keine externen Libs. Für Production sklearn.GaussianProcessRegressor
oder Hyperopt verwenden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class HPOResult:
    best_params: np.ndarray
    best_value: float
    history_x: list[np.ndarray]
    history_y: list[float]


def _rbf_kernel(
    X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0
) -> np.ndarray:
    """Squared-exponential kernel."""
    dist_sq = (
        np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
    )
    return np.exp(-0.5 * dist_sq / length_scale**2)


def _gp_posterior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    length_scale: float = 1.0,
    noise: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """GP posterior mean + variance at X_test."""
    K = _rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(len(X_train))
    K_s = _rbf_kernel(X_train, X_test, length_scale)
    K_ss = _rbf_kernel(X_test, X_test, length_scale) + noise * np.eye(len(X_test))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mean = K_s.T @ alpha
    v = np.linalg.solve(L, K_s)
    var = np.diag(K_ss) - np.sum(v * v, axis=0)
    return mean, np.maximum(var, 1e-12)


def _expected_improvement(
    mean: np.ndarray, var: np.ndarray, y_best: float, xi: float = 0.01
) -> np.ndarray:
    """EI(x) = (μ(x) - y_best - ξ) Φ(z) + σ(x) φ(z), z = (μ - y_best - ξ) / σ.

    For minimization, we negate.
    """
    sigma = np.sqrt(np.maximum(var, 1e-12))
    imp = y_best - mean - xi  # minimization
    z = imp / sigma
    # Normal CDF + PDF
    from math import erf, exp, pi, sqrt

    cdf = np.array([0.5 * (1 + erf(zi / sqrt(2))) for zi in z])
    pdf = np.array([exp(-0.5 * zi * zi) / sqrt(2 * pi) for zi in z])
    return imp * cdf + sigma * pdf


def bayesian_optimize(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    n_iter: int = 30,
    n_init: int = 5,
    length_scale: float = 1.0,
    seed: int = 42,
) -> HPOResult:
    """Minimize black-box objective via Bayesian Optimization.

    Args:
        objective: callable f(x) -> scalar to minimize.
        bounds: list of (low, high) per dimension.
        n_iter: total evaluations.
        n_init: initial random samples before BO.
        length_scale: GP kernel length-scale.
        seed: RNG seed.

    Returns:
        HPOResult with best_params + history.
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])

    # Standardize to unit cube
    def _to_unit(x):
        return (x - lows) / (highs - lows)

    def _from_unit(u):
        return lows + u * (highs - lows)

    history_x: list[np.ndarray] = []
    history_y: list[float] = []

    # Initial random samples
    for _ in range(n_init):
        u = rng.uniform(0, 1, d)
        x = _from_unit(u)
        y = float(objective(x))
        history_x.append(x)
        history_y.append(y)

    for _ in range(n_iter - n_init):
        X_unit = np.array([_to_unit(x) for x in history_x])
        y_arr = np.array(history_y)
        y_best = float(y_arr.min())

        # Random candidates
        cand_u = rng.uniform(0, 1, (200, d))
        try:
            mean, var = _gp_posterior(X_unit, y_arr, cand_u, length_scale=length_scale)
        except np.linalg.LinAlgError:
            cand_u = rng.uniform(0, 1, (50, d))
            mean, var = _gp_posterior(
                X_unit, y_arr, cand_u, length_scale=length_scale + 0.1
            )
        ei = _expected_improvement(mean, var, y_best)
        next_idx = int(np.argmax(ei))
        next_x = _from_unit(cand_u[next_idx])
        next_y = float(objective(next_x))
        history_x.append(next_x)
        history_y.append(next_y)

    best_idx = int(np.argmin(history_y))
    return HPOResult(
        best_params=history_x[best_idx],
        best_value=history_y[best_idx],
        history_x=history_x,
        history_y=history_y,
    )


__all__ = ["HPOResult", "bayesian_optimize"]
