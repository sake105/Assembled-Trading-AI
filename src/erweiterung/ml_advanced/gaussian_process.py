"""Gaussian-Process-Regression mit Composite-Kernels.

Theorie
-------
GP = nonparametrisches Bayesian-Modell für Regression. Posterior gibt
predicted mean + variance. Kernel-Wahl entscheidet Charakter:
- RBF: smooth functions.
- Matern: flexibler (ν=1.5 oder 2.5 für moderate Glättung).
- Periodic: cyclic patterns.
- White-Noise: explicit i.i.d. noise level.
- Composite (Sum/Product): kombiniere für komplexere Strukturen.

Anwendung
---------
- Yield-Curve-Interpolation mit Uncertainty
- Option-Pricing-Surface mit Sparse-Strikes
- Time-Series-Forecasting mit predictive intervals

Implementation
--------------
Pure NumPy. Marginal-Likelihood-Optimization via grid-search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class GPFit:
    X_train: np.ndarray
    y_train: np.ndarray
    kernel: Callable
    noise: float
    L: np.ndarray  # Cholesky of K
    alpha: np.ndarray  # K^-1 y


def rbf_kernel(length_scale: float = 1.0, variance: float = 1.0) -> Callable:
    """Squared-Exponential / RBF kernel."""

    def k(X1, X2):
        d2 = (
            np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        )
        return variance * np.exp(-0.5 * d2 / length_scale**2)

    return k


def matern_kernel(
    length_scale: float = 1.0, nu: float = 1.5, variance: float = 1.0
) -> Callable:
    """Matern kernel (ν = 1.5 or 2.5 standard)."""

    def k(X1, X2):
        d2 = (
            np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        )
        d = np.sqrt(np.maximum(d2, 0))
        if nu == 1.5:
            scale = np.sqrt(3) * d / length_scale
            return variance * (1 + scale) * np.exp(-scale)
        if nu == 2.5:
            scale = np.sqrt(5) * d / length_scale
            return variance * (1 + scale + scale**2 / 3) * np.exp(-scale)
        # General: would need K_ν special function — use ν=1.5 fallback
        scale = np.sqrt(3) * d / length_scale
        return variance * (1 + scale) * np.exp(-scale)

    return k


def periodic_kernel(
    period: float = 1.0, length_scale: float = 1.0, variance: float = 1.0
) -> Callable:
    """Periodic kernel — for cyclic phenomena."""

    def k(X1, X2):
        d = np.abs(X1[:, None, :] - X2[None, :, :]).sum(axis=2)
        return variance * np.exp(-2 * np.sin(np.pi * d / period) ** 2 / length_scale**2)

    return k


def fit_gp(
    X: np.ndarray, y: np.ndarray, kernel: Callable, noise: float = 1e-3
) -> GPFit:
    """Fit GP via Cholesky-decomp of K + σ²I.

    Args:
        X: training inputs (n, d).
        y: training targets (n,).
        kernel: kernel-function (callable).
        noise: σ² to add to diagonal.

    Returns:
        GPFit.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    K = kernel(X, X) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return GPFit(X_train=X, y_train=y, kernel=kernel, noise=noise, L=L, alpha=alpha)


def gp_predict(fit: GPFit, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GP posterior at test points.

    Returns:
        (mean, variance) for each test point.
    """
    X_test = np.asarray(X_test, dtype=float)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    K_s = fit.kernel(fit.X_train, X_test)
    K_ss = fit.kernel(X_test, X_test) + fit.noise * np.eye(len(X_test))
    mean = K_s.T @ fit.alpha
    v = np.linalg.solve(fit.L, K_s)
    var = np.diag(K_ss) - np.sum(v * v, axis=0)
    return mean, np.maximum(var, 1e-12)


def gp_marginal_log_likelihood(fit: GPFit) -> float:
    """log p(y|X, θ) — used for hyperparameter optimization."""
    n = len(fit.y_train)
    return float(
        -0.5 * fit.y_train @ fit.alpha
        - np.sum(np.log(np.diag(fit.L)))
        - 0.5 * n * np.log(2 * np.pi)
    )


def grid_search_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    length_scales: list[float],
    variances: list[float],
    noise_levels: list[float],
    kernel_type: str = "rbf",
) -> dict:
    """Find best GP-hyperparams via grid + log-marginal-likelihood maximization."""
    best_ll = -np.inf
    best = None
    for ls in length_scales:
        for v in variances:
            for n in noise_levels:
                if kernel_type == "rbf":
                    k = rbf_kernel(ls, v)
                elif kernel_type == "matern":
                    k = matern_kernel(ls, 1.5, v)
                else:
                    continue
                try:
                    fit = fit_gp(X, y, k, n)
                    ll = gp_marginal_log_likelihood(fit)
                    if ll > best_ll:
                        best_ll = ll
                        best = {
                            "length_scale": ls,
                            "variance": v,
                            "noise": n,
                            "log_likelihood": ll,
                            "kernel_type": kernel_type,
                        }
                except (np.linalg.LinAlgError, ValueError):
                    continue
    return best or {}


__all__ = [
    "GPFit",
    "rbf_kernel",
    "matern_kernel",
    "periodic_kernel",
    "fit_gp",
    "gp_predict",
    "gp_marginal_log_likelihood",
    "grid_search_hyperparams",
]
