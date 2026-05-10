"""Recursive Least Squares with Forgetting Factor.

Theorie
-------
Standard-OLS schätzt Beta auf einer fixed window. RLS aktualisiert Beta
**recursively** mit jedem neuen (x, y) — und kann durch Forgetting-Faktor
λ < 1 ältere Beobachtungen exponentiell entgewichten:

    P_t = (1/λ) (P_{t-1} - P_{t-1} x_t x_t' P_{t-1} / (λ + x_t' P_{t-1} x_t))
    K_t = P_{t-1} x_t / (λ + x_t' P_{t-1} x_t)
    β_t = β_{t-1} + K_t (y_t - x_t' β_{t-1})

Anwendung
---------
- Time-varying Beta in Faktor-Modellen
- Adaptive Hedge-Ratios in Pairs-Trading
- Online Kalman-Filter-Approximation
"""

from __future__ import annotations

import numpy as np


class RecursiveLeastSquares:
    """Online RLS-Regression mit Forgetting-Faktor."""

    def __init__(self, n_features: int, lam: float = 0.99, P0: float = 1e3):
        self.n_features = n_features
        self.lam = lam
        self.beta = np.zeros(n_features)
        self.P = P0 * np.eye(n_features)
        self.n_updates = 0

    def predict(self, x: np.ndarray) -> float:
        return float(self.beta @ x)

    def partial_fit(self, x: np.ndarray, y: float) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        Px = self.P @ x
        denom = self.lam + float(x @ Px)
        K = Px / denom
        err = y - float(self.beta @ x)
        self.beta = self.beta + K * err
        # Update covariance
        self.P = (self.P - np.outer(K, Px)) / self.lam
        self.n_updates += 1


def online_rls_predict(X: np.ndarray, y: np.ndarray, lam: float = 0.99) -> np.ndarray:
    """Run RLS over a stream, return per-step predictions (made BEFORE update)."""
    n, d = X.shape
    model = RecursiveLeastSquares(d, lam=lam)
    preds = np.zeros(n)
    for t in range(n):
        preds[t] = model.predict(X[t])
        model.partial_fit(X[t], y[t])
    return preds


__all__ = ["RecursiveLeastSquares", "online_rls_predict"]
