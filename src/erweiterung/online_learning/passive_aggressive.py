"""Passive-Aggressive Regression (Crammer et al. 2006).

Theorie
-------
Online-Lernen: jedes neues (x, y) updated den Gewichtsvektor minimal so dass
der Loss verschwindet, sonst bleibt w unverändert ("passive" wenn ε klein,
"aggressive" wenn ε > epsilon).

PA-I (mit slack):
    τ_t = min(C, max(0, |y_t - w·x_t| - ε) / ||x_t||²)
    w_{t+1} = w_t + sign(y_t - w·x_t) · τ_t · x_t

Vorteil
-------
- Real-time-Updates, kein Re-Training nötig
- Robust gegen Concept-Drift (kontinuierliche Adaption)
- O(d) per Step

Reference
---------
Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S., Singer, Y. (2006).
Online passive-aggressive algorithms. *Journal of Machine Learning Research* 7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PAConfig:
    epsilon: float = 0.01
    C: float = 1.0  # aggressiveness


class PassiveAggressiveRegressor:
    """Online passive-aggressive regression."""

    def __init__(self, n_features: int, config: PAConfig | None = None):
        self.n_features = n_features
        self.config = config or PAConfig()
        self.w = np.zeros(n_features)
        self.bias = 0.0
        self.n_updates = 0

    def predict(self, x: np.ndarray) -> float:
        return float(self.w @ x + self.bias)

    def partial_fit(self, x: np.ndarray, y: float) -> None:
        x = np.asarray(x, dtype=float)
        pred = self.predict(x)
        err = y - pred
        loss = max(0.0, abs(err) - self.config.epsilon)
        if loss == 0:
            return
        x_norm_sq = float(x @ x) + 1.0  # +1 for bias
        if x_norm_sq <= 0:
            return
        # PA-I
        tau = min(self.config.C, loss / x_norm_sq)
        sign_err = 1 if err >= 0 else -1
        self.w += sign_err * tau * x
        self.bias += sign_err * tau
        self.n_updates += 1


def online_predict_sequence(
    X: np.ndarray, y: np.ndarray, config: PAConfig | None = None
) -> np.ndarray:
    """Run online PA-regressor over a stream, return per-step predictions
    (out-of-sample, made BEFORE seeing each y).
    """
    n, d = X.shape
    model = PassiveAggressiveRegressor(d, config)
    preds = np.zeros(n)
    for t in range(n):
        preds[t] = model.predict(X[t])
        model.partial_fit(X[t], y[t])
    return preds


__all__ = ["PAConfig", "PassiveAggressiveRegressor", "online_predict_sequence"]
