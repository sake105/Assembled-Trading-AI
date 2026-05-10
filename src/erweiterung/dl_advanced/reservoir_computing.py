"""Reservoir-Computing / Echo-State-Network — chaotic time-series forecast.

Reference
---------
Jaeger, H. (2001). The "echo state" approach to analysing and training
recurrent neural networks.

Idee
----
Random RNN with **fixed** weights. Only the readout-layer is trained
(Ridge-Regression). Drastisch schneller als BPTT-LSTM und oft konkurrenzfähig
auf chaotic series (Lorenz, finance returns).

Implementation: pure numpy — kein Deep-Learning-Framework nötig.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ESNConfig:
    n_reservoir: int = 200
    spectral_radius: float = 0.95
    sparsity: float = 0.10  # fraction of non-zero weights
    leak_rate: float = 0.3
    input_scaling: float = 1.0
    ridge_alpha: float = 1e-4
    seed: int = 42


def fit_predict_esn(
    train_series: np.ndarray,
    test_series: np.ndarray | None = None,
    horizon: int = 1,
    config: ESNConfig | None = None,
) -> np.ndarray:
    """Train an Echo State Network and forecast.

    Args:
        train_series: 1-D training series.
        test_series: optional held-out series for one-step-ahead.
        horizon: forecast horizon.
        config: ESNConfig.

    Returns:
        Array of forecasts of length len(test_series) (or `horizon`).
    """
    cfg = config or ESNConfig()
    rng = np.random.default_rng(cfg.seed)
    N = cfg.n_reservoir

    # Reservoir matrix W
    W = rng.standard_normal((N, N))
    mask = rng.random((N, N)) > cfg.sparsity
    W[mask] = 0
    eigvals = np.linalg.eigvals(W)
    rho = max(abs(eigvals))
    if rho > 0:
        W = W * (cfg.spectral_radius / rho)
    # Input matrix
    Win = rng.standard_normal((N, 1)) * cfg.input_scaling
    # Bias
    bias = rng.uniform(-0.1, 0.1, size=(N,))

    train = np.asarray(train_series, dtype=float)
    train = train[~np.isnan(train)]
    if len(train) < 50:
        raise ValueError("need >= 50 train samples")

    # Drive reservoir
    h = np.zeros(N)
    states = np.zeros((len(train), N))
    for t, x_t in enumerate(train):
        h = (1 - cfg.leak_rate) * h + cfg.leak_rate * np.tanh(
            W @ h + Win.flatten() * x_t + bias
        )
        states[t] = h

    # Train ridge readout: target = next-step
    X = states[:-horizon]
    y = train[horizon:]
    A = X.T @ X + cfg.ridge_alpha * np.eye(N)
    Wout = np.linalg.solve(A, X.T @ y)

    # Predict
    if test_series is None:
        # Free-running on train end (last state)
        h_curr = states[-1].copy()
        forecasts = []
        x_curr = train[-1]
        for _ in range(horizon):
            h_curr = (1 - cfg.leak_rate) * h_curr + cfg.leak_rate * np.tanh(
                W @ h_curr + Win.flatten() * x_curr + bias
            )
            y_pred = float(h_curr @ Wout)
            forecasts.append(y_pred)
            x_curr = y_pred
        return np.array(forecasts)

    # Teacher-forced one-step-ahead on test
    test = np.asarray(test_series, dtype=float)
    h_curr = states[-1].copy()
    out = np.zeros(len(test))
    prev_x = train[-1]
    for t in range(len(test)):
        h_curr = (1 - cfg.leak_rate) * h_curr + cfg.leak_rate * np.tanh(
            W @ h_curr + Win.flatten() * prev_x + bias
        )
        out[t] = float(h_curr @ Wout)
        prev_x = test[t]
    return out


__all__ = ["ESNConfig", "fit_predict_esn"]
