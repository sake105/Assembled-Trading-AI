"""Autoencoder für Multi-Asset-Return-Anomaly-Detection.

Idee
----
Trainiere AE so, dass er normale Cross-Section-Tagesreturns gut rekonstruiert.
Hohe Reconstruction-Errors = Anomaly = potenzielles Stress-Event.

Anwendung
---------
- Crisis-Score (komplementär zu Correlation-Breakdown).
- Outlier-Filter für Faktor-Trainings-Sets.

Architektur (Default)
---------------------
Input  (n_assets,) → Linear(64) → ReLU → Linear(16) → ReLU
                  → Linear(64) → ReLU → Linear(n_assets) → Output

Kompakter Bottleneck; Tied-Weights optional.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AEConfig:
    hidden_dim: int = 64
    bottleneck: int = 16
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 30


def _import_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        return torch, nn
    except ImportError as e:
        raise RuntimeError("torch required") from e


def train_autoencoder(returns_wide: pd.DataFrame, config: AEConfig | None = None):
    """Train an autoencoder on cross-sectional returns.

    Args:
        returns_wide: DataFrame (time × assets), one row = one day.
        config: AEConfig.

    Returns:
        (model, scaler_mean, scaler_std).
    """
    torch, nn = _import_torch()
    config = config or AEConfig()

    df = returns_wide.dropna()
    if df.empty:
        raise ValueError("empty returns")
    arr = df.values.astype(np.float32)
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0) + 1e-9
    arr_norm = (arr - mu) / sd
    n, d = arr_norm.shape

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.bottleneck),
                nn.ReLU(),
            )
            self.dec = nn.Sequential(
                nn.Linear(config.bottleneck, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, d),
            )

        def forward(self, x):
            return self.dec(self.enc(x))

    model = AE()
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    X_t = torch.from_numpy(arr_norm)

    for epoch in range(config.epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, config.batch_size):
            idx = perm[i : i + config.batch_size]
            xb = X_t[idx]
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, xb)
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0)
    return model, mu, sd


def reconstruction_error(
    model, returns_wide: pd.DataFrame, mu: np.ndarray, sd: np.ndarray
) -> pd.Series:
    """Compute per-day reconstruction MSE."""
    torch, _ = _import_torch()
    arr = (returns_wide.values - mu) / sd
    X_t = torch.from_numpy(arr.astype(np.float32))
    model.eval()
    with torch.no_grad():
        pred = model(X_t).numpy()
    err = ((arr - pred) ** 2).mean(axis=1)
    return pd.Series(err, index=returns_wide.index, name="reconstruction_error")


__all__ = ["AEConfig", "train_autoencoder", "reconstruction_error"]
