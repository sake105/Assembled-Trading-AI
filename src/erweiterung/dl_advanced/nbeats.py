"""N-BEATS — Neural Basis Expansion Analysis (Oreshkin et al. 2020).

Reference
---------
Oreshkin, B. et al. (2020). *N-BEATS: Neural basis expansion analysis for
interpretable time series forecasting.* ICLR 2020.

Idee
----
Stack of fully-connected residual blocks. Each block produces a "backcast"
(reconstructs input window) and a "forecast" (predicts horizon). Backcast is
subtracted before next block — enabling *residual learning over time-series*.

Vorteil
-------
- State-of-the-art generic time-series forecaster (auf M4-competition top-3)
- Pure MLP (kein RNN/Attention) -> sehr schnelles Training
- Interpretable basis expansion (Trend + Seasonality stacks möglich)

Architektur
-----------
[Input (B, L)] -> Block_1 -> backcast_1 + forecast_1
              -> Input - backcast_1 -> Block_2 -> backcast_2 + forecast_2
              -> ...
Final forecast = Σ forecast_i.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NBeatsConfig:
    seq_len: int = 60
    forecast_horizon: int = 5
    n_blocks: int = 3
    hidden_dim: int = 128
    n_layers_per_block: int = 4
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


def build_nbeats(config: NBeatsConfig):
    torch, nn = _import_torch()

    class NBeatsBlock(nn.Module):
        def __init__(
            self, input_size: int, theta_size: int, hidden_dim: int, n_layers: int
        ):
            super().__init__()
            layers = []
            in_dim = input_size
            for _ in range(n_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            self.fc = nn.Sequential(*layers)
            self.backcast_head = nn.Linear(hidden_dim, input_size)
            self.forecast_head = nn.Linear(hidden_dim, theta_size)

        def forward(self, x):
            h = self.fc(x)
            return self.backcast_head(h), self.forecast_head(h)

    class NBeats(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    NBeatsBlock(
                        input_size=config.seq_len,
                        theta_size=config.forecast_horizon,
                        hidden_dim=config.hidden_dim,
                        n_layers=config.n_layers_per_block,
                    )
                    for _ in range(config.n_blocks)
                ]
            )

        def forward(self, x):  # x: (B, L)
            residual = x
            forecast_total = torch.zeros(
                x.size(0), config.forecast_horizon, device=x.device
            )
            for block in self.blocks:
                backcast, forecast = block(residual)
                residual = residual - backcast
                forecast_total = forecast_total + forecast
            return forecast_total

    return NBeats()


def train_nbeats(
    series: pd.Series, config: NBeatsConfig | None = None, val_frac: float = 0.2
):
    """Train N-BEATS on a univariate series."""
    torch, nn = _import_torch()
    config = config or NBeatsConfig()
    arr = pd.Series(series).dropna().values.astype(np.float32)
    L = config.seq_len
    H = config.forecast_horizon
    if len(arr) < L + H + 50:
        raise ValueError("not enough data")
    X_list, y_list = [], []
    for i in range(len(arr) - L - H + 1):
        X_list.append(arr[i : i + L])
        y_list.append(arr[i + L : i + L + H])
    X = np.stack(X_list)
    y = np.stack(y_list)
    n_train = int(len(X) * (1 - val_frac))
    X_tr = torch.from_numpy(X[:n_train])
    y_tr = torch.from_numpy(y[:n_train])
    X_va = torch.from_numpy(X[n_train:])
    y_va = torch.from_numpy(y[n_train:])

    model = build_nbeats(config)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    train_losses, val_losses = [], []
    for _ in range(config.epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        ep = 0.0
        for i in range(0, len(X_tr), config.batch_size):
            idx = perm[i : i + config.batch_size]
            optim.zero_grad()
            pred = model(X_tr[idx])
            loss = loss_fn(pred, y_tr[idx])
            loss.backward()
            optim.step()
            ep += loss.item() * idx.numel()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(X_va), y_va).item()
        train_losses.append(ep / len(X_tr))
        val_losses.append(v)
    return model, train_losses, val_losses


__all__ = ["NBeatsConfig", "build_nbeats", "train_nbeats"]
