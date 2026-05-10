"""LSTM für Univariate-Returns-Forecasting.

Standard-LSTM mit Dropout + Layer-Norm. Robuste Baseline für Sequenz-Modelle,
schnell zu trainieren, Reference für PatchTST-Vergleich.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LSTMConfig:
    seq_len: int = 60
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    forecast_horizon: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 25


def _import_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        return torch, nn
    except ImportError as e:
        raise RuntimeError("torch required") from e


def build_lstm(config: LSTMConfig, n_features: int = 1):
    torch, nn = _import_torch()

    class LSTMReturn(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=config.hidden_dim,
                num_layers=config.n_layers,
                dropout=config.dropout if config.n_layers > 1 else 0,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(config.hidden_dim)
            self.head = nn.Linear(config.hidden_dim, config.forecast_horizon)

        def forward(self, x):
            out, _ = self.lstm(x)
            last = self.norm(out[:, -1, :])
            return self.head(last)

    return LSTMReturn()


def train_lstm(
    series: pd.Series, config: LSTMConfig | None = None, val_frac: float = 0.2
):
    torch, nn = _import_torch()
    config = config or LSTMConfig()
    arr = pd.Series(series).dropna().values.astype(np.float32)
    L = config.seq_len
    H = config.forecast_horizon
    if len(arr) < L + H + 50:
        raise ValueError("not enough data")
    X_list, y_list = [], []
    for i in range(len(arr) - L - H + 1):
        X_list.append(arr[i : i + L].reshape(L, 1))
        y_list.append(arr[i + L : i + L + H])
    X = np.stack(X_list)
    y = np.stack(y_list)
    n_train = int(len(X) * (1 - val_frac))
    X_tr, y_tr = torch.from_numpy(X[:n_train]), torch.from_numpy(y[:n_train])
    X_va, y_va = torch.from_numpy(X[n_train:]), torch.from_numpy(y[n_train:])
    model = build_lstm(config, n_features=1)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    train_losses, val_losses = [], []
    for _ in range(config.epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        ep_loss = 0.0
        for i in range(0, len(X_tr), config.batch_size):
            idx = perm[i : i + config.batch_size]
            optim.zero_grad()
            pred = model(X_tr[idx])
            loss = loss_fn(pred, y_tr[idx])
            loss.backward()
            optim.step()
            ep_loss += loss.item() * idx.numel()
        model.eval()
        with torch.no_grad():
            v_loss = loss_fn(model(X_va), y_va).item()
        train_losses.append(ep_loss / len(X_tr))
        val_losses.append(v_loss)
    return model, train_losses, val_losses


__all__ = ["LSTMConfig", "build_lstm", "train_lstm"]
