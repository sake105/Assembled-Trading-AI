"""PatchTST — Time-Series Transformer mit Patch-Tokenization.

Reference
---------
Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers.* ICLR 2023.

Idee
----
Statt jeden einzelnen Time-Step als Token zu sehen, partitioniere die Sequenz in
overlapping Patches der Länge P. Das erhöht
- die effektive Sequenzlänge (kürzere Token-Sequence)
- die lokale Strukturerfassung
- die Trainings-Effizienz

Architektur
-----------
[Input (B, L, C)] → patching (B, n_patches, P*C)
                  → Linear-Embed (B, n_patches, d_model)
                  → Pos-Encoding
                  → Transformer-Encoder (n_layers × MHA + FFN)
                  → Flatten + Linear → Forecast (B, H)

Verwendung
----------
Forecast von Returns: input r_{t-L+1, ..., t} → predict r_{t+1, ..., t+H}.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PatchTSTConfig:
    seq_len: int = 96
    patch_len: int = 16
    stride: int = 8
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    forecast_horizon: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 30


def _import_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        return torch, nn
    except ImportError as e:
        raise RuntimeError("torch required: pip install torch") from e


def build_patch_tst(config: PatchTSTConfig, n_features: int = 1):
    """Construct a PatchTST nn.Module."""
    torch, nn = _import_torch()

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class PatchTST(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.n_patches = (config.seq_len - config.patch_len) // config.stride + 1
            self.patch_embed = nn.Linear(config.patch_len * n_features, config.d_model)
            self.pos_enc = PositionalEncoding(
                config.d_model, max_len=self.n_patches + 1
            )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=4 * config.d_model,
                dropout=config.dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.n_layers)
            self.head = nn.Linear(
                config.d_model * self.n_patches, config.forecast_horizon
            )

        def forward(self, x):  # x: (B, L, C)
            # patching
            B, L, C = x.shape
            patches = []
            for i in range(self.n_patches):
                s = i * self.config.stride
                e = s + self.config.patch_len
                patches.append(x[:, s:e, :].reshape(B, -1))
            patched = torch.stack(patches, dim=1)  # (B, n_patches, patch_len*C)
            tokens = self.patch_embed(patched)
            tokens = self.pos_enc(tokens)
            encoded = self.encoder(tokens)  # (B, n_patches, d_model)
            flat = encoded.reshape(B, -1)
            return self.head(flat)  # (B, H)

    return PatchTST()


def train_patch_tst(
    series: pd.Series,
    config: Optional[PatchTSTConfig] = None,
    val_frac: float = 0.2,
    verbose: bool = False,
):
    """Trainiere PatchTST auf einer Returns-Series.

    Args:
        series: pandas Series of returns.
        config: PatchTSTConfig.
        val_frac: validation-fraction (last X%).
        verbose: print epoch losses.

    Returns:
        Tuple (model, train_losses, val_losses).
    """
    torch, nn = _import_torch()
    config = config or PatchTSTConfig()

    arr = pd.Series(series).dropna().values.astype(np.float32)
    L = config.seq_len
    H = config.forecast_horizon
    if len(arr) < L + H + 50:
        raise ValueError(f"need >= {L + H + 50} samples")

    # build (X, y) windows
    X_list, y_list = [], []
    for i in range(len(arr) - L - H + 1):
        X_list.append(arr[i : i + L].reshape(L, 1))
        y_list.append(arr[i + L : i + L + H])
    X = np.stack(X_list)
    y = np.stack(y_list)

    n_train = int(len(X) * (1 - val_frac))
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_va, y_va = X[n_train:], y[n_train:]

    model = build_patch_tst(config, n_features=1)
    optim = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)
    X_va_t = torch.from_numpy(X_va)
    y_va_t = torch.from_numpy(y_va)

    train_losses, val_losses = [], []
    for epoch in range(config.epochs):
        model.train()
        # mini-batches
        perm = torch.randperm(len(X_tr_t))
        ep_loss = 0.0
        for i in range(0, len(X_tr_t), config.batch_size):
            idx = perm[i : i + config.batch_size]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            ep_loss += loss.item() * xb.size(0)
        ep_loss /= len(X_tr_t)
        # val
        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t)
            v_loss = loss_fn(val_pred, y_va_t).item()
        train_losses.append(ep_loss)
        val_losses.append(v_loss)
        if verbose:
            logger.info("[ep %d] train=%.6f val=%.6f", epoch, ep_loss, v_loss)
    return model, train_losses, val_losses


__all__ = ["PatchTSTConfig", "build_patch_tst", "train_patch_tst"]
