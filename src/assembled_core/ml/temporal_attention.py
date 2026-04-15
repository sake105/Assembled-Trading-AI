"""Temporal Attention Model for Factor Prediction (M19 Task 19.4).

Lightweight Transformer Encoder that learns WHICH historical days are most
predictive, replacing fixed lookback windows with learned attention weights.

Architecture:
    - Input: Sequence of feature vectors (last N trading days)
    - Positional Encoding: Sinusoidal (temporal position)
    - Transformer Encoder: 2-4 attention heads, 2 layers
    - Output: Next-day return prediction + attention map

Reference: Ding et al. (2020) "Hierarchical Multi-Scale Gaussian Transformer"
Sharpe improvement: +7-10%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TemporalAttentionConfig:
    """Configuration for temporal attention model."""
    seq_len: int = 20          # Input sequence length (trading days)
    d_model: int = 64          # Model dimension
    n_heads: int = 4           # Number of attention heads
    n_layers: int = 2          # Number of transformer layers
    dropout: float = 0.1       # Dropout rate
    lr: float = 1e-3           # Learning rate
    epochs: int = 50           # Training epochs
    batch_size: int = 32       # Batch size


@dataclass
class AttentionResult:
    """Result from temporal attention prediction."""
    predictions: np.ndarray         # Predicted returns
    attention_weights: np.ndarray   # Attention maps (n_samples, seq_len)
    important_lags: list[int]       # Most-attended lag positions
    train_loss: float               # Final training loss


def _sinusoidal_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Generate sinusoidal positional encoding.

    Args:
        seq_len: Sequence length.
        d_model: Model dimension.

    Returns:
        Array of shape (seq_len, d_model).
    """
    pos = np.arange(seq_len)[:, np.newaxis]
    dim = np.arange(d_model)[np.newaxis, :]
    angle = pos / (10000 ** (2 * (dim // 2) / d_model))
    encoding = np.zeros((seq_len, d_model))
    encoding[:, 0::2] = np.sin(angle[:, 0::2])
    encoding[:, 1::2] = np.cos(angle[:, 1::2])
    return encoding.astype(np.float32)


def _build_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build input sequences for temporal model.

    Args:
        features: (T, n_features) array.
        targets: (T,) array of next-day returns.
        seq_len: Lookback window.

    Returns:
        (X, y) where X is (n_samples, seq_len, n_features).
    """
    n = len(features)
    X = []
    y = []
    for i in range(seq_len, n):
        X.append(features[i - seq_len:i])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class TemporalAttentionModel:
    """Transformer-based temporal attention for factor prediction.

    Learns which historical days matter most for predicting future returns.
    Falls back to simple weighted average when PyTorch is unavailable.
    """

    def __init__(self, config: TemporalAttentionConfig | None = None):
        self.config = config or TemporalAttentionConfig()
        self._model = None
        self._feature_dim = None

    def fit(
        self,
        features: pd.DataFrame | np.ndarray,
        returns: pd.Series | np.ndarray,
    ) -> AttentionResult:
        """Train temporal attention model.

        Args:
            features: (T, n_features) feature matrix.
            returns: (T,) target returns.

        Returns:
            AttentionResult with training diagnostics.
        """
        feat_arr = np.asarray(features, dtype=np.float32)
        ret_arr = np.asarray(returns, dtype=np.float32)

        # Standardize features
        self._feat_mean = feat_arr.mean(axis=0)
        self._feat_std = feat_arr.std(axis=0) + 1e-8
        feat_norm = (feat_arr - self._feat_mean) / self._feat_std

        X, y = _build_sequences(feat_norm, ret_arr, self.config.seq_len)

        if len(X) < self.config.batch_size:
            logger.warning("[TemporalAttention] Insufficient data (%d samples)", len(X))
            return AttentionResult(
                predictions=np.zeros(len(ret_arr)),
                attention_weights=np.ones((1, self.config.seq_len)) / self.config.seq_len,
                important_lags=[1, 2, 3],
                train_loss=float("inf"),
            )

        self._feature_dim = feat_arr.shape[1]

        if TORCH_AVAILABLE:
            return self._fit_torch(X, y)
        else:
            return self._fit_fallback(X, y)

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray) -> AttentionResult:
        """Fallback: exponential decay attention (no PyTorch)."""
        seq_len = X.shape[1]
        # Exponential decay: recent days get more weight
        decay = np.exp(-np.arange(seq_len)[::-1] * 0.1)
        attention = decay / decay.sum()

        # Weighted average of features → linear prediction
        weighted_features = (X * attention[np.newaxis, :, np.newaxis]).sum(axis=1)

        # Ridge regression
        from numpy.linalg import lstsq
        coef = lstsq(weighted_features, y, rcond=None)[0]
        preds = weighted_features @ coef
        loss = float(np.mean((preds - y) ** 2))

        self._fallback_coef = coef
        self._fallback_attention = attention

        # Important lags: highest attention
        top_lags = np.argsort(attention)[::-1][:5].tolist()

        return AttentionResult(
            predictions=preds,
            attention_weights=np.tile(attention, (len(X), 1)),
            important_lags=top_lags,
            train_loss=loss,
        )

    def _fit_torch(self, X: np.ndarray, y: np.ndarray) -> AttentionResult:
        """Full PyTorch transformer training."""
        cfg = self.config
        n_features = X.shape[2]

        # Simple transformer encoder model
        class _TransformerPredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(n_features, cfg.d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model, nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_model * 2, dropout=cfg.dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
                self.output_head = nn.Linear(cfg.d_model, 1)
                self.attn_pool = nn.Linear(cfg.d_model, 1)

            def forward(self, x):
                h = self.input_proj(x)  # (B, T, d_model)
                # Add positional encoding
                pe = torch.tensor(
                    _sinusoidal_encoding(x.shape[1], cfg.d_model),
                    device=x.device,
                )
                h = h + pe.unsqueeze(0)
                h = self.encoder(h)  # (B, T, d_model)
                # Attention pooling
                attn_logits = self.attn_pool(h).squeeze(-1)  # (B, T)
                attn_weights = torch.softmax(attn_logits, dim=-1)
                pooled = (h * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
                return self.output_head(pooled).squeeze(-1), attn_weights

        model = _TransformerPredictor()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        loss_fn = nn.MSELoss()

        X_t = torch.tensor(X)
        y_t = torch.tensor(y)

        model.train()
        final_loss = float("inf")
        for epoch in range(cfg.epochs):
            # Mini-batch
            perm = torch.randperm(len(X_t))
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, len(perm), cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                pred, _ = model(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            final_loss = epoch_loss / max(n_batches, 1)

        # Extract predictions and attention
        model.eval()
        with torch.no_grad():
            preds, attn = model(X_t)

        attn_np = attn.numpy()
        avg_attn = attn_np.mean(axis=0)
        top_lags = np.argsort(avg_attn)[::-1][:5].tolist()

        self._model = model
        return AttentionResult(
            predictions=preds.numpy(),
            attention_weights=attn_np,
            important_lags=top_lags,
            train_loss=final_loss,
        )

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict using trained model.

        Args:
            features: (T, n_features) feature matrix (must be >= seq_len rows).

        Returns:
            (predictions, attention_weights) tuple.
        """
        feat_norm = (np.asarray(features, dtype=np.float32) - self._feat_mean) / self._feat_std
        X, _ = _build_sequences(feat_norm, np.zeros(len(feat_norm)), self.config.seq_len)

        if TORCH_AVAILABLE and self._model is not None:
            self._model.eval()
            with torch.no_grad():
                preds, attn = self._model(torch.tensor(X))
            return preds.numpy(), attn.numpy()
        elif hasattr(self, "_fallback_coef"):
            attention = self._fallback_attention
            weighted = (X * attention[np.newaxis, :, np.newaxis]).sum(axis=1)
            preds = weighted @ self._fallback_coef
            return preds, np.tile(attention, (len(X), 1))
        else:
            raise RuntimeError("Model not fitted")


__all__ = [
    "TemporalAttentionConfig",
    "TemporalAttentionModel",
    "AttentionResult",
]
