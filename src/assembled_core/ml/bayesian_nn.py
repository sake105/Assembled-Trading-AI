"""Bayesian Neural Network via MC Dropout.

Provides epistemic uncertainty estimates via Monte Carlo Dropout:
- Train a standard MLP with dropout layers
- At inference, run N forward passes with dropout enabled
- Mean of predictions = point estimate
- Std of predictions = epistemic uncertainty

High uncertainty → out-of-distribution → reduce position size.

Requires: torch (optional). Falls back to sklearn MLPRegressor if unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass
class BNNPrediction:
    """Prediction with uncertainty from Bayesian NN."""
    mean: np.ndarray
    std: np.ndarray
    samples: np.ndarray | None = None  # (n_samples, n_predictions)

    @property
    def confidence(self) -> np.ndarray:
        """Inverse uncertainty: higher = more confident."""
        return 1.0 / (self.std + 1e-9)

    @property
    def sharpe_sizing(self) -> np.ndarray:
        """Optimal sizing ∝ mean / std (Sharpe of each prediction)."""
        return self.mean / (self.std + 1e-9)


class MCDropoutMLP:
    """MLP with MC Dropout for Bayesian uncertainty estimation.

    Uses PyTorch if available, otherwise falls back to sklearn.

    Args:
        hidden_sizes: Hidden layer sizes (default: [64, 32, 16]).
        dropout_rate: Dropout probability (default: 0.2).
        n_mc_samples: Number of MC forward passes at inference (default: 50).
        learning_rate: Learning rate (default: 1e-3).
        n_epochs: Training epochs (default: 100).
        batch_size: Mini-batch size (default: 64).
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.2,
        n_mc_samples: int = 50,
        learning_rate: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 64,
    ) -> None:
        self.hidden_sizes = hidden_sizes or [64, 32, 16]
        self.dropout_rate = dropout_rate
        self.n_mc_samples = n_mc_samples
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self._model = None
        self._fitted = False
        self._use_torch = TORCH_AVAILABLE

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MCDropoutMLP":
        """Train the BNN.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        if self._use_torch:
            self._fit_torch(X, y)
        else:
            self._fit_sklearn(X, y)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> BNNPrediction:
        """Predict with MC Dropout uncertainty estimation.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            BNNPrediction with mean, std, and optional samples.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)

        if self._use_torch:
            return self._predict_torch(X)
        return self._predict_sklearn(X)

    # ------- PyTorch implementation -------

    def _build_torch_model(self, n_features: int) -> "nn.Module":
        layers = []
        in_dim = n_features
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def _fit_torch(self, X: np.ndarray, y: np.ndarray) -> None:
        model = self._build_torch_model(X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y).unsqueeze(1)

        model.train()
        n = len(X)
        for epoch in range(self.n_epochs):
            indices = torch.randperm(n)
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                pred = model(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self._model = model
        logger.info("[BNN] Torch model trained for %d epochs", self.n_epochs)

    def _predict_torch(self, X: np.ndarray) -> BNNPrediction:
        model = self._model
        model.train()  # Keep dropout active

        X_t = torch.from_numpy(X)
        samples = []
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                pred = model(X_t).numpy().ravel()
                samples.append(pred)

        samples_arr = np.array(samples)  # (n_mc, n_samples)
        return BNNPrediction(
            mean=samples_arr.mean(axis=0),
            std=samples_arr.std(axis=0),
            samples=samples_arr,
        )

    # ------- sklearn fallback -------

    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.neural_network import MLPRegressor  # type: ignore

        self._model = MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_sizes),
            max_iter=self.n_epochs,
            random_state=42,
        )
        self._model.fit(X, y)
        self._X_train = X
        self._y_train = y
        logger.info("[BNN] sklearn fallback model trained")

    def _predict_sklearn(self, X: np.ndarray) -> BNNPrediction:
        """Approximate MC Dropout via bootstrap resampling."""
        rng = np.random.default_rng(42)
        n_train = len(self._X_train)
        samples = []

        for _ in range(min(self.n_mc_samples, 20)):
            idx = rng.choice(n_train, size=int(n_train * 0.8), replace=True)
            from sklearn.neural_network import MLPRegressor  # type: ignore
            m = MLPRegressor(
                hidden_layer_sizes=tuple(self.hidden_sizes),
                max_iter=max(50, self.n_epochs // 2),
                random_state=rng.integers(0, 10000),
            )
            m.fit(self._X_train[idx], self._y_train[idx])
            samples.append(m.predict(X))

        samples_arr = np.array(samples)
        return BNNPrediction(
            mean=samples_arr.mean(axis=0),
            std=samples_arr.std(axis=0),
            samples=samples_arr,
        )
