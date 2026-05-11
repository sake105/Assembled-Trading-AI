"""Variational Autoencoder für Asset-Return-Kompression.

Theorie
-------
Encoder φ(x): x → (μ_z, log_σ²_z), latent z ~ N(μ_z, σ_z²).
Decoder ψ(z): z → x̂.

ELBO-Loss:
    L = E_q[log p(x|z)] − KL(q(z|x) || p(z))

mit p(z) = N(0, I) prior.

Anwendung
---------
- Asset-Return-Embedding (Dimensionality-Reduction)
- Anomaly-Detection via Reconstruction-Loss
- Synthetic-Data-Generation (Sample z ~ N(0,I), decode)

Implementation
--------------
PyTorch — lazy-import. Fällt auf RuntimeError wenn torch nicht installiert.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VAEConfig:
    input_dim: int
    latent_dim: int = 4
    hidden_dim: int = 32
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    kl_weight: float = 1.0


def _import_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as F  # type: ignore

        return torch, nn, F
    except ImportError as e:
        raise RuntimeError("torch required") from e


def build_vae(config: VAEConfig):
    torch, nn, F = _import_torch()

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
            self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.mu = nn.Linear(config.hidden_dim, config.latent_dim)
            self.log_var = nn.Linear(config.hidden_dim, config.latent_dim)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return self.mu(h), self.log_var(h)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(config.latent_dim, config.hidden_dim)
            self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.out = nn.Linear(config.hidden_dim, config.input_dim)

        def forward(self, z):
            h = F.relu(self.fc1(z))
            h = F.relu(self.fc2(h))
            return self.out(h)

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def reparametrize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, log_var = self.encoder(x)
            z = self.reparametrize(mu, log_var)
            return self.decoder(z), mu, log_var

    return VAE()


def train_vae(
    returns_panel: pd.DataFrame,
    config: VAEConfig | None = None,
    verbose: bool = False,
):
    """Train VAE on multi-asset returns.

    Args:
        returns_panel: DataFrame (T, N) — T days × N assets.
        config: VAEConfig.

    Returns:
        (model, scaler_mean, scaler_std, training_losses).
    """
    torch, nn, F = _import_torch()
    if config is None:
        config = VAEConfig(input_dim=returns_panel.shape[1])

    arr = returns_panel.dropna().values.astype(np.float32)
    if len(arr) < 50:
        raise ValueError("not enough data")
    mu_x = arr.mean(axis=0)
    sd_x = arr.std(axis=0) + 1e-9
    X = (arr - mu_x) / sd_x

    model = build_vae(config)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    X_t = torch.from_numpy(X)
    losses = []
    for epoch in range(config.epochs):
        perm = torch.randperm(len(X_t))
        ep_loss = 0.0
        for i in range(0, len(X_t), config.batch_size):
            idx = perm[i : i + config.batch_size]
            xb = X_t[idx]
            optim.zero_grad()
            recon, mu, log_var = model(xb)
            recon_loss = F.mse_loss(recon, xb, reduction="sum")
            kl = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var))
            loss = recon_loss + config.kl_weight * kl
            loss.backward()
            optim.step()
            ep_loss += loss.item()
        losses.append(ep_loss / len(X_t))
    return model, mu_x, sd_x, losses


def vae_anomaly_score(
    model, returns_panel: pd.DataFrame, mu_x: np.ndarray, sd_x: np.ndarray
) -> pd.Series:
    """Reconstruction-error per timepoint."""
    torch, _, _ = _import_torch()
    arr = (returns_panel.values - mu_x) / sd_x
    X_t = torch.from_numpy(arr.astype(np.float32))
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(X_t)
    err = ((arr - recon.numpy()) ** 2).mean(axis=1)
    return pd.Series(err, index=returns_panel.index, name="vae_anomaly_score")


def vae_sample_synthetic(
    model, mu_x: np.ndarray, sd_x: np.ndarray, n_samples: int = 100
) -> np.ndarray:
    """Generate synthetic returns by sampling z ~ N(0, I) → decode."""
    torch, _, _ = _import_torch()
    config_latent = model.decoder.fc1.in_features  # type: ignore
    z = torch.randn(n_samples, config_latent)
    model.eval()
    with torch.no_grad():
        x_recon = model.decoder(z).numpy()
    return x_recon * sd_x + mu_x


__all__ = [
    "VAEConfig",
    "build_vae",
    "train_vae",
    "vae_anomaly_score",
    "vae_sample_synthetic",
]
