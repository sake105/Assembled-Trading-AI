"""Particle Filter — Sequential Monte Carlo Bayesian Filtering.

Theorie
-------
Generalisierung von Kalman: kann **nicht-lineare und nicht-Gauss'sche**
State-Space-Modelle handeln.

Algorithmus (Bootstrap-Filter, Gordon et al. 1993)
--------------------------------------------------
1. Initialize N particles from prior.
2. For each observation y_t:
   a) Propagate particles via transition-density.
   b) Weight each particle via likelihood p(y_t | x_t^(i)).
   c) Resample particles proportional to weights (systematic resampling).
3. Posterior mean / quantiles from particle-empirical-distribution.

Anwendung
---------
- Stochastische-Volatility-Modelle (z. B. Heston) mit unbeobachteter Vola.
- Non-linear state-space tracking.
- Robust filtering bei outliers.

Reference
---------
- Gordon, N., Salmond, D. & Smith, A. (1993). Novel approach to non-linear/
  non-Gaussian Bayesian state estimation.
- Doucet, A., Godsill, S. & Andrieu, C. (2000). On Sequential Monte Carlo
  Sampling Methods for Bayesian Filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class ParticleFilter:
    n_particles: int
    transition: Callable  # f(x, t, rng) -> new x
    likelihood: Callable  # f(x, y, t) -> log p(y|x)
    particles: np.ndarray = field(default=None)
    log_weights: np.ndarray = field(default=None)

    def initialize(self, init_sampler: Callable, rng: np.random.Generator):
        self.particles = np.array([init_sampler(rng) for _ in range(self.n_particles)])
        self.log_weights = np.zeros(self.n_particles)

    def step(self, y: float, t: int, rng: np.random.Generator):
        # 1) Propagate
        for i in range(self.n_particles):
            self.particles[i] = self.transition(self.particles[i], t, rng)
        # 2) Weight via log-likelihood
        log_lik = np.array([self.likelihood(x, y, t) for x in self.particles])
        self.log_weights = log_lik
        # 3) Resample
        max_lw = np.max(self.log_weights)
        weights = np.exp(self.log_weights - max_lw)
        weights /= weights.sum()
        # Effective sample size
        ess = 1.0 / np.sum(weights**2)
        if ess < self.n_particles / 2:
            # Systematic resampling
            indices = self._systematic_resample(weights, rng)
            self.particles = self.particles[indices]
            self.log_weights = np.zeros(self.n_particles)

    def _systematic_resample(
        self, weights: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        n = len(weights)
        positions = (np.arange(n) + rng.uniform()) / n
        cumsum = np.cumsum(weights)
        indices = np.zeros(n, dtype=int)
        i, j = 0, 0
        while i < n:
            if positions[i] < cumsum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices

    def posterior_mean(self):
        return float(self.particles.mean())

    def posterior_quantile(self, q: float):
        return float(np.quantile(self.particles, q))


def stoch_vol_particle_filter_example(
    returns: np.ndarray,
    n_particles: int = 500,
    phi: float = 0.95,  # AR(1) persistence
    sigma_eta: float = 0.2,  # vol-of-vol
    mu_h: float = 0.0,  # long-run log-vola mean
    seed: int = 42,
):
    """Stochastic Volatility: r_t = exp(h_t/2) ε_t, h_t = μ + φ(h_{t-1}-μ) + η_t.

    Returns:
        Dict with smoothed_h, posterior mean of latent log-vola per timepoint.
    """
    rng = np.random.default_rng(seed)

    def transition(h, t, rng):
        return mu_h + phi * (h - mu_h) + sigma_eta * rng.standard_normal()

    def likelihood(h, y, t):
        # r ~ N(0, exp(h))
        var = np.exp(h)
        return -0.5 * (np.log(2 * np.pi * var) + y * y / var)

    def init_sampler(rng):
        return mu_h + rng.standard_normal() * sigma_eta

    pf = ParticleFilter(n_particles, transition, likelihood)
    pf.initialize(init_sampler, rng)
    posterior_h = []
    for t, y in enumerate(returns):
        pf.step(y, t, rng)
        posterior_h.append(pf.posterior_mean())
    return {
        "posterior_h": np.array(posterior_h),
        "exp_vola": np.exp(np.array(posterior_h) / 2),
    }


__all__ = ["ParticleFilter", "stoch_vol_particle_filter_example"]
