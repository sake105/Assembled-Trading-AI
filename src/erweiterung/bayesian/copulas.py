"""Copula-Modelle für Joint-Distribution-Modellierung.

Theorie
-------
Ein Copula trennt **Marginal-Verteilungen** und **Abhängigkeitsstruktur**.
    F(x_1, ..., x_n) = C(F_1(x_1), ..., F_n(x_n))

Copula-Familien
---------------
- **Gaussian**: symmetrisch, keine Tail-Dependence
- **Student-t**: symmetrisch, **gleiche Upper- und Lower-Tail-Dependence**
- **Clayton**: nur Lower-Tail-Dependence (Crisis-correlated downturn)
- **Gumbel**: nur Upper-Tail-Dependence

Anwendung in Trading
--------------------
- Stress-Tests: Sample aus Tail-Dependent-Copula simuliert Crisis-Szenarien
  realistischer als Multivariate-Normal
- Pairs-Trading: Copula-Residual-Spreads (statt Cointegration)
- Portfolio-Tail-Risk-Modellierung

Reference
---------
- Joe, H. (2014). *Dependence Modeling with Copulas*. CRC.
- McNeil/Frey/Embrechts (2005). *Quantitative Risk Management*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CopulaFit:
    family: str
    param: float
    n_samples: int
    log_likelihood: float


def empirical_cdf(x: np.ndarray) -> np.ndarray:
    """Empirical CDF — uniformly distributed margins."""
    n = len(x)
    ranks = np.argsort(np.argsort(x))
    return (ranks + 1) / (n + 1)


def kendalls_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau (rank correlation)."""
    n = len(x)
    if n < 3:
        return float("nan")
    # O(n²) — fine for moderate n
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    total = concordant + discordant
    return float((concordant - discordant) / total) if total > 0 else 0.0


def fit_clayton_copula(u: np.ndarray, v: np.ndarray) -> CopulaFit:
    """Clayton-Copula via Kendall's tau MoM.

    Clayton: τ = θ / (θ + 2)  =>  θ = 2τ / (1 - τ).
    """
    tau = kendalls_tau(u, v)
    if tau >= 1:
        theta = 50.0
    elif tau <= 0:
        theta = 1e-6
    else:
        theta = 2 * tau / (1 - tau)
    # log-lik: log c(u,v) = log(θ+1) -(θ+1)(log u + log v) - (1/θ + 2) log(u^-θ + v^-θ - 1)
    u_safe = np.clip(u, 1e-9, 1 - 1e-9)
    v_safe = np.clip(v, 1e-9, 1 - 1e-9)
    inner = u_safe**-theta + v_safe**-theta - 1
    inner = np.maximum(inner, 1e-12)
    log_c = (
        np.log(theta + 1)
        - (theta + 1) * (np.log(u_safe) + np.log(v_safe))
        - (1 / theta + 2) * np.log(inner)
    )
    return CopulaFit(
        family="Clayton",
        param=theta,
        n_samples=len(u),
        log_likelihood=float(log_c.sum()),
    )


def fit_gumbel_copula(u: np.ndarray, v: np.ndarray) -> CopulaFit:
    """Gumbel-Copula via Kendall's tau MoM.

    Gumbel: τ = 1 - 1/θ  =>  θ = 1/(1-τ).  θ ≥ 1.
    """
    tau = kendalls_tau(u, v)
    if tau <= 0:
        theta = 1.0
    else:
        theta = 1.0 / (1 - tau)
    return CopulaFit(
        family="Gumbel", param=theta, n_samples=len(u), log_likelihood=float("nan")
    )


def fit_gaussian_copula(u: np.ndarray, v: np.ndarray) -> CopulaFit:
    """Gaussian-Copula: ρ = sin(π/2 · τ)."""
    tau = kendalls_tau(u, v)
    rho = float(np.sin(np.pi / 2 * tau))
    rho = float(np.clip(rho, -0.999, 0.999))
    # log lik: complicated — return placeholder
    return CopulaFit(
        family="Gaussian", param=rho, n_samples=len(u), log_likelihood=float("nan")
    )


def sample_clayton(
    theta: float, n: int, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Bivariate Clayton-Copula Sampling via inverse CDF method."""
    rng = rng or np.random.default_rng()
    # Marshall-Olkin algorithm
    # gamma(1/θ) frailty
    if theta <= 0:
        u = rng.uniform(size=n)
        v = rng.uniform(size=n)
        return u, v
    # 1) sample x ~ Gamma(1/θ, 1)
    x = rng.gamma(shape=1 / theta, scale=1.0, size=n)
    # 2) sample u, v ~ exp(1)
    e1 = rng.exponential(size=n)
    e2 = rng.exponential(size=n)
    u = (1 + e1 / x) ** (-1 / theta)
    v = (1 + e2 / x) ** (-1 / theta)
    return u, v


def sample_gaussian_copula(rho: float, n: int, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1 - rho * rho) * rng.standard_normal(n)
    # to uniform: Φ
    u = 0.5 * (1 + np.tanh(z1 * np.sqrt(2 / np.pi)))
    v = 0.5 * (1 + np.tanh(z2 * np.sqrt(2 / np.pi)))
    return u, v


def upper_tail_dependence(family: str, param: float) -> float:
    """λ_U für die Familie."""
    if family == "Gumbel":
        return 2 - 2 ** (1 / param) if param >= 1 else 0.0
    if family == "Clayton":
        return 0.0
    if family == "Gaussian":
        return 0.0
    if family == "StudentT":
        # depends on df + rho — placeholder
        return float("nan")
    return float("nan")


def lower_tail_dependence(family: str, param: float) -> float:
    """λ_L für die Familie."""
    if family == "Clayton":
        return 2 ** (-1 / param) if param > 0 else 0.0
    if family == "Gumbel":
        return 0.0
    if family == "Gaussian":
        return 0.0
    return float("nan")


__all__ = [
    "CopulaFit",
    "empirical_cdf",
    "kendalls_tau",
    "fit_clayton_copula",
    "fit_gumbel_copula",
    "fit_gaussian_copula",
    "sample_clayton",
    "sample_gaussian_copula",
    "upper_tail_dependence",
    "lower_tail_dependence",
]
