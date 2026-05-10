"""Black-Litterman Modell mit quantitativen Views.

Theorie
-------
Black-Litterman (1992) löst das Hauptproblem von Markowitz-MVO:
- Markowitz benötigt erwartete Returns als Input -> meist sehr ungenau.
- Kleine Schätzfehler => extreme Gewichte.

BL-Lösung
---------
1. Starte mit **markt-implizierten Returns** Π = δ·Σ·w_mkt (CAPM-Reverse-Optim).
2. Investor-Views (P, Q) repräsentieren: "Asset A wird Asset B um Q outperformen mit
   Confidence Ω".
3. Posterior-Returns blend: μ_BL = ((τΣ)⁻¹ + Pᵀ Ω⁻¹ P)⁻¹ ((τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q).

Vorteil: Stabilere Gewichte als rohe MVO; Views können quantitativ sein
(z. B. aus Signal-Modell).

Referenzen
----------
- Black, F. & Litterman, R. (1992). Global Portfolio Optimization. *FAJ* 48(5).
- Idzorek, T. (2007). A Step-by-Step Guide to the Black-Litterman Model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BLViews:
    """Investor-Views.

    Attributes:
        P: matrix (k_views × n_assets), each row a linear combination of assets.
        Q: vector (k_views,) of expected outperformance.
        omega: covariance matrix of view errors (k_views × k_views).
            Wenn None: Idzorek-Heuristik mit confidence levels.
        confidence: list (k_views,) ∈ [0, 1] für Idzorek-Heuristik.
    """

    P: np.ndarray
    Q: np.ndarray
    omega: Optional[np.ndarray] = None
    confidence: Optional[list[float]] = None


def market_implied_returns(
    cov: pd.DataFrame, market_weights: pd.Series, risk_aversion: float = 2.5
) -> pd.Series:
    """Π = δ * Σ * w_mkt (CAPM-Reverse Optim).

    Args:
        cov: Asset-Covariance-Matrix (annualisiert).
        market_weights: Markt-Cap-Weights, Σ = 1.
        risk_aversion: typischerweise 2-3 für Equity-Markets.

    Returns:
        Series mit market-implied Excess-Returns.
    """
    pi = risk_aversion * cov.values @ market_weights.reindex(cov.index).fillna(0).values
    return pd.Series(pi, index=cov.index)


def black_litterman_posterior(
    cov: pd.DataFrame,
    pi: pd.Series,
    views: BLViews,
    tau: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame]:
    """Berechne Posterior-Returns und Posterior-Cov nach Black-Litterman.

    Args:
        cov: Prior Asset-Covariance.
        pi: Prior Returns (e.g. from ``market_implied_returns``).
        views: Investor-Views.
        tau: Skalierung der Prior-Unsicherheit (Black-Litterman default 0.025–0.1).

    Returns:
        (mu_bl, sigma_bl): Posterior Returns + Posterior Cov.
    """
    n = len(pi)
    Sigma = cov.values
    P = views.P
    Q = views.Q
    if P.shape[1] != n:
        raise ValueError(f"P shape ({P.shape}) inconsistent with n_assets={n}")

    omega = views.omega
    if omega is None:
        if views.confidence is None:
            # default: omega = diag(P @ tau*Sigma @ P^T)
            inner = P @ (tau * Sigma) @ P.T
            omega = np.diag(np.diag(inner))
        else:
            # Idzorek heuristic
            tau_sigma_pt = tau * Sigma @ P.T
            omega = np.zeros((P.shape[0], P.shape[0]))
            for i, c in enumerate(views.confidence):
                if c <= 0:
                    omega[i, i] = 1e6  # essentially no info
                    continue
                tilt_var = float(P[i] @ tau_sigma_pt[:, i])
                # Idzorek: ω_i = (1/c - 1) * P_i @ tau*Σ @ P_iᵀ
                omega[i, i] = (1.0 / c - 1.0) * tilt_var

    tau_sigma_inv = np.linalg.pinv(tau * Sigma)
    omega_inv = np.linalg.pinv(omega)
    A = tau_sigma_inv + P.T @ omega_inv @ P
    A_inv = np.linalg.pinv(A)
    mu_bl = A_inv @ (tau_sigma_inv @ pi.values + P.T @ omega_inv @ Q)
    sigma_bl = Sigma + A_inv  # post-cov

    return (
        pd.Series(mu_bl, index=cov.index),
        pd.DataFrame(sigma_bl, index=cov.index, columns=cov.columns),
    )


def mean_variance_optimal_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = 2.5,
    long_only: bool = False,
    max_weight: float = 0.20,
) -> pd.Series:
    """Mean-Variance Optimization: w = (1/λ) * Σ⁻¹ * μ, with constraints.

    Args:
        mu: Erwartete Returns (z. B. aus BL-Posterior).
        cov: Cov-Matrix.
        risk_aversion: λ.
        long_only: clip negative weights to 0 + renormalize.
        max_weight: Cap pro Asset.

    Returns:
        Series of weights summing to 1.
    """
    sigma_inv = np.linalg.pinv(cov.values)
    w = (1.0 / risk_aversion) * sigma_inv @ mu.values
    w = pd.Series(w, index=cov.index)
    if long_only:
        w = w.clip(lower=0)
    # Iteratively cap and re-distribute the over-allocation to assets below cap
    for _ in range(50):
        if w.sum() <= 0:
            break
        w = w / w.sum()
        excess_mask = w > max_weight
        if not excess_mask.any():
            break
        excess_total = (w[excess_mask] - max_weight).sum()
        w[excess_mask] = max_weight
        # distribute excess_total proportionally to remaining (non-capped, positive) weights
        rem_mask = (w < max_weight) & (w > 0 if long_only else True)
        if rem_mask.sum() == 0:
            break
        rem_w = w[rem_mask]
        if rem_w.sum() <= 0:
            break
        w[rem_mask] = rem_w + excess_total * (rem_w / rem_w.sum())
    return w


__all__ = [
    "BLViews",
    "market_implied_returns",
    "black_litterman_posterior",
    "mean_variance_optimal_weights",
]
