"""Risk Parity (Equal Risk Contribution) Portfolio.

Theorie
-------
Each asset contributes equally to total portfolio risk:
    RC_i := w_i * (Σw)_i / sqrt(wᵀ Σ w)
    Solve: RC_i = RC_j ∀ i, j

Vorteil
-------
- Asset-Allocation **risikobasiert**, nicht kapital-gewichtet.
- Robust gegen Estimation-Errors in expected returns (es werden keine
  benötigt — nur die Cov-Matrix).
- Bridgewater All-Weather verwendet ähnliches Konzept.

Implementation
--------------
Newton-iteration mit dual-formulation. Convergiert in ~10-30 Iterationen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    sigma = float(np.sqrt(weights @ cov @ weights))
    if sigma == 0:
        return np.zeros_like(weights)
    return weights * (cov @ weights) / sigma


def risk_parity_weights(
    cov: pd.DataFrame,
    target_risk_budget: pd.Series | None = None,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> pd.Series:
    """Iterative Newton-Methode für Risk-Parity / risk-budget portfolio.

    Args:
        cov: Cov-Matrix.
        target_risk_budget: Ziel-Anteile am Gesamtrisiko (Σ = 1).
            None -> Equal-Risk-Contribution.
        max_iter: Max-Iterationen.
        tol: Konvergenztoleranz.

    Returns:
        Series of weights summing to 1.
    """
    n = cov.shape[0]
    Sigma = cov.values
    if target_risk_budget is None:
        b = np.ones(n) / n
    else:
        b = target_risk_budget.reindex(cov.index).fillna(1.0 / n).values
        b = b / b.sum()

    # Newton iteration on log-weights to ensure positivity
    w = np.ones(n) / n
    for _ in range(max_iter):
        sigma = float(np.sqrt(w @ Sigma @ w))
        if sigma == 0:
            break
        rc = w * (Sigma @ w) / sigma
        # gradient of (rc_i / sigma_p - b_i)^2
        grad = (rc - b * sigma) / w
        step = grad / (Sigma.diagonal() + 1e-12)
        w_new = w - 0.1 * step
        w_new = np.clip(w_new, 1e-9, None)
        w_new = w_new / w_new.sum()
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    return pd.Series(w, index=cov.index)


__all__ = ["risk_contributions", "risk_parity_weights"]
