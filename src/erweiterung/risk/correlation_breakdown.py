"""Correlation-Breakdown Detection — Crisis-Identification.

DUPLIKAT-HINWEIS
================
Mainline hat ``src/assembled_core/risk/correlation_guard.py`` (317 LoC) mit
Online-Detection und Risk-Action-Hooks. Für Production die mainline.

Diese Variante kombiniert APC + Eigenvalue-Concentration zu einem Composite-
Crisis-Score und ist forschungsorientiert.

Theorie
-------
In Stress-Phasen (z. B. März 2020, Sept 2008) brechen historisch niedrige
Korrelationen zusammen: alles fällt korreliert. Klassische Diversifikation
versagt genau dann, wenn man sie am meisten bräuchte ("correlation 1 in a crisis").

Indikatoren
-----------
1. **Average Pairwise Correlation** (APC): Σ_pairs(ρ) / N_pairs.
   Hohe APC = Crisis-Modus.
2. **Eigenvalue-Concentration**: Erster Eigenwert der Cov-Matrix dominiert.
3. **Mahalanobis-Distance** zum normalen Korrelations-Regime.

Anwendung
---------
Als Risk-Overlay: Wenn APC > Threshold => Reduce Diversification-Annahme
=> Reduce Total Exposure / Switch to Defensives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def average_pairwise_correlation(returns: pd.DataFrame) -> float:
    """APC = Mean(ρ_ij) für alle i ≠ j."""
    if returns.empty or returns.shape[1] < 2:
        return float("nan")
    corr = returns.corr()
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(corr.values[mask].mean())


def rolling_apc(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Rolling APC."""
    if returns.empty:
        return pd.Series(dtype=float)
    out = pd.Series(np.nan, index=returns.index)
    for end in range(window, len(returns) + 1):
        sub = returns.iloc[end - window : end]
        out.iloc[end - 1] = average_pairwise_correlation(sub)
    return out


def first_eigenvalue_share(returns: pd.DataFrame) -> float:
    """λ_1 / Σ λ_i — Anteil des ersten PCA-Faktors an Gesamtvarianz."""
    if returns.empty or returns.shape[1] < 2:
        return float("nan")
    cov = returns.cov().values
    try:
        evals = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return float("nan")
    total = float(evals.sum())
    if total <= 0:
        return float("nan")
    return float(evals.max() / total)


def crisis_score(
    returns: pd.DataFrame, window: int = 60, apc_threshold: float = 0.6
) -> pd.DataFrame:
    """Composite Crisis-Score: APC + Eigenvalue-Concentration.

    Returns:
        DataFrame [date, apc, lambda1_share, crisis_score].
        ``crisis_score ∈ [0, 1]``: 1 = full crisis.
    """
    if returns.empty:
        return pd.DataFrame()
    apc = rolling_apc(returns, window=window)
    lam = pd.Series(np.nan, index=returns.index)
    for end in range(window, len(returns) + 1):
        sub = returns.iloc[end - window : end]
        lam.iloc[end - 1] = first_eigenvalue_share(sub)

    # Sigmoid-mapping
    apc_score = 1 / (1 + np.exp(-(apc - apc_threshold) * 10))
    lam_score = 1 / (1 + np.exp(-(lam - 0.5) * 10))
    crisis = (0.6 * apc_score.fillna(0) + 0.4 * lam_score.fillna(0)).clip(0, 1)
    return pd.DataFrame({"apc": apc, "lambda1_share": lam, "crisis_score": crisis})


__all__ = [
    "average_pairwise_correlation",
    "rolling_apc",
    "first_eigenvalue_share",
    "crisis_score",
]
