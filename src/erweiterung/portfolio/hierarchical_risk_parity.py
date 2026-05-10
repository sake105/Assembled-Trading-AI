"""Hierarchical Risk Parity (HRP) — Lopez de Prado, 2016.

DUPLIKAT-HINWEIS
================
Eine produktive Implementierung dieser Methode existiert bereits unter::

    src/assembled_core/portfolio/hierarchical_risk_parity.py

Die mainline-Version (313 LoC) nutzt scipy für hierarchisches Clustering und
bietet zusätzlich ``hrp_with_turnover_control`` und Vergleiche gegen Equal-
Weight. Für **Production** sollte die mainline-Version verwendet werden.

Diese Erweiterungs-Variante ist eine **NumPy-only-Forschungs-Implementierung**
(keine scipy-Abhängigkeit), bewusst kompakt für didaktische Klarheit. Sie
existiert weil:
1. Das ERWEITERUNG-Paket eine zero-extra-deps-Forschungs-Sandbox ist.
2. Die Demo- und Pipeline-Skripte deterministische Reproduzierbarkeit ohne
   scipy benötigen.

Theorie
-------
Klassische Mean-Variance-Optimization (Markowitz) ist instabil:
- Sample-Cov-Matrix singulär oder ill-conditioned bei N >~ T/2.
- Kleine Schätzfehler => extreme Gewichte.
- "Garbage-In-Garbage-Out".

HRP-Algorithmus (Lopez de Prado, *J. Portfolio Management* 2016):
1. Tree-Clustering (single linkage) der Korrelationsmatrix.
2. Quasi-Diagonal: rearrange covariance to put similar assets together.
3. Recursive Bisection: split tree, allocate inverse-variance, recurse.

Vorteil
-------
- Funktioniert ohne Matrix-Invertierung.
- Robust gegen Schätzfehler.
- Empirisch stabil out-of-sample (siehe Raffinot 2017, Kasa 2019).

Referenzen
----------
- Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform
  Out-of-Sample. *J. Portfolio Management* 42(4): 59-69.
- Raffinot, T. (2017). Hierarchical Clustering-Based Asset Allocation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Distance matrix: ``d_ij = sqrt((1 - ρ_ij) / 2)``."""
    d = ((1 - corr) / 2.0).clip(lower=0)
    return d.pow(0.5)


def _single_linkage_cluster(dist: pd.DataFrame) -> list[tuple[int, int, float]]:
    """Single-Linkage hierarchical clustering. Returns linkage list of (i, j, distance)."""
    n = dist.shape[0]
    distances = dist.values.copy()
    np.fill_diagonal(distances, np.inf)

    linkage_list: list[tuple[int, int, float]] = []
    next_id = n
    active = list(range(n))
    label_map = {i: i for i in range(n)}

    while len(active) > 1:
        # find min distance pair
        sub = np.array([[distances[i, j] for j in active] for i in active])
        idx = np.argmin(sub)
        i_idx, j_idx = idx // len(active), idx % len(active)
        i, j = active[i_idx], active[j_idx]
        d = distances[i, j]
        linkage_list.append((label_map[i], label_map[j], d))
        # merge: update distances row/col
        new_id = next_id
        next_id += 1
        for k in active:
            if k in (i, j):
                continue
            new_dist = min(distances[i, k], distances[j, k])
            distances[i, k] = new_dist
            distances[k, i] = new_dist
        label_map[i] = new_id
        active.remove(j)
    return linkage_list


def quasi_diag_order(linkage: list[tuple[int, int, float]], n: int) -> list[int]:
    """Reorder leaves into quasi-diagonal order following the cluster tree."""
    if not linkage:
        return list(range(n))
    # Build tree as dict
    tree: dict[int, tuple[int, int]] = {}
    for k, (a, b, _d) in enumerate(linkage):
        new_id = n + k
        tree[new_id] = (a, b)
    root = n + len(linkage) - 1

    order: list[int] = []

    def _walk(node: int) -> None:
        if node < n:
            order.append(node)
            return
        a, b = tree[node]
        _walk(a)
        _walk(b)

    _walk(root)
    return order


def _ivp_weights(cov: np.ndarray) -> np.ndarray:
    """Inverse-Variance-Portfolio: w_i ∝ 1 / σ_i²."""
    iv = 1.0 / np.diag(cov)
    iv = iv / iv.sum()
    return iv


def _cluster_var(cov: np.ndarray, indices: list[int]) -> float:
    """Variance of inverse-variance-weighted cluster."""
    sub = cov[np.ix_(indices, indices)]
    w = _ivp_weights(sub)
    return float(w @ sub @ w)


def hrp_weights(
    returns: pd.DataFrame,
    cov_method: str = "sample",
    shrinkage: float = 0.0,
) -> pd.Series:
    """Berechne HRP-Portfolio-Gewichte.

    Args:
        returns: DataFrame mit Index = date, Columns = Assets.
        cov_method: 'sample' | 'ledoit_wolf' (falls scikit-learn vorhanden).
        shrinkage: Manueller Shrinkage-Faktor [0, 1] gegen ``α·I + (1-α)·Σ``.

    Returns:
        Series of weights summing to 1.0, indexed by asset.
    """
    if returns.empty or returns.shape[1] < 2:
        return pd.Series(dtype=float)

    cov = _estimate_covariance(returns, method=cov_method, shrinkage=shrinkage)
    corr = cov.div(np.sqrt(np.diag(cov)).reshape(-1, 1)).div(
        np.sqrt(np.diag(cov)), axis=1
    )
    dist = correlation_distance(corr)

    linkage = _single_linkage_cluster(dist)
    order = quasi_diag_order(linkage, n=len(corr))

    cov_arr = cov.values
    n = cov.shape[0]
    # We re-order indices so clusters can be split as adjacent halves
    cov_arr_q = cov_arr[np.ix_(order, order)]
    weights_q = np.ones(n)

    work: list[list[int]] = [list(range(n))]
    while work:
        cl = work.pop()
        if len(cl) <= 1:
            continue
        mid = len(cl) // 2
        left = cl[:mid]
        right = cl[mid:]
        v_left = _cluster_var(cov_arr_q, left)
        v_right = _cluster_var(cov_arr_q, right)
        alpha = 1.0 - v_left / (v_left + v_right) if (v_left + v_right) > 0 else 0.5
        for idx in left:
            weights_q[idx] *= alpha
        for idx in right:
            weights_q[idx] *= 1.0 - alpha
        work.append(left)
        work.append(right)

    # un-permute
    final = np.zeros(n)
    for q_idx, orig_idx in enumerate(order):
        final[orig_idx] = weights_q[q_idx]

    final = final / final.sum() if final.sum() > 0 else final
    return pd.Series(final, index=cov.index)


def _estimate_covariance(
    returns: pd.DataFrame, method: str = "sample", shrinkage: float = 0.0
) -> pd.DataFrame:
    """Robuste Cov-Schätzung mit optionalem Shrinkage."""
    if method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf  # type: ignore

            lw = LedoitWolf()
            lw.fit(returns.dropna().values)
            cov = pd.DataFrame(
                lw.covariance_, index=returns.columns, columns=returns.columns
            )
            return cov
        except ImportError:
            logger.warning("[hrp] sklearn missing; falling back to sample cov")

    cov = returns.cov()
    if shrinkage > 0:
        diag_mean = float(np.diag(cov).mean())
        identity_like = np.eye(len(cov)) * diag_mean
        cov = (1 - shrinkage) * cov + shrinkage * pd.DataFrame(
            identity_like, index=cov.index, columns=cov.columns
        )
    return cov


__all__ = ["correlation_distance", "quasi_diag_order", "hrp_weights"]
