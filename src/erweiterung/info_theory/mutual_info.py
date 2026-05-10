"""Mutual Information + Transfer-Entropy für nichtlineare Beziehungen.

Theorie
-------
- **Mutual Information** I(X; Y) = ∫∫ p(x,y) log(p(x,y)/(p(x) p(y))) dx dy
  Misst beliebige (nicht nur lineare) Abhängigkeit. 0 ⇔ Unabhängigkeit.

- **Transfer-Entropy** TE(X→Y) = I(Y_t+1 ; X_t | Y_t)
  Directed Information-Flow von X nach Y, kontrolliert für Eigenverhalten.

Anwendung
---------
- Feature-Selection: select features mit höchstem MI(X_i, y), aber niedrigem MI(X_i, X_j)
  → Maximum-Relevance Minimum-Redundancy (mRMR).
- Lead-Lag-Detection mit nichtlinearen Beziehungen (statt nur Granger).
- Cross-Asset-Information-Flow während Krisen.

Implementation
--------------
- Histogram-basiert (default): einfach, schnell, für eindimensionale X.
- KDE-basiert: optional, präziser, langsamer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mutual_info_histogram(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """Histogram-basiertes MI in nats.

    I(X;Y) = H(X) + H(Y) - H(X,Y)
    mit Shannon-Entropy auf Histogramm.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 30:
        return float("nan")

    p_xy, _, _ = np.histogram2d(x, y, bins=n_bins, density=False)
    p_xy = p_xy / p_xy.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    # Avoid log(0)
    eps = 1e-12
    h_x = -np.sum(p_x[p_x > 0] * np.log(p_x[p_x > 0] + eps))
    h_y = -np.sum(p_y[p_y > 0] * np.log(p_y[p_y > 0] + eps))
    h_xy = -np.sum(p_xy[p_xy > 0] * np.log(p_xy[p_xy > 0] + eps))
    return float(max(h_x + h_y - h_xy, 0.0))


def normalized_mutual_info(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """NMI ∈ [0, 1] — normalisiertes MI."""
    mi = mutual_info_histogram(x, y, n_bins)
    h_x = mutual_info_histogram(x, x, n_bins)
    h_y = mutual_info_histogram(y, y, n_bins)
    denom = np.sqrt(h_x * h_y)
    return mi / denom if denom > 0 else 0.0


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P||Q) = Σ p log(p/q). Discrete distributions."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("shapes mismatch")
    mask = (p > 0) & (q > 0)
    if not mask.any():
        return float("nan")
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def transfer_entropy(
    source: np.ndarray, target: np.ndarray, lag: int = 1, n_bins: int = 8
) -> float:
    """Transfer-Entropy TE(source → target).

    TE = I(target_t+1; source_t | target_t)
       = H(target_t+1, target_t) + H(target_t, source_t)
       - H(target_t+1, target_t, source_t) - H(target_t)

    Args:
        source, target: 1-D arrays.
        lag: time-lag for source.
        n_bins: discretization bins.

    Returns:
        TE in nats.
    """
    s = np.asarray(source, dtype=float)
    t = np.asarray(target, dtype=float)
    if len(s) != len(t) or len(s) < 100:
        return float("nan")

    # Build x_t, y_t, y_{t+1}
    y_curr = t[:-1]
    y_next = t[1:]
    x_lag = s[: len(y_curr)]  # source at time t (shifted to align with y_curr)
    if lag != 1:
        x_lag = np.concatenate(
            [np.full(lag - 1, np.nan), s[: -lag + 1 if lag > 1 else None]]
        )[: len(y_curr)]

    mask = ~(np.isnan(y_curr) | np.isnan(y_next) | np.isnan(x_lag))
    y_curr, y_next, x_lag = y_curr[mask], y_next[mask], x_lag[mask]
    if len(y_curr) < 50:
        return float("nan")

    # Discretize
    def _digitize(arr, bins):
        edges = np.histogram_bin_edges(arr, bins=bins)
        return np.clip(np.digitize(arr, edges) - 1, 0, bins - 1)

    y_d = _digitize(y_curr, n_bins)
    y_n = _digitize(y_next, n_bins)
    x_d = _digitize(x_lag, n_bins)

    # Joint distributions
    def _entropy_joint(*arrays):
        stacked = np.stack(arrays, axis=1)
        # use unique-rows for counting
        _, counts = np.unique(stacked, axis=0, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log(p + 1e-12))

    h_yt_y = _entropy_joint(y_d, y_n)
    h_yt_x = _entropy_joint(y_d, x_d)
    h_yt_y_x = _entropy_joint(y_d, y_n, x_d)
    h_y = _entropy_joint(y_d)

    te = h_yt_y + h_yt_x - h_yt_y_x - h_y
    return float(max(te, 0.0))


def mrmr_feature_selection(
    X: pd.DataFrame, y: pd.Series, n_select: int = 10, n_bins: int = 20
) -> list[str]:
    """Maximum-Relevance Minimum-Redundancy Feature-Selection (Peng et al. 2005).

    1. Score je Feature: MI(X_i, y) − mean MI(X_i, X_j_selected).
    2. Greedy-Auswahl bis n_select.
    """
    y_v = pd.Series(y).dropna().values
    common_idx = pd.Series(y).dropna().index
    Xf = X.loc[common_idx].dropna()
    common_idx = Xf.index
    y_v = pd.Series(y).loc[common_idx].values

    relevance = {}
    for col in Xf.columns:
        relevance[col] = mutual_info_histogram(Xf[col].values, y_v, n_bins=n_bins)

    selected: list[str] = []
    remaining = list(Xf.columns)
    while len(selected) < n_select and remaining:
        best_score = -np.inf
        best_feat: str | None = None
        for col in remaining:
            rel = relevance[col]
            redundancy = 0.0
            if selected:
                redundancy = np.mean(
                    [
                        mutual_info_histogram(
                            Xf[col].values, Xf[s].values, n_bins=n_bins
                        )
                        for s in selected
                    ]
                )
            score = rel - redundancy
            if score > best_score:
                best_score = score
                best_feat = col
        if best_feat is None:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
    return selected


__all__ = [
    "mutual_info_histogram",
    "normalized_mutual_info",
    "kl_divergence",
    "transfer_entropy",
    "mrmr_feature_selection",
]
