"""Korrelations-Graph + Mantegna MST + Spectral Clustering.

Reference
---------
- Mantegna, R. (1999). Hierarchical structure in financial markets.
  *European Physical Journal B* 11.
- Tumminello, M. et al. (2010). Correlation, hierarchies, and networks in
  financial markets.

Idee
----
Korrelations-Matrix wird via Distanz d_ij = √(2(1-ρ_ij)) zu vollständigem
Graphen. **Mantegna-MST** extrahiert den Sub-Graphen mit minimaler Total-Distanz
(N-1 Kanten) — zeigt strukturelle Hierarchie der Assets.

Anwendung
---------
- Visualization der Asset-Beziehungen
- Cluster-basierte Diversifikation
- Crisis-Detection (MST-Topology ändert sich in Krisen)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MSTEdge:
    a: str
    b: str
    distance: float


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """d_ij = √(2(1 - ρ_ij)). Symmetric, d_ii = 0, d_ij ∈ [0, 2]."""
    d = np.sqrt(np.maximum(2 * (1 - corr.values), 0))
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def mst_kruskal(distance: pd.DataFrame) -> list[MSTEdge]:
    """Kruskal's MST (Mantegna 1999).

    Returns:
        Liste der N-1 Kanten (sortiert nach Distanz ascending).
    """
    n = distance.shape[0]
    names = list(distance.index)
    # Generate all edges (i<j)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((float(distance.iat[i, j]), i, j))
    edges.sort()

    # Union-find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True

    mst = []
    for d, i, j in edges:
        if union(i, j):
            mst.append(MSTEdge(a=names[i], b=names[j], distance=d))
            if len(mst) >= n - 1:
                break
    return mst


def asset_degrees_in_mst(mst: list[MSTEdge]) -> dict[str, int]:
    """Degree-Centrality jeder Node im MST.

    Hohe Degree = Hub. Hubs verlieren in Krisen oft am meisten (Information-Flow).
    """
    deg: dict[str, int] = {}
    for e in mst:
        deg[e.a] = deg.get(e.a, 0) + 1
        deg[e.b] = deg.get(e.b, 0) + 1
    return deg


def spectral_clustering_assets(
    corr: pd.DataFrame, n_clusters: int = 4, seed: int = 42
) -> dict[str, int]:
    """Spectral Clustering auf Korrelations-Matrix.

    Wandelt Korrelation in Adjacency, berechnet Graph-Laplacian, extrahiert
    Top-K Eigenvectors, KMeans im niedrigdim. Raum.

    Returns:
        Dict {asset_name: cluster_id}.
    """
    A = np.maximum(corr.values, 0)  # only positive correlations as similarities
    np.fill_diagonal(A, 0)
    # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    d = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(len(d)) - (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]
    L = 0.5 * (L + L.T)
    eigvals, eigvecs = np.linalg.eigh(L)
    # Take smallest n_clusters eigenvectors
    sel = eigvecs[:, :n_clusters]
    # Row-normalize
    norms = np.linalg.norm(sel, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    sel = sel / norms

    # Simple KMeans (NumPy-only)
    rng = np.random.default_rng(seed)
    n = sel.shape[0]
    if n_clusters >= n:
        return {name: i for i, name in enumerate(corr.index)}
    init_idx = rng.choice(n, n_clusters, replace=False)
    centers = sel[init_idx]
    labels = np.zeros(n, dtype=int)
    for _ in range(50):
        # Assign
        dists = np.linalg.norm(sel[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update centers
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = sel[mask].mean(axis=0)

    return {name: int(labels[i]) for i, name in enumerate(corr.index)}


def cluster_diversification_weights(
    cluster_assignment: dict[str, int], n_clusters: int
) -> pd.Series:
    """Equal-weight zwischen Clustern, dann equal-weight innerhalb.

    Sehr robust gegen Cov-Schätzfehler.
    """
    assets_by_cluster: dict[int, list[str]] = {}
    for asset, c in cluster_assignment.items():
        assets_by_cluster.setdefault(c, []).append(asset)
    w = {}
    cluster_weight = 1.0 / len(assets_by_cluster)
    for c, assets in assets_by_cluster.items():
        per_asset = cluster_weight / len(assets)
        for a in assets:
            w[a] = per_asset
    return pd.Series(w)


__all__ = [
    "MSTEdge",
    "correlation_distance",
    "mst_kruskal",
    "asset_degrees_in_mst",
    "spectral_clustering_assets",
    "cluster_diversification_weights",
]
