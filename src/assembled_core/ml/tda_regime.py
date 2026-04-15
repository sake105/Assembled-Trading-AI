"""Topological Data Analysis for Regime Detection (M25 Task 25.3).

Uses persistent homology to detect market regimes from the topology
of the return point-cloud. Model-free alternative to HMM that detects
regime transitions earlier (Gidea et al. 2018).

Falls back to simple rolling statistics when giotto-tda is unavailable.

Reference: Gidea & Katz (2018) "Topological data analysis of financial time series"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from giotto.homology import VietorisRipsPersistence
    from giotto.diagrams import PersistenceEntropy, BettiCurve
    GIOTTO_AVAILABLE = True
except ImportError:
    GIOTTO_AVAILABLE = False


@dataclass
class TDAFeatures:
    """Topological features extracted from return data."""
    h0_persistence: float      # Connected components persistence (clustering)
    h1_persistence: float      # Loop persistence (cyclical patterns)
    persistence_entropy: float  # Entropy of persistence diagram
    betti_0: int               # Number of connected components
    betti_1: int               # Number of loops
    norm_persistence: float    # Normalized total persistence
    regime_signal: float       # Composite regime signal (-1=crisis, 0=normal, 1=trending)


def _build_point_cloud(
    returns: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 1,
) -> np.ndarray:
    """Build time-delay embedding point cloud from returns.

    Takens' theorem: time-delay embedding reconstructs attractor topology.

    Args:
        returns: (T,) return series.
        embedding_dim: Embedding dimension.
        delay: Time delay.

    Returns:
        (T - (dim-1)*delay, dim) point cloud.
    """
    T = len(returns)
    n_points = T - (embedding_dim - 1) * delay
    if n_points < 10:
        return np.zeros((1, embedding_dim))

    cloud = np.zeros((n_points, embedding_dim))
    for d in range(embedding_dim):
        start = d * delay
        cloud[:, d] = returns[start:start + n_points]

    return cloud


def compute_persistence_features(
    point_cloud: np.ndarray,
    max_homology_dim: int = 1,
) -> tuple[list[np.ndarray], dict]:
    """Compute persistent homology features.

    Args:
        point_cloud: (N, d) point cloud.
        max_homology_dim: Maximum homology dimension (0=components, 1=loops).

    Returns:
        (persistence_diagrams, feature_dict).
    """
    if GIOTTO_AVAILABLE:
        return _compute_giotto(point_cloud, max_homology_dim)
    return _compute_fallback(point_cloud)


def _compute_giotto(point_cloud: np.ndarray, max_dim: int) -> tuple[list, dict]:
    """Compute features using giotto-tda."""
    vr = VietorisRipsPersistence(homology_dimensions=list(range(max_dim + 1)))
    diagrams = vr.fit_transform(point_cloud[np.newaxis, :, :])[0]

    # Extract features
    pe = PersistenceEntropy()
    entropy = pe.fit_transform(diagrams[np.newaxis, :])[0]

    bc = BettiCurve()
    betti = bc.fit_transform(diagrams[np.newaxis, :])[0]

    features = {
        "persistence_entropy": float(entropy.sum()),
        "betti_0": int(betti[0].max()) if len(betti) > 0 else 0,
        "betti_1": int(betti[1].max()) if len(betti) > 1 else 0,
    }

    return [diagrams], features


def _compute_fallback(point_cloud: np.ndarray) -> tuple[list, dict]:
    """Fallback: approximate topological features from pairwise distances."""
    from scipy.spatial.distance import pdist, squareform

    if len(point_cloud) < 3:
        return [], {"persistence_entropy": 0, "betti_0": 1, "betti_1": 0}

    # Pairwise distances
    dists = pdist(point_cloud)
    dist_matrix = squareform(dists)

    # Approximate H0: number of clusters at various thresholds
    thresholds = np.percentile(dists, [25, 50, 75])
    n_components = []
    for thresh in thresholds:
        # Simple connected components via adjacency
        adj = dist_matrix < thresh
        visited = np.zeros(len(point_cloud), dtype=bool)
        components = 0
        for i in range(len(point_cloud)):
            if not visited[i]:
                # BFS
                stack = [i]
                while stack:
                    node = stack.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    neighbors = np.where(adj[node] & ~visited)[0]
                    stack.extend(neighbors.tolist())
                components += 1
        n_components.append(components)

    # H0 persistence: spread of component counts across thresholds
    h0_persist = float(np.std(n_components))

    # H1 proxy: ratio of short to long distances (loops create intermediate distances)
    median_dist = np.median(dists)
    short_frac = (dists < median_dist * 0.5).sum() / max(len(dists), 1)
    h1_proxy = short_frac  # Higher = more structure = possible loops

    # Persistence entropy proxy
    dist_hist, _ = np.histogram(dists, bins=20, density=True)
    dist_hist = dist_hist + 1e-10
    dist_hist = dist_hist / dist_hist.sum()
    entropy = -float(np.sum(dist_hist * np.log(dist_hist)))

    features = {
        "persistence_entropy": round(entropy, 4),
        "betti_0": n_components[1] if len(n_components) > 1 else 1,
        "betti_1": int(h1_proxy > 0.3),
    }

    return [], features


def extract_tda_features(
    returns: pd.Series | np.ndarray,
    window: int = 60,
    embedding_dim: int = 3,
    delay: int = 1,
) -> TDAFeatures:
    """Extract TDA features from a return window.

    Args:
        returns: Return series (at least `window` points).
        window: Window size for point cloud.
        embedding_dim: Time-delay embedding dimension.
        delay: Time delay for embedding.

    Returns:
        TDAFeatures dataclass.
    """
    ret_arr = np.asarray(returns, dtype=float)[-window:]

    # Build point cloud
    cloud = _build_point_cloud(ret_arr, embedding_dim, delay)

    # Compute persistence
    _, features = compute_persistence_features(cloud)

    # Derive regime signal
    entropy = features.get("persistence_entropy", 0)
    betti_0 = features.get("betti_0", 1)
    betti_1 = features.get("betti_1", 0)

    # High entropy + many components = crisis/fragmented market
    # Low entropy + few components = trending/stable
    # Loops (H1) = cyclical/mean-reverting
    h0_persistence = float(betti_0) / max(window // embedding_dim, 1)
    h1_persistence = float(betti_1)
    norm_persistence = h0_persistence + h1_persistence

    # Regime signal
    if h0_persistence > 0.5:  # Many clusters → fragmented → crisis
        regime_signal = -1.0
    elif betti_1 > 0:  # Loops → mean-reverting
        regime_signal = 0.0
    else:  # Few clusters, no loops → trending
        regime_signal = 1.0

    return TDAFeatures(
        h0_persistence=round(h0_persistence, 4),
        h1_persistence=round(h1_persistence, 4),
        persistence_entropy=round(entropy, 4),
        betti_0=betti_0,
        betti_1=betti_1,
        norm_persistence=round(norm_persistence, 4),
        regime_signal=regime_signal,
    )


def rolling_tda_features(
    returns: pd.Series,
    window: int = 60,
    step: int = 20,
) -> pd.DataFrame:
    """Compute rolling TDA features.

    Args:
        returns: Daily return series.
        window: Rolling window.
        step: Step size between computations.

    Returns:
        DataFrame with TDA features.
    """
    results = []
    dates = returns.index

    for i in range(window, len(returns), step):
        ret_window = returns.iloc[i - window:i].values
        tda = extract_tda_features(ret_window, window)
        results.append({
            "date": dates[i - 1],
            "h0_persistence": tda.h0_persistence,
            "h1_persistence": tda.h1_persistence,
            "persistence_entropy": tda.persistence_entropy,
            "betti_0": tda.betti_0,
            "betti_1": tda.betti_1,
            "regime_signal": tda.regime_signal,
        })

    return pd.DataFrame(results).set_index("date") if results else pd.DataFrame()


__all__ = [
    "TDAFeatures",
    "extract_tda_features",
    "rolling_tda_features",
    "compute_persistence_features",
]
