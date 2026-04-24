"""Graph-Based Models — Cross-Asset Signal Propagation (M30).

Implements graph-based analysis for capturing inter-asset relationships:
  1. Supply Chain Graph: propagate signals through customer-supplier links
  2. Sector Correlation Graph: detect signal diffusion across sectors
  3. Lead-Lag Graph: identify which assets lead others
  4. Graph Centrality Signals: use PageRank/eigenvector centrality as factors

The key insight: assets are not independent. A shock to a major supplier
propagates to its customers with a lag. Graph models capture these
network effects that traditional factor models miss.

Reference:
    Cohen, L. & Frazzini, A. (2008). "Economic Links and Predictable Returns."
    Rapach, D. et al. (2019). "Industry Return Predictability."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the asset graph.

    Attributes:
        symbol: Ticker symbol.
        sector: Sector classification.
        centrality: Centrality score (PageRank or eigenvector).
        degree: Number of connections.
    """

    symbol: str
    sector: str = ""
    centrality: float = 0.0
    degree: int = 0


@dataclass
class GraphEdge:
    """An edge in the asset graph.

    Attributes:
        source: Source node symbol.
        target: Target node symbol.
        weight: Edge weight (correlation, supply chain strength, etc.).
        edge_type: Type of relationship.
        lag: Temporal lag in periods (for lead-lag relationships).
    """

    source: str
    target: str
    weight: float
    edge_type: str = "correlation"
    lag: int = 0


@dataclass
class GraphSignal:
    """Signal derived from graph analysis.

    Attributes:
        symbol: Ticker symbol.
        propagated_score: Signal propagated from connected nodes.
        centrality_score: Centrality-based importance factor.
        lead_lag_score: Score from lead-lag relationships.
        composite: Blended graph signal.
    """

    symbol: str
    propagated_score: float
    centrality_score: float
    lead_lag_score: float
    composite: float


def build_correlation_graph(
    returns_df: pd.DataFrame,
    min_correlation: float = 0.3,
    lookback: int = 60,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Build an asset graph from return correlations.

    Two assets are connected if their return correlation exceeds
    the threshold over the lookback period.

    Args:
        returns_df: DataFrame with symbols as columns, dates as rows.
        min_correlation: Minimum absolute correlation for an edge.
        lookback: Lookback window in periods.

    Returns:
        Tuple of (nodes, edges).
    """
    df = returns_df.tail(lookback).dropna(axis=1, thresh=lookback // 2)
    symbols = list(df.columns)

    if len(symbols) < 2:
        return [GraphNode(symbol=s) for s in symbols], []

    corr = df.corr()
    nodes = []
    edges = []

    for sym in symbols:
        nodes.append(GraphNode(symbol=sym))

    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1:]:
            c = corr.loc[sym_a, sym_b]
            if abs(c) >= min_correlation:
                edges.append(GraphEdge(
                    source=sym_a, target=sym_b,
                    weight=round(float(c), 4),
                    edge_type="correlation",
                ))

    # Compute degree
    degree_map: dict[str, int] = {s: 0 for s in symbols}
    for edge in edges:
        degree_map[edge.source] = degree_map.get(edge.source, 0) + 1
        degree_map[edge.target] = degree_map.get(edge.target, 0) + 1

    for node in nodes:
        node.degree = degree_map.get(node.symbol, 0)

    logger.info(
        "[GraphModels] Built correlation graph: %d nodes, %d edges (threshold=%.2f)",
        len(nodes), len(edges), min_correlation,
    )

    return nodes, edges


def compute_pagerank(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    damping: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Compute PageRank centrality for the asset graph.

    Higher PageRank = more "influential" asset in the network.

    Args:
        nodes: Graph nodes.
        edges: Graph edges.
        damping: Damping factor (default: 0.85).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Dict of symbol -> PageRank score.
    """
    symbols = [n.symbol for n in nodes]
    n = len(symbols)
    if n == 0:
        return {}

    idx = {s: i for i, s in enumerate(symbols)}

    # Build adjacency matrix
    adj = np.zeros((n, n))
    for edge in edges:
        if edge.source in idx and edge.target in idx:
            i, j = idx[edge.source], idx[edge.target]
            w = abs(edge.weight)
            adj[i, j] = w
            adj[j, i] = w  # undirected

    # Normalize columns (for transition matrix)
    col_sums = adj.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    M = adj / col_sums

    # Power iteration
    pr = np.ones(n) / n
    for _ in range(max_iter):
        pr_new = (1 - damping) / n + damping * M @ pr
        if np.abs(pr_new - pr).sum() < tol:
            pr = pr_new
            break
        pr = pr_new

    return {symbols[i]: round(float(pr[i]), 6) for i in range(n)}


def detect_lead_lag(
    returns_df: pd.DataFrame,
    max_lag: int = 5,
    min_correlation: float = 0.15,
) -> list[GraphEdge]:
    """Detect lead-lag relationships between assets.

    For each pair, check if lagged returns of A predict returns of B
    better than contemporaneous correlation.

    Args:
        returns_df: Returns DataFrame (symbols as columns).
        max_lag: Maximum lag to test.
        min_correlation: Minimum lagged correlation for a lead-lag edge.

    Returns:
        List of GraphEdge with lag information.
    """
    df = returns_df.dropna(axis=1, thresh=len(returns_df) // 2)
    symbols = list(df.columns)
    edges = []

    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1:]:
            a = df[sym_a].values
            b = df[sym_b].values

            best_corr = 0.0
            best_lag = 0
            best_leader = sym_a

            for lag in range(1, max_lag + 1):
                # Does A lead B?
                if len(a) > lag:
                    c_ab = np.corrcoef(a[:-lag], b[lag:])[0, 1]
                    if np.isfinite(c_ab) and abs(c_ab) > abs(best_corr):
                        best_corr = c_ab
                        best_lag = lag
                        best_leader = sym_a

                    # Does B lead A?
                    c_ba = np.corrcoef(b[:-lag], a[lag:])[0, 1]
                    if np.isfinite(c_ba) and abs(c_ba) > abs(best_corr):
                        best_corr = c_ba
                        best_lag = lag
                        best_leader = sym_b

            if abs(best_corr) >= min_correlation:
                follower = sym_b if best_leader == sym_a else sym_a
                edges.append(GraphEdge(
                    source=best_leader,
                    target=follower,
                    weight=round(float(best_corr), 4),
                    edge_type="lead_lag",
                    lag=best_lag,
                ))

    logger.info("[GraphModels] Detected %d lead-lag relationships", len(edges))
    return edges


def propagate_signals(
    signals: dict[str, float],
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    propagation_decay: float = 0.5,
) -> dict[str, float]:
    """Propagate signals through the graph.

    Each node receives a weighted sum of its neighbors' signals,
    decayed by distance and edge weight.

    Args:
        signals: Dict of symbol -> raw signal score.
        nodes: Graph nodes.
        edges: Graph edges.
        propagation_decay: How much signal decays per hop (0-1).

    Returns:
        Dict of symbol -> propagated signal (original + neighbor influence).
    """
    symbols = {n.symbol for n in nodes}
    propagated = dict(signals)

    # Build adjacency list
    neighbors: dict[str, list[tuple[str, float]]] = {s: [] for s in symbols}
    for edge in edges:
        if edge.source in neighbors and edge.target in neighbors:
            neighbors[edge.source].append((edge.target, edge.weight))
            neighbors[edge.target].append((edge.source, edge.weight))

    # Single-hop propagation
    for sym in symbols:
        neighbor_signal = 0.0
        total_weight = 0.0
        for neighbor, weight in neighbors.get(sym, []):
            if neighbor in signals:
                neighbor_signal += signals[neighbor] * abs(weight)
                total_weight += abs(weight)

        if total_weight > 0:
            propagated_component = neighbor_signal / total_weight
            original = signals.get(sym, 0.0)
            propagated[sym] = round(
                original + propagation_decay * propagated_component, 6,
            )

    return propagated


def generate_graph_signals(
    returns_df: pd.DataFrame,
    raw_signals: dict[str, float] | None = None,
    min_correlation: float = 0.3,
    lookback: int = 60,
) -> list[GraphSignal]:
    """Generate graph-based signals for all assets.

    Combines correlation graph, PageRank centrality, lead-lag detection,
    and signal propagation into composite graph signals.

    Args:
        returns_df: Returns DataFrame (symbols as columns).
        raw_signals: Optional raw signal scores to propagate.
        min_correlation: Correlation threshold for edges.
        lookback: Lookback for correlation computation.

    Returns:
        List of GraphSignal for each symbol.
    """
    nodes, edges = build_correlation_graph(returns_df, min_correlation, lookback)
    symbols = [n.symbol for n in nodes]

    if not symbols:
        return []

    # PageRank centrality
    pagerank = compute_pagerank(nodes, edges)

    # Lead-lag edges
    ll_edges = detect_lead_lag(returns_df[symbols], max_lag=3)

    # Build lead-lag scores: followers of strong leaders get a boost
    ll_scores: dict[str, float] = {s: 0.0 for s in symbols}
    for edge in ll_edges:
        if raw_signals and edge.source in raw_signals:
            ll_scores[edge.target] = round(
                float(raw_signals[edge.source] * edge.weight * 0.3), 6,
            )

    # Propagate signals
    if raw_signals is None:
        raw_signals = {s: 0.0 for s in symbols}
    propagated = propagate_signals(raw_signals, nodes, edges)

    # Z-score centrality
    pr_vals = np.array([pagerank.get(s, 0.0) for s in symbols])
    pr_mean = pr_vals.mean()
    pr_std = pr_vals.std()
    if pr_std < 1e-10:
        pr_std = 1.0

    results = []
    for sym in symbols:
        pr_z = (pagerank.get(sym, 0.0) - pr_mean) / pr_std
        prop = propagated.get(sym, 0.0)
        ll = ll_scores.get(sym, 0.0)

        composite = 0.5 * prop + 0.3 * pr_z + 0.2 * ll
        composite = float(np.clip(composite, -3.0, 3.0))

        results.append(GraphSignal(
            symbol=sym,
            propagated_score=round(prop, 4),
            centrality_score=round(float(pr_z), 4),
            lead_lag_score=round(ll, 4),
            composite=round(composite, 4),
        ))

    return results


__all__ = [
    "GraphNode",
    "GraphEdge",
    "GraphSignal",
    "build_correlation_graph",
    "compute_pagerank",
    "detect_lead_lag",
    "propagate_signals",
    "generate_graph_signals",
]
