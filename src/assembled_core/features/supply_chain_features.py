"""Supply-chain and geopolitical network features per asset.

Extracts quantitative features from the intel pipeline's dependency graph,
shipping lane analysis, and sanctions modelling:

- ``supply_chain_depth``: Number of hops from raw inputs to end-customer
  in the dependency graph (deeper = more fragile).
- ``single_source_dependency``: max(edge_weight) / sum(edge_weights) for
  inbound edges (high → dependent on one supplier).
- ``chokepoint_exposure``: Weighted sum of chokepoint world-trade shares
  from ``shipping_lanes.CHOKEPOINT_WORLD_TRADE_SHARE``.
- ``sanctions_vulnerability``: 1 - sanctions_resilience (from nation_profiles).
- ``network_centrality``: Eigenvector centrality in the dependency graph
  (high → systemically important, but also systemically exposed).

These features quantify supply-chain and geopolitical fragility that
traditional financial factors miss entirely.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_supply_chain_depth(
    dependency_edges: list[tuple[str, str, float]],
    target_symbols: list[str],
) -> dict[str, int]:
    """Compute supply-chain depth (max path length to any sink node).

    Args:
        dependency_edges: List of (source, target, weight) tuples
            representing the supply-chain graph.
        target_symbols: Symbols to compute depth for.

    Returns:
        Dict mapping symbol → max depth (0 if leaf/no edges).
    """
    # Build adjacency list
    adj: dict[str, list[str]] = {}
    for src, tgt, _ in dependency_edges:
        adj.setdefault(src, []).append(tgt)

    def _bfs_depth(start: str) -> int:
        visited = {start}
        queue = [(start, 0)]
        max_d = 0
        while queue:
            node, depth = queue.pop(0)
            max_d = max(max_d, depth)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return max_d

    return {sym: _bfs_depth(sym) for sym in target_symbols}


def compute_single_source_dependency(
    dependency_edges: list[tuple[str, str, float]],
    target_symbols: list[str],
) -> dict[str, float]:
    """Compute single-source dependency score per symbol.

    Score = max(inbound_edge_weight) / sum(inbound_edge_weights).
    High score (near 1.0) means the symbol depends heavily on one supplier.

    Args:
        dependency_edges: List of (source, target, weight) tuples.
        target_symbols: Symbols to compute for.

    Returns:
        Dict mapping symbol → single-source dependency score [0, 1].
    """
    inbound: dict[str, list[float]] = {}
    for src, tgt, w in dependency_edges:
        inbound.setdefault(tgt, []).append(w)

    result = {}
    for sym in target_symbols:
        weights = inbound.get(sym, [])
        if not weights:
            result[sym] = 0.0
            continue
        total = sum(weights)
        result[sym] = max(weights) / total if total > 0 else 0.0

    return result


def compute_chokepoint_exposure(
    symbol_country_map: dict[str, list[str]],
    country_chokepoint_shares: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute chokepoint exposure per symbol.

    For each symbol, looks up its operating countries and their dependence
    on maritime chokepoints (Suez, Malacca, Hormuz, etc.).

    Args:
        symbol_country_map: Dict mapping symbol → list of country codes
            where the company operates or sources from.
        country_chokepoint_shares: Dict mapping country → {chokepoint: share}.
            ``share`` is fraction of trade transiting that chokepoint.

    Returns:
        Dict mapping symbol → aggregate chokepoint exposure [0, 1+].
    """
    result = {}
    for sym, countries in symbol_country_map.items():
        total_exposure = 0.0
        for country in countries:
            cp_shares = country_chokepoint_shares.get(country, {})
            total_exposure += sum(cp_shares.values())
        # Normalize by number of countries
        result[sym] = total_exposure / max(len(countries), 1)

    return result


def compute_sanctions_vulnerability(
    symbol_country_map: dict[str, list[str]],
    country_resilience: dict[str, float],
) -> dict[str, float]:
    """Compute sanctions vulnerability per symbol.

    Vulnerability = 1 - avg(sanctions_resilience) across operating countries.
    High vulnerability means the company is exposed to countries with low
    sanctions resilience (easily disrupted by new sanctions).

    Args:
        symbol_country_map: Dict mapping symbol → list of country codes.
        country_resilience: Dict mapping country → resilience score [0, 1].

    Returns:
        Dict mapping symbol → vulnerability score [0, 1].
    """
    result = {}
    for sym, countries in symbol_country_map.items():
        resiliences = [country_resilience.get(c, 0.5) for c in countries]
        avg_resilience = float(np.mean(resiliences)) if resiliences else 0.5
        result[sym] = round(1.0 - avg_resilience, 4)
    return result


def compute_network_centrality(
    dependency_edges: list[tuple[str, str, float]],
    target_symbols: list[str],
    *,
    max_iterations: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Compute eigenvector centrality in the dependency network.

    High centrality → the symbol is a hub in the supply chain
    (systemically important but also systemically exposed to contagion).

    Uses power iteration to approximate the leading eigenvector of the
    adjacency matrix.

    Args:
        dependency_edges: List of (source, target, weight) tuples.
        target_symbols: Symbols to return centrality for.
        max_iterations: Max iterations for power method.
        tol: Convergence tolerance.

    Returns:
        Dict mapping symbol → centrality score [0, 1].
    """
    # Collect all nodes
    all_nodes = set()
    for src, tgt, _ in dependency_edges:
        all_nodes.add(src)
        all_nodes.add(tgt)

    if not all_nodes:
        return {s: 0.0 for s in target_symbols}

    node_list = sorted(all_nodes)
    node_idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    # Build adjacency matrix
    A = np.zeros((n, n))
    for src, tgt, w in dependency_edges:
        i, j = node_idx[src], node_idx[tgt]
        A[i, j] += w
        A[j, i] += w  # undirected for centrality

    # Power iteration
    x = np.ones(n) / n
    for _ in range(max_iterations):
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-15:
            break
        x_new /= norm
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    # Normalize to [0, 1]
    max_val = x.max()
    if max_val > 0:
        x = x / max_val

    return {
        s: round(float(x[node_idx[s]]), 6) if s in node_idx else 0.0
        for s in target_symbols
    }


def build_supply_chain_features(
    symbols: list[str],
    dependency_edges: list[tuple[str, str, float]] | None = None,
    symbol_country_map: dict[str, list[str]] | None = None,
    country_chokepoint_shares: dict[str, dict[str, float]] | None = None,
    country_resilience: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build all supply-chain features into a single DataFrame.

    Args:
        symbols: Symbols to compute features for.
        dependency_edges: Supply-chain graph edges.
        symbol_country_map: Symbol → operating countries.
        country_chokepoint_shares: Country → chokepoint shares.
        country_resilience: Country → sanctions resilience.

    Returns:
        DataFrame with symbol as index and feature columns.
    """
    features = pd.DataFrame(index=symbols)

    if dependency_edges:
        features["supply_chain_depth"] = pd.Series(
            compute_supply_chain_depth(dependency_edges, symbols)
        )
        features["single_source_dep"] = pd.Series(
            compute_single_source_dependency(dependency_edges, symbols)
        )
        features["network_centrality"] = pd.Series(
            compute_network_centrality(dependency_edges, symbols)
        )
    else:
        features["supply_chain_depth"] = 0
        features["single_source_dep"] = 0.0
        features["network_centrality"] = 0.0

    if symbol_country_map and country_chokepoint_shares:
        features["chokepoint_exposure"] = pd.Series(
            compute_chokepoint_exposure(symbol_country_map, country_chokepoint_shares)
        )
    else:
        features["chokepoint_exposure"] = 0.0

    if symbol_country_map and country_resilience:
        features["sanctions_vulnerability"] = pd.Series(
            compute_sanctions_vulnerability(symbol_country_map, country_resilience)
        )
    else:
        features["sanctions_vulnerability"] = 0.0

    return features


def propagate_returns_through_chain(
    returns: pd.DataFrame,
    dependency_edges: list[tuple[str, str, float]],
    lag_days: int = 1,
    decay: float = 0.5,
    max_hops: int = 2,
) -> pd.DataFrame:
    """Propagate returns through supply chain graph (Cohen & Frazzini 2008, Task 18.6).

    When a major customer rallies, its suppliers tend to follow with a lag.
    Computes lagged weighted-average return of connected nodes.

    Args:
        returns: DataFrame with symbols as columns, dates as index.
        dependency_edges: List of (supplier, customer, weight) tuples.
        lag_days: How many days the signal propagates (default 1).
        decay: Weight decay per hop (default 0.5 = half per hop).
        max_hops: Maximum hops for propagation (default 2).

    Returns:
        DataFrame with propagated return features per symbol.

    Reference: Cohen & Frazzini (2008) "Economic Links and Predictable Returns"
    Alpha: +60-140 bps/year
    """
    symbols = list(returns.columns)
    sym_set = set(symbols)

    # Build adjacency both directions
    forward: dict[str, list[tuple[str, float]]] = {}
    reverse: dict[str, list[tuple[str, float]]] = {}
    for src, dst, w in dependency_edges:
        if src in sym_set:
            forward.setdefault(src, []).append((dst, w))
        if dst in sym_set:
            reverse.setdefault(dst, []).append((src, w))

    result = pd.DataFrame(0.0, index=returns.index, columns=symbols)

    for sym in symbols:
        connected: dict[str, float] = {}

        neighbors_1: list[tuple[str, float]] = []
        for dst, w in forward.get(sym, []):
            if dst in sym_set:
                neighbors_1.append((dst, w))
        for src, w in reverse.get(sym, []):
            if src in sym_set:
                neighbors_1.append((src, w))

        for neighbor, w in neighbors_1:
            connected[neighbor] = connected.get(neighbor, 0) + w

        if max_hops >= 2:
            for n1, w1 in neighbors_1:
                for dst, w2 in forward.get(n1, []):
                    if dst in sym_set and dst != sym:
                        connected[dst] = connected.get(dst, 0) + w1 * w2 * decay
                for src, w2 in reverse.get(n1, []):
                    if src in sym_set and src != sym:
                        connected[src] = connected.get(src, 0) + w1 * w2 * decay

        if not connected:
            continue

        total_w = sum(connected.values())
        if total_w < 1e-10:
            continue

        weighted_ret = (
            sum(
                connected[n] * returns[n].shift(lag_days).fillna(0)
                for n in connected
                if n in returns.columns
            )
            / total_w
        )

        result[sym] = weighted_ret

    return result


__all__ = [
    "build_supply_chain_features",
    "compute_chokepoint_exposure",
    "compute_network_centrality",
    "compute_sanctions_vulnerability",
    "compute_single_source_dependency",
    "compute_supply_chain_depth",
    "propagate_returns_through_chain",
]
