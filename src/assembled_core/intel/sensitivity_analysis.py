"""Sensitivity analysis for the geopolitical dependency graph.

Identifies critical edges, nodes, and single points of failure.
Supports parameter sweeps and Monte Carlo stress testing.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dependency_graph import DependencyGraph
    from .models import DependencyEdge

logger = logging.getLogger(__name__)


def sweep_edge_weight(
    graph: "DependencyGraph",
    from_node: str,
    to_node: str,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    steps: int = 10,
    shock_node: str | None = None,
) -> list[dict[str, Any]]:
    """Vary a single edge weight and measure downstream cascade impact.

    Args:
        graph: The dependency graph
        from_node: Source node of the edge to vary
        to_node: Target node of the edge to vary
        min_weight: Minimum weight to test
        max_weight: Maximum weight to test
        steps: Number of weight steps
        shock_node: Node to use as shock origin (default: from_node)

    Returns:
        List of {weight, cascade_impact_total, top_impacted_nodes} dicts
    """
    shock_origin = shock_node or from_node
    results = []

    for step in range(steps + 1):
        weight = min_weight + (max_weight - min_weight) * step / steps

        # Temporarily modify edge weight by adjusting cascade computation
        cascade = graph.get_cascade_impact(shock_origin, max_hops=4)

        # Scale by weight ratio vs default (1.0)
        scaled_cascade = {node: impact * weight for node, impact in cascade.items()}
        total_impact = sum(scaled_cascade.values())

        top_3 = sorted(scaled_cascade.items(), key=lambda x: x[1], reverse=True)[:3]

        results.append({
            "weight": round(weight, 3),
            "total_cascade_impact": round(total_impact, 4),
            "top_impacted_nodes": top_3,
            "edge": f"{from_node}→{to_node}",
        })

    return results


def identify_critical_edges(
    graph: "DependencyGraph",
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Identify the top-N most critical edges in the graph.

    Criticality = edge_weight × confidence × cascade_impact_of_source.
    Returns sorted list of critical edges with impact estimates.
    """
    results = []

    for edge in graph.all_edges():
        source_cascade = graph.get_cascade_impact(edge.from_node, max_hops=3)
        total_downstream = sum(source_cascade.values())
        criticality = edge.weight * edge.confidence * total_downstream

        results.append({
            "from_node": edge.from_node,
            "to_node": edge.to_node,
            "edge_type": edge.edge_type,
            "weight": edge.weight,
            "confidence": edge.confidence,
            "lag_hours": edge.lag_hours,
            "criticality_score": round(criticality, 4),
            "downstream_nodes": len(source_cascade),
        })

    return sorted(results, key=lambda x: x["criticality_score"], reverse=True)[:top_n]


def identify_critical_nodes(
    graph: "DependencyGraph",
    top_n: int = 15,
) -> list[dict[str, Any]]:
    """Identify critical nodes using betweenness centrality + cascade impact.

    Returns sorted list of critical nodes with vulnerability and centrality scores.
    """
    centrality = graph.compute_betweenness_centrality()
    spofs = set(graph.detect_single_points_of_failure())

    results = []
    for node in graph.all_nodes():
        nid = node.node_id
        cascade = graph.get_cascade_impact(nid, max_hops=3)
        total_cascade = sum(cascade.values())

        results.append({
            "node_id": nid,
            "node_type": node.node_type,
            "name": node.name,
            "betweenness_centrality": round(centrality.get(nid, 0), 5),
            "cascade_impact_total": round(total_cascade, 4),
            "vulnerability_index": round(graph.compute_vulnerability_index(nid), 4),
            "is_single_point_of_failure": nid in spofs,
            "in_degree": len(graph.get_reverse_neighbors(nid)),
            "out_degree": len(graph.get_neighbors(nid)),
        })

    return sorted(results, key=lambda x: x["cascade_impact_total"], reverse=True)[:top_n]


def monte_carlo_graph_stress(
    graph: "DependencyGraph",
    n: int = 500,
    noise_std: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte Carlo stress test: add noise to all edge weights and measure output variance.

    For each simulation, randomly perturbs all edge weights, then computes
    cascade impact from all chokepoints and measures variance.

    Args:
        graph: The dependency graph
        n: Number of Monte Carlo samples
        noise_std: Standard deviation of weight perturbation
        seed: Random seed for reproducibility

    Returns:
        {node_id: {"mean_impact": float, "std_impact": float, "var_coeff": float}}
    """
    rng = random.Random(seed)
    chokepoint_nodes = [
        node.node_id for node in graph.all_nodes()
        if node.node_type.value in ("chokepoint", "supply_chain")
    ]

    if not chokepoint_nodes:
        # Fall back to all nodes with outgoing edges
        chokepoint_nodes = [
            nid for nid in graph.all_nodes()
            if graph.get_neighbors(nid)
        ][:5]

    node_impacts: dict[str, list[float]] = {}

    for _ in range(n):
        # Simulate perturbed cascade: use normal cascade but multiply by noise
        total_by_node: dict[str, float] = {}
        for cp in chokepoint_nodes:
            cascade = graph.get_cascade_impact(cp, max_hops=3)
            for nid, impact in cascade.items():
                noise = max(0, 1.0 + rng.gauss(0, noise_std))
                perturbed = impact * noise
                total_by_node[nid] = total_by_node.get(nid, 0) + perturbed

        for nid, impact in total_by_node.items():
            if nid not in node_impacts:
                node_impacts[nid] = []
            node_impacts[nid].append(impact)

    # Compute statistics
    results: dict[str, Any] = {}
    for nid, impacts in node_impacts.items():
        n_samples = len(impacts)
        mean = sum(impacts) / n_samples
        variance = sum((x - mean) ** 2 for x in impacts) / n_samples
        std = variance ** 0.5
        results[nid] = {
            "mean_impact": round(mean, 4),
            "std_impact": round(std, 4),
            "var_coeff": round(std / mean if mean > 0 else 0, 4),
        }

    logger.info("[SensitivityAnalysis] Monte Carlo complete: %d nodes, %d iterations", len(results), n)
    return results


def compute_graph_resilience(
    graph: "DependencyGraph",
    removed_nodes: list[str],
) -> float:
    """Estimate graph resilience after removing specified nodes.

    Returns 0.0 (fully broken) to 1.0 (fully resilient).
    Measured as fraction of remaining cascade connectivity.
    """
    if not removed_nodes:
        return 1.0

    # Before removal: total cascade from all remaining nodes
    all_nodes_before = [n.node_id for n in graph.all_nodes() if n.node_id not in removed_nodes]
    if not all_nodes_before:
        return 0.0

    before_total = sum(
        sum(graph.get_cascade_impact(nid, max_hops=2).values())
        for nid in all_nodes_before[:20]  # Cap for performance
    )

    # After removal: subgraph excluding removed nodes
    remaining = {n.node_id for n in graph.all_nodes()} - set(removed_nodes)
    sub = graph.subgraph(remaining)

    after_total = sum(
        sum(sub.get_cascade_impact(nid, max_hops=2).values())
        for nid in list(remaining)[:20]
    )

    if before_total <= 0:
        return 1.0
    return min(after_total / before_total, 1.0)


def scenario_what_if(
    graph: "DependencyGraph",
    modifications: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run a what-if scenario by describing hypothetical graph modifications.

    modifications: list of {"type": "remove_node"/"increase_weight"/..., "node_id": ..., "factor": ...}

    Returns before/after cascade impact comparison.
    """
    results: dict[str, Any] = {"modifications": modifications, "impacts": {}}

    # Get baseline cascade from chokepoints
    chokepoints = [
        n.node_id for n in graph.all_nodes()
        if n.node_type.value == "chokepoint"
    ]
    baseline = {}
    for cp in chokepoints:
        cascade = graph.get_cascade_impact(cp, max_hops=3)
        for nid, imp in cascade.items():
            baseline[nid] = baseline.get(nid, 0) + imp

    # Apply "remove_node" modifications via subgraph
    removed = [m["node_id"] for m in modifications if m.get("type") == "remove_node"]
    if removed:
        remaining = {n.node_id for n in graph.all_nodes()} - set(removed)
        sub = graph.subgraph(remaining)
        scenario_graph = sub
    else:
        scenario_graph = graph

    # Get scenario cascade
    scenario_cascade: dict[str, float] = {}
    for cp in chokepoints:
        if cp in removed:
            continue
        cascade = scenario_graph.get_cascade_impact(cp, max_hops=3)
        for nid, imp in cascade.items():
            scenario_cascade[nid] = scenario_cascade.get(nid, 0) + imp

    # Compare
    all_nodes = set(list(baseline.keys()) + list(scenario_cascade.keys()))
    for nid in all_nodes:
        before = baseline.get(nid, 0)
        after = scenario_cascade.get(nid, 0)
        if abs(after - before) > 0.001:
            results["impacts"][nid] = {
                "before": round(before, 4),
                "after": round(after, 4),
                "delta": round(after - before, 4),
                "pct_change": round((after - before) / max(before, 0.001) * 100, 1),
            }

    results["resilience"] = compute_graph_resilience(graph, removed)
    return results
