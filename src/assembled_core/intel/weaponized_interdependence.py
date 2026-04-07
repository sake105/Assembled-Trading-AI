"""Weaponized Interdependence scoring (Farrell & Newman 2019).

Quantifies asymmetric dependencies in global networks that can be
"weaponized" by states controlling choke points.  Two key dimensions:

1. **Sensitivity** — how dependent is A on trade/services from B?
   ``sensitivity(A→B) = trade_volume(A→B) / total_imports(A)``

2. **Vulnerability** — how much damage can B inflict by cutting off A?
   ``vulnerability(A→B) = market_share(B_in_A) × (1/substitutability) × centrality(B)``

3. **WI Score** — asymmetry of dependence:
   ``wi_score = vulnerability / max(sensitivity, epsilon)``

Known weaponizable chokepoints: US dollar system (SWIFT), semiconductor
tooling (ASML/US), rare earths (China), energy (Russia→EU), internet
backbone.

The Panoptikon Effect captures nodes that control information flow
(SWIFT, internet exchanges, TSMC).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WIScore:
    """Weaponized Interdependence score for a directed node pair."""

    source: str  # the node that could be weaponized against
    target: str  # the node that would suffer
    sensitivity: float  # how dependent target is on source
    vulnerability: float  # how much damage source can inflict
    wi_score: float  # vulnerability / sensitivity asymmetry
    substitutability: float  # how easy to replace source (0=impossible, 1=easy)
    is_chokepoint: bool  # source is a network chokepoint


@dataclass
class PanoptikonNode:
    """Node with information/flow control power (Panoptikon Effect)."""

    node: str
    betweenness_centrality: float
    flow_control_score: float  # fraction of total network flow passing through
    information_weapon_score: float  # ability to surveil or disrupt via information


# Known weaponizable relationships (static knowledge base)
KNOWN_WI_PAIRS: list[dict] = [
    {
        "source": "US", "target": "CN", "domain": "semiconductors",
        "description": "US controls semiconductor tooling (ASML, Applied Materials, Lam Research)",
        "substitutability": 0.1, "estimated_vulnerability": 0.9,
    },
    {
        "source": "US", "target": "WORLD", "domain": "dollar_system",
        "description": "US controls SWIFT access and dollar clearing",
        "substitutability": 0.15, "estimated_vulnerability": 0.85,
    },
    {
        "source": "CN", "target": "WORLD", "domain": "rare_earths",
        "description": "China controls ~60% of rare earth mining, ~90% of processing",
        "substitutability": 0.2, "estimated_vulnerability": 0.8,
    },
    {
        "source": "TW", "target": "WORLD", "domain": "advanced_chips",
        "description": "TSMC produces ~90% of advanced (<7nm) semiconductors",
        "substitutability": 0.05, "estimated_vulnerability": 0.95,
    },
    {
        "source": "RU", "target": "EU", "domain": "energy",
        "description": "Russia was primary gas supplier to EU (pre-2022 ~40%)",
        "substitutability": 0.35, "estimated_vulnerability": 0.65,
    },
    {
        "source": "CN", "target": "WORLD", "domain": "solar_panels",
        "description": "China produces ~80% of global solar panel supply chain",
        "substitutability": 0.15, "estimated_vulnerability": 0.75,
    },
    {
        "source": "US", "target": "CN", "domain": "ai_chips",
        "description": "US export controls on advanced AI chips (A100/H100+)",
        "substitutability": 0.1, "estimated_vulnerability": 0.85,
    },
]


def compute_wi_score(
    trade_volume_a_to_b: float,
    total_imports_a: float,
    market_share_b_in_a: float,
    substitutability: float,
    centrality_b: float,
) -> WIScore:
    """Compute WI score for a directed pair.

    Args:
        trade_volume_a_to_b: Value of trade from A to B.
        total_imports_a: Total imports of A from all sources.
        market_share_b_in_a: B's market share in A's imports for the
            relevant commodity/service.
        substitutability: Ease of replacing B (0 = impossible, 1 = trivial).
        centrality_b: Network centrality of B (0-1).

    Returns:
        WIScore dataclass.
    """
    eps = 0.01

    sensitivity = trade_volume_a_to_b / max(total_imports_a, eps)
    vulnerability = market_share_b_in_a * (1.0 / max(substitutability, eps)) * centrality_b
    # Normalize vulnerability to reasonable range
    vulnerability = min(vulnerability, 10.0)

    wi = vulnerability / max(sensitivity, eps)

    return WIScore(
        source="B",
        target="A",
        sensitivity=round(sensitivity, 4),
        vulnerability=round(vulnerability, 4),
        wi_score=round(wi, 4),
        substitutability=round(substitutability, 4),
        is_chokepoint=vulnerability > 2.0 and substitutability < 0.3,
    )


def compute_panoptikon_scores(
    adjacency: dict[str, dict[str, float]],
) -> list[PanoptikonNode]:
    """Identify Panoptikon nodes — those that control information/flow.

    Uses betweenness centrality on the adjacency graph to find nodes
    through which a disproportionate share of network traffic flows.

    Args:
        adjacency: Dict of dicts representing weighted directed graph.
            ``adjacency[src][tgt] = weight``.

    Returns:
        List of PanoptikonNode sorted by flow_control_score descending.
    """
    nodes = set()
    for src, targets in adjacency.items():
        nodes.add(src)
        nodes.update(targets.keys())
    nodes = sorted(nodes)

    if len(nodes) < 3:
        return []

    # Compute betweenness centrality via simplified BFS
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    betweenness = np.zeros(n)

    for s_idx, s_node in enumerate(nodes):
        # BFS from s_node
        dist = np.full(n, np.inf)
        dist[s_idx] = 0
        n_paths = np.zeros(n)
        n_paths[s_idx] = 1
        order = []
        queue = [s_idx]

        while queue:
            current = queue.pop(0)
            order.append(current)
            current_node = nodes[current]
            for neighbor, _ in adjacency.get(current_node, {}).items():
                n_idx = node_idx[neighbor]
                if dist[n_idx] == np.inf:
                    dist[n_idx] = dist[current] + 1
                    queue.append(n_idx)
                if dist[n_idx] == dist[current] + 1:
                    n_paths[n_idx] += n_paths[current]

        # Back-propagate dependencies
        delta = np.zeros(n)
        for v in reversed(order):
            v_node = nodes[v]
            for neighbor, _ in adjacency.get(v_node, {}).items():
                w = node_idx[neighbor]
                if dist[w] == dist[v] + 1 and n_paths[w] > 0:
                    delta[v] += (n_paths[v] / n_paths[w]) * (1 + delta[w])
            if v != s_idx:
                betweenness[v] += delta[v]

    # Normalize
    max_bc = betweenness.max()
    if max_bc > 0:
        betweenness /= max_bc

    # Total flow through each node
    total_flow = sum(
        w for targets in adjacency.values() for w in targets.values()
    )

    results = []
    for i, node in enumerate(nodes):
        node_flow = sum(adjacency.get(node, {}).values())
        flow_pct = node_flow / total_flow if total_flow > 0 else 0

        results.append(PanoptikonNode(
            node=node,
            betweenness_centrality=round(float(betweenness[i]), 4),
            flow_control_score=round(float(flow_pct), 4),
            information_weapon_score=round(float(betweenness[i] * (1 + flow_pct)), 4),
        ))

    results.sort(key=lambda x: x.information_weapon_score, reverse=True)
    return results


def get_known_wi_pairs() -> list[dict]:
    """Return the static knowledge base of known WI relationships."""
    return KNOWN_WI_PAIRS.copy()


def score_symbol_wi_exposure(
    symbol: str,
    symbol_sectors: dict[str, str],
    symbol_countries: dict[str, list[str]],
) -> dict[str, float]:
    """Score a symbol's exposure to known weaponized interdependence risks.

    Maps the symbol's sector and country exposure to known WI pairs and
    returns a composite exposure score.

    Args:
        symbol: Symbol to score.
        symbol_sectors: Dict mapping symbol → sector.
        symbol_countries: Dict mapping symbol → list of country codes.

    Returns:
        Dict with ``wi_exposure``, ``wi_semiconductor_risk``,
        ``wi_energy_risk``, ``wi_dollar_risk``.
    """
    sector = symbol_sectors.get(symbol, "unknown")
    countries = set(symbol_countries.get(symbol, []))

    scores = {
        "wi_exposure": 0.0,
        "wi_semiconductor_risk": 0.0,
        "wi_energy_risk": 0.0,
        "wi_dollar_risk": 0.0,
    }

    for pair in KNOWN_WI_PAIRS:
        domain = pair["domain"]
        vuln = pair["estimated_vulnerability"]

        # Semiconductor exposure
        if domain in ("semiconductors", "advanced_chips", "ai_chips"):
            if sector in ("Technology", "Semiconductors", "Electronics"):
                scores["wi_semiconductor_risk"] = max(
                    scores["wi_semiconductor_risk"], vuln * 0.8
                )
            if "CN" in countries or "TW" in countries:
                scores["wi_semiconductor_risk"] = max(
                    scores["wi_semiconductor_risk"], vuln * 0.6
                )

        # Energy exposure
        if domain == "energy":
            if "EU" in countries or any(c in countries for c in ["DE", "FR", "IT", "PL"]):
                scores["wi_energy_risk"] = max(
                    scores["wi_energy_risk"], vuln * 0.7
                )

        # Dollar system exposure
        if domain == "dollar_system":
            if any(c in countries for c in ["CN", "RU", "IR", "KP"]):
                scores["wi_dollar_risk"] = max(
                    scores["wi_dollar_risk"], vuln * 0.8
                )

    # Composite
    scores["wi_exposure"] = round(
        0.4 * scores["wi_semiconductor_risk"]
        + 0.3 * scores["wi_energy_risk"]
        + 0.3 * scores["wi_dollar_risk"],
        4,
    )

    return scores


__all__ = [
    "PanoptikonNode",
    "WIScore",
    "compute_panoptikon_scores",
    "compute_wi_score",
    "get_known_wi_pairs",
    "score_symbol_wi_exposure",
]
