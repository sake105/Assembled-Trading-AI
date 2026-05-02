"""Load and traverse the geopolitical dependency graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import DependencyEdge, DependencyNode, EdgeType, NodeType


class DependencyGraph:
    """In-memory representation of the geopolitical dependency graph."""

    def __init__(self) -> None:
        self._nodes: dict[str, DependencyNode] = {}
        # adjacency: from_node -> list of (edge, to_node)
        self._adj_out: dict[str, list[tuple[DependencyEdge, DependencyNode]]] = {}
        # reverse adjacency: to_node -> list of (edge, from_node)
        self._adj_in: dict[str, list[tuple[DependencyEdge, DependencyNode]]] = {}

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def add_node(self, node: DependencyNode) -> None:
        self._nodes[node.node_id] = node
        if node.node_id not in self._adj_out:
            self._adj_out[node.node_id] = []
        if node.node_id not in self._adj_in:
            self._adj_in[node.node_id] = []

    def add_edge(self, edge: DependencyEdge) -> None:
        from_node = self._nodes.get(edge.from_node)
        to_node = self._nodes.get(edge.to_node)
        # Allow edges where source or target node is not yet defined (virtual nodes)
        if from_node is None:
            from_node = DependencyNode(
                node_id=edge.from_node,
                node_type=NodeType.ASSET,
                name=edge.from_node,
            )
            self._nodes[edge.from_node] = from_node
            self._adj_out[edge.from_node] = []
            self._adj_in[edge.from_node] = []
        if to_node is None:
            to_node = DependencyNode(
                node_id=edge.to_node,
                node_type=NodeType.ASSET,
                name=edge.to_node,
            )
            self._nodes[edge.to_node] = to_node
            self._adj_out[edge.to_node] = []
            self._adj_in[edge.to_node] = []

        if edge.from_node not in self._adj_out:
            self._adj_out[edge.from_node] = []
        if edge.to_node not in self._adj_in:
            self._adj_in[edge.to_node] = []

        self._adj_out[edge.from_node].append((edge, to_node))
        self._adj_in[edge.to_node].append((edge, from_node))

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> DependencyNode | None:
        return self._nodes.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        edge_types: list[EdgeType] | None = None,
    ) -> list[tuple[DependencyEdge, DependencyNode]]:
        """Return outgoing neighbors. Optionally filter by edge type."""
        neighbors = self._adj_out.get(node_id, [])
        if edge_types is None:
            return list(neighbors)
        return [(e, n) for e, n in neighbors if e.edge_type in edge_types]

    def get_asset_nodes(self) -> list[DependencyNode]:
        """Return nodes that have tradeable assets (sector, asset, macro_index)."""
        tradeable_types = {NodeType.SECTOR, NodeType.ASSET, NodeType.MACRO_INDEX}
        return [n for n in self._nodes.values() if n.node_type in tradeable_types]

    def find_paths(
        self,
        from_node: str,
        to_node: str,
        max_depth: int = 4,
    ) -> list[list[str]]:
        """BFS to find all paths from from_node to to_node up to max_depth edges."""
        if from_node not in self._nodes or to_node not in self._nodes:
            return []

        results: list[list[str]] = []
        # Queue of (current_node_id, path_so_far)
        queue: list[tuple[str, list[str]]] = [(from_node, [from_node])]

        while queue:
            current, path = queue.pop(0)
            if current == to_node and len(path) > 1:
                results.append(path)
                continue
            if len(path) > max_depth:
                continue
            for edge, neighbor in self._adj_out.get(current, []):
                if neighbor.node_id not in path:  # avoid cycles
                    queue.append((neighbor.node_id, path + [neighbor.node_id]))

        return results

    def all_nodes(self) -> list[DependencyNode]:
        return list(self._nodes.values())

    def all_edges(self) -> list[DependencyEdge]:
        edges = []
        for edge_list in self._adj_out.values():
            for edge, _ in edge_list:
                edges.append(edge)
        return edges

    # ------------------------------------------------------------------
    # M15 extended query API
    # ------------------------------------------------------------------

    def get_nodes_by_type(self, node_type: NodeType) -> list[DependencyNode]:
        """Return all nodes of a given type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def get_reverse_neighbors(
        self,
        node_id: str,
        edge_types: list[EdgeType] | None = None,
    ) -> list[tuple[DependencyEdge, DependencyNode]]:
        """Return incoming neighbors (who points to this node)."""
        neighbors = self._adj_in.get(node_id, [])
        if edge_types is None:
            return list(neighbors)
        return [(e, n) for e, n in neighbors if e.edge_type in edge_types]

    def subgraph(self, node_ids: set[str]) -> "DependencyGraph":
        """Extract a subgraph containing only the specified node IDs."""
        sub = DependencyGraph()
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node is not None:
                sub.add_node(node)
        for nid in node_ids:
            for edge, neighbor in self._adj_out.get(nid, []):
                if neighbor.node_id in node_ids:
                    sub.add_edge(edge)
        return sub

    def compute_betweenness_centrality(self) -> dict[str, float]:
        """Approximate betweenness centrality for all nodes.

        Uses BFS from every node to estimate how often each node lies on
        shortest paths. Identifies critical bottleneck nodes in the network.
        """
        node_ids = list(self._nodes.keys())
        centrality: dict[str, float] = {nid: 0.0 for nid in node_ids}
        n = len(node_ids)
        if n < 3:
            return centrality

        for source in node_ids:
            # BFS shortest-path tree
            dist: dict[str, int] = {source: 0}
            paths: dict[str, int] = {source: 1}
            pred: dict[str, list[str]] = {nid: [] for nid in node_ids}
            queue = [source]
            order: list[str] = []

            while queue:
                current = queue.pop(0)
                order.append(current)
                for _, neighbor in self._adj_out.get(current, []):
                    nid = neighbor.node_id
                    if nid not in dist:
                        dist[nid] = dist[current] + 1
                        paths[nid] = 0
                        queue.append(nid)
                    if dist.get(nid, -1) == dist[current] + 1:
                        paths[nid] = paths.get(nid, 0) + paths[current]
                        pred[nid].append(current)

            # Accumulate dependencies (Brandes algorithm)
            delta: dict[str, float] = {nid: 0.0 for nid in node_ids}
            for w in reversed(order):
                for v in pred[w]:
                    if paths.get(w, 0) > 0:
                        delta[v] += (paths.get(v, 0) / paths[w]) * (1.0 + delta[w])
                if w != source:
                    centrality[w] += delta[w]

        # Normalize
        norm = max(1, (n - 1) * (n - 2))
        return {nid: v / norm for nid, v in centrality.items()}

    def compute_vulnerability_index(self, node_id: str) -> float:
        """Compute how vulnerable a node is based on in-degree concentration.

        A node that depends on few critical suppliers has high vulnerability.
        Returns 0.0 (resilient) to 1.0 (extremely vulnerable).
        """
        in_edges = self._adj_in.get(node_id, [])
        if not in_edges:
            return 0.0
        weights = [e.weight for e, _ in in_edges]
        if not weights:
            return 0.0
        max_w = max(weights)
        avg_w = sum(weights) / len(weights)
        concentration = max_w / max(sum(weights), 1e-9)
        return min(1.0, concentration * avg_w * len(weights) / max(len(weights), 1))

    def detect_single_points_of_failure(self) -> list[str]:
        """Identify nodes whose removal would disconnect significant subgraphs.

        Returns node IDs sorted by criticality (highest first).
        """
        centrality = self.compute_betweenness_centrality()
        threshold = 0.05  # top 5% centrality
        if not centrality:
            return []
        max_c = max(centrality.values()) if centrality else 0
        if max_c == 0:
            return []
        return sorted(
            [nid for nid, c in centrality.items() if c >= threshold * max_c],
            key=lambda nid: centrality[nid],
            reverse=True,
        )

    def get_cascade_impact(
        self,
        node_id: str,
        max_hops: int = 5,
    ) -> dict[str, float]:
        """Estimate downstream cascade impact from a node disruption.

        Returns {node_id: impact_score} for all reachable nodes,
        with impact decaying per hop (0.85 dampening).
        """
        dampening = 0.85
        impacts: dict[str, float] = {}
        queue: list[tuple[str, float, int]] = [(node_id, 1.0, 0)]
        visited: set[str] = {node_id}

        while queue:
            current, impact, hops = queue.pop(0)
            if hops > max_hops:
                continue
            for edge, neighbor in self._adj_out.get(current, []):
                nid = neighbor.node_id
                propagated = impact * edge.weight * edge.confidence * dampening
                if propagated < 0.01:
                    continue
                if nid not in visited or propagated > impacts.get(nid, 0):
                    impacts[nid] = max(impacts.get(nid, 0), propagated)
                    if nid not in visited:
                        visited.add(nid)
                        queue.append((nid, propagated, hops + 1))

        return dict(sorted(impacts.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _parse_node(raw: dict[str, Any]) -> DependencyNode:
    return DependencyNode(
        node_id=raw["node_id"],
        node_type=NodeType(raw["node_type"]),
        name=raw["name"],
        attributes=raw.get("attributes", {}),
    )


def _parse_edge(raw: dict[str, Any]) -> DependencyEdge:
    return DependencyEdge(
        from_node=raw["from_node"],
        to_node=raw["to_node"],
        edge_type=EdgeType(raw["edge_type"]),
        weight=float(raw["weight"]),
        direction=raw["direction"],
        lag_hours=int(raw["lag_hours"]),
        confidence=float(raw.get("confidence", 1.0)),
        source_refs=raw.get("source_refs", []),
    )


def load_graph(path: str | Path) -> DependencyGraph:
    """Load a DependencyGraph from a YAML file."""
    import logging as _logging
    _log = _logging.getLogger(__name__)

    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"[DependencyGraph] Malformed YAML in {path}: {exc}") from exc

    graph = DependencyGraph()

    for raw_node in data.get("nodes", []):
        try:
            graph.add_node(_parse_node(raw_node))
        except (KeyError, ValueError, TypeError) as exc:
            _log.warning("[DependencyGraph] Skipping malformed node %s: %s", raw_node.get("node_id", "?"), exc)

    for raw_edge in data.get("edges", []):
        try:
            graph.add_edge(_parse_edge(raw_edge))
        except (KeyError, ValueError, TypeError) as exc:
            _log.warning("[DependencyGraph] Skipping malformed edge %s→%s: %s", raw_edge.get("from_node", "?"), raw_edge.get("to_node", "?"), exc)

    return graph
