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
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    graph = DependencyGraph()

    for raw_node in data.get("nodes", []):
        graph.add_node(_parse_node(raw_node))

    for raw_edge in data.get("edges", []):
        graph.add_edge(_parse_edge(raw_edge))

    return graph
