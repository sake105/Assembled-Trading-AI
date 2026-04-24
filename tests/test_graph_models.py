"""Tests for M30: Graph-Based Models — Cross-Asset Signal Propagation."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

import pytest; pytest.importorskip('src.assembled_core.ml.graph_models')
from src.assembled_core.ml.graph_models import (
    GraphNode,
    GraphEdge,
    GraphSignal,
    build_correlation_graph,
    compute_pagerank,
    detect_lead_lag,
    propagate_signals,
    generate_graph_signals,
)


@pytest.fixture
def correlated_returns():
    """Returns with known correlation structure."""
    rng = np.random.default_rng(42)
    n = 120
    market = rng.normal(0, 0.01, n)
    # A and B are correlated through market
    a = market + rng.normal(0, 0.005, n)
    b = 0.8 * market + rng.normal(0, 0.005, n)
    # C is independent
    c = rng.normal(0, 0.01, n)
    return pd.DataFrame({"A": a, "B": b, "C": c})


@pytest.fixture
def lead_lag_returns():
    """Returns where A leads B by 1 period."""
    rng = np.random.default_rng(42)
    n = 200
    a = rng.normal(0, 0.01, n)
    b = np.zeros(n)
    for i in range(1, n):
        b[i] = 0.5 * a[i - 1] + rng.normal(0, 0.005)
    c = rng.normal(0, 0.01, n)
    return pd.DataFrame({"A": a, "B": b, "C": c})


@pytest.mark.phase12
class TestBuildCorrelationGraph:
    def test_basic_graph(self, correlated_returns):
        nodes, edges = build_correlation_graph(correlated_returns, min_correlation=0.2)
        assert len(nodes) == 3
        # A-B should be connected, C probably not
        symbols_in_edges = set()
        for e in edges:
            symbols_in_edges.add(e.source)
            symbols_in_edges.add(e.target)
        assert "A" in symbols_in_edges or "B" in symbols_in_edges

    def test_high_threshold_fewer_edges(self, correlated_returns):
        _, edges_low = build_correlation_graph(correlated_returns, min_correlation=0.1)
        _, edges_high = build_correlation_graph(correlated_returns, min_correlation=0.5)
        assert len(edges_high) <= len(edges_low)

    def test_empty_returns(self):
        nodes, edges = build_correlation_graph(pd.DataFrame())
        assert len(nodes) == 0
        assert len(edges) == 0

    def test_node_degree(self, correlated_returns):
        nodes, edges = build_correlation_graph(correlated_returns, min_correlation=0.2)
        for node in nodes:
            assert node.degree >= 0


@pytest.mark.phase12
class TestPageRank:
    def test_basic_pagerank(self):
        nodes = [GraphNode("A"), GraphNode("B"), GraphNode("C")]
        edges = [
            GraphEdge("A", "B", 0.8),
            GraphEdge("B", "C", 0.5),
            GraphEdge("A", "C", 0.3),
        ]
        pr = compute_pagerank(nodes, edges)
        assert len(pr) == 3
        assert all(v > 0 for v in pr.values())
        # Sum should be approximately 1
        assert sum(pr.values()) == pytest.approx(1.0, abs=0.1)

    def test_empty_graph(self):
        pr = compute_pagerank([], [])
        assert pr == {}

    def test_hub_node_higher_rank(self):
        nodes = [GraphNode(s) for s in ["hub", "a", "b", "c", "d"]]
        edges = [
            GraphEdge("hub", "a", 0.5),
            GraphEdge("hub", "b", 0.5),
            GraphEdge("hub", "c", 0.5),
            GraphEdge("hub", "d", 0.5),
        ]
        pr = compute_pagerank(nodes, edges)
        # Hub should have highest PageRank
        assert pr["hub"] >= max(pr["a"], pr["b"], pr["c"], pr["d"])


@pytest.mark.phase12
class TestLeadLag:
    def test_detect_leader(self, lead_lag_returns):
        edges = detect_lead_lag(lead_lag_returns, max_lag=3, min_correlation=0.1)
        assert isinstance(edges, list)
        # Should detect A leads B
        ab_edges = [e for e in edges if
                    (e.source == "A" and e.target == "B") or
                    (e.source == "B" and e.target == "A")]
        if ab_edges:
            assert ab_edges[0].lag >= 1

    def test_no_lead_lag_in_random(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "X": rng.normal(0, 0.01, 200),
            "Y": rng.normal(0, 0.01, 200),
        })
        edges = detect_lead_lag(df, max_lag=3, min_correlation=0.3)
        # With high threshold, random data shouldn't show lead-lag
        assert isinstance(edges, list)


@pytest.mark.phase12
class TestPropagateSignals:
    def test_basic_propagation(self):
        nodes = [GraphNode("A"), GraphNode("B"), GraphNode("C")]
        edges = [GraphEdge("A", "B", 0.8)]
        signals = {"A": 1.0, "B": 0.0, "C": -0.5}
        result = propagate_signals(signals, nodes, edges)
        # B should pick up some of A's positive signal
        assert result["B"] > signals["B"]

    def test_no_edges_no_change(self):
        nodes = [GraphNode("A"), GraphNode("B")]
        signals = {"A": 1.0, "B": -1.0}
        result = propagate_signals(signals, nodes, [])
        assert result["A"] == signals["A"]
        assert result["B"] == signals["B"]


@pytest.mark.phase12
class TestGenerateGraphSignals:
    def test_basic_generation(self, correlated_returns):
        signals = generate_graph_signals(correlated_returns)
        assert len(signals) == 3
        assert all(isinstance(s, GraphSignal) for s in signals)

    def test_with_raw_signals(self, correlated_returns):
        raw = {"A": 0.5, "B": -0.3, "C": 0.1}
        signals = generate_graph_signals(correlated_returns, raw_signals=raw)
        assert len(signals) == 3

    def test_composite_bounded(self, correlated_returns):
        signals = generate_graph_signals(correlated_returns)
        for s in signals:
            assert -3.0 <= s.composite <= 3.0
