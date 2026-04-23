"""Tests for wave-61 module wiring into trading_cycle.py.

Covers:
  Step 8.60 — ml.gnn_stocks (GNNConfig / build_stock_graph / GNNEmbedder)
  Step 8.61 — ml.graph_models (build_correlation_graph / generate_graph_signals)
  Step 8.62 — ml.maml (MAMLConfig / MAMLPredictor)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.gnn_stocks import (
    GNNConfig,
    StockGraph,
    build_stock_graph,
    TORCH_AVAILABLE,
)
from src.assembled_core.ml.graph_models import (
    GraphNode,
    GraphEdge,
    build_correlation_graph,
    compute_pagerank,
    generate_graph_signals,
)
from src.assembled_core.ml.maml import (
    MAMLConfig,
    MAMLPredictor,
    MAMLResult,
    TORCH_AVAILABLE as MAML_TORCH,
)


# ---------------------------------------------------------------------------
# gnn_stocks (Step 8.60)
# ---------------------------------------------------------------------------

def test_gnn_config_creates():
    cfg = GNNConfig()
    assert isinstance(cfg, GNNConfig)


def test_gnn_config_defaults():
    cfg = GNNConfig()
    assert cfg.embedding_dim > 0
    assert cfg.n_layers > 0


def test_gnn_torch_flag():
    assert isinstance(TORCH_AVAILABLE, bool)


def test_build_stock_graph_returns_graph():
    rng = np.random.default_rng(0)
    symbols = ["AAPL", "MSFT", "GOOG"]
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    returns = pd.DataFrame(rng.normal(0, 0.01, (60, 3)), index=idx, columns=symbols)
    graph = build_stock_graph(returns)
    assert isinstance(graph, StockGraph)


def test_build_stock_graph_has_adjacency():
    rng = np.random.default_rng(0)
    symbols = ["A", "B", "C"]
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    returns = pd.DataFrame(rng.normal(0, 0.01, (60, 3)), index=idx, columns=symbols)
    graph = build_stock_graph(returns)
    assert graph.adjacency.shape == (3, 3)


def test_build_stock_graph_with_sector_map():
    rng = np.random.default_rng(0)
    symbols = ["AAPL", "MSFT", "XOM"]
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    returns = pd.DataFrame(rng.normal(0, 0.01, (60, 3)), index=idx, columns=symbols)
    sector_map = {"AAPL": "tech", "MSFT": "tech", "XOM": "energy"}
    graph = build_stock_graph(returns, sector_map=sector_map)
    assert graph.adjacency[0, 1] > 0  # tech-tech edge


# ---------------------------------------------------------------------------
# graph_models (Step 8.61)
# ---------------------------------------------------------------------------

def test_build_correlation_graph_returns_nodes_edges():
    rng = np.random.default_rng(0)
    symbols = ["A", "B", "C", "D"]
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    returns = pd.DataFrame(rng.normal(0, 0.01, (80, 4)), index=idx, columns=symbols)
    nodes, edges = build_correlation_graph(returns)
    assert isinstance(nodes, list)
    assert isinstance(edges, list)
    assert all(isinstance(n, GraphNode) for n in nodes)


def test_compute_pagerank_returns_dict():
    nodes = [GraphNode(symbol="A", sector="tech", centrality=0.0, degree=1),
             GraphNode(symbol="B", sector="tech", centrality=0.0, degree=1)]
    edges = [GraphEdge(source="A", target="B", weight=0.5, edge_type="correlation")]
    result = compute_pagerank(nodes, edges)
    assert isinstance(result, dict)
    assert "A" in result and "B" in result


def test_generate_graph_signals_returns_list():
    rng = np.random.default_rng(0)
    symbols = ["A", "B", "C"]
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    returns = pd.DataFrame(rng.normal(0, 0.01, (80, 3)), index=idx, columns=symbols)
    signals = generate_graph_signals(returns)
    assert isinstance(signals, list)


# ---------------------------------------------------------------------------
# maml (Step 8.62)
# ---------------------------------------------------------------------------

def test_maml_config_creates():
    cfg = MAMLConfig()
    assert isinstance(cfg, MAMLConfig)


def test_maml_config_defaults():
    cfg = MAMLConfig()
    assert cfg.inner_lr > 0
    assert cfg.inner_steps > 0


def test_maml_torch_flag():
    assert isinstance(MAML_TORCH, bool)


def test_maml_predictor_creates():
    model = MAMLPredictor()
    assert isinstance(model, MAMLPredictor)


def test_maml_predictor_meta_train_fallback():
    rng = np.random.default_rng(0)
    regime_data = {
        "BULL": (rng.normal(0, 1, (30, 4)).astype(np.float32), rng.normal(0, 1, 30).astype(np.float32)),
        "BEAR": (rng.normal(0, 1, (30, 4)).astype(np.float32), rng.normal(0, 1, 30).astype(np.float32)),
    }
    model = MAMLPredictor(config=MAMLConfig(n_meta_epochs=2))
    result = model.meta_train(regime_data)
    assert isinstance(result, MAMLResult)
