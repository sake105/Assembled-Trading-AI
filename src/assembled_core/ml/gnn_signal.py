"""GNN (Graph Neural Network) signal generator — stub / skeleton.

Tier 4 item: GNN requires PyTorch Geometric and substantial training data
(co-movement graph, earnings surprise propagation, supply-chain links).
This stub exposes the intended interface so downstream code can wire against
it without waiting for the full implementation.

The stub returns zero signals and raises NotImplementedError for training.
Replace the stub body with actual PyTorch Geometric code when:
  1. PyG + CUDA environment is confirmed.
  2. Co-movement adjacency matrix pipeline is ready.
  3. Node feature engineering (returns, vol, sentiment, macro) is defined.

References:
  - Kipf & Welling (2016) "Semi-Supervised Classification with GCN"
  - Hamilton et al. (2017) "Inductive Representation Learning on Large Graphs" (GraphSAGE)
  - Xu et al. (2018) "How Powerful are Graph Neural Networks?" (GIN)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
_PYG_AVAILABLE = False

try:
    import torch  # type: ignore[import]
    _TORCH_AVAILABLE = True
    import torch_geometric  # type: ignore[import]
    _PYG_AVAILABLE = True
except ImportError:
    pass


@dataclass
class GNNConfig:
    """Configuration for the GNN signal model."""
    n_node_features: int = 16       # per-symbol feature vector dimension
    hidden_dim: int = 64            # GNN hidden layer width
    n_layers: int = 3               # message-passing rounds
    output_dim: int = 1             # signal per node (1 = scalar alpha score)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 100


@dataclass
class GNNSignalResult:
    """Output of the GNN signal generator."""
    symbols: list[str]
    scores: dict[str, float]        # symbol → alpha score
    n_edges: int                    # edges in the co-movement graph
    backend: str                    # "pytorch_geometric" or "stub"


class GNNSignalModel:
    """Graph Neural Network alpha signal generator.

    This is a skeleton implementation. The interface is stable; the internals
    will be filled in when PyTorch Geometric is available in the environment.

    Args:
        config: Model configuration.
        adjacency_threshold: Minimum absolute correlation to create a graph edge.
    """

    def __init__(
        self,
        config: GNNConfig | None = None,
        adjacency_threshold: float = 0.4,
    ) -> None:
        self.config = config or GNNConfig()
        self._adj_threshold = adjacency_threshold
        self._model: Any = None
        self._is_trained = False
        self._symbols: list[str] = []

        if not _PYG_AVAILABLE:
            logger.info(
                "[GNNSignal] torch-geometric not installed — stub mode active. "
                "Install with: pip install torch-geometric"
            )

    def build_graph(
        self,
        returns_matrix: np.ndarray,
        symbols: list[str],
    ) -> tuple[Any, Any]:
        """Build adjacency list from correlation of return series.

        Args:
            returns_matrix: shape (T, N) — T days, N symbols.
            symbols: Symbol labels for each column.

        Returns:
            (edge_index, edge_weights) suitable for PyG — or (None, None) in stub mode.
        """
        if not _TORCH_AVAILABLE:
            return None, None

        import torch
        n = returns_matrix.shape[1]
        corr = np.corrcoef(returns_matrix.T)
        edges_src, edges_dst, weights = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) >= self._adj_threshold:
                    edges_src += [i, j]
                    edges_dst += [j, i]
                    weights += [corr[i, j], corr[i, j]]

        if not edges_src:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_weights = torch.tensor(weights, dtype=torch.float)
        return edge_index, edge_weights

    def fit(
        self,
        returns_matrix: np.ndarray,
        symbols: list[str],
        labels: np.ndarray | None = None,
    ) -> None:
        """Train the GNN on historical return data.

        Args:
            returns_matrix: (T, N) daily returns.
            symbols: N symbol names.
            labels: (N,) target labels (e.g. next-period alpha). If None,
                    unsupervised pre-training mode (not yet implemented).

        Raises:
            NotImplementedError: Always in stub mode.
        """
        if not _PYG_AVAILABLE:
            raise NotImplementedError(
                "GNNSignalModel.fit() requires torch-geometric. "
                "Install it, then implement the GCN/GraphSAGE forward pass."
            )
        self._symbols = symbols
        self._is_trained = False
        raise NotImplementedError("Full GNN training not yet implemented. See stub docstring.")

    def predict(
        self,
        node_features: np.ndarray,
        symbols: list[str] | None = None,
        returns_matrix: np.ndarray | None = None,
    ) -> GNNSignalResult:
        """Generate alpha scores for each node (symbol).

        Args:
            node_features: (N, F) feature matrix for current period.
            symbols: Symbol names; falls back to symbols from fit().
            returns_matrix: If provided, rebuild graph from current correlations.

        Returns:
            GNNSignalResult with zero scores in stub mode.
        """
        syms = symbols or self._symbols or [f"SYM_{i}" for i in range(len(node_features))]

        if not _PYG_AVAILABLE or not self._is_trained:
            logger.debug("[GNNSignal] stub predict — returning zero scores for %d symbols", len(syms))
            return GNNSignalResult(
                symbols=syms,
                scores={s: 0.0 for s in syms},
                n_edges=0,
                backend="stub",
            )

        # Full implementation placeholder (unreachable in current stub)
        raise NotImplementedError("GNN inference not implemented — stub mode only.")

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def pyg_available(self) -> bool:
        return _PYG_AVAILABLE


TORCH_AVAILABLE = _TORCH_AVAILABLE
PYG_AVAILABLE = _PYG_AVAILABLE
