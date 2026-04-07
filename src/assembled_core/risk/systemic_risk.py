"""Systemic Risk via Network Analysis (Plan 7.9).

Pairwise Granger causality → directed graph → centrality scores.
Identifies systemically important assets.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_return_network_centrality(
    returns: pd.DataFrame,
    correlation_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute network centrality from return correlations.

    Simplified version: uses correlation > threshold as edge criterion,
    then computes degree centrality.

    Args:
        returns: Returns DataFrame (columns = symbols).
        correlation_threshold: Min correlation for edge.

    Returns:
        Symbol → centrality score (0-1).
    """
    corr = returns.corr()
    n = len(corr)
    symbols = list(corr.columns)

    centrality: dict[str, float] = {}
    for i, sym in enumerate(symbols):
        connections = sum(
            1 for j in range(n) if i != j and abs(corr.iloc[i, j]) > correlation_threshold
        )
        centrality[sym] = round(connections / max(n - 1, 1), 4)

    return centrality


__all__ = ["compute_return_network_centrality"]
