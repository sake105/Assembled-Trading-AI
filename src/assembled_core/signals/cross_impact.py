"""Cross-impact graph for sector news propagation.

From 30_NEWS_TA_FUSION.md §Layer 4 — optional cross-impact via Pearson-correlation graph.

When news hits ticker A, sentiment propagates with weight = correlation to
connected tickers in the graph. Uses Ledoit-Wolf shrinkage for robustness.

Note: networkx and scikit-learn are required. Both are optional extras.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def build_cross_impact_graph(
    returns_df: "pd.DataFrame",
    window: int = 60,
    threshold: float = 0.5,
):
    """Build a ticker correlation graph via Ledoit-Wolf covariance shrinkage.

    Args:
        returns_df: DataFrame of daily returns, columns = tickers.
        window: Lookback window (most recent N rows used).
        threshold: Min absolute correlation for an edge to be added.

    Returns:
        networkx.Graph with edges weighted by Pearson correlation, or None
        if networkx / sklearn are not installed.
    """
    try:
        import networkx as nx
        from sklearn.covariance import LedoitWolf
    except ImportError:
        logger.warning("networkx or sklearn not available — build_cross_impact_graph returns None")
        return None

    data = returns_df.tail(window).dropna(axis=1, how="any")
    if data.shape[0] < 5 or data.shape[1] < 2:
        logger.warning("Insufficient data for cross-impact graph (%s rows, %s tickers)", data.shape[0], data.shape[1])
        return nx.Graph()

    lw = LedoitWolf()
    lw.fit(data)
    cov = lw.covariance_

    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 0.0)  # no self-loops

    G = nx.Graph()
    tickers = list(data.columns)
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i < j and abs(corr[i, j]) > threshold:
                G.add_edge(t1, t2, weight=float(corr[i, j]))
    return G


def propagate_through_graph(
    ticker: str,
    news_features: dict,
    graph,
    sentiment_key: str = "aggregate_z",
    decay: float = 0.5,
) -> float:
    """Propagate ticker sentiment to neighbors via graph edge weights.

    Args:
        ticker: Source ticker with news event.
        news_features: Dict with at least ``sentiment_key`` for the source ticker.
        graph: networkx.Graph from build_cross_impact_graph.
        sentiment_key: Key to use as source sentiment.
        decay: Correlation weight dampening factor.

    Returns:
        Weighted-average sector sentiment in [-1, +1]. Returns 0.0 if graph
        is None or ticker not in graph.
    """
    if graph is None or ticker not in graph:
        return 0.0
    source_sentiment = float(news_features.get(sentiment_key, 0.0))
    neighbors = list(graph.neighbors(ticker))
    if not neighbors:
        return 0.0
    weights = [abs(graph[ticker][n].get("weight", 0.0)) for n in neighbors]
    total = sum(weights)
    if total == 0:
        return 0.0
    return source_sentiment * decay * sum(w / total for w in weights)


__all__ = ["build_cross_impact_graph", "propagate_through_graph"]
