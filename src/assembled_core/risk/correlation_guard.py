"""Correlation / cluster guard: detect and scale down concentrated correlated clusters.

When a portfolio holds multiple positively correlated positions their effective
risk is concentrated beyond what individual weights suggest. This module:

  1. Computes pairwise return correlations for the held symbols.
  2. Groups symbols into clusters (positive correlation >= threshold).
  3. Detects clusters whose combined weight exceeds ``max_cluster_weight``.
  4. Proportionally scales down all positions in over-concentrated clusters.

This is a pre-order sizing overlay, not a state machine. It has no side effects
and is deterministic per call.

M6-T07: implement correlation / cluster guard.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _pivot_returns(
    prices: pd.DataFrame,
    symbols: list[str],
    lookback_days: int,
) -> pd.DataFrame:
    """Build a (bar × symbol) returns DataFrame for correlation computation.

    Requires a ``timestamp`` column in prices for deterministic ordering.
    Returns empty DataFrame if prices or timestamp are absent.
    """
    if prices is None or prices.empty:
        return pd.DataFrame()
    if "timestamp" not in prices.columns or "close" not in prices.columns:
        return pd.DataFrame()

    rows = prices[prices["symbol"].isin(symbols)].copy()
    if rows.empty:
        return pd.DataFrame()

    rows = rows.sort_values(["symbol", "timestamp"])
    pivot = rows.pivot_table(
        index="timestamp",
        columns="symbol",
        values="close",
        aggfunc="last",
    )
    pivot = pivot.iloc[-lookback_days:] if len(pivot) > lookback_days else pivot
    returns = pivot.pct_change().dropna(how="all")
    return returns


def compute_correlation_matrix(
    prices: pd.DataFrame,
    symbols: list[str],
    lookback_days: int = 60,
) -> pd.DataFrame:
    """Compute pairwise return correlation matrix for the given symbols.

    Args:
        prices: DataFrame with columns ``timestamp``, ``symbol``, ``close``.
        symbols: Symbols to include.
        lookback_days: Number of recent bars to use (default 60).

    Returns:
        Square correlation DataFrame (symbols × symbols).
        Empty DataFrame if fewer than 2 symbols or insufficient data (< 3 bars).
    """
    if prices is None or prices.empty or len(symbols) < 2:
        return pd.DataFrame()
    if "close" not in prices.columns or "symbol" not in prices.columns:
        return pd.DataFrame()

    returns = _pivot_returns(prices, symbols, lookback_days)
    valid_cols = [c for c in symbols if c in returns.columns]
    if len(valid_cols) < 2 or len(returns) < 3:
        return pd.DataFrame()

    return returns[valid_cols].corr()


def detect_correlated_clusters(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.70,
) -> list[list[str]]:
    """Group symbols into positively correlated clusters.

    Two symbols join the same cluster if their correlation is >= threshold
    (positive correlation only; negative / hedging correlations are excluded).
    Uses union-find to transitively group connected symbols.

    Args:
        corr_matrix: Square correlation DataFrame (symbols × symbols).
        threshold: Minimum positive correlation for grouping (default 0.70).

    Returns:
        List of clusters. Each cluster is a sorted list of symbol strings.
        Single-symbol "clusters" are excluded — only groups of 2+ are returned.
    """
    if corr_matrix is None or corr_matrix.empty:
        return []

    symbols = list(corr_matrix.columns)
    parent = {s: s for s in symbols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for i, sym_i in enumerate(symbols):
        for sym_j in symbols[i + 1:]:
            val = corr_matrix.loc[sym_i, sym_j]
            if pd.isna(val):
                continue
            if float(val) >= threshold:  # positive correlation only
                union(sym_i, sym_j)

    groups: dict[str, list[str]] = {}
    for sym in symbols:
        root = find(sym)
        groups.setdefault(root, []).append(sym)

    return [sorted(members) for members in groups.values() if len(members) >= 2]


def apply_correlation_guard(
    target_weights: dict[str, float],
    prices: pd.DataFrame,
    policy: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    """Apply correlation guard: proportionally scale down over-concentrated clusters.

    Args:
        target_weights: symbol → weight mapping.
        prices: Price DataFrame with ``timestamp``, ``symbol``, ``close``.
        policy: Policy dict. Reads from ``correlation_guard`` section:
            - enabled (bool, default True)
            - threshold (float, default 0.70)
            - max_cluster_weight (float, default 0.40)
            - lookback_days (int, default 60)

    Returns:
        (adjusted_weights, reasons).
        ``adjusted_weights`` is a new dict (original is never mutated).
        ``reasons`` is a list of human-readable explanation strings.
        Returns originals unchanged if disabled, < 2 symbols, or no cluster risk.
    """
    cg = (policy or {}).get("correlation_guard") or {}
    if not cg.get("enabled", True):
        return dict(target_weights), []

    if not target_weights:
        return {}, []

    symbols = list(target_weights.keys())
    if len(symbols) < 2:
        return dict(target_weights), []

    threshold = float(cg.get("threshold", 0.70) or 0.70)
    max_cluster_weight = float(cg.get("max_cluster_weight", 0.40) or 0.40)
    lookback_days = int(cg.get("lookback_days", 60) or 60)

    corr_matrix = compute_correlation_matrix(prices, symbols, lookback_days)
    if corr_matrix.empty:
        return dict(target_weights), []

    clusters = detect_correlated_clusters(corr_matrix, threshold=threshold)
    if not clusters:
        return dict(target_weights), []

    adjusted = dict(target_weights)
    reasons: list[str] = []

    for cluster in clusters:
        cluster_weight = sum(adjusted.get(sym, 0.0) for sym in cluster)
        if cluster_weight <= max_cluster_weight + 1e-9:
            continue
        scale = max_cluster_weight / cluster_weight
        for sym in cluster:
            adjusted[sym] = adjusted.get(sym, 0.0) * scale
        reasons.append(
            f"correlation_guard: cluster {cluster} weight={cluster_weight:.4f} "
            f"> max={max_cluster_weight:.4f}, scaled by {scale:.4f}"
        )

    return adjusted, reasons


__all__ = [
    "compute_correlation_matrix",
    "detect_correlated_clusters",
    "apply_correlation_guard",
]
