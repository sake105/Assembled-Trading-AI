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

    # Track NaN-pair ratio so callers can detect a silently-degraded
    # correlation matrix. Pairs with NaN correlation are necessarily
    # excluded from clustering; a matrix that is mostly NaN (short
    # history, new listings, missing bars) produces zero clusters and
    # the guard trivially passes — that degradation used to be invisible.
    n_total = 0
    n_nan = 0
    for i, sym_i in enumerate(symbols):
        for sym_j in symbols[i + 1 :]:
            n_total += 1
            val = corr_matrix.loc[sym_i, sym_j]
            if pd.isna(val):
                n_nan += 1
                continue
            if float(val) >= threshold:  # positive correlation only
                union(sym_i, sym_j)

    if n_total > 0 and n_nan / n_total > 0.25:
        import logging
        logging.getLogger(__name__).warning(
            "[CorrGuard] degraded correlation matrix: %d/%d pairs NaN (%.1f%%) — "
            "clustering may under-detect concentration",
            n_nan, n_total, 100.0 * n_nan / n_total,
        )

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
    if not cg.get("enabled", False):
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

    # Sprint 3 / C8: optional tail-dependence tightening via Clayton copula.
    # When enabled, if portfolio avg lower-tail dependence exceeds the trigger,
    # tighten `max_cluster_weight` by `tightening_factor` so clusters that look
    # fine under linear correlation but crash jointly are reined in.
    reasons_tail: list[str] = []
    tail_cfg = cg.get("tail_dependence") or {}
    if tail_cfg.get("enabled", False) and len(symbols) >= 2:
        try:
            from src.assembled_core.ml.copula_models import compute_portfolio_tail_risk

            returns_wide = _pivot_returns(prices, symbols, lookback_days)
            if not returns_wide.empty and returns_wide.shape[1] >= 2:
                tail_metrics = compute_portfolio_tail_risk(returns_wide)
                avg_td = float(tail_metrics.get("avg_lower_tail_dep", 0.0) or 0.0)
                trigger = float(tail_cfg.get("trigger", 0.5) or 0.5)
                tightening_factor = float(tail_cfg.get("tightening_factor", 1.3) or 1.3)
                if avg_td > trigger and tightening_factor > 1.0:
                    new_cap = max_cluster_weight / tightening_factor
                    reasons_tail.append(
                        f"correlation_guard: avg_lower_tail_dep={avg_td:.3f} > "
                        f"trigger={trigger:.2f}, max_cluster_weight "
                        f"{max_cluster_weight:.3f} -> {new_cap:.3f}"
                    )
                    max_cluster_weight = new_cap
        except Exception as exc:  # noqa: BLE001 - defensive, optional scipy
            reasons_tail.append(f"correlation_guard: tail_dependence skipped ({exc})")

    clusters = detect_correlated_clusters(corr_matrix, threshold=threshold)
    if not clusters:
        return dict(target_weights), []

    adjusted = dict(target_weights)
    reasons: list[str] = list(reasons_tail)

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


def compute_avg_correlation(corr_matrix: pd.DataFrame) -> float:
    """Compute the average pairwise correlation (off-diagonal).

    Returns 0.0 if matrix is empty or has fewer than 2 symbols.
    Higher values indicate correlation convergence (crisis regime).
    """
    if corr_matrix is None or corr_matrix.empty or len(corr_matrix) < 2:
        return 0.0
    import numpy as np
    mask = ~np.eye(len(corr_matrix), dtype=bool)
    vals = corr_matrix.values[mask]
    vals = vals[~pd.isna(vals)]
    if len(vals) == 0:
        return 0.0
    return float(vals.mean())


def detect_correlation_regime_shift(
    prices: pd.DataFrame,
    symbols: list[str],
    *,
    short_window: int = 20,
    long_window: int = 120,
    shift_threshold: float = 0.15,
) -> dict[str, float | bool]:
    """Detect if correlations are regime-shifting toward convergence.

    Compares short-window average correlation vs long-window average.
    When short >> long, diversification is collapsing (crisis signal).

    Args:
        prices: Price DataFrame with timestamp, symbol, close.
        symbols: Symbols to analyze.
        short_window: Recent lookback for short-term correlation.
        long_window: Baseline lookback for long-term correlation.
        shift_threshold: Difference threshold to flag a shift.

    Returns:
        Dict with:
            avg_corr_short: short-window average pairwise correlation
            avg_corr_long: long-window average pairwise correlation
            shift: difference (short - long)
            regime_shift_detected: True if shift > threshold
            exposure_scale: suggested exposure scaling (1.0 = no change, <1 = reduce)
    """
    corr_short = compute_correlation_matrix(prices, symbols, short_window)
    corr_long = compute_correlation_matrix(prices, symbols, long_window)

    avg_short = compute_avg_correlation(corr_short)
    avg_long = compute_avg_correlation(corr_long)
    shift = avg_short - avg_long

    detected = shift > shift_threshold
    # Suggest exposure reduction proportional to shift magnitude
    exposure_scale = 1.0
    if detected:
        exposure_scale = max(0.5, 1.0 - (shift - shift_threshold) * 2.0)

    return {
        "avg_corr_short": round(avg_short, 4),
        "avg_corr_long": round(avg_long, 4),
        "shift": round(shift, 4),
        "regime_shift_detected": detected,
        "exposure_scale": round(exposure_scale, 4),
    }


__all__ = [
    "compute_avg_correlation",
    "compute_correlation_matrix",
    "detect_correlated_clusters",
    "detect_correlation_regime_shift",
    "apply_correlation_guard",
]
