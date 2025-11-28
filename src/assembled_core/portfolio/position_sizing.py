"""Position sizing module.

This module provides position sizing strategies for EOD trading.
It determines target positions based on signals and available capital.

Zukünftige Integration:
- Nutzt pipeline.portfolio.simulate_with_costs für Backtesting
- Erweitert um weitere Sizing-Strategien (Kelly Criterion, Risk Parity, etc.)
"""
from __future__ import annotations

import pandas as pd


def compute_target_positions(
    signals: pd.DataFrame,
    total_capital: float = 1.0,
    top_n: int | None = None,
    equal_weight: bool = True
) -> pd.DataFrame:
    """Compute target positions from trading signals.
    
    This function determines target positions (weights or quantities) based on:
    - Signal scores (if available)
    - Top-N selection (if top_n is specified)
    - Equal weighting (if equal_weight=True) or score-based weighting
    
    Args:
        signals: DataFrame with columns: symbol, direction (and optionally score)
            direction: "LONG" or "FLAT"
            score: Signal strength (0.0 to 1.0), optional
        total_capital: Total capital available (default: 1.0 for normalized weights)
        top_n: Optional maximum number of positions to select (default: None = all LONG signals)
        equal_weight: If True, use equal weights (1/N). If False, use score-based weights (default: True)
    
    Returns:
        DataFrame with columns: symbol, target_weight, target_qty
        target_weight: Target weight (0.0 to 1.0)
        target_qty: Target quantity (in units, if total_capital represents actual capital)
        Sorted by symbol
    
    Raises:
        ValueError: If signals DataFrame is empty or missing required columns
    """
    if signals.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    
    # Ensure required columns
    required = ["symbol", "direction"]
    missing = [c for c in required if c not in signals.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(signals.columns)}")
    
    # Filter for LONG signals only
    long_signals = signals[signals["direction"] == "LONG"].copy()
    
    if long_signals.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    
    # Select top N by score if specified
    if top_n is not None and top_n > 0:
        if "score" in long_signals.columns:
            # Sort by score descending, take top N
            long_signals = long_signals.nlargest(top_n, "score")
        else:
            # If no score, just take first N
            long_signals = long_signals.head(top_n)
    
    # Compute weights
    n_positions = len(long_signals)
    
    if equal_weight:
        # Equal weighting: 1/N for each position
        long_signals["target_weight"] = 1.0 / n_positions
    else:
        # Score-based weighting (normalize scores to sum to 1.0)
        if "score" in long_signals.columns:
            total_score = long_signals["score"].sum()
            if total_score > 0:
                long_signals["target_weight"] = long_signals["score"] / total_score
            else:
                # Fallback to equal weight if all scores are zero
                long_signals["target_weight"] = 1.0 / n_positions
        else:
            # Fallback to equal weight if no scores
            long_signals["target_weight"] = 1.0 / n_positions
    
    # Compute target quantities (if total_capital represents actual capital)
    # For normalized weights (total_capital=1.0), target_qty = target_weight
    # For actual capital, target_qty would need current prices (not available here)
    # So we set target_qty = target_weight * total_capital as a placeholder
    long_signals["target_qty"] = long_signals["target_weight"] * total_capital
    
    # Select and sort output columns
    result = long_signals[["symbol", "target_weight", "target_qty"]].copy()
    result = result.sort_values("symbol").reset_index(drop=True)
    
    return result


def compute_target_positions_from_trend_signals(
    trend_signals: pd.DataFrame,
    total_capital: float = 1.0,
    top_n: int | None = None,
    min_score: float = 0.0
) -> pd.DataFrame:
    """Compute target positions from trend signals (convenience function).
    
    This is a convenience wrapper around compute_target_positions that:
    - Filters signals by minimum score
    - Uses score-based weighting (not equal weight)
    
    Args:
        trend_signals: DataFrame with columns: symbol, direction, score
            (from signals.rules_trend.generate_trend_signals)
        total_capital: Total capital available (default: 1.0)
        top_n: Optional maximum number of positions (default: None)
        min_score: Minimum score threshold (default: 0.0)
    
    Returns:
        DataFrame with columns: symbol, target_weight, target_qty
    """
    # Filter by minimum score
    if "score" in trend_signals.columns:
        filtered = trend_signals[trend_signals["score"] >= min_score].copy()
    else:
        filtered = trend_signals.copy()
    
    # Use score-based weighting (not equal weight)
    return compute_target_positions(
        filtered,
        total_capital=total_capital,
        top_n=top_n,
        equal_weight=False
    )
