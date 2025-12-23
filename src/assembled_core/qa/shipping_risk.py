"""Shipping and systemic risk analysis module.

This module provides functions to compute shipping exposure and systemic risk flags
for portfolios based on shipping route congestion data and portfolio positions.

Key Functions:
- compute_shipping_exposure: Compute portfolio-weighted shipping metrics
- compute_systemic_risk_flags: Generate risk flags based on shipping exposure

Example usage:
    >>> from src.assembled_core.qa.shipping_risk import compute_shipping_exposure
    >>>
    >>> portfolio = pd.DataFrame({
    ...     "symbol": ["AAPL", "MSFT", "GOOGL"],
    ...     "weight": [0.4, 0.3, 0.3]
    ... })
    >>>
    >>> shipping_features = pd.DataFrame({
    ...     "symbol": ["AAPL", "MSFT", "GOOGL"],
    ...     "shipping_congestion_score": [75.0, 45.0, 30.0]
    ... })
    >>>
    >>> exposure = compute_shipping_exposure(portfolio, shipping_features)
    >>> print(f"Avg Congestion: {exposure['avg_shipping_congestion']:.2f}")
"""

from __future__ import annotations

from collections import Counter

import pandas as pd


def compute_shipping_exposure(
    portfolio_positions: pd.DataFrame,
    shipping_features: pd.DataFrame,
    congestion_threshold: float = 70.0,
) -> dict[str, float | list[str]]:
    """Compute portfolio shipping exposure metrics.

    This function aggregates shipping-related metrics across portfolio positions,
    weighted by position weights or values.

    Args:
        portfolio_positions: DataFrame with columns:
            - timestamp (optional): Timestamp of positions
            - symbol: Stock symbol
            - weight (or value): Position weight (0-1) or absolute value
            Additional columns are ignored
        shipping_features: DataFrame with columns:
            - symbol: Stock symbol
            - shipping_congestion_score: Current congestion score (0-100)
            - shipping_ships_count (optional): Number of ships
            - route_id (optional): Route identifier
            - port_from, port_to (optional): Port codes
        congestion_threshold: Threshold for "high congestion" (default: 70.0)

    Returns:
        Dictionary with metrics:
        - avg_shipping_congestion: Portfolio-weighted average congestion score
        - high_congestion_weight: Sum of weights for positions with congestion > threshold
        - top_routes: List of most common route_ids in portfolio (if route_id available)
        - exposed_symbols: List of symbols with congestion > threshold

    Example:
        >>> portfolio = pd.DataFrame({
        ...     "symbol": ["AAPL", "MSFT"],
        ...     "weight": [0.6, 0.4]
        ... })
        >>> shipping = pd.DataFrame({
        ...     "symbol": ["AAPL", "MSFT"],
        ...     "shipping_congestion_score": [80.0, 50.0]
        ... })
        >>> exposure = compute_shipping_exposure(portfolio, shipping)
        >>> print(exposure["avg_shipping_congestion"])  # 0.6 * 80 + 0.4 * 50 = 68.0
    """
    if portfolio_positions.empty:
        return {
            "avg_shipping_congestion": 0.0,
            "high_congestion_weight": 0.0,
            "top_routes": [],
            "exposed_symbols": [],
        }

    if shipping_features.empty:
        return {
            "avg_shipping_congestion": 0.0,
            "high_congestion_weight": 0.0,
            "top_routes": [],
            "exposed_symbols": [],
        }

    # Determine weight column
    if "weight" in portfolio_positions.columns:
        weight_col = "weight"
    elif "value" in portfolio_positions.columns:
        # Normalize values to weights
        portfolio_positions = portfolio_positions.copy()
        total_value = portfolio_positions["value"].sum()
        if total_value > 0:
            portfolio_positions["weight"] = portfolio_positions["value"] / total_value
        else:
            portfolio_positions["weight"] = 0.0
        weight_col = "weight"
    else:
        # Default: equal weights
        portfolio_positions = portfolio_positions.copy()
        portfolio_positions["weight"] = 1.0 / len(portfolio_positions)
        weight_col = "weight"

    # Merge portfolio with shipping features
    merged = portfolio_positions[["symbol", weight_col]].merge(
        shipping_features[["symbol", "shipping_congestion_score"]],
        on="symbol",
        how="left",
    )

    # Fill missing congestion scores with 0 (no shipping exposure)
    merged["shipping_congestion_score"] = merged["shipping_congestion_score"].fillna(
        0.0
    )

    # Compute weighted average congestion
    avg_shipping_congestion = float(
        (merged["weight"] * merged["shipping_congestion_score"]).sum()
    )

    # Compute high congestion weight
    high_congestion_mask = merged["shipping_congestion_score"] > congestion_threshold
    high_congestion_weight = float(merged.loc[high_congestion_mask, weight_col].sum())

    # Get exposed symbols
    exposed_symbols = merged.loc[high_congestion_mask, "symbol"].tolist()

    # Compute top routes (if route_id available)
    top_routes = []
    if "route_id" in shipping_features.columns:
        # Get routes for exposed symbols
        exposed_shipping = shipping_features[
            shipping_features["symbol"].isin(exposed_symbols)
        ]
        if not exposed_shipping.empty:
            route_counter = Counter(exposed_shipping["route_id"].dropna())
            top_routes = [route for route, _ in route_counter.most_common(5)]

    return {
        "avg_shipping_congestion": avg_shipping_congestion,
        "high_congestion_weight": high_congestion_weight,
        "top_routes": top_routes,
        "exposed_symbols": exposed_symbols,
    }


def compute_systemic_risk_flags(
    shipping_exposure: dict[str, float | list[str]],
    high_congestion_threshold: float = 70.0,
    high_exposure_threshold: float = 0.3,
) -> dict[str, bool | str]:
    """Compute systemic risk flags based on shipping exposure.

    This function generates boolean flags and risk indicators based on
    portfolio shipping exposure metrics.

    Args:
        shipping_exposure: Output from compute_shipping_exposure()
        high_congestion_threshold: Threshold for "high congestion" flag (default: 70.0)
        high_exposure_threshold: Threshold for "high exposure" (weight fraction, default: 0.3)

    Returns:
        Dictionary with risk flags:
        - high_shipping_risk: True if avg congestion > threshold or high exposure > threshold
        - exposed_to_blockade_routes: True if any exposed symbols have high congestion
        - risk_level: "LOW", "MEDIUM", or "HIGH" based on exposure metrics
        - risk_reason: Human-readable reason for risk level

    Example:
        >>> exposure = {
        ...     "avg_shipping_congestion": 75.0,
        ...     "high_congestion_weight": 0.4,
        ...     "exposed_symbols": ["AAPL", "MSFT"]
        ... }
        >>> flags = compute_systemic_risk_flags(exposure)
        >>> print(flags["risk_level"])  # "HIGH"
    """
    avg_congestion = float(shipping_exposure.get("avg_shipping_congestion", 0.0))
    high_exposure_weight = float(shipping_exposure.get("high_congestion_weight", 0.0))
    exposed_symbols = shipping_exposure.get("exposed_symbols", [])

    # Determine risk flags
    high_shipping_risk = (
        avg_congestion > high_congestion_threshold
        or high_exposure_weight > high_exposure_threshold
    )

    exposed_to_blockade_routes = len(exposed_symbols) > 0

    # Determine risk level
    if (
        avg_congestion > high_congestion_threshold
        and high_exposure_weight > high_exposure_threshold
    ):
        risk_level = "HIGH"
        risk_reason = f"High average congestion ({avg_congestion:.1f}) and high exposure weight ({high_exposure_weight:.1%})"
    elif (
        avg_congestion > high_congestion_threshold
        or high_exposure_weight > high_exposure_threshold
    ):
        risk_level = "MEDIUM"
        if avg_congestion > high_congestion_threshold:
            risk_reason = f"High average congestion ({avg_congestion:.1f})"
        else:
            risk_reason = f"High exposure weight ({high_exposure_weight:.1%})"
    else:
        risk_level = "LOW"
        risk_reason = f"Low congestion ({avg_congestion:.1f}) and exposure ({high_exposure_weight:.1%})"

    return {
        "high_shipping_risk": high_shipping_risk,
        "exposed_to_blockade_routes": exposed_to_blockade_routes,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
    }
