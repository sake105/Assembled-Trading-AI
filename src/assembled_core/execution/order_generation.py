"""Order generation module.

This module generates orders by comparing current positions to target positions.
It extends the basic order generation from pipeline.orders.

Zukünftige Integration:
- Nutzt pipeline.orders.signals_to_orders als Basis
- Erweitert um Position-Sizing (portfolio.position_sizing)
- Bietet verschiedene Order-Typen (Market, Limit, Stop-Loss)
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.assembled_core.portfolio.position_sizing import compute_target_positions


def generate_orders_from_targets(
    target_positions: pd.DataFrame,
    current_positions: pd.DataFrame | None = None,
    timestamp: datetime | None = None,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate orders to transition from current to target positions.

    This function compares current positions to target positions and generates
    orders to achieve the target portfolio.

    Args:
        target_positions: DataFrame with columns: symbol, target_weight, target_qty
            (from portfolio.position_sizing.compute_target_positions)
        current_positions: Optional DataFrame with columns: symbol, qty
            If None, assumes all positions are zero (starting from scratch)
        timestamp: Order timestamp (default: current UTC time)
        prices: Optional DataFrame with columns: symbol, close (for price lookup)
            If None, price will be set to 0.0 (must be filled later)

    Returns:
        DataFrame with columns: timestamp, symbol, side, qty, price
        side: "BUY" or "SELL"
        qty: Quantity (always positive)
        price: Order price (from prices if available, else 0.0)
        Sorted by symbol

    Raises:
        ValueError: If required columns are missing
    """
    if target_positions.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    # Ensure required columns
    required = ["symbol", "target_qty"]
    missing = [c for c in required if c not in target_positions.columns]
    if missing:
        raise ValueError(f"Missing required columns in target_positions: {missing}")

    # Use current timestamp if not provided
    if timestamp is None:
        timestamp = pd.Timestamp.utcnow()

    # Get current positions (default to zero if not provided)
    if current_positions is None:
        current_positions = pd.DataFrame(columns=["symbol", "qty"])

    # Ensure current_positions has required columns
    if "symbol" not in current_positions.columns:
        current_positions = pd.DataFrame(columns=["symbol", "qty"])
    if "qty" not in current_positions.columns:
        current_positions["qty"] = 0.0

    # Merge target and current positions
    merged = target_positions[["symbol", "target_qty"]].merge(
        current_positions[["symbol", "qty"]],
        on="symbol",
        how="outer",
        suffixes=("_target", "_current"),
    )

    # Fill NaN with 0.0 (symbols not in current or target)
    # Ensure columns are float type before filling to avoid FutureWarning
    if "target_qty" in merged.columns:
        merged["target_qty"] = merged["target_qty"].astype(float).fillna(0.0)
    if "qty" in merged.columns:
        merged["qty"] = merged["qty"].astype(float).fillna(0.0)

    # Compute quantity delta
    merged["qty_delta"] = merged["target_qty"] - merged["qty"]

    # Filter for non-zero deltas
    orders = merged[merged["qty_delta"].abs() > 1e-6].copy()

    if orders.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    # Determine side and quantity
    orders["side"] = orders["qty_delta"].apply(lambda x: "BUY" if x > 0 else "SELL")
    orders["qty"] = orders["qty_delta"].abs()

    # Get prices if available
    if prices is not None and "close" in prices.columns and "symbol" in prices.columns:
        # Use latest price per symbol
        latest_prices = prices.groupby("symbol")["close"].last()
        orders["price"] = orders["symbol"].map(latest_prices).fillna(0.0)
    else:
        orders["price"] = 0.0

    # Select output columns
    result = orders[["symbol", "side", "qty", "price"]].copy()
    result["timestamp"] = timestamp

    # Reorder columns
    result = result[["timestamp", "symbol", "side", "qty", "price"]]
    result = result.sort_values("symbol").reset_index(drop=True)

    return result


def generate_orders_from_signals(
    signals: pd.DataFrame,
    total_capital: float = 1.0,
    top_n: int | None = None,
    current_positions: pd.DataFrame | None = None,
    timestamp: datetime | None = None,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate orders directly from signals (convenience function).

    This function combines position sizing and order generation in one step.

    Args:
        signals: DataFrame with columns: symbol, direction (and optionally score)
        total_capital: Total capital available (default: 1.0)
        top_n: Optional maximum number of positions (default: None)
        current_positions: Optional current positions DataFrame
        timestamp: Order timestamp (default: current UTC time)
        prices: Optional prices DataFrame for price lookup

    Returns:
        DataFrame with columns: timestamp, symbol, side, qty, price
    """
    # Compute target positions
    targets = compute_target_positions(
        signals, total_capital=total_capital, top_n=top_n, equal_weight=True
    )

    # Generate orders
    return generate_orders_from_targets(
        targets, current_positions=current_positions, timestamp=timestamp, prices=prices
    )
