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

import numpy as np
import pandas as pd

from src.assembled_core.portfolio.position_sizing import compute_target_positions


def generate_orders_from_targets_fast(
    target_positions: pd.DataFrame,
    current_positions: pd.DataFrame | None = None,
    timestamp: datetime | None = None,
    prices_latest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fast-path order generation when target and current positions are already aligned.
    
    This function avoids expensive pandas merge operations by assuming that:
    - target_positions and current_positions have the same symbols in the same order
    - Both DataFrames are sorted by symbol
    - Columns are minimal: symbol, target_qty (or qty for current)
    
    This is optimized for performance-critical backtest loops where positions
    are already aligned (e.g., from TradingCycle result).
    
    Args:
        target_positions: DataFrame with columns: symbol, target_qty (or qty)
            Must be sorted by symbol
        current_positions: Optional DataFrame with columns: symbol, qty
            Must have same symbols in same order as target_positions (or empty)
            If None, assumes all positions are zero
        timestamp: Order timestamp (default: current UTC time)
        prices_latest: Optional DataFrame with columns: symbol, close (one row per symbol)
            For fast price lookup (already aligned/latest per symbol)
            
    Returns:
        DataFrame with columns: timestamp, symbol, side, qty, price
        side: "BUY" or "SELL"
        qty: Quantity (always positive)
        price: Order price (from prices_latest if available, else 0.0)
        Sorted by symbol
        
    Raises:
        ValueError: If alignment assumption is violated (symbols don't match)
    """
    if target_positions.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    
    # Extract columns (handle both "target_qty" and "qty" in target)
    if "target_qty" in target_positions.columns:
        target_qty_col = "target_qty"
    elif "qty" in target_positions.columns:
        target_qty_col = "qty"
    else:
        raise ValueError("target_positions must have 'target_qty' or 'qty' column")
    
    # Ensure target is sorted by symbol (required for alignment)
    if not target_positions["symbol"].is_monotonic_increasing:
        target_positions = target_positions.sort_values("symbol").reset_index(drop=True)
    
    # Use current timestamp if not provided
    if timestamp is None:
        timestamp = pd.Timestamp.utcnow()
    
    # Extract numpy arrays directly (no merge, no sort)
    symbols = target_positions["symbol"].values
    target_qty = target_positions[target_qty_col].values.astype(np.float64)
    
    # Get current quantities (assume aligned if current_positions provided)
    if current_positions is None or current_positions.empty:
        current_qty = np.zeros(len(symbols), dtype=np.float64)
    else:
        # Ensure current is sorted by symbol (required for alignment check)
        if not current_positions["symbol"].is_monotonic_increasing:
            current_positions = current_positions.sort_values("symbol").reset_index(drop=True)
        # Fast-path: assume same symbols in same order
        if len(current_positions) != len(target_positions):
            raise ValueError(
                "Fast-path requires current_positions to have same length as target_positions"
            )
        current_symbols = current_positions["symbol"].values
        if not np.array_equal(current_symbols, symbols):
            raise ValueError(
                "Fast-path requires current_positions to have same symbols in same order as target_positions"
            )
        current_qty = current_positions["qty"].values.astype(np.float64)
    
    # Compute delta vectorially (numpy)
    qty_delta = target_qty - current_qty
    
    # Filter for non-zero deltas (vectorized)
    abs_delta = np.abs(qty_delta)
    non_zero_mask = abs_delta > 1e-6
    
    if not np.any(non_zero_mask):
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    
    # Extract non-zero deltas
    symbols_filtered = symbols[non_zero_mask]
    qty_delta_filtered = qty_delta[non_zero_mask]
    
    # Determine side and quantity vectorially (numpy)
    sides = np.where(qty_delta_filtered > 0, "BUY", "SELL")
    qtys = np.abs(qty_delta_filtered)
    
    # Get prices if available (fast lookup from aligned prices_latest)
    if prices_latest is not None and "close" in prices_latest.columns and "symbol" in prices_latest.columns:
        # Build price mapping (assuming prices_latest is already aligned or small)
        price_map = dict(zip(prices_latest["symbol"].values, prices_latest["close"].values))
        prices_array = np.array([price_map.get(sym, 0.0) for sym in symbols_filtered], dtype=np.float64)
    else:
        prices_array = np.zeros(len(symbols_filtered), dtype=np.float64)
    
    # Build DataFrame directly (no pandas operations except construction)
    result = pd.DataFrame({
        "timestamp": timestamp,
        "symbol": symbols_filtered,
        "side": sides,
        "qty": qtys,
        "price": prices_array,
    })
    
    # Ensure columns are in correct order
    result = result[["timestamp", "symbol", "side", "qty", "price"]]
    
    return result


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

    # Try fast-path if positions are already aligned
    # Fast-path condition: same symbols in same order, no missing symbols
    target_symbols = set(target_positions["symbol"].unique())
    if current_positions is not None and not current_positions.empty:
        current_symbols = set(current_positions["symbol"].unique())
        # Fast-path: exact same symbols, can use aligned arrays
        if target_symbols == current_symbols:
            target_sorted = target_positions[["symbol", "target_qty"]].sort_values("symbol").reset_index(drop=True)
            current_sorted = current_positions[["symbol", "qty"]].sort_values("symbol").reset_index(drop=True)
            # Check if symbols match exactly (same order after sort)
            if (target_sorted["symbol"].values == current_sorted["symbol"].values).all():
                # Use fast-path with prices_latest (extract latest per symbol if prices provided)
                prices_latest = None
                if prices is not None and "close" in prices.columns and "symbol" in prices.columns:
                    prices_latest = prices.groupby("symbol", group_keys=False)["close"].last().reset_index()
                    prices_latest = prices_latest[prices_latest["symbol"].isin(target_symbols)].sort_values("symbol").reset_index(drop=True)
                try:
                    return generate_orders_from_targets_fast(
                        target_sorted,
                        current_positions=current_sorted,
                        timestamp=timestamp,
                        prices_latest=prices_latest,
                    )
                except (ValueError, KeyError):
                    # Fallback to merge-based path if fast-path fails
                    pass
    
    # Fallback to merge-based path (handles misaligned or missing symbols)
    # Ensure both DataFrames are sorted by symbol for stable alignment
    target_sorted = target_positions[["symbol", "target_qty"]].sort_values("symbol").reset_index(drop=True)
    current_sorted = current_positions[["symbol", "qty"]].sort_values("symbol").reset_index(drop=True) if current_positions is not None and not current_positions.empty else pd.DataFrame(columns=["symbol", "qty"])

    # Merge target and current positions (outer join to include all symbols)
    merged = target_sorted.merge(
        current_sorted,
        on="symbol",
        how="outer",
        suffixes=("_target", "_current"),
    )

    # Ensure stable symbol order (sorted)
    merged = merged.sort_values("symbol").reset_index(drop=True)

    # Fill NaN with 0.0 (symbols not in current or target)
    # Ensure columns are float type before filling to avoid FutureWarning
    if "target_qty" in merged.columns:
        merged["target_qty"] = merged["target_qty"].astype(float).fillna(0.0)
    if "qty" in merged.columns:
        merged["qty"] = merged["qty"].astype(float).fillna(0.0)

    # Compute quantity delta vectorially (aligned arrays)
    qty_delta = merged["target_qty"].values - merged["qty"].values
    merged["qty_delta"] = qty_delta

    # Filter for non-zero deltas (vectorized)
    abs_delta = np.abs(qty_delta)
    non_zero_mask = abs_delta > 1e-6
    orders = merged[non_zero_mask].copy()

    if orders.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    # Determine side and quantity vectorially (no apply loop)
    # BUY if delta > 0, SELL if delta < 0
    qty_delta_filtered = orders["qty_delta"].values
    orders["side"] = np.where(qty_delta_filtered > 0, "BUY", "SELL")
    orders["qty"] = np.abs(qty_delta_filtered)

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
