"""Optional Numba-accelerated kernels for backtest loops.

This module provides Numba-accelerated functions for performance-critical loops.
If Numba is not available, the code falls back to pure NumPy implementations.

All functions use @njit decorator for maximum performance.
"""

from __future__ import annotations

import numpy as np

# Try to import Numba, set flag if available
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator if Numba is not available
    def njit(*args, **kwargs):
        """Dummy decorator when Numba is not available."""
        def decorator(func):
            return func
        return decorator


@njit(cache=True)
def compute_mark_to_market_numba(
    positions_array: np.ndarray, prices_array: np.ndarray
) -> float:
    """Compute mark-to-market value using vectorized operations (Numba-accelerated).

    This function computes: sum(position_shares * price) for all valid (non-NaN) prices.

    Args:
        positions_array: 1D array of position quantities (aligned with symbols)
        prices_array: 1D array of prices (aligned with symbols, may contain NaNs)

    Returns:
        Mark-to-market value (sum of position_shares * price for valid prices)

    Note:
        This function is optimized with Numba for performance.
        NaN prices are automatically filtered out.
    """
    mtm = 0.0
    n = len(positions_array)
    for i in range(n):
        price = prices_array[i]
        if not np.isnan(price):
            mtm += positions_array[i] * price
    return mtm


@njit(cache=True)
def compute_equity_curve_numba(
    cash_array: np.ndarray,
    positions_matrix: np.ndarray,
    prices_matrix: np.ndarray,
) -> np.ndarray:
    """Compute equity curve for all timestamps using vectorized operations (Numba-accelerated).

    This function computes: equity[t] = cash[t] + sum(positions[t, s] * prices[t, s]) for all t.

    Args:
        cash_array: 1D array of cash values per timestamp (length: n_timestamps)
        positions_matrix: 2D array of positions [n_timestamps, n_symbols]
        prices_matrix: 2D array of prices [n_timestamps, n_symbols] (may contain NaNs)

    Returns:
        1D array of equity values per timestamp (length: n_timestamps)

    Note:
        This function is optimized with Numba for performance.
        NaN prices are automatically filtered out.
    """
    n_timestamps = len(cash_array)
    n_symbols = positions_matrix.shape[1]
    equity = np.zeros(n_timestamps, dtype=np.float64)

    for t in range(n_timestamps):
        cash = cash_array[t]
        mtm = 0.0
        for s in range(n_symbols):
            price = prices_matrix[t, s]
            if not np.isnan(price):
                mtm += positions_matrix[t, s] * price
        equity[t] = cash + mtm

    return equity


@njit(cache=True)
def aggregate_position_deltas_numba(
    symbol_indices: np.ndarray, deltas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate position deltas by symbol index (Numba-accelerated).

    This function aggregates multiple orders for the same symbol into a single delta.

    Args:
        symbol_indices: 1D array of symbol indices (integers, 0-based)
        deltas: 1D array of position deltas (floats)

    Returns:
        Tuple of (unique_indices, aggregated_deltas)
        - unique_indices: 1D array of unique symbol indices
        - aggregated_deltas: 1D array of aggregated deltas (sum per symbol)

    Note:
        This function is optimized with Numba for performance.
        It uses a simple accumulation approach (not hash-based for Numba compatibility).
    """
    if len(symbol_indices) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    # Find max index to determine array size
    max_idx = int(np.max(symbol_indices))
    # Create accumulator array (size: max_idx + 1)
    accumulator = np.zeros(max_idx + 1, dtype=np.float64)

    # Accumulate deltas
    for i in range(len(symbol_indices)):
        idx = int(symbol_indices[i])
        accumulator[idx] += deltas[i]

    # Extract non-zero deltas
    unique_indices = []
    aggregated_deltas = []
    for idx in range(len(accumulator)):
        if abs(accumulator[idx]) > 1e-10:  # Threshold for non-zero
            unique_indices.append(idx)
            aggregated_deltas.append(accumulator[idx])

    return np.array(unique_indices, dtype=np.int32), np.array(aggregated_deltas, dtype=np.float64)


@njit(cache=True)
def apply_fills_cash_delta_numba(
    sides: np.ndarray,  # 0=BUY, 1=SELL
    qtys: np.ndarray,
    prices: np.ndarray,
    spread_w: float,
    impact_w: float,
    commission_bps: float,
) -> float:
    """Compute total cash delta from fills using Numba JIT.
    
    This function computes the cash impact of executing orders with costs:
    - BUY: cash decreases by (qty * fill_price + fee)
    - SELL: cash increases by (qty * fill_price - fee)
    
    Args:
        sides: Array of order sides as integers (0=BUY, 1=SELL)
        qtys: Array of order quantities (always positive)
        prices: Array of order prices
        spread_w: Spread weight (multiplier for bid/ask spread)
        impact_w: Market impact weight (multiplier for price impact)
        commission_bps: Commission in basis points
        
    Returns:
        Total cash delta (sum of all order cash impacts)
        
    Note:
        This function is optimized with Numba for performance.
        Costs are computed as:
        - Fill price: price * (1 + spread_w + impact_w) for BUY, price * (1 - spread_w - impact_w) for SELL
        - Fee: commission_bps * notional
    """
    n = len(sides)
    if n == 0:
        return 0.0
    
    spread_factor = spread_w * 1e-4  # Convert to decimal
    impact_factor = impact_w * 1e-4
    commission_factor = commission_bps * 1e-4
    
    total_delta = 0.0
    for i in range(n):
        side = sides[i]
        qty = qtys[i]
        price = prices[i]
        
        # Skip invalid prices
        if np.isnan(price) or qty <= 0.0:
            continue
        
        # Compute fill price with costs
        if side == 0:  # BUY
            fill_price = price * (1.0 + spread_factor + impact_factor)
        else:  # SELL
            fill_price = price * (1.0 - spread_factor - impact_factor)
        
        # Compute notional and fee
        notional = qty * price
        fee = notional * commission_factor
        
        # Compute cash delta
        if side == 0:  # BUY: cash decreases
            cash_delta = -(qty * fill_price + fee)
        else:  # SELL: cash increases
            cash_delta = +(qty * fill_price - fee)
        
        total_delta += cash_delta
    
    return total_delta


@njit(cache=True)
def apply_fills_position_deltas_numba(
    sides: np.ndarray,  # 0=BUY, 1=SELL
    qtys: np.ndarray,
    symbol_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute position deltas from fills using Numba JIT.
    
    This function computes position changes from orders and aggregates by symbol:
    - BUY: position increases by qty
    - SELL: position decreases by qty
    
    Args:
        sides: Array of order sides as integers (0=BUY, 1=SELL)
        qtys: Array of order quantities (always positive)
        symbol_indices: Array of symbol indices (integers, 0-based)
        
    Returns:
        Tuple of (unique_symbol_indices, aggregated_deltas)
        - unique_symbol_indices: Array of unique symbol indices
        - aggregated_deltas: Array of aggregated position deltas (sum per symbol)
        
    Note:
        This function is optimized with Numba for performance.
        Position deltas are aggregated by symbol index.
    """
    n = len(sides)
    if n == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
    
    # Compute position deltas
    deltas = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if sides[i] == 0:  # BUY
            deltas[i] = qtys[i]
        else:  # SELL
            deltas[i] = -qtys[i]
    
    # Aggregate by symbol index (reuse existing function)
    return aggregate_position_deltas_numba(symbol_indices, deltas)

