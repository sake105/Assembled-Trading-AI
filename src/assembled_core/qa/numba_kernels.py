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

