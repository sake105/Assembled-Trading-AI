"""Numba-accelerated functions for backtest engine performance optimization.

This module provides JIT-compiled functions for tight loops in the backtest engine.
All functions are optional - if numba is not available, the backtest engine falls
back to pure Python/pandas implementations.

Functions in this module operate on numpy arrays rather than DataFrames for
optimal performance with Numba JIT compilation.
"""
from __future__ import annotations

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a dummy decorator if numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    numba = type("MockNumba", (), {"njit": staticmethod(njit)})()


if NUMBA_AVAILABLE:
    import numpy as np

    @numba.njit
    def compute_position_deltas_numba(
        sides: np.ndarray,  # array of strings ("BUY" or "SELL")
        qtys: np.ndarray,   # array of floats (always positive)
    ) -> np.ndarray:
        """Compute position deltas from orders using Numba JIT.
        
        Args:
            sides: Array of order sides ("BUY" or "SELL") as integer codes
                   (0 = BUY, 1 = SELL) for Numba compatibility
            qtys: Array of order quantities (always positive)
        
        Returns:
            Array of position deltas (+qty for BUY, -qty for SELL)
        
        Note:
            This function uses integer codes for sides (0=BUY, 1=SELL) since
            Numba doesn't support string comparisons efficiently.
        """
        n = len(sides)
        deltas = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if sides[i] == 0:  # BUY
                deltas[i] = qtys[i]
            else:  # SELL
                deltas[i] = -qtys[i]
        return deltas

    @numba.njit
    def aggregate_position_deltas_numba(
        symbols: np.ndarray,  # array of symbol indices (integers)
        deltas: np.ndarray,   # array of position deltas
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate position deltas by symbol using Numba JIT.
        
        Args:
            symbols: Array of symbol indices (integers, 0-based)
            deltas: Array of position deltas
        
        Returns:
            Tuple of (unique_symbols, aggregated_deltas)
            unique_symbols: Sorted array of unique symbol indices
            aggregated_deltas: Sum of deltas per symbol (same order as unique_symbols)
        
        Note:
            This function assumes symbol indices are in a reasonable range (0 to ~1000).
            For larger universes, consider using a hash-based approach.
        """
        # Find unique symbols
        unique_symbols = np.unique(symbols)
        n_unique = len(unique_symbols)
        aggregated = np.zeros(n_unique, dtype=np.float64)
        
        # Aggregate deltas per symbol (more efficient: single pass)
        for j in range(len(symbols)):
            sym = symbols[j]
            # Find index of sym in unique_symbols
            for i in range(n_unique):
                if unique_symbols[i] == sym:
                    aggregated[i] += deltas[j]
                    break
        
        return unique_symbols, aggregated

else:
    # Fallback implementations (pure Python, not optimized)
    import numpy as np

    def compute_position_deltas_numba(
        sides: np.ndarray,
        qtys: np.ndarray,
    ) -> np.ndarray:
        """Fallback implementation when numba is not available."""
        deltas = np.zeros(len(sides), dtype=np.float64)
        for i in range(len(sides)):
            if sides[i] == 0:  # BUY
                deltas[i] = qtys[i]
            else:  # SELL
                deltas[i] = -qtys[i]
        return deltas

    def aggregate_position_deltas_numba(
        symbols: np.ndarray,
        deltas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback implementation when numba is not available."""
        unique_symbols = np.unique(symbols)
        aggregated = np.zeros(len(unique_symbols), dtype=np.float64)
        
        for i, sym in enumerate(unique_symbols):
            mask = symbols == sym
            aggregated[i] = np.sum(deltas[mask])
        
        return unique_symbols, aggregated

