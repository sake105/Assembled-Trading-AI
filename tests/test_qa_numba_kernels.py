"""Unit tests for optional Numba-accelerated kernels.

This test module verifies that:
- Numba kernels work correctly when Numba is available
- Code falls back gracefully to pure NumPy when Numba is not available
- Results are identical between Numba and NumPy implementations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.pipeline.backtest import _update_equity_mark_to_market


def test_equity_mark_to_market_without_numba() -> None:
    """Test that equity calculation works without Numba (fallback to pure NumPy)."""
    # Create price pivot
    timestamps = [pd.Timestamp("2024-01-01", tz="UTC")]
    symbols = ["AAPL", "MSFT"]
    price_data = {
        "timestamp": timestamps * 2,
        "symbol": ["AAPL", "MSFT"],
        "close": [100.0, 200.0],
    }
    prices_df = pd.DataFrame(price_data)
    price_pivot = prices_df.pivot(index="timestamp", columns="symbol", values="close")

    positions = {"AAPL": 10.0, "MSFT": 5.0}
    cash = 1000.0

    # Test with use_numba=False (force pure NumPy)
    equity = _update_equity_mark_to_market(
        timestamp=timestamps[0],
        cash=cash,
        positions=positions,
        price_pivot=price_pivot,
        symbols=symbols,
        use_numba=False,
    )

    # Expected: cash + (10 * 100) + (5 * 200) = 1000 + 1000 + 1000 = 3000
    expected = cash + (10.0 * 100.0) + (5.0 * 200.0)
    assert abs(equity - expected) < 1e-6


def test_equity_mark_to_market_with_numba_optional() -> None:
    """Test that equity calculation works with optional Numba (uses Numba if available)."""
    # Create price pivot
    timestamps = [pd.Timestamp("2024-01-01", tz="UTC")]
    symbols = ["AAPL", "MSFT", "GOOGL"]
    price_data = {
        "timestamp": timestamps * 3,
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "close": [100.0, 200.0, 150.0],
    }
    prices_df = pd.DataFrame(price_data)
    price_pivot = prices_df.pivot(index="timestamp", columns="symbol", values="close")

    positions = {"AAPL": 10.0, "MSFT": 5.0, "GOOGL": 3.0}
    cash = 1000.0

    # Test with use_numba=True (will use Numba if available, fallback otherwise)
    equity = _update_equity_mark_to_market(
        timestamp=timestamps[0],
        cash=cash,
        positions=positions,
        price_pivot=price_pivot,
        symbols=symbols,
        use_numba=True,
    )

    # Expected: cash + (10 * 100) + (5 * 200) + (3 * 150) = 1000 + 1000 + 1000 + 450 = 3450
    expected = cash + (10.0 * 100.0) + (5.0 * 200.0) + (3.0 * 150.0)
    assert abs(equity - expected) < 1e-6


def test_equity_mark_to_market_numba_vs_numpy_identical() -> None:
    """Test that Numba and NumPy implementations produce identical results."""
    # Create price pivot
    timestamps = [pd.Timestamp("2024-01-01", tz="UTC")]
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    price_data = {
        "timestamp": timestamps * 4,
        "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"],
        "close": [100.0, 200.0, 150.0, 250.0],
    }
    prices_df = pd.DataFrame(price_data)
    price_pivot = prices_df.pivot(index="timestamp", columns="symbol", values="close")

    positions = {"AAPL": 10.0, "MSFT": 5.0, "GOOGL": 3.0, "TSLA": 2.0}
    cash = 1000.0

    # Test with NumPy (use_numba=False)
    equity_numpy = _update_equity_mark_to_market(
        timestamp=timestamps[0],
        cash=cash,
        positions=positions,
        price_pivot=price_pivot,
        symbols=symbols,
        use_numba=False,
    )

    # Test with Numba (use_numba=True, will fallback if not available)
    equity_numba = _update_equity_mark_to_market(
        timestamp=timestamps[0],
        cash=cash,
        positions=positions,
        price_pivot=price_pivot,
        symbols=symbols,
        use_numba=True,
    )

    # Results should be identical (within numerical precision)
    assert abs(equity_numpy - equity_numba) < 1e-10


def test_equity_mark_to_market_nan_prices() -> None:
    """Test that NaN prices are handled correctly with both NumPy and Numba paths."""
    timestamps = [pd.Timestamp("2024-01-01", tz="UTC")]
    symbols = ["AAPL", "MSFT", "GOOGL"]
    price_data = {
        "timestamp": timestamps * 3,
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "close": [100.0, np.nan, 150.0],
    }
    prices_df = pd.DataFrame(price_data)
    price_pivot = prices_df.pivot(index="timestamp", columns="symbol", values="close")

    positions = {"AAPL": 10.0, "MSFT": 5.0, "GOOGL": 3.0}
    cash = 1000.0

    # Test with NumPy
    equity_numpy = _update_equity_mark_to_market(
        timestamp=timestamps[0],
        cash=cash,
        positions=positions,
        price_pivot=price_pivot,
        symbols=symbols,
        use_numba=False,
    )

    # Test with Numba (optional)
    equity_numba = _update_equity_mark_to_market(
        timestamp=timestamps[0],
        cash=cash,
        positions=positions,
        price_pivot=price_pivot,
        symbols=symbols,
        use_numba=True,
    )

    # Expected: cash + (10 * 100) + (5 * NaN) + (3 * 150) = 1000 + 1000 + 0 + 450 = 2450
    expected = cash + (10.0 * 100.0) + (3.0 * 150.0)  # MSFT excluded due to NaN

    assert abs(equity_numpy - expected) < 1e-6
    assert abs(equity_numba - expected) < 1e-6


@pytest.mark.skipif(
    not hasattr(__import__("src.assembled_core.qa.numba_kernels", fromlist=["NUMBA_AVAILABLE"]), "NUMBA_AVAILABLE")
    or not __import__("src.assembled_core.qa.numba_kernels", fromlist=["NUMBA_AVAILABLE"]).NUMBA_AVAILABLE,
    reason="Numba not available",
)
def test_numba_kernels_direct() -> None:
    """Test Numba kernels directly (only runs if Numba is available)."""
    from src.assembled_core.qa.numba_kernels import (
        compute_mark_to_market_numba,
        NUMBA_AVAILABLE,
    )

    assert NUMBA_AVAILABLE, "This test should only run if Numba is available"

    # Test mark-to-market kernel
    positions_array = np.array([10.0, 5.0, 3.0], dtype=np.float64)
    prices_array = np.array([100.0, 200.0, 150.0], dtype=np.float64)

    mtm = compute_mark_to_market_numba(positions_array, prices_array)
    expected = (10.0 * 100.0) + (5.0 * 200.0) + (3.0 * 150.0)
    assert abs(mtm - expected) < 1e-6

    # Test with NaN prices
    prices_array_nan = np.array([100.0, np.nan, 150.0], dtype=np.float64)
    mtm_nan = compute_mark_to_market_numba(positions_array, prices_array_nan)
    expected_nan = (10.0 * 100.0) + (3.0 * 150.0)  # NaN filtered out
    assert abs(mtm_nan - expected_nan) < 1e-6


def test_numba_kernels_import_without_numba() -> None:
    """Test that numba_kernels module can be imported even without Numba installed."""
    # This should not raise an ImportError
    from src.assembled_core.qa.numba_kernels import NUMBA_AVAILABLE

    # NUMBA_AVAILABLE should be False if Numba is not installed
    # (but the import should still work)
    assert isinstance(NUMBA_AVAILABLE, bool)

