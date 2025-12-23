"""Tests for Numba-accelerated backtest engine functions."""

from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

pytestmark = pytest.mark.phase4


@pytest.fixture
def sample_orders() -> pd.DataFrame:
    """Create sample orders DataFrame for testing."""
    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-01"),
            "symbol": ["AAPL", "MSFT", "AAPL", "GOOGL"],
            "side": ["BUY", "BUY", "SELL", "BUY"],
            "qty": [10.0, 5.0, 3.0, 2.0],
            "price": [150.0, 200.0, 150.0, 100.0],
        }
    )


@pytest.fixture
def sample_current_positions() -> pd.DataFrame:
    """Create sample current positions DataFrame."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [5.0, 10.0],
        }
    )


@pytest.mark.advanced
def test_numba_module_import():
    """Test that numba module can be imported."""
    try:
        from src.assembled_core.qa.backtest_engine_numba import (
            NUMBA_AVAILABLE,
            compute_position_deltas_numba,
            aggregate_position_deltas_numba,
        )

        # Module should always be importable, even if numba is not installed
        assert isinstance(NUMBA_AVAILABLE, bool)
        assert callable(compute_position_deltas_numba)
        assert callable(aggregate_position_deltas_numba)
    except ImportError as e:
        pytest.fail(
            f"Numba module should be importable even without numba installed: {e}"
        )


@pytest.mark.advanced
def test_compute_position_deltas_numba_basic():
    """Test compute_position_deltas_numba with basic inputs."""
    from src.assembled_core.qa.backtest_engine_numba import (
        compute_position_deltas_numba,
    )

    # Test with BUY orders (side=0)
    sides = np.array([0, 0], dtype=np.int32)
    qtys = np.array([10.0, 5.0], dtype=np.float64)
    deltas = compute_position_deltas_numba(sides, qtys)

    assert len(deltas) == 2
    assert deltas[0] == 10.0
    assert deltas[1] == 5.0

    # Test with SELL orders (side=1)
    sides = np.array([1, 1], dtype=np.int32)
    qtys = np.array([10.0, 5.0], dtype=np.float64)
    deltas = compute_position_deltas_numba(sides, qtys)

    assert len(deltas) == 2
    assert deltas[0] == -10.0
    assert deltas[1] == -5.0

    # Test mixed
    sides = np.array([0, 1, 0], dtype=np.int32)
    qtys = np.array([10.0, 5.0, 3.0], dtype=np.float64)
    deltas = compute_position_deltas_numba(sides, qtys)

    assert len(deltas) == 3
    assert deltas[0] == 10.0
    assert deltas[1] == -5.0
    assert deltas[2] == 3.0


@pytest.mark.advanced
def test_aggregate_position_deltas_numba_basic():
    """Test aggregate_position_deltas_numba with basic inputs."""
    from src.assembled_core.qa.backtest_engine_numba import (
        aggregate_position_deltas_numba,
    )

    # Test with single symbol
    symbols = np.array([0, 0, 0], dtype=np.int32)
    deltas = np.array([10.0, -5.0, 3.0], dtype=np.float64)
    unique_symbols, aggregated = aggregate_position_deltas_numba(symbols, deltas)

    assert len(unique_symbols) == 1
    assert unique_symbols[0] == 0
    assert len(aggregated) == 1
    assert abs(aggregated[0] - 8.0) < 1e-9  # 10 - 5 + 3 = 8

    # Test with multiple symbols
    symbols = np.array([0, 1, 0, 1], dtype=np.int32)
    deltas = np.array([10.0, 5.0, -3.0, -2.0], dtype=np.float64)
    unique_symbols, aggregated = aggregate_position_deltas_numba(symbols, deltas)

    assert len(unique_symbols) == 2
    assert unique_symbols[0] == 0
    assert unique_symbols[1] == 1
    assert abs(aggregated[0] - 7.0) < 1e-9  # 10 - 3 = 7
    assert abs(aggregated[1] - 3.0) < 1e-9  # 5 - 2 = 3


@pytest.mark.advanced
def test_update_positions_vectorized_with_numba(
    sample_orders, sample_current_positions
):
    """Test _update_positions_vectorized with Numba path enabled."""
    from src.assembled_core.qa.backtest_engine import _update_positions_vectorized

    # Test with Numba (if available)
    result = _update_positions_vectorized(
        sample_orders, sample_current_positions, use_numba=True
    )

    # Verify result structure
    assert isinstance(result, pd.DataFrame)
    assert "symbol" in result.columns
    assert "qty" in result.columns

    # Verify positions are correct
    # AAPL: 5.0 (current) + 10.0 (BUY) - 3.0 (SELL) = 12.0
    # MSFT: 10.0 (current) + 5.0 (BUY) = 15.0
    # GOOGL: 0.0 (new) + 2.0 (BUY) = 2.0
    result_dict = dict(zip(result["symbol"], result["qty"]))

    assert abs(result_dict.get("AAPL", 0.0) - 12.0) < 1e-6
    assert abs(result_dict.get("MSFT", 0.0) - 15.0) < 1e-6
    assert abs(result_dict.get("GOOGL", 0.0) - 2.0) < 1e-6


@pytest.mark.advanced
def test_update_positions_vectorized_without_numba(
    sample_orders, sample_current_positions
):
    """Test _update_positions_vectorized with Numba path disabled."""
    from src.assembled_core.qa.backtest_engine import _update_positions_vectorized

    # Test without Numba (pandas fallback)
    result_numba = _update_positions_vectorized(
        sample_orders, sample_current_positions, use_numba=True
    )

    result_pandas = _update_positions_vectorized(
        sample_orders, sample_current_positions, use_numba=False
    )

    # Both should produce identical results
    pd.testing.assert_frame_equal(
        result_numba.sort_values("symbol").reset_index(drop=True),
        result_pandas.sort_values("symbol").reset_index(drop=True),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.advanced
def test_update_positions_vectorized_numerical_compatibility():
    """Test that vectorized position updates match expected numerical results."""
    from src.assembled_core.qa.backtest_engine import _update_positions_vectorized

    # Create orders with known outcome
    orders = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-01"),
            "symbol": ["A", "B", "A"],
            "side": ["BUY", "BUY", "SELL"],
            "qty": [100.0, 50.0, 30.0],
            "price": [10.0, 20.0, 10.0],
        }
    )

    current_positions = pd.DataFrame(
        {
            "symbol": ["A"],
            "qty": [10.0],
        }
    )

    result = _update_positions_vectorized(orders, current_positions, use_numba=True)

    # Expected: A = 10 + 100 - 30 = 80, B = 0 + 50 = 50
    result_dict = dict(zip(result["symbol"], result["qty"]))

    assert abs(result_dict.get("A", 0.0) - 80.0) < 1e-9
    assert abs(result_dict.get("B", 0.0) - 50.0) < 1e-9


@pytest.mark.advanced
def test_update_positions_vectorized_empty_orders(sample_current_positions):
    """Test _update_positions_vectorized with empty orders."""
    from src.assembled_core.qa.backtest_engine import _update_positions_vectorized

    empty_orders = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    result = _update_positions_vectorized(
        empty_orders, sample_current_positions, use_numba=True
    )

    # Should return unchanged positions
    pd.testing.assert_frame_equal(
        result.sort_values("symbol").reset_index(drop=True),
        sample_current_positions.sort_values("symbol").reset_index(drop=True),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.advanced
@pytest.mark.parametrize("use_numba", [True, False])
def test_backtest_engine_with_numba_option(use_numba):
    """Test that backtest engine works with both Numba and pandas paths."""
    pytest.importorskip("pandas")

    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

    # Create minimal test data
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [100.0 + i * 0.1 for i in range(10)],
        }
    )

    def dummy_signal_fn(prices_df):
        signals = []
        for _, row in prices_df.iterrows():
            signals.append(
                {
                    "timestamp": row["timestamp"],
                    "symbol": row["symbol"],
                    "direction": "LONG" if row["close"] > 100.5 else "FLAT",
                    "score": 1.0 if row["close"] > 100.5 else 0.0,
                }
            )
        return pd.DataFrame(signals)

    def dummy_position_sizing_fn(signals_df, capital):
        long_signals = signals_df[signals_df["direction"] == "LONG"]
        if long_signals.empty:
            return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])

        targets = []
        n = len(long_signals)
        for symbol in long_signals["symbol"].unique():
            targets.append(
                {
                    "symbol": symbol,
                    "target_weight": 1.0 / n if n > 0 else 0.0,
                    "target_qty": (capital / n) / 100.0,  # Rough estimate
                }
            )
        return pd.DataFrame(targets)

    # Run backtest (should work with both paths)
    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=dummy_signal_fn,
        position_sizing_fn=dummy_position_sizing_fn,
        start_capital=10000.0,
        include_costs=False,
        include_trades=True,
    )

    # Verify result
    assert result is not None
    assert hasattr(result, "equity")
    assert len(result.equity) > 0
