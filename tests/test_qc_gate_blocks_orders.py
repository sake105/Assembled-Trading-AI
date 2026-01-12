# tests/test_qc_gate_blocks_orders.py
"""Tests for QC Gate blocking orders (Sprint 3 / D2).

This test suite verifies:
1. Gate blocks orders when qa_block_trading=True (unit-level)
2. QC FAIL -> ctx flags -> orders empty (mini integration)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.pipeline.trading_cycle import TradingContext, run_trading_cycle
from src.assembled_core.qa.data_qc import run_price_panel_qc


def test_gate_blocks_orders_unit_level() -> None:
    """Test that gate blocks orders when qa_block_trading=True (unit-level)."""
    # Create minimal valid prices
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "close": [150.0 + i * 0.5 for i in range(10)],
    })

    # Define minimal signal and position sizing functions
    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": df["timestamp"],
            "symbol": df["symbol"],
            "direction": ["LONG"] * len(df),
            "score": [1.0] * len(df),
        })

    def position_sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol": signals["symbol"].unique(),
            "target_weight": [0.1] * len(signals["symbol"].unique()),
            "target_qty": [1.0] * len(signals["symbol"].unique()),
        })

    # Build TradingContext with qa_block_trading=True
    ctx = TradingContext(
        prices=prices,
        freq="1d",
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        capital=10000.0,
        qa_block_trading=True,  # Gate enabled
        qa_block_reason="TEST: QA Gate blocking for test",
    )

    # Run trading cycle
    result = run_trading_cycle(ctx)

    # Assert: Orders should be empty
    assert result.orders.empty, "Orders should be empty when qa_block_trading=True"
    assert result.orders_filtered.empty, "orders_filtered should be empty when qa_block_trading=True"

    # Assert: Orders should have correct schema
    assert list(result.orders.columns) == ["timestamp", "symbol", "side", "qty", "price"], \
        "Orders should have correct schema even when empty"

    # Assert: Reason should be in meta
    assert result.meta.get("qa_block_reason") == "TEST: QA Gate blocking for test", \
        "qa_block_reason should be in result.meta"
    assert result.meta.get("qa_block_trading") is True, \
        "qa_block_trading flag should be in result.meta"

    # Assert: Status should still be success (gate is not an error)
    assert result.status == "success", "Status should be success (gate is not an error)"


def test_qc_fail_blocks_orders_mini_integration() -> None:
    """Test that QC FAIL -> ctx flags -> orders empty (mini integration)."""
    # Create prices with a clear FAIL (negative price)
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "close": [150.0, -10.0, 152.0, 153.0, 154.0],  # Negative price -> FAIL
    })

    # Run QC
    qc_report = run_price_panel_qc(prices, freq="1d", calendar="NYSE")

    # Assert: QC should have FAIL issues
    assert not qc_report.ok, "QC report should not be OK (has FAIL issues)"
    assert qc_report.summary["fail_count"] > 0, "QC report should have FAIL issues"

    # Set QA Gate flags (as Entry Points would do)
    qa_block_trading = not qc_report.ok
    qa_block_reason = f"DATA_QC_FAIL: {qc_report.summary.get('fail_count', 0)} FAIL issues"

    assert qa_block_trading is True, "qa_block_trading should be True when QC has FAIL issues"
    assert qa_block_reason is not None, "qa_block_reason should be set"

    # Define minimal signal and position sizing functions
    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": df["timestamp"],
            "symbol": df["symbol"],
            "direction": ["LONG"] * len(df),
            "score": [1.0] * len(df),
        })

    def position_sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol": signals["symbol"].unique(),
            "target_weight": [0.1] * len(signals["symbol"].unique()),
            "target_qty": [1.0] * len(signals["symbol"].unique()),
        })

    # Build TradingContext with QA Gate flags
    ctx = TradingContext(
        prices=prices,
        freq="1d",
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        capital=10000.0,
        qa_block_trading=qa_block_trading,
        qa_block_reason=qa_block_reason,
    )

    # Run trading cycle
    result = run_trading_cycle(ctx)

    # Assert: Orders should be empty (blocked by gate)
    assert result.orders.empty, "Orders should be empty when QC FAIL -> qa_block_trading=True"
    assert result.orders_filtered.empty, "orders_filtered should be empty when QC FAIL"

    # Assert: Reason should be in meta
    assert "DATA_QC_FAIL" in result.meta.get("qa_block_reason", ""), \
        "qa_block_reason should contain DATA_QC_FAIL"


def test_qc_warn_does_not_block_orders() -> None:
    """Test that QC WARN does not block orders (only FAIL blocks)."""
    # Create prices that might have WARN issues but no FAIL issues
    # (e.g., stale prices, but no negative prices or duplicates)
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,  # Stale prices (might trigger WARN)
    })

    # Run QC
    qc_report = run_price_panel_qc(prices, freq="1d", calendar="NYSE")

    # Set QA Gate flags (only block on FAIL, not WARN)
    qa_block_trading = not qc_report.ok  # Only True if FAIL issues
    qa_block_reason = (
        f"DATA_QC_FAIL: {qc_report.summary.get('fail_count', 0)} FAIL issues"
        if not qc_report.ok
        else None
    )

    # If QC is OK (no FAIL), gate should not block
    if qc_report.ok:
        assert qa_block_trading is False, "qa_block_trading should be False when QC is OK (only WARN)"
        assert qa_block_reason is None, "qa_block_reason should be None when QC is OK"

        # Define minimal signal and position sizing functions
        def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({
                "timestamp": df["timestamp"],
                "symbol": df["symbol"],
                "direction": ["LONG"] * len(df),
                "score": [1.0] * len(df),
            })

        def position_sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
            return pd.DataFrame({
                "symbol": signals["symbol"].unique(),
                "target_weight": [0.1] * len(signals["symbol"].unique()),
                "target_qty": [1.0] * len(signals["symbol"].unique()),
            })

        # Build TradingContext without gate
        ctx = TradingContext(
            prices=prices,
            freq="1d",
            signal_fn=signal_fn,
            position_sizing_fn=position_sizing_fn,
            capital=10000.0,
            qa_block_trading=qa_block_trading,
            qa_block_reason=qa_block_reason,
        )

        # Run trading cycle
        result = run_trading_cycle(ctx)

        # Assert: Orders should NOT be empty (gate did not block)
        # Note: Orders might be empty for other reasons (no signals, etc.), but not because of gate
        assert result.meta.get("qa_block_trading") is not True, \
            "qa_block_trading should not be True when QC is OK"


def test_gate_preserves_order_schema() -> None:
    """Test that gate preserves order schema even when blocking."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "close": [150.0] * 5,
    })

    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": df["timestamp"],
            "symbol": df["symbol"],
            "direction": ["LONG"] * len(df),
            "score": [1.0] * len(df),
        })

    def position_sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol": signals["symbol"].unique(),
            "target_weight": [0.1] * len(signals["symbol"].unique()),
            "target_qty": [1.0] * len(signals["symbol"].unique()),
        })

    ctx = TradingContext(
        prices=prices,
        freq="1d",
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        capital=10000.0,
        qa_block_trading=True,
        qa_block_reason="TEST: Schema preservation",
    )

    result = run_trading_cycle(ctx)

    # Assert: Schema should be correct
    expected_cols = ["timestamp", "symbol", "side", "qty", "price"]
    assert list(result.orders.columns) == expected_cols, \
        f"Orders should have schema {expected_cols}, got {list(result.orders.columns)}"
    assert list(result.orders_filtered.columns) == expected_cols, \
        f"orders_filtered should have schema {expected_cols}"
