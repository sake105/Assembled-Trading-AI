"""Smoke tests for reconciliation engine (Sprint 13 L3)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.accounting.reconciliation import reconcile_ledger_vs_broker


def test_exact_match_ok():
    """Test that exact match returns ok=True."""
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )

    assert result["ok"] is True
    assert result["cash_match"] is True
    assert abs(result["cash_diff"]) < 1e-6
    assert len(result["position_diffs_df"]) == 0
    assert len(result["missing_in_ledger"]) == 0
    assert len(result["missing_in_broker"]) == 0
    assert "OK" in result["message"]


def test_qty_mismatch_detected():
    """Test that qty mismatch is detected."""
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 55.0],  # MSFT mismatch
        }
    )

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
        fail_fast=False,  # intentional mismatch for assertion testing
    )

    assert result["ok"] is False
    assert len(result["position_diffs_df"]) == 1
    assert result["position_diffs_df"].iloc[0]["symbol"] == "MSFT"
    assert abs(result["position_diffs_df"].iloc[0]["diff_qty"] - (-5.0)) < 1e-6
    assert "mismatch" in result["message"].lower()


def test_cash_mismatch_detected():
    """Test that cash mismatch is detected."""
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [100.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [100.0],
        }
    )

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10001.0,  # Cash mismatch
        fail_fast=False,  # intentional mismatch for assertion testing
    )

    assert result["ok"] is False
    assert result["cash_match"] is False
    assert abs(result["cash_diff"] - (-1.0)) < 1e-6
    assert "cash" in result["message"].lower()


def test_missing_symbol_detected():
    """Test that missing symbols are detected."""
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],  # MSFT missing
            "qty": [100.0],
        }
    )

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
        fail_fast=False,  # intentional mismatch for assertion testing
    )

    assert result["ok"] is False
    assert len(result["missing_in_broker"]) == 1
    assert result["missing_in_broker"][0] == "MSFT"
    assert "missing" in result["message"].lower()

    # Test missing in ledger
    ledger_positions2 = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [100.0],
        }
    )
    broker_positions2 = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],  # MSFT in broker but not ledger
            "qty": [100.0, 50.0],
        }
    )

    result2 = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions2,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions2,
        broker_cash=10000.0,
        fail_fast=False,  # intentional mismatch for assertion testing
    )

    assert result2["ok"] is False
    assert len(result2["missing_in_ledger"]) == 1
    assert result2["missing_in_ledger"][0] == "MSFT"


def test_tolerance_behavior_deterministic():
    """Test that tolerance behavior is deterministic."""
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [100.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [100.0 + 1e-9],  # Very small difference
        }
    )

    # With default tolerance (qty_tol=1e-8), this should be treated as match
    result1 = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
        qty_tol=1e-8,
    )

    assert result1["ok"] is True
    assert len(result1["position_diffs_df"]) == 0

    # With stricter tolerance (qty_tol=1e-10), this should be detected as mismatch
    result2 = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
        qty_tol=1e-10,
        fail_fast=False,  # intentional mismatch for assertion testing
    )

    assert result2["ok"] is False
    assert len(result2["position_diffs_df"]) == 1

    # Cash tolerance test
    result3 = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0 + 1e-7,  # Very small cash difference
        cash_tol=1e-6,
    )

    assert result3["ok"] is True
    assert result3["cash_match"] is True

    result4 = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0 + 1e-5,  # Larger cash difference
        cash_tol=1e-6,
        fail_fast=False,  # intentional mismatch for assertion testing
    )

    assert result4["ok"] is False
    assert result4["cash_match"] is False


def test_fail_fast_raises():
    """Test that fail_fast=True raises ValueError on mismatch."""
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [100.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [105.0],  # Mismatch
        }
    )

    with pytest.raises(ValueError, match="Reconciliation FAILED"):
        reconcile_ledger_vs_broker(
            ledger_positions_df=ledger_positions,
            ledger_cash=10000.0,
            broker_positions_df=broker_positions,
            broker_cash=10000.0,
            fail_fast=True,
        )

    # Exact match should not raise
    broker_positions2 = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [100.0],
        }
    )

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions2,
        broker_cash=10000.0,
        fail_fast=True,
    )

    assert result["ok"] is True


def test_empty_positions():
    """Test reconciliation with empty positions."""
    ledger_positions = pd.DataFrame(columns=["symbol", "qty"])
    broker_positions = pd.DataFrame(columns=["symbol", "qty"])

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )

    assert result["ok"] is True
    assert len(result["position_diffs_df"]) == 0
    assert len(result["missing_in_ledger"]) == 0
    assert len(result["missing_in_broker"]) == 0


def test_symbol_trimming():
    """Test that symbol trimming works correctly."""
    ledger_positions = pd.DataFrame(
        {
            "symbol": [" AAPL ", "MSFT"],  # Extra spaces
            "qty": [100.0, 50.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", " MSFT "],  # Extra spaces
            "qty": [100.0, 50.0],
        }
    )

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )

    # Should match after trimming
    assert result["ok"] is True


def test_zero_positions_ignored():
    """Test that zero positions are ignored."""
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "qty": [100.0, 0.0, 50.0],  # MSFT is zero
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "GOOGL"],
            "qty": [100.0, 50.0],
        }
    )

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )

    # Should match (zero positions ignored)
    assert result["ok"] is True
    assert len(result["missing_in_ledger"]) == 0
    assert len(result["missing_in_broker"]) == 0


def test_tiny_residual_qty_ignored():
    """Test that tiny residual qty (1e-10) is ignored via qty_tol."""
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "qty": [100.0, 50.0, 1e-10],  # GOOGL has tiny residual
        }
    )

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
        qty_tol=1e-8,  # Default tolerance should filter out 1e-10
    )

    # Should match (tiny residual ignored)
    assert result["ok"] is True
    assert len(result["missing_in_ledger"]) == 0
    assert len(result["missing_in_broker"]) == 0

    # Test with stricter tolerance - should detect tiny residual
    result2 = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
        qty_tol=1e-12,  # Stricter tolerance
        fail_fast=False,  # intentional mismatch for assertion testing
    )

    # Should detect missing in ledger (GOOGL not filtered out)
    assert result2["ok"] is False
    assert len(result2["missing_in_ledger"]) == 1
    assert result2["missing_in_ledger"][0] == "GOOGL"


def test_missing_lists_deterministically_sorted():
    """Test that missing_in_ledger and missing_in_broker are deterministically sorted."""
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["GOOGL", "TSLA", "NVDA"],  # Different symbols, unsorted
            "qty": [200.0, 150.0, 75.0],
        }
    )

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
        fail_fast=False,  # intentional mismatch for sorting test
    )

    # missing_in_ledger should be sorted (GOOGL, NVDA, TSLA)
    assert len(result["missing_in_ledger"]) == 3
    assert result["missing_in_ledger"] == sorted(result["missing_in_ledger"])
    assert result["missing_in_ledger"] == ["GOOGL", "NVDA", "TSLA"]

    # missing_in_broker should be sorted (AAPL, MSFT)
    assert len(result["missing_in_broker"]) == 2
    assert result["missing_in_broker"] == sorted(result["missing_in_broker"])
    assert result["missing_in_broker"] == ["AAPL", "MSFT"]

    # Run again to ensure deterministic
    result2 = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
        fail_fast=False,  # intentional mismatch for sorting test
    )

    assert result2["missing_in_ledger"] == result["missing_in_ledger"]
    assert result2["missing_in_broker"] == result["missing_in_broker"]


def test_fail_fast_message_contains_key_differences():
    """Test that fail_fast message contains cash_diff and all affected symbols."""
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "qty": [100.0, 50.0, 25.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "TSLA"],
            "qty": [105.0, 45.0, 30.0],  # Mismatches + missing
        }
    )

    with pytest.raises(ValueError) as exc_info:
        reconcile_ledger_vs_broker(
            ledger_positions_df=ledger_positions,
            ledger_cash=10000.0,
            broker_positions_df=broker_positions,
            broker_cash=10050.0,  # Cash mismatch
            fail_fast=True,
        )

    message = str(exc_info.value)

    # Message should contain cash_diff
    assert "diff=" in message or "Cash mismatch" in message
    assert "cash" in message.lower()

    # Message should contain all affected symbols
    # Position mismatches: AAPL, MSFT
    # Missing in ledger: TSLA
    # Missing in broker: GOOGL
    assert "AAPL" in message or "MSFT" in message
    assert "TSLA" in message or "GOOGL" in message

    # Message should be deterministic (sorted symbols)
    # Run again to verify
    with pytest.raises(ValueError) as exc_info2:
        reconcile_ledger_vs_broker(
            ledger_positions_df=ledger_positions,
            ledger_cash=10000.0,
            broker_positions_df=broker_positions,
            broker_cash=10050.0,
            fail_fast=True,
        )

    assert str(exc_info2.value) == message  # Deterministic message
