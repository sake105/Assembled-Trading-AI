# tests/test_corporate_actions.py
"""Tests for Corporate Actions Handling (Sprint 4 / C1).

This test suite verifies:
1. Split adjustment creates research prices without fake crashes
2. Dividend cashflow computation generates ledger-ready events
3. Integration with prices and positions
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.corporate_actions import (
    apply_splits_for_research_prices,
    compute_dividend_cashflows,
)


def test_split_adjustment_no_fake_crash() -> None:
    """Test that 2:1 split doesn't create fake -50% crash in research returns."""
    # Create prices: 2:1 split on day 3
    # Day 1-2: before split (high prices)
    # Day 3-5: after split (low prices, but same value)
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "close": [200.0, 205.0, 100.0, 102.0, 104.0],  # 2:1 split on day 3
    })
    
    # Create split action
    actions = pd.DataFrame({
        "symbol": ["AAPL"],
        "action_type": ["SPLIT"],
        "effective_date": [pd.Timestamp("2024-01-03", tz="UTC")],
        "split_ratio": [2.0],  # 2:1 split
    })
    
    # Apply split adjustment
    result = apply_splits_for_research_prices(prices, actions)
    
    # Verify close_research column exists
    assert "close_research" in result.columns, "close_research column should exist"
    
    # Verify close is unchanged (for trading)
    assert result["close"].equals(prices["close"]), "close should be unchanged"
    
    # Verify split adjustment: prices before split are multiplied by 0.5
    # Day 1: 200.0 * 0.5 = 100.0
    # Day 2: 205.0 * 0.5 = 102.5
    # Day 3-5: unchanged (100.0, 102.0, 104.0)
    expected_research = [100.0, 102.5, 100.0, 102.0, 104.0]
    assert result["close_research"].tolist() == pytest.approx(
        expected_research, rel=1e-6
    ), "close_research should be split-adjusted"
    
    # Verify returns with close_research show no fake crash
    returns_research = result["close_research"].pct_change().dropna()
    # Day 1->2: (102.5 - 100.0) / 100.0 = +2.5%
    # Day 2->3: (100.0 - 102.5) / 102.5 = -2.44% (small, not -50%!)
    # Day 3->4: (102.0 - 100.0) / 100.0 = +2.0%
    # Day 4->5: (104.0 - 102.0) / 102.0 = +1.96%
    
    # No return should be around -50% (fake crash)
    assert all(returns_research > -0.1), "No fake crash in research returns"
    
    # Compare with unadjusted returns (would show -50% crash)
    returns_unadjusted = result["close"].pct_change().dropna()
    # Day 2->3: (100.0 - 205.0) / 205.0 = -51.2% (fake crash!)
    assert any(returns_unadjusted < -0.5), "Unadjusted returns show fake crash"


def test_split_adjustment_multiple_splits() -> None:
    """Test that multiple splits are applied correctly."""
    # Create prices with two splits
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=7, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 7,
        "close": [200.0, 205.0, 100.0, 102.0, 50.0, 51.0, 52.0],  # 2:1 split on day 3, 2:1 split on day 5
    })
    
    # Create two split actions
    actions = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "action_type": ["SPLIT", "SPLIT"],
        "effective_date": [
            pd.Timestamp("2024-01-03", tz="UTC"),
            pd.Timestamp("2024-01-05", tz="UTC"),
        ],
        "split_ratio": [2.0, 2.0],  # Both 2:1 splits
    })
    
    # Apply split adjustment
    result = apply_splits_for_research_prices(prices, actions)
    
    # Verify adjustments:
    # Day 1-2: adjusted by 0.5 (first split) and 0.5 (second split) = 0.25
    # Day 3-4: adjusted by 0.5 (second split only)
    # Day 5-7: unchanged
    expected_research = [50.0, 51.25, 50.0, 51.0, 50.0, 51.0, 52.0]
    assert result["close_research"].tolist() == pytest.approx(
        expected_research, rel=1e-6
    ), "Multiple splits should be applied correctly"


def test_dividend_cashflow_computation() -> None:
    """Test that dividend cashflows are computed correctly."""
    # Create positions
    positions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [100.0, 50.0],
    })
    
    # Create dividend action
    actions = pd.DataFrame({
        "symbol": ["AAPL"],
        "action_type": ["DIVIDEND"],
        "effective_date": [pd.Timestamp("2024-01-15", tz="UTC")],
        "dividend_cash": [0.25],  # $0.25 per share
    })
    
    # Compute cashflows
    cashflows = compute_dividend_cashflows(positions, actions)
    
    # Verify schema
    required_cols = {"timestamp", "symbol", "cashflow_type", "amount"}
    assert set(cashflows.columns) == required_cols, "Cashflow schema should match"
    
    # Verify only AAPL gets dividend (MSFT has no dividend action)
    assert len(cashflows) == 1, "Should have one cashflow event"
    assert cashflows.iloc[0]["symbol"] == "AAPL", "Cashflow should be for AAPL"
    assert cashflows.iloc[0]["cashflow_type"] == "DIVIDEND", "Cashflow type should be DIVIDEND"
    assert cashflows.iloc[0]["amount"] == pytest.approx(25.0, rel=1e-6), "Amount should be 100 * 0.25 = 25.0"
    assert cashflows.iloc[0]["timestamp"] == pd.Timestamp("2024-01-15", tz="UTC"), "Timestamp should match"


def test_dividend_cashflow_multiple_dividends() -> None:
    """Test that multiple dividends are computed correctly."""
    # Create positions
    positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [100.0],
    })
    
    # Create two dividend actions
    actions = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "action_type": ["DIVIDEND", "DIVIDEND"],
        "effective_date": [
            pd.Timestamp("2024-01-15", tz="UTC"),
            pd.Timestamp("2024-04-15", tz="UTC"),
        ],
        "dividend_cash": [0.25, 0.30],  # $0.25 and $0.30 per share
    })
    
    # Compute cashflows
    cashflows = compute_dividend_cashflows(positions, actions)
    
    # Verify two cashflow events
    assert len(cashflows) == 2, "Should have two cashflow events"
    
    # Verify amounts
    amounts = cashflows["amount"].tolist()
    assert amounts == pytest.approx([25.0, 30.0], rel=1e-6), "Amounts should be 100 * 0.25 and 100 * 0.30"
    
    # Verify sorted by timestamp
    assert cashflows["timestamp"].is_monotonic_increasing, "Cashflows should be sorted by timestamp"


def test_dividend_cashflow_as_of_filter() -> None:
    """Test that as_of filter works correctly."""
    # Create positions
    positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [100.0],
    })
    
    # Create two dividend actions
    actions = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "action_type": ["DIVIDEND", "DIVIDEND"],
        "effective_date": [
            pd.Timestamp("2024-01-15", tz="UTC"),
            pd.Timestamp("2024-04-15", tz="UTC"),
        ],
        "dividend_cash": [0.25, 0.30],
    })
    
    # Compute cashflows with as_of filter (only first dividend)
    as_of = pd.Timestamp("2024-02-01", tz="UTC")
    cashflows = compute_dividend_cashflows(positions, actions, as_of=as_of)
    
    # Verify only first dividend is included
    assert len(cashflows) == 1, "Should have one cashflow event (filtered by as_of)"
    assert cashflows.iloc[0]["timestamp"] == pd.Timestamp("2024-01-15", tz="UTC"), "Should be first dividend"


def test_split_adjustment_empty_actions() -> None:
    """Test that empty actions return unchanged prices."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 3,
        "close": [100.0, 102.0, 104.0],
    })
    
    actions = pd.DataFrame(columns=["symbol", "action_type", "effective_date", "split_ratio"])
    
    result = apply_splits_for_research_prices(prices, actions)
    
    # Verify close_research = close (no adjustment)
    assert result["close_research"].equals(result["close"]), "close_research should equal close when no actions"


def test_dividend_cashflow_empty_positions() -> None:
    """Test that empty positions return empty cashflows."""
    positions = pd.DataFrame(columns=["symbol", "qty"])
    
    actions = pd.DataFrame({
        "symbol": ["AAPL"],
        "action_type": ["DIVIDEND"],
        "effective_date": [pd.Timestamp("2024-01-15", tz="UTC")],
        "dividend_cash": [0.25],
    })
    
    cashflows = compute_dividend_cashflows(positions, actions)
    
    # Verify empty result
    assert cashflows.empty, "Should return empty cashflows when no positions"


def test_split_adjustment_validation() -> None:
    """Test that validation errors are raised for invalid inputs."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 3,
        "close": [100.0, 102.0, 104.0],
    })
    
    # Missing required column
    actions_invalid = pd.DataFrame({
        "symbol": ["AAPL"],
        "action_type": ["SPLIT"],
        # Missing effective_date and split_ratio
    })
    
    with pytest.raises(ValueError, match="missing required columns"):
        apply_splits_for_research_prices(prices, actions_invalid)
    
    # Non-SPLIT action
    actions_dividend = pd.DataFrame({
        "symbol": ["AAPL"],
        "action_type": ["DIVIDEND"],  # Wrong type
        "effective_date": [pd.Timestamp("2024-01-15", tz="UTC")],
        "split_ratio": [2.0],
    })
    
    with pytest.raises(ValueError, match="must contain only SPLIT actions"):
        apply_splits_for_research_prices(prices, actions_dividend)


def test_dividend_cashflow_validation() -> None:
    """Test that validation errors are raised for invalid inputs."""
    positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [100.0],
    })
    
    # Missing required column
    actions_invalid = pd.DataFrame({
        "symbol": ["AAPL"],
        "action_type": ["DIVIDEND"],
        # Missing effective_date and dividend_cash
    })
    
    with pytest.raises(ValueError, match="missing required columns"):
        compute_dividend_cashflows(positions, actions_invalid)
    
    # Non-DIVIDEND action
    actions_split = pd.DataFrame({
        "symbol": ["AAPL"],
        "action_type": ["SPLIT"],  # Wrong type
        "effective_date": [pd.Timestamp("2024-01-15", tz="UTC")],
        "dividend_cash": [0.25],
    })
    
    with pytest.raises(ValueError, match="must contain only DIVIDEND actions"):
        compute_dividend_cashflows(positions, actions_split)
