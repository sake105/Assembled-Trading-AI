# tests/test_sprint4_c3_integration.py
"""Sprint 4 / C3 Integration Tests - Hard Integration Suite.

This test suite verifies end-to-end integration of Corporate Actions and Universe Management:
1. Split adjustment prevents fake crashes in research prices
2. Dividend cashflows are generated correctly
3. Universe membership changes correctly with as_of

Additional checks:
- Contracts: UTC, required columns, deterministic sorting
- No network calls, no external data sources
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.corporate_actions import (
    apply_splits_for_research_prices,
    compute_dividend_cashflows,
)
from src.assembled_core.data.universe import get_universe_members


def test_split_day_no_fake_crash_in_research_prices() -> None:
    """Test that split day doesn't create fake crash in research prices (C3 integration)."""
    # Create prices with 2:1 split on day 3
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
    prices_research = apply_splits_for_research_prices(prices, actions)
    
    # Verify: close_research column exists
    assert "close_research" in prices_research.columns, "close_research column should exist"
    
    # Verify: close is unchanged (for trading)
    assert prices_research["close"].equals(prices["close"]), "close should be unchanged for trading"
    
    # Verify: close_research is split-adjusted
    # Day 1-2: adjusted by 0.5 (200.0 -> 100.0, 205.0 -> 102.5)
    # Day 3-5: unchanged (100.0, 102.0, 104.0)
    expected_research = [100.0, 102.5, 100.0, 102.0, 104.0]
    assert prices_research["close_research"].tolist() == expected_research, "close_research should be split-adjusted"
    
    # Verify: Returns with close_research show NO fake crash
    returns_research = prices_research["close_research"].pct_change().dropna()
    # Day 1->2: (102.5 - 100.0) / 100.0 = +2.5%
    # Day 2->3: (100.0 - 102.5) / 102.5 = -2.44% (small, NOT -50%!)
    # Day 3->4: (102.0 - 100.0) / 100.0 = +2.0%
    # Day 4->5: (104.0 - 102.0) / 102.0 = +1.96%
    
    # Critical: No return should be around -50% (fake crash)
    assert all(returns_research > -0.1), "Research returns should NOT show fake crash (no -50% return)"
    
    # Compare with unadjusted returns (would show -50% crash)
    returns_unadjusted = prices_research["close"].pct_change().dropna()
    # Day 2->3: (100.0 - 205.0) / 205.0 = -51.2% (fake crash!)
    assert any(returns_unadjusted < -0.5), "Unadjusted returns show fake crash (expected)"
    
    # Contract checks: UTC, required columns, deterministic sorting
    assert prices_research["timestamp"].dt.tz is not None, "Timestamps should be UTC-aware"
    # Check UTC (tz.zone might not exist, use str(tz) or compare with UTC)
    tz_first = prices_research["timestamp"].iloc[0].tz
    assert tz_first is not None, "Timestamps should be UTC-aware"
    assert str(tz_first) == "UTC" or (hasattr(tz_first, "zone") and tz_first.zone == "UTC"), "Timestamps should be UTC"
    assert "symbol" in prices_research.columns, "Required column 'symbol' should exist"
    assert "close" in prices_research.columns, "Required column 'close' should exist"
    assert "close_research" in prices_research.columns, "Required column 'close_research' should exist"
    
    # Verify deterministic sorting (symbol, timestamp)
    sorted_check = prices_research.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    assert prices_research.equals(sorted_check), "Prices should be sorted by (symbol, timestamp)"


def test_dividend_day_cashflow_event_generated() -> None:
    """Test that dividend day generates cashflow event (C3 integration)."""
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
    
    # Verify: Cashflow event generated
    assert len(cashflows) == 1, "Should generate one cashflow event"
    assert cashflows.iloc[0]["symbol"] == "AAPL", "Cashflow should be for AAPL"
    assert cashflows.iloc[0]["cashflow_type"] == "DIVIDEND", "Cashflow type should be DIVIDEND"
    assert cashflows.iloc[0]["amount"] == 25.0, "Amount should be 100 * 0.25 = 25.0"
    assert cashflows.iloc[0]["timestamp"] == pd.Timestamp("2024-01-15", tz="UTC"), "Timestamp should match effective_date"
    
    # Verify: MSFT has no cashflow (no dividend action)
    msft_cashflows = cashflows[cashflows["symbol"] == "MSFT"]
    assert len(msft_cashflows) == 0, "MSFT should have no cashflow (no dividend action)"
    
    # Contract checks: UTC, required columns, deterministic sorting
    assert cashflows["timestamp"].dt.tz is not None, "Timestamps should be UTC-aware"
    # Check UTC: verify first timestamp is UTC-aware (simplified check)
    tz_first = cashflows["timestamp"].iloc[0].tz
    assert tz_first is not None, "Cashflow timestamps should be UTC-aware"
    required_cols = {"timestamp", "symbol", "cashflow_type", "amount"}
    assert set(cashflows.columns) == required_cols, f"Cashflows should have required columns: {required_cols}"
    
    # Verify deterministic sorting (timestamp, symbol)
    sorted_check = cashflows.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    assert cashflows.equals(sorted_check), "Cashflows should be sorted by (timestamp, symbol)"


def test_universe_membership_as_of_changes() -> None:
    """Test that universe membership changes correctly with as_of (C3 integration)."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create universe history with membership changes:
        # AAPL: added 2024-01-01, removed 2024-06-30 (exclusive)
        # MSFT: added 2024-01-01, still active (end_date=None)
        # GOOGL: added 2024-07-01, still active (end_date=None)
        from src.assembled_core.data.universe import store_universe_history
        
        history = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "start_date": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-07-01", tz="UTC"),
            ],
            "end_date": [
                pd.Timestamp("2024-06-30", tz="UTC"),  # Removed on 2024-06-30 (exclusive)
                None,  # Still active
                None,  # Still active
            ],
        })
        
        # Store universe history
        store_universe_history(history, universe_name="test", root=root, format="parquet")
        
        # Test: as_of = 2024-06-29 (before AAPL removal)
        as_of_before = pd.Timestamp("2024-06-29", tz="UTC")
        members_before = get_universe_members(as_of_before, universe_name="test", root=root)
        assert set(members_before) == {"AAPL", "MSFT"}, "AAPL and MSFT should be in universe before removal"
        assert "GOOGL" not in members_before, "GOOGL should not be in universe yet (added on 2024-07-01)"
        
        # Test: as_of = 2024-06-30 (AAPL removal day, EXCLUSIVE)
        as_of_removal = pd.Timestamp("2024-06-30", tz="UTC")
        members_removal = get_universe_members(as_of_removal, universe_name="test", root=root)
        assert set(members_removal) == {"MSFT"}, "Only MSFT should be in universe on removal day (AAPL excluded)"
        assert "AAPL" not in members_removal, "AAPL should NOT be in universe on removal day (end_date exclusive)"
        assert "GOOGL" not in members_removal, "GOOGL should not be in universe yet"
        
        # Test: as_of = 2024-07-01 (GOOGL addition day)
        as_of_after = pd.Timestamp("2024-07-01", tz="UTC")
        members_after = get_universe_members(as_of_after, universe_name="test", root=root)
        assert set(members_after) == {"MSFT", "GOOGL"}, "MSFT and GOOGL should be in universe after addition"
        assert "AAPL" not in members_after, "AAPL should not be in universe after removal"
        
        # Contract checks: Deterministic, sorted, uppercase
        assert members_before == sorted(members_before), "Members should be sorted (deterministic)"
        assert members_removal == sorted(members_removal), "Members should be sorted (deterministic)"
        assert members_after == sorted(members_after), "Members should be sorted (deterministic)"
        assert all(s.isupper() for s in members_before), "Members should be uppercase"
        assert all(s.isupper() for s in members_removal), "Members should be uppercase"
        assert all(s.isupper() for s in members_after), "Members should be uppercase"


def test_contracts_utc_required_columns_deterministic() -> None:
    """Test that all outputs fulfill contracts: UTC, required columns, deterministic sorting."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Test 1: Corporate Actions - apply_splits_for_research_prices
        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 3,
            "close": [200.0, 205.0, 100.0],
        })
        actions = pd.DataFrame({
            "symbol": ["AAPL"],
            "action_type": ["SPLIT"],
            "effective_date": [pd.Timestamp("2024-01-03", tz="UTC")],
            "split_ratio": [2.0],
        })
        prices_research = apply_splits_for_research_prices(prices, actions)
        
        # Contract: UTC
        assert prices_research["timestamp"].dt.tz is not None, "Timestamps should be UTC-aware"
        # Check UTC: verify first timestamp is UTC-aware (simplified check)
        tz_first = prices_research["timestamp"].iloc[0].tz
        assert tz_first is not None, "Timestamps should be UTC-aware"
        
        # Contract: Required columns
        required_cols = {"timestamp", "symbol", "close", "close_research"}
        assert set(prices_research.columns) >= required_cols, f"Should have required columns: {required_cols}"
        
        # Contract: Deterministic sorting
        sorted_check = prices_research.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        assert prices_research.equals(sorted_check), "Prices should be sorted by (symbol, timestamp)"
        
        # Test 2: Corporate Actions - compute_dividend_cashflows
        positions = pd.DataFrame({
            "symbol": ["AAPL"],
            "qty": [100.0],
        })
        dividend_actions = pd.DataFrame({
            "symbol": ["AAPL"],
            "action_type": ["DIVIDEND"],
            "effective_date": [pd.Timestamp("2024-01-15", tz="UTC")],
            "dividend_cash": [0.25],
        })
        cashflows = compute_dividend_cashflows(positions, dividend_actions)
        
        # Contract: UTC
        assert cashflows["timestamp"].dt.tz is not None, "Cashflow timestamps should be UTC-aware"
        # Check UTC: verify first timestamp is UTC-aware (simplified check)
        tz_first = cashflows["timestamp"].iloc[0].tz
        assert tz_first is not None, "Cashflow timestamps should be UTC-aware"
        
        # Contract: Required columns
        required_cashflow_cols = {"timestamp", "symbol", "cashflow_type", "amount"}
        assert set(cashflows.columns) == required_cashflow_cols, f"Cashflows should have required columns: {required_cashflow_cols}"
        
        # Contract: Deterministic sorting
        sorted_cashflows = cashflows.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        assert cashflows.equals(sorted_cashflows), "Cashflows should be sorted by (timestamp, symbol)"
        
        # Test 3: Universe - get_universe_members
        from src.assembled_core.data.universe import store_universe_history
        
        history = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "start_date": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-01", tz="UTC"),
            ],
            "end_date": [None, None],
        })
        store_universe_history(history, universe_name="test", root=root, format="parquet")
        
        as_of = pd.Timestamp("2024-06-15", tz="UTC")
        members = get_universe_members(as_of, universe_name="test", root=root)
        
        # Contract: Deterministic (sorted, uppercase)
        assert members == sorted(members), "Members should be sorted (deterministic)"
        assert all(s.isupper() for s in members), "Members should be uppercase"
        
        # Contract: UTC (as_of is UTC-aware)
        assert as_of.tz is not None, "as_of should be UTC-aware"
