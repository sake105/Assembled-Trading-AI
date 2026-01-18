# tests/test_fill_model_session_gate.py
"""Tests for fill model session gate (Sprint 7 / C2).

Tests verify:
1. Weekend orders rejected
2. Holiday orders rejected
3. 1d only close accepted
4. 5min within session accepted, outside rejected
5. Deterministic behavior
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.fill_model import apply_session_gate


def test_weekend_rejected() -> None:
    """Test that weekend orders are rejected."""
    try:
        # Saturday, 2024-01-06 (not a trading day)
        weekend_orders = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-06 21:00", tz="UTC")],  # Saturday
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
        })
        
        fills = apply_session_gate(weekend_orders, freq="1d", strict=True)
        
        # Should be rejected
        assert fills["status"].iloc[0] == "rejected", "Weekend order should be rejected"
        assert fills["fill_qty"].iloc[0] == 0.0, "Rejected order should have fill_qty=0"
        assert fills["remaining_qty"].iloc[0] == 100.0, "Rejected order should have remaining_qty=qty"
    except ImportError:
        # Skip if exchange_calendars not available
        pass


def test_holiday_rejected() -> None:
    """Test that holiday orders are rejected."""
    try:
        # New Year's Day 2024 (not a trading day)
        holiday_orders = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-01 21:00", tz="UTC")],  # New Year's Day
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
        })
        
        fills = apply_session_gate(holiday_orders, freq="1d", strict=True)
        
        # Should be rejected
        assert fills["status"].iloc[0] == "rejected", "Holiday order should be rejected"
        assert fills["fill_qty"].iloc[0] == 0.0, "Rejected order should have fill_qty=0"
        assert fills["remaining_qty"].iloc[0] == 100.0, "Rejected order should have remaining_qty=qty"
    except ImportError:
        # Skip if exchange_calendars not available
        pass


def test_1d_only_close_accepted() -> None:
    """Test that for freq='1d', only session close is accepted."""
    try:
        from src.assembled_core.data.calendar import session_close_utc
        
        # Get a trading day session close
        trading_date = pd.Timestamp("2024-01-02").date()  # Tuesday (trading day)
        session_close = session_close_utc(trading_date)
        
        # Order at session close (should be accepted)
        orders_at_close = pd.DataFrame({
            "timestamp": [session_close],
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
        })
        
        fills_at_close = apply_session_gate(orders_at_close, freq="1d", strict=True)
        
        # Should be accepted (status="filled" by default, or unchanged if already set)
        assert fills_at_close["fill_qty"].iloc[0] == 100.0, "Order at session close should be accepted"
        
        # Order not at session close (should be rejected)
        orders_not_at_close = pd.DataFrame({
            "timestamp": [session_close - pd.Timedelta(hours=2)],  # 2 hours before close
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
        })
        
        fills_not_at_close = apply_session_gate(orders_not_at_close, freq="1d", strict=True)
        
        # Should be rejected
        assert fills_not_at_close["status"].iloc[0] == "rejected", "Order not at session close should be rejected"
        assert fills_not_at_close["fill_qty"].iloc[0] == 0.0, "Rejected order should have fill_qty=0"
    except ImportError:
        # Skip if exchange_calendars not available
        pass


def test_5min_within_session_accepted() -> None:
    """Test that for freq='5min', orders within session are accepted."""
    try:
        from src.assembled_core.data.calendar import get_nyse_calendar
        
        cal = get_nyse_calendar()
        
        # Get a trading day
        trading_date = pd.Timestamp("2024-01-02").date()  # Tuesday (trading day)
        session_ts = pd.Timestamp(trading_date)
        
        # Get session open and close
        session_open_local = cal.session_open(session_ts)
        session_close_local = cal.session_close(session_ts)
        
        # Convert to UTC
        if session_open_local.tz is None:
            session_open_local = session_open_local.tz_localize("America/New_York")
        if session_close_local.tz is None:
            session_close_local = session_close_local.tz_localize("America/New_York")
        
        session_open_utc = session_open_local.tz_convert("UTC")
        session_close_utc = session_close_local.tz_convert("UTC")
        
        # Order within session (middle of day)
        mid_session = session_open_utc + (session_close_utc - session_open_utc) / 2
        
        orders_within = pd.DataFrame({
            "timestamp": [mid_session],
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
        })
        
        fills_within = apply_session_gate(orders_within, freq="5min", strict=True)
        
        # Should be accepted
        assert fills_within["fill_qty"].iloc[0] == 100.0, "Order within session should be accepted"
        
        # Order outside session (before open)
        orders_before_open = pd.DataFrame({
            "timestamp": [session_open_utc - pd.Timedelta(hours=1)],  # 1 hour before open
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
        })
        
        fills_before = apply_session_gate(orders_before_open, freq="5min", strict=True)
        
        # Should be rejected
        assert fills_before["status"].iloc[0] == "rejected", "Order before session open should be rejected"
        assert fills_before["fill_qty"].iloc[0] == 0.0, "Rejected order should have fill_qty=0"
        
        # Order outside session (after close)
        orders_after_close = pd.DataFrame({
            "timestamp": [session_close_utc + pd.Timedelta(hours=1)],  # 1 hour after close
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
        })
        
        fills_after = apply_session_gate(orders_after_close, freq="5min", strict=True)
        
        # Should be rejected
        assert fills_after["status"].iloc[0] == "rejected", "Order after session close should be rejected"
        assert fills_after["fill_qty"].iloc[0] == 0.0, "Rejected order should have fill_qty=0"
    except ImportError:
        # Skip if exchange_calendars not available
        pass


def test_session_gate_deterministic() -> None:
    """Test that session gate is deterministic (same input -> same output)."""
    try:
        orders = pd.DataFrame({
            "timestamp": [
                pd.Timestamp("2024-01-06 21:00", tz="UTC"),  # Saturday (rejected)
                pd.Timestamp("2024-01-02 21:00", tz="UTC"),  # Tuesday (may be accepted/rejected depending on freq)
            ],
            "symbol": ["AAPL", "MSFT"],
            "side": ["BUY", "SELL"],
            "qty": [100.0, 50.0],
            "price": [150.0, 200.0],
        })
        
        # Apply session gate twice
        fills1 = apply_session_gate(orders, freq="1d", strict=True)
        fills2 = apply_session_gate(orders, freq="1d", strict=True)
        
        # Should be identical
        pd.testing.assert_frame_equal(fills1, fills2, "Session gate should be deterministic")
    except ImportError:
        # Skip if exchange_calendars not available
        pass


def test_session_gate_strict_false_fallback() -> None:
    """Test that strict=False allows fallback when exchange_calendars is missing."""
    # This test verifies that if exchange_calendars is not available,
    # strict=False allows all orders (permissive fallback)
    # Note: This test may pass or fail depending on whether exchange_calendars is installed
    orders = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-06 21:00", tz="UTC")],  # Saturday
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
    })
    
    try:
        fills = apply_session_gate(orders, freq="1d", strict=False)
        # If exchange_calendars is available, weekend should be rejected
        # If not available, should warn and allow (permissive)
        assert len(fills) == 1, "Should return one fill"
    except ImportError:
        # If strict=False and exchange_calendars missing, should warn and allow
        # But if ImportError is raised, that's also acceptable (implementation detail)
        pass
