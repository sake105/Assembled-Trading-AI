# tests/test_risk_exposure_engine.py
"""Tests for exposure engine (Sprint 8).

Tests verify:
1. Target positions computation (current + orders)
2. Exposure metrics (notional, weight, gross/net)
3. SELL reduces exposure correctly
4. Missing price handling (deterministic)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.risk.exposure_engine import (
    compute_exposures,
    compute_target_positions,
)


def test_compute_target_positions_known_toy_portfolio() -> None:
    """Test target positions computation with known toy portfolio."""
    # Current positions: AAPL=100, MSFT=50
    current = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [100.0, 50.0],
    })

    # Orders: SELL 50 AAPL, BUY 200 MSFT, BUY 100 GOOGL
    orders = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "side": ["SELL", "BUY", "BUY"],
        "qty": [50.0, 200.0, 100.0],
    })

    target = compute_target_positions(current, orders)

    # Expected: AAPL=50 (100-50), MSFT=250 (50+200), GOOGL=100 (0+100)
    expected = pd.DataFrame({
        "symbol": ["AAPL", "GOOGL", "MSFT"],
        "target_qty": [50.0, 100.0, 250.0],
    })

    pd.testing.assert_frame_equal(
        target.sort_values("symbol").reset_index(drop=True),
        expected.sort_values("symbol").reset_index(drop=True),
        check_dtype=False,
    )


def test_compute_target_positions_empty_current() -> None:
    """Test target positions with empty current positions."""
    current = pd.DataFrame(columns=["symbol", "qty"])

    orders = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "BUY"],
        "qty": [100.0, 50.0],
    })

    target = compute_target_positions(current, orders)

    # Expected: AAPL=100, MSFT=50
    expected = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "target_qty": [100.0, 50.0],
    })

    pd.testing.assert_frame_equal(
        target.sort_values("symbol").reset_index(drop=True),
        expected.sort_values("symbol").reset_index(drop=True),
        check_dtype=False,
    )


def test_compute_target_positions_empty_orders() -> None:
    """Test target positions with empty orders."""
    current = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [100.0, 50.0],
    })

    orders = pd.DataFrame(columns=["symbol", "side", "qty"])

    target = compute_target_positions(current, orders)

    # Expected: same as current
    expected = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "target_qty": [100.0, 50.0],
    })

    pd.testing.assert_frame_equal(
        target.sort_values("symbol").reset_index(drop=True),
        expected.sort_values("symbol").reset_index(drop=True),
        check_dtype=False,
    )


def test_compute_target_positions_multiple_orders_same_symbol() -> None:
    """Test target positions with multiple orders for same symbol."""
    current = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [100.0],
    })

    # Multiple orders for AAPL: BUY 50, SELL 30, BUY 20
    orders = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "side": ["BUY", "SELL", "BUY"],
        "qty": [50.0, 30.0, 20.0],
    })

    target = compute_target_positions(current, orders)

    # Expected: 100 + 50 - 30 + 20 = 140
    expected = pd.DataFrame({
        "symbol": ["AAPL"],
        "target_qty": [140.0],
    })

    pd.testing.assert_frame_equal(
        target,
        expected,
        check_dtype=False,
    )


def test_compute_exposures_gross_net_weights_correct() -> None:
    """Test that exposure metrics (gross/net/weights) are computed correctly."""
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "target_qty": [100.0, -50.0, 200.0],
    })

    prices = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "close": [150.0, 200.0, 100.0],
    })

    equity = 10000.0

    exposures, summary = compute_exposures(target, prices, equity)

    # Exposures are sorted by symbol: AAPL, GOOGL, MSFT
    # Expected notional: AAPL=15000, GOOGL=20000, MSFT=-10000
    aapl_row = exposures[exposures["symbol"] == "AAPL"].iloc[0]
    googl_row = exposures[exposures["symbol"] == "GOOGL"].iloc[0]
    msft_row = exposures[exposures["symbol"] == "MSFT"].iloc[0]

    assert abs(aapl_row["notional"] - 15000.0) < 1e-10, "AAPL notional should be 15000"
    assert abs(googl_row["notional"] - 20000.0) < 1e-10, "GOOGL notional should be 20000"
    assert abs(msft_row["notional"] - (-10000.0)) < 1e-10, "MSFT notional should be -10000"

    # Expected weights: AAPL=1.5, GOOGL=2.0, MSFT=-1.0
    assert abs(aapl_row["weight"] - 1.5) < 1e-10, "AAPL weight should be 1.5"
    assert abs(googl_row["weight"] - 2.0) < 1e-10, "GOOGL weight should be 2.0"
    assert abs(msft_row["weight"] - (-1.0)) < 1e-10, "MSFT weight should be -1.0"

    # Expected summary:
    # gross_exposure = 15000 + 10000 + 20000 = 45000
    # net_exposure = 15000 - 10000 + 20000 = 25000
    assert abs(summary.gross_exposure - 45000.0) < 1e-10, "Gross exposure should be 45000"
    assert abs(summary.net_exposure - 25000.0) < 1e-10, "Net exposure should be 25000"
    assert abs(summary.gross_exposure_pct - 450.0) < 1e-10, "Gross exposure % should be 450%"
    assert abs(summary.net_exposure_pct - 250.0) < 1e-10, "Net exposure % should be 250%"
    assert summary.n_positions == 3, "Should have 3 positions"


def test_compute_exposures_sell_reduces_exposure() -> None:
    """Test that SELL orders reduce exposure correctly."""
    # Current: AAPL=100
    current = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [100.0],
    })

    # Order: SELL 50 AAPL
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["SELL"],
        "qty": [50.0],
    })

    target = compute_target_positions(current, orders)

    prices = pd.DataFrame({
        "symbol": ["AAPL"],
        "close": [150.0],
    })

    equity = 10000.0

    exposures, summary = compute_exposures(target, prices, equity)

    # Expected: target_qty=50, notional=7500, weight=0.75
    assert abs(exposures["target_qty"].iloc[0] - 50.0) < 1e-10, "Target qty should be 50"
    assert abs(exposures["notional"].iloc[0] - 7500.0) < 1e-10, "Notional should be 7500"
    assert abs(exposures["weight"].iloc[0] - 0.75) < 1e-10, "Weight should be 0.75"
    assert abs(summary.gross_exposure - 7500.0) < 1e-10, "Gross exposure should be 7500"
    assert abs(summary.net_exposure - 7500.0) < 1e-10, "Net exposure should be 7500"


def test_compute_exposures_missing_price_raise() -> None:
    """Test that missing price raises ValueError when handling='raise'."""
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "target_qty": [100.0, 50.0],
    })

    # Missing price for MSFT
    prices = pd.DataFrame({
        "symbol": ["AAPL"],
        "close": [150.0],
    })

    equity = 10000.0

    with pytest.raises(ValueError, match="Missing price for symbols"):
        compute_exposures(target, prices, equity, missing_price_handling="raise")


def test_compute_exposures_missing_price_zero() -> None:
    """Test that missing price uses 0.0 when handling='zero' (deterministic)."""
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "target_qty": [100.0, 50.0],
    })

    # Missing price for MSFT
    prices = pd.DataFrame({
        "symbol": ["AAPL"],
        "close": [150.0],
    })

    equity = 10000.0

    exposures, summary = compute_exposures(target, prices, equity, missing_price_handling="zero")

    # AAPL should have correct price, MSFT should have price=0.0
    aapl_row = exposures[exposures["symbol"] == "AAPL"].iloc[0]
    msft_row = exposures[exposures["symbol"] == "MSFT"].iloc[0]

    assert abs(aapl_row["price"] - 150.0) < 1e-10, "AAPL price should be 150.0"
    assert abs(msft_row["price"] - 0.0) < 1e-10, "MSFT price should be 0.0 (fallback)"
    assert abs(msft_row["notional"] - 0.0) < 1e-10, "MSFT notional should be 0.0"
    assert abs(msft_row["weight"] - 0.0) < 1e-10, "MSFT weight should be 0.0"


def test_compute_exposures_deterministic_ordering() -> None:
    """Test that exposures are sorted deterministically by symbol."""
    target = pd.DataFrame({
        "symbol": ["MSFT", "AAPL", "GOOGL"],
        "target_qty": [50.0, 100.0, 200.0],
    })

    prices = pd.DataFrame({
        "symbol": ["MSFT", "AAPL", "GOOGL"],
        "close": [200.0, 150.0, 100.0],
    })

    equity = 10000.0

    exposures1, _ = compute_exposures(target, prices, equity)
    exposures2, _ = compute_exposures(target, prices, equity)

    # Should be sorted by symbol (AAPL, GOOGL, MSFT)
    expected_symbols = ["AAPL", "GOOGL", "MSFT"]
    assert exposures1["symbol"].tolist() == expected_symbols, "Should be sorted by symbol"
    assert exposures2["symbol"].tolist() == expected_symbols, "Should be deterministic"

    # Results should be identical
    pd.testing.assert_frame_equal(exposures1, exposures2, "Results should be deterministic")


def test_compute_exposures_empty_target() -> None:
    """Test that empty target positions returns empty exposures and zero summary."""
    target = pd.DataFrame(columns=["symbol", "target_qty"])

    prices = pd.DataFrame({
        "symbol": ["AAPL"],
        "close": [150.0],
    })

    equity = 10000.0

    exposures, summary = compute_exposures(target, prices, equity)

    assert exposures.empty, "Exposures should be empty"
    assert summary.gross_exposure == 0.0, "Gross exposure should be 0"
    assert summary.net_exposure == 0.0, "Net exposure should be 0"
    assert summary.n_positions == 0, "Should have 0 positions"


def test_compute_exposures_equity_zero_raises() -> None:
    """Test that equity <= 0 raises ValueError."""
    target = pd.DataFrame({
        "symbol": ["AAPL"],
        "target_qty": [100.0],
    })

    prices = pd.DataFrame({
        "symbol": ["AAPL"],
        "close": [150.0],
    })

    with pytest.raises(ValueError, match="equity must be > 0"):
        compute_exposures(target, prices, equity=0.0)

    with pytest.raises(ValueError, match="equity must be > 0"):
        compute_exposures(target, prices, equity=-1000.0)


def test_compute_exposures_uses_price_column_if_close_missing() -> None:
    """Test that compute_exposures uses 'price' column if 'close' is missing."""
    target = pd.DataFrame({
        "symbol": ["AAPL"],
        "target_qty": [100.0],
    })

    prices = pd.DataFrame({
        "symbol": ["AAPL"],
        "price": [150.0],  # Use 'price' instead of 'close'
    })

    equity = 10000.0

    exposures, _ = compute_exposures(target, prices, equity)

    assert abs(exposures["price"].iloc[0] - 150.0) < 1e-10, "Should use 'price' column"


def test_compute_exposures_short_positions() -> None:
    """Test that short positions (negative target_qty) are handled correctly."""
    target = pd.DataFrame({
        "symbol": ["AAPL"],
        "target_qty": [-100.0],  # Short position
    })

    prices = pd.DataFrame({
        "symbol": ["AAPL"],
        "close": [150.0],
    })

    equity = 10000.0

    exposures, summary = compute_exposures(target, prices, equity)

    # Short position: notional should be negative
    assert abs(exposures["notional"].iloc[0] - (-15000.0)) < 1e-10, "Notional should be -15000"
    assert abs(exposures["weight"].iloc[0] - (-1.5)) < 1e-10, "Weight should be -1.5"
    assert abs(summary.gross_exposure - 15000.0) < 1e-10, "Gross exposure should be 15000 (abs)"
    assert abs(summary.net_exposure - (-15000.0)) < 1e-10, "Net exposure should be -15000"
