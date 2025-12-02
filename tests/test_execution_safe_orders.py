# tests/test_execution_safe_orders.py
"""Tests for SAFE-Bridge order generation."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.order_generation import (
    generate_orders_from_signals,
    generate_orders_from_targets
)
from src.assembled_core.execution.safe_bridge import (
    write_safe_orders_csv,
    write_safe_orders_csv_from_targets
)


def create_sample_target_positions() -> pd.DataFrame:
    """Create sample target positions."""
    data = [
        {"symbol": "AAPL", "target_weight": 0.5, "target_qty": 0.5},
        {"symbol": "MSFT", "target_weight": 0.3, "target_qty": 0.3},
        {"symbol": "GOOGL", "target_weight": 0.2, "target_qty": 0.2},
    ]
    return pd.DataFrame(data)


def create_sample_prices() -> pd.DataFrame:
    """Create sample prices."""
    from datetime import datetime
    
    base = datetime(2025, 1, 1, 0, 0, 0)
    data = []
    for sym in ["AAPL", "MSFT", "GOOGL"]:
        price = 100.0 if sym == "AAPL" else (200.0 if sym == "MSFT" else 150.0)
        data.append({
            "timestamp": base,
            "symbol": sym,
            "close": price
        })
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_generate_orders_from_targets_no_current():
    """Test order generation from targets with no current positions."""
    targets = create_sample_target_positions()
    orders = generate_orders_from_targets(targets)
    
    # Should generate BUY orders for all targets
    assert len(orders) == 3
    assert (orders["side"] == "BUY").all()
    assert set(orders["symbol"].values) == {"AAPL", "MSFT", "GOOGL"}
    
    # Quantities should match target_qty
    for _, row in orders.iterrows():
        target_qty = targets[targets["symbol"] == row["symbol"]]["target_qty"].iloc[0]
        assert row["qty"] == pytest.approx(target_qty)


def test_generate_orders_from_targets_with_current():
    """Test order generation with existing positions."""
    targets = create_sample_target_positions()
    current = pd.DataFrame([
        {"symbol": "AAPL", "qty": 0.3},  # Need to buy 0.2 more
        {"symbol": "MSFT", "qty": 0.5},  # Need to sell 0.2
        {"symbol": "TSLA", "qty": 0.1},  # Need to sell all (not in targets)
    ])
    
    orders = generate_orders_from_targets(targets, current_positions=current)
    
    # Should have orders for:
    # - AAPL: BUY 0.2
    # - MSFT: SELL 0.2
    # - GOOGL: BUY 0.2 (new position)
    # - TSLA: SELL 0.1 (close position)
    assert len(orders) == 4
    
    aapl_order = orders[orders["symbol"] == "AAPL"].iloc[0]
    assert aapl_order["side"] == "BUY"
    assert aapl_order["qty"] == pytest.approx(0.2)
    
    msft_order = orders[orders["symbol"] == "MSFT"].iloc[0]
    assert msft_order["side"] == "SELL"
    assert msft_order["qty"] == pytest.approx(0.2)
    
    googl_order = orders[orders["symbol"] == "GOOGL"].iloc[0]
    assert googl_order["side"] == "BUY"
    assert googl_order["qty"] == pytest.approx(0.2)
    
    tsla_order = orders[orders["symbol"] == "TSLA"].iloc[0]
    assert tsla_order["side"] == "SELL"
    assert tsla_order["qty"] == pytest.approx(0.1)


def test_generate_orders_from_targets_with_prices():
    """Test order generation with price lookup."""
    targets = create_sample_target_positions()
    prices = create_sample_prices()
    
    orders = generate_orders_from_targets(targets, prices=prices)
    
    # Should have prices from prices DataFrame
    assert (orders["price"] > 0).all()
    
    aapl_price = orders[orders["symbol"] == "AAPL"]["price"].iloc[0]
    assert aapl_price == pytest.approx(100.0)


def test_write_safe_orders_csv(tmp_path: Path):
    """Test writing SAFE-Bridge CSV file."""
    orders = pd.DataFrame([
        {"timestamp": datetime(2025, 1, 1), "symbol": "AAPL", "side": "BUY", "qty": 10.0, "price": 100.0},
        {"timestamp": datetime(2025, 1, 1), "symbol": "MSFT", "side": "SELL", "qty": 5.0, "price": 200.0},
    ])
    
    output_path = tmp_path / "orders_20250101.csv"
    result_path = write_safe_orders_csv(orders, output_path=output_path)
    
    assert result_path == output_path
    assert output_path.exists()
    
    # Read and verify CSV
    df = pd.read_csv(output_path)
    
    # Check columns
    assert list(df.columns) == ["Ticker", "Side", "Quantity", "PriceType", "Comment"]
    
    # Check data
    assert len(df) == 2
    assert set(df["Ticker"].values) == {"AAPL", "MSFT"}
    assert set(df["Side"].values) == {"BUY", "SELL"}
    assert df["PriceType"].iloc[0] == "MARKET"
    assert df["Comment"].iloc[0] == "EOD Strategy"


def test_write_safe_orders_csv_date_filename(tmp_path: Path, monkeypatch):
    """Test SAFE-Bridge CSV with date-based filename."""
    
    # Monkeypatch OUTPUT_DIR to tmp_path
    monkeypatch.setattr("src.assembled_core.execution.safe_bridge.OUTPUT_DIR", tmp_path)
    
    orders = pd.DataFrame([
        {"timestamp": datetime(2025, 11, 28), "symbol": "AAPL", "side": "BUY", "qty": 10.0, "price": 100.0},
    ])
    
    test_date = datetime(2025, 11, 28)
    result_path = write_safe_orders_csv(orders, date=test_date)
    
    # Should create orders_20251128.csv
    expected_path = tmp_path / "orders_20251128.csv"
    assert result_path == expected_path
    assert expected_path.exists()


def test_write_safe_orders_csv_empty(tmp_path: Path):
    """Test writing empty SAFE-Bridge CSV."""
    orders = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    
    output_path = tmp_path / "orders_empty.csv"
    result_path = write_safe_orders_csv(orders, output_path=output_path)
    
    assert result_path.exists()
    
    # Read and verify empty CSV has correct schema
    df = pd.read_csv(output_path)
    assert list(df.columns) == ["Ticker", "Side", "Quantity", "PriceType", "Comment"]
    assert len(df) == 0


def test_write_safe_orders_csv_from_targets(tmp_path: Path, monkeypatch):
    """Test writing SAFE-Bridge CSV from target positions."""
    
    # Monkeypatch OUTPUT_DIR to tmp_path
    monkeypatch.setattr("src.assembled_core.execution.safe_bridge.OUTPUT_DIR", tmp_path)
    
    targets = create_sample_target_positions()
    prices = create_sample_prices()
    
    test_date = datetime(2025, 11, 28)
    result_path = write_safe_orders_csv_from_targets(
        targets,
        prices=prices,
        date=test_date
    )
    
    # Should create orders_20251128.csv
    expected_path = tmp_path / "orders_20251128.csv"
    assert result_path == expected_path
    assert expected_path.exists()
    
    # Read and verify
    df = pd.read_csv(expected_path)
    assert len(df) == 3
    assert set(df["Ticker"].values) == {"AAPL", "MSFT", "GOOGL"}
    assert (df["Side"] == "BUY").all()  # All new positions


def test_generate_orders_from_signals():
    """Test convenience function for generating orders from signals."""
    signals = pd.DataFrame([
        {"symbol": "AAPL", "direction": "LONG", "score": 0.8},
        {"symbol": "MSFT", "direction": "LONG", "score": 0.6},
        {"symbol": "GOOGL", "direction": "FLAT", "score": 0.4},
    ])
    
    orders = generate_orders_from_signals(signals, total_capital=1.0, top_n=2)
    
    # Should generate orders for top 2 LONG signals (AAPL, MSFT)
    assert len(orders) == 2
    assert set(orders["symbol"].values) == {"AAPL", "MSFT"}
    assert (orders["side"] == "BUY").all()


def test_validate_safe_orders_df_quantity_zero(tmp_path: Path):
    """Test that orders with Quantity <= 0 fail validation."""
    from src.assembled_core.execution.safe_bridge import validate_safe_orders_df
    
    # Create orders with zero quantity
    df = pd.DataFrame([
        {"Ticker": "AAPL", "Side": "BUY", "Quantity": 0.0, "PriceType": "MARKET", "Comment": "Test"},
        {"Ticker": "MSFT", "Side": "BUY", "Quantity": -1.0, "PriceType": "MARKET", "Comment": "Test"},
        {"Ticker": "GOOGL", "Side": "BUY", "Quantity": 10.0, "PriceType": "MARKET", "Comment": "Test"},
    ])
    
    result = validate_safe_orders_df(df)
    
    # Should have issues (zero/negative quantities)
    assert len(result["issues"]) > 0
    assert "Quantity <= 0" in " ".join(result["issues"])
    
    # Cleaned DataFrame should only have GOOGL (valid)
    assert len(result["df_cleaned"]) == 1
    assert result["df_cleaned"]["Ticker"].iloc[0] == "GOOGL"
    
    # Should still be valid (has at least one valid row)
    assert result["valid"] is True


def test_validate_safe_orders_df_all_invalid(tmp_path: Path):
    """Test that if all orders are invalid, validation fails."""
    from src.assembled_core.execution.safe_bridge import validate_safe_orders_df
    
    # All orders have zero quantity
    df = pd.DataFrame([
        {"Ticker": "AAPL", "Side": "BUY", "Quantity": 0.0, "PriceType": "MARKET", "Comment": "Test"},
        {"Ticker": "MSFT", "Side": "BUY", "Quantity": -1.0, "PriceType": "MARKET", "Comment": "Test"},
    ])
    
    result = validate_safe_orders_df(df)
    
    # Should be invalid (all rows removed)
    assert result["valid"] is False
    assert len(result["df_cleaned"]) == 0
    assert "no valid orders remain" in " ".join(result["issues"]).lower()


def test_validate_safe_orders_df_invalid_side(tmp_path: Path):
    """Test that orders with invalid Side fail validation."""
    from src.assembled_core.execution.safe_bridge import validate_safe_orders_df
    
    # Create orders with invalid side
    df = pd.DataFrame([
        {"Ticker": "AAPL", "Side": "INVALID", "Quantity": 10.0, "PriceType": "MARKET", "Comment": "Test"},
        {"Ticker": "MSFT", "Side": "BUY", "Quantity": 10.0, "PriceType": "MARKET", "Comment": "Test"},
    ])
    
    result = validate_safe_orders_df(df)
    
    # Should have issues (invalid side)
    assert len(result["issues"]) > 0
    assert "invalid Side" in " ".join(result["issues"])
    
    # Cleaned DataFrame should only have MSFT (valid)
    assert len(result["df_cleaned"]) == 1
    assert result["df_cleaned"]["Ticker"].iloc[0] == "MSFT"


def test_validate_safe_orders_df_duplicates(tmp_path: Path):
    """Test that duplicate orders are aggregated."""
    from src.assembled_core.execution.safe_bridge import validate_safe_orders_df
    
    # Create duplicate orders (same Ticker, Side, Comment)
    df = pd.DataFrame([
        {"Ticker": "AAPL", "Side": "BUY", "Quantity": 5.0, "PriceType": "MARKET", "Comment": "Test"},
        {"Ticker": "AAPL", "Side": "BUY", "Quantity": 3.0, "PriceType": "MARKET", "Comment": "Test"},
        {"Ticker": "MSFT", "Side": "BUY", "Quantity": 10.0, "PriceType": "MARKET", "Comment": "Test"},
    ])
    
    result = validate_safe_orders_df(df)
    
    # Should have warning about duplicates
    assert len(result["issues"]) > 0
    assert "duplicate" in " ".join(result["issues"]).lower() or "aggregated" in " ".join(result["issues"]).lower()
    
    # Should be valid (duplicates aggregated)
    assert result["valid"] is True
    
    # Should have 2 rows (AAPL aggregated, MSFT separate)
    assert len(result["df_cleaned"]) == 2
    
    # AAPL quantity should be summed (5.0 + 3.0 = 8.0)
    aapl_row = result["df_cleaned"][result["df_cleaned"]["Ticker"] == "AAPL"]
    assert len(aapl_row) == 1
    assert aapl_row["Quantity"].iloc[0] == pytest.approx(8.0)


def test_write_safe_orders_csv_validation_fails(tmp_path: Path):
    """Test that write_safe_orders_csv raises ValueError when validation fails."""
    from src.assembled_core.execution.safe_bridge import write_safe_orders_csv
    
    # Create orders with all invalid quantities
    orders = pd.DataFrame([
        {"timestamp": datetime(2025, 1, 1), "symbol": "AAPL", "side": "BUY", "qty": 0.0, "price": 100.0},
        {"timestamp": datetime(2025, 1, 1), "symbol": "MSFT", "side": "BUY", "qty": -1.0, "price": 200.0},
    ])
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="SAFE orders validation failed"):
        write_safe_orders_csv(orders, output_path=tmp_path / "orders_invalid.csv")
    
    # File should not be created
    assert not (tmp_path / "orders_invalid.csv").exists()

