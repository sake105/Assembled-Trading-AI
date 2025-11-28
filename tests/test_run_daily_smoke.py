# tests/test_run_daily_smoke.py
"""Smoke tests for run_daily EOD-MVP runner."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_daily import run_daily_eod


def create_sample_price_data(tmp_path: Path, symbols: list[str] = None) -> Path:
    """Create sample price data for testing.
    
    Args:
        tmp_path: Temporary directory path
        symbols: List of symbols to include (default: ["AAPL", "MSFT", "GOOGL"])
    
    Returns:
        Path to created parquet file
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL"]
    
    base = datetime(2025, 1, 1, 0, 0, 0)
    data = []
    
    for sym in symbols:
        price_base = 100.0 if sym == "AAPL" else (200.0 if sym == "MSFT" else 150.0)
        for i in range(50):  # 50 days of data
            ts = base + timedelta(days=i)
            open_p = price_base + i * 0.1
            high_p = open_p + 0.5
            low_p = open_p - 0.3
            close_p = open_p + 0.2
            volume = 1000000.0 + i * 10000.0
            
            data.append({
                "timestamp": ts,
                "symbol": sym,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume
            })
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Save to parquet
    price_file = tmp_path / "sample_prices.parquet"
    price_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(price_file, index=False)
    
    return price_file


def test_run_daily_eod_smoke(tmp_path: Path, monkeypatch):
    """Test run_daily_eod with sample data."""
    from src.assembled_core.config import OUTPUT_DIR
    
    # Monkeypatch OUTPUT_DIR to tmp_path
    monkeypatch.setattr("src.assembled_core.execution.safe_bridge.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("src.assembled_core.config.OUTPUT_DIR", tmp_path)
    
    # Create sample price data
    price_file = create_sample_price_data(tmp_path)
    
    # Run daily EOD
    test_date = datetime(2025, 1, 15)
    safe_path = run_daily_eod(
        date_str=test_date.strftime("%Y-%m-%d"),
        price_file=price_file,
        output_dir=tmp_path,
        total_capital=1.0,
        top_n=2,
        ma_fast=20,
        ma_slow=50
    )
    
    # Check that SAFE orders file was created
    assert safe_path.exists(), f"SAFE orders file should exist: {safe_path}"
    
    # Check filename format: orders_YYYYMMDD.csv
    expected_filename = f"orders_{test_date.strftime('%Y%m%d')}.csv"
    assert safe_path.name == expected_filename, \
        f"Filename should be {expected_filename}, got {safe_path.name}"
    
    # Read and verify CSV
    df = pd.read_csv(safe_path)
    
    # Check columns
    expected_cols = ["Ticker", "Side", "Quantity", "PriceType", "Comment"]
    assert list(df.columns) == expected_cols, \
        f"Columns should be {expected_cols}, got {list(df.columns)}"
    
    # Check that we have some orders (may be empty if no signals, but structure should be correct)
    # At minimum, the file should exist with correct schema
    assert len(df) >= 0, "Should have non-negative number of orders"
    
    # If we have orders, check they're valid
    if len(df) > 0:
        assert df["Side"].isin(["BUY", "SELL"]).all(), "Sides should be BUY or SELL"
        assert (df["Quantity"] > 0).all(), "Quantities should be positive"
        assert df["PriceType"].iloc[0] == "MARKET", "PriceType should be MARKET"
        assert df["Comment"].iloc[0] == "EOD Strategy - Daily MVP", "Comment should match"


def test_run_daily_eod_with_universe(tmp_path: Path, monkeypatch):
    """Test run_daily_eod with universe file."""
    from src.assembled_core.config import OUTPUT_DIR, get_base_dir
    
    # Monkeypatch OUTPUT_DIR
    monkeypatch.setattr("src.assembled_core.execution.safe_bridge.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("src.assembled_core.config.OUTPUT_DIR", tmp_path)
    
    # Create sample price data
    price_file = create_sample_price_data(tmp_path)
    
    # Create universe file
    universe_file = tmp_path / "test_universe.txt"
    universe_file.write_text("AAPL\nMSFT\nGOOGL\n", encoding="utf-8")
    
    # Run daily EOD
    test_date = datetime(2025, 1, 15)
    safe_path = run_daily_eod(
        date_str=test_date.strftime("%Y-%m-%d"),
        universe_file=universe_file,
        price_file=price_file,
        output_dir=tmp_path,
        total_capital=1.0,
        top_n=None  # All signals
    )
    
    # Check that file was created
    assert safe_path.exists()


def test_run_daily_eod_universe_missing_symbols(tmp_path: Path, monkeypatch, capsys):
    """Test that symbols in universe without data are warned and dropped."""
    from src.assembled_core.config import OUTPUT_DIR
    
    # Monkeypatch OUTPUT_DIR
    monkeypatch.setattr("src.assembled_core.execution.safe_bridge.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("src.assembled_core.config.OUTPUT_DIR", tmp_path)
    
    # Create price data for only 1 symbol (AAPL)
    price_file = create_sample_price_data(tmp_path, symbols=["AAPL"])
    
    # Create universe file with 3 symbols (but only AAPL has data)
    universe_file = tmp_path / "test_universe.txt"
    universe_file.write_text("AAPL\nMSFT\nGOOGL\n", encoding="utf-8")
    
    # Run daily EOD
    test_date = datetime(2025, 1, 15)
    safe_path = run_daily_eod(
        date_str=test_date.strftime("%Y-%m-%d"),
        universe_file=universe_file,
        price_file=price_file,
        output_dir=tmp_path,
        total_capital=1.0
    )
    
    # Check that WARNING was logged (via logging)
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "WARNING" in output or "warning" in output.lower()
    assert "MSFT" in output or "GOOGL" in output
    
    # Check that SAFE file was created (with orders only for AAPL)
    assert safe_path.exists()
    
    # Read SAFE file
    df = pd.read_csv(safe_path)
    
    # If orders exist, they should only be for AAPL
    if len(df) > 0:
        assert (df["Ticker"] == "AAPL").all(), "Orders should only be for AAPL (only symbol with data)"


def test_run_daily_eod_no_symbols_after_filtering(tmp_path: Path, monkeypatch, capsys):
    """Test that script exits cleanly when no symbols remain after filtering."""
    from src.assembled_core.config import OUTPUT_DIR
    
    # Monkeypatch OUTPUT_DIR
    monkeypatch.setattr("src.assembled_core.execution.safe_bridge.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("src.assembled_core.config.OUTPUT_DIR", tmp_path)
    
    # Create price data for symbols NOT in universe
    price_file = create_sample_price_data(tmp_path, symbols=["TSLA", "NVDA"])
    
    # Create universe file with different symbols
    universe_file = tmp_path / "test_universe.txt"
    universe_file.write_text("AAPL\nMSFT\nGOOGL\n", encoding="utf-8")
    
    # Run daily EOD - should exit with error
    test_date = datetime(2025, 1, 15)
    
    with pytest.raises(SystemExit) as exc_info:
        run_daily_eod(
            date_str=test_date.strftime("%Y-%m-%d"),
            universe_file=universe_file,
            price_file=price_file,
            output_dir=tmp_path,
            total_capital=1.0
        )
    
    # Should exit with code 1 (error)
    assert exc_info.value.code == 1
    
    # Check that error message was logged (via logging)
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "ERROR" in output or "error" in output.lower()
    assert "No valid symbols" in output or "no price data" in output.lower() or "no valid symbols" in output.lower()


def test_run_daily_eod_invalid_date():
    """Test that invalid date raises SystemExit."""
    with pytest.raises(SystemExit) as exc_info:
        run_daily_eod(date_str="invalid-date")
    
    assert exc_info.value.code == 1


def test_run_daily_eod_missing_price_file(tmp_path: Path, monkeypatch):
    """Test that missing price file raises SystemExit."""
    from src.assembled_core.config import OUTPUT_DIR
    
    monkeypatch.setattr("src.assembled_core.config.OUTPUT_DIR", tmp_path)
    
    with pytest.raises(SystemExit) as exc_info:
        run_daily_eod(
            date_str="2025-01-15",
            price_file=tmp_path / "nonexistent.parquet",
            output_dir=tmp_path
        )
    
    assert exc_info.value.code == 1


def test_run_daily_eod_missing_universe_file(tmp_path: Path, monkeypatch):
    """Test that missing universe file raises SystemExit."""
    from src.assembled_core.config import OUTPUT_DIR
    
    monkeypatch.setattr("src.assembled_core.config.OUTPUT_DIR", tmp_path)
    
    # Create price file
    price_file = create_sample_price_data(tmp_path)
    
    # Non-existent universe file
    universe_file = tmp_path / "nonexistent_universe.txt"
    
    with pytest.raises(SystemExit) as exc_info:
        run_daily_eod(
            date_str="2025-01-15",
            universe_file=universe_file,
            price_file=price_file,
            output_dir=tmp_path
        )
    
    assert exc_info.value.code == 1


def test_run_daily_eod_empty_price_data(tmp_path: Path, monkeypatch):
    """Test that empty price data raises SystemExit."""
    from src.assembled_core.config import OUTPUT_DIR
    
    monkeypatch.setattr("src.assembled_core.execution.safe_bridge.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("src.assembled_core.config.OUTPUT_DIR", tmp_path)
    
    # Create empty price file
    price_file = tmp_path / "empty_prices.parquet"
    empty_df = pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    empty_df.to_parquet(price_file, index=False)
    
    with pytest.raises(SystemExit) as exc_info:
        run_daily_eod(
            date_str="2025-01-15",
            price_file=price_file,
            output_dir=tmp_path
        )
    
    assert exc_info.value.code == 1


def test_run_daily_eod_invalid_orders_validation(tmp_path: Path, monkeypatch, capsys):
    """Test that run_daily handles invalid orders (e.g., Quantity=0) correctly."""
    from src.assembled_core.config import OUTPUT_DIR
    from src.assembled_core.execution.order_generation import generate_orders_from_signals
    
    # Monkeypatch OUTPUT_DIR
    monkeypatch.setattr("src.assembled_core.execution.safe_bridge.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("src.assembled_core.config.OUTPUT_DIR", tmp_path)
    
    # Create price data
    price_file = create_sample_price_data(tmp_path, symbols=["AAPL"])
    
    # Mock generate_orders_from_signals to return invalid orders (Quantity=0)
    def mock_generate_orders_invalid(*args, **kwargs):
        return pd.DataFrame([
            {"timestamp": datetime(2025, 1, 15), "symbol": "AAPL", "side": "BUY", "qty": 0.0, "price": 100.0}
        ])
    
    # Patch the function
    import src.assembled_core.execution.order_generation as order_gen
    original_func = order_gen.generate_orders_from_signals
    order_gen.generate_orders_from_signals = mock_generate_orders_invalid
    
    try:
        # Run daily EOD - should fail with validation error
        test_date = datetime(2025, 1, 15)
        with pytest.raises(SystemExit) as exc_info:
            run_daily_eod(
                date_str=test_date.strftime("%Y-%m-%d"),
                price_file=price_file,
                output_dir=tmp_path,
                total_capital=1.0
            )
        
        # Should exit with code 1
        assert exc_info.value.code == 1
        
        # Check that error message was logged
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "validation failed" in output.lower() or "validation" in output.lower()
        
        # File should not be created
        expected_file = tmp_path / f"orders_{test_date.strftime('%Y%m%d')}.csv"
        assert not expected_file.exists()
    finally:
        # Restore original function
        order_gen.generate_orders_from_signals = original_func
