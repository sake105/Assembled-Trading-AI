# tests/test_logging_cli.py
"""Tests for CLI logging and error handling."""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.logging_utils import setup_logging, get_logger


def test_setup_logging():
    """Test that setup_logging creates a logger with correct level."""
    logger = setup_logging(level="INFO")
    
    assert logger is not None
    assert logger.name == "assembled_core"
    assert logger.level == 20  # INFO level


def test_logger_levels(capsys):
    """Test that logger outputs messages at different levels."""
    logger = setup_logging(level="INFO")
    
    logger.debug("Debug message (should not appear)")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    captured = capsys.readouterr()
    output = captured.out + captured.err
    
    # Debug should not appear (level is INFO)
    assert "Debug message" not in output
    
    # Other levels should appear
    assert "Info message" in output
    assert "Warning message" in output
    assert "Error message" in output
    
    # Check format: [LEVEL] message
    assert "[INFO]" in output
    assert "[WARNING]" in output
    assert "[ERROR]" in output


def test_run_daily_logging_patterns(tmp_path: Path, monkeypatch, capsys):
    """Test that run_daily logs expected patterns."""
    from src.assembled_core.config import OUTPUT_DIR
    from scripts.run_daily import create_sample_price_data
    
    # Monkeypatch OUTPUT_DIR
    monkeypatch.setattr("src.assembled_core.execution.safe_bridge.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("src.assembled_core.config.OUTPUT_DIR", tmp_path)
    
    # Create sample price data
    from datetime import datetime, timedelta
    import pandas as pd
    
    base = datetime(2025, 1, 1, 0, 0, 0)
    symbols = ["AAPL"]
    data = []
    
    for sym in symbols:
        for i in range(50):
            ts = base + timedelta(days=i)
            data.append({
                "timestamp": ts,
                "symbol": sym,
                "open": 100.0 + i * 0.1,
                "high": 100.5 + i * 0.1,
                "low": 99.7 + i * 0.1,
                "close": 100.2 + i * 0.1,
                "volume": 1000000.0
            })
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    price_file = tmp_path / "sample_prices.parquet"
    df.to_parquet(price_file, index=False)
    
    # Import and run
    from scripts.run_daily import run_daily_eod
    
    test_date = datetime(2025, 1, 15)
    run_daily_eod(
        date_str=test_date.strftime("%Y-%m-%d"),
        price_file=price_file,
        output_dir=tmp_path,
        total_capital=1.0
    )
    
    # Check logging patterns
    captured = capsys.readouterr()
    output = captured.out + captured.err
    
    # Should have "Starting EOD-MVP" message
    assert "Starting EOD-MVP" in output or "Starting" in output
    
    # Should have step messages
    assert "Step" in output or "Loading" in output or "Computing" in output
    
    # Should have success message
    assert "SUCCESS" in output or "completed" in output.lower()


def test_run_eod_pipeline_logging_patterns(tmp_path: Path, monkeypatch, capsys):
    """Test that run_eod_pipeline logs expected patterns."""
    from src.assembled_core.config import OUTPUT_DIR
    from src.assembled_core.pipeline.orchestrator import run_eod_pipeline
    
    # Monkeypatch OUTPUT_DIR
    monkeypatch.setattr("src.assembled_core.config.OUTPUT_DIR", tmp_path)
    
    # Create minimal price data
    from datetime import datetime, timedelta
    import pandas as pd
    
    base = datetime(2025, 1, 1, 0, 0, 0)
    data = []
    for i in range(10):
        ts = base + timedelta(days=i)
        data.append({
            "timestamp": pd.Timestamp(ts, tz="UTC"),
            "symbol": "AAPL",
            "close": 100.0 + i * 0.1
        })
    
    df = pd.DataFrame(data)
    
    # Save to parquet
    price_file = tmp_path / "aggregates" / "5min.parquet"
    price_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(price_file, index=False)
    
    # Setup logging
    from src.assembled_core.logging_utils import setup_logging
    logger = setup_logging(level="INFO")
    
    # Run pipeline (may fail, but should log)
    try:
        run_eod_pipeline(
            freq="5min",
            start_capital=10000.0,
            skip_backtest=True,
            skip_portfolio=True,
            skip_qa=True,
            output_dir=tmp_path,
            price_file=str(price_file)
        )
    except Exception:
        pass  # May fail, but we're checking logging
    
    # Check logging patterns
    captured = capsys.readouterr()
    output = captured.out + captured.err
    
    # Should have step messages
    assert "Step" in output or "Execute" in output or "Pipeline" in output

