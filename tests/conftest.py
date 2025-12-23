# tests/conftest.py
"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_output_dir(tmp_path: Path):
    """Create a temporary output directory for tests.

    This fixture creates a temporary directory that can be used as OUTPUT_DIR
    in tests, ensuring tests don't interfere with each other or production data.

    Usage:
        def test_something(tmp_output_dir, monkeypatch):
            monkeypatch.setattr("src.assembled_core.config.OUTPUT_DIR", tmp_output_dir)
            # ... test code ...
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_universe(tmp_path: Path) -> Path:
    """Create a sample universe file for testing.

    Returns:
        Path to a temporary universe file with sample symbols
    """
    universe_file = tmp_path / "test_universe.txt"
    universe_file.write_text("AAPL\nMSFT\nGOOGL\n", encoding="utf-8")
    return universe_file


@pytest.fixture
def sample_price_data(tmp_path: Path):
    """Create sample price data for testing.

    Creates a minimal parquet file with price data for testing.

    Returns:
        Path to created parquet file
    """
    from datetime import datetime, timedelta
    import pandas as pd

    base = datetime(2025, 1, 1, 0, 0, 0)
    symbols = ["AAPL", "MSFT", "GOOGL"]
    data = []

    for sym in symbols:
        price_base = 100.0 if sym == "AAPL" else (200.0 if sym == "MSFT" else 150.0)
        for i in range(50):  # 50 days of data
            ts = base + timedelta(days=i)
            data.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "open": price_base + i * 0.1,
                    "high": price_base + i * 0.1 + 0.5,
                    "low": price_base + i * 0.1 - 0.3,
                    "close": price_base + i * 0.1 + 0.2,
                    "volume": 1000000.0 + i * 10000.0,
                }
            )

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Save to parquet
    price_file = tmp_path / "sample_prices.parquet"
    price_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(price_file, index=False)

    return price_file
