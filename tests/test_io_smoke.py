# tests/test_io_smoke.py
"""Smoke tests for I/O functions using existing output files."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.pipeline.io import (
    load_orders,
    load_prices,
    load_prices_with_fallback,
)


def test_load_prices_1d_if_exists():
    """Test loading daily prices if file exists."""
    try:
        df = load_prices("1d")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "Price DataFrame should not be empty"
        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "close" in df.columns
        assert df["timestamp"].dtype.name.startswith("datetime")
        assert df["close"].dtype == "float64"
    except FileNotFoundError:
        pytest.skip("output/aggregates/daily.parquet not found (skip if no data)")


def test_load_prices_5min_if_exists():
    """Test loading 5min prices if file exists."""
    try:
        df = load_prices("5min")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "Price DataFrame should not be empty"
        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "close" in df.columns
    except FileNotFoundError:
        pytest.skip("output/aggregates/5min.parquet not found (skip if no data)")


def test_load_prices_with_fallback_5min():
    """Test loading 5min prices with fallback paths."""
    try:
        df = load_prices_with_fallback("5min")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "Price DataFrame should not be empty"
        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "close" in df.columns
    except FileNotFoundError:
        pytest.skip("No 5min price file found in any fallback location")


def test_load_orders_1d_if_exists():
    """Test loading daily orders if file exists."""
    try:
        df = load_orders("1d", strict=False)
        assert isinstance(df, pd.DataFrame)
        # May be empty if no orders generated
        if not df.empty:
            assert "timestamp" in df.columns
            assert "symbol" in df.columns
            assert "side" in df.columns
            assert "qty" in df.columns
            assert "price" in df.columns
    except FileNotFoundError:
        pytest.skip("output/orders_1d.csv not found (skip if no orders)")


def test_load_orders_5min_if_exists():
    """Test loading 5min orders if file exists."""
    try:
        df = load_orders("5min", strict=False)
        assert isinstance(df, pd.DataFrame)
        # May be empty if no orders generated
        if not df.empty:
            assert "timestamp" in df.columns
            assert "symbol" in df.columns
            assert "side" in df.columns
            assert "qty" in df.columns
            assert "price" in df.columns
    except FileNotFoundError:
        pytest.skip("output/orders_5min.csv not found (skip if no orders)")
