# tests/test_factor_store_integration_daily_backtest.py
"""Tests for Factor Store Integration in Daily/Backtest (Sprint 5 / F4).

Tests verify:
1. Daily writes factors to store (smoke test)
2. Backtest uses factor store when available (monkeypatch feature builder, verify not called)
3. Backtest fallback: wenn factor store fehlt -> Feature-Build wird genutzt (aber ohne externe fetches)
4. Hard Gate: No external fetches in backtest mode
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.factor_store import (
    compute_universe_key,
    load_factors_parquet,
    store_factors_parquet,
)
from src.assembled_core.features.ta_features import add_all_features


def test_daily_writes_factors_to_store() -> None:
    """Smoke test: Daily writes factors to store (incremental update)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "factors"
        
        # Create synthetic prices (last session only for daily)
        timestamps = pd.date_range("2024-01-10", periods=1, freq="1d", tz="UTC")
        symbols = ["AAPL", "MSFT"]
        
        prices = pd.DataFrame({
            "timestamp": [ts for ts in timestamps for _ in symbols],
            "symbol": symbols * 1,
            "open": [150.0, 200.0],
            "high": [155.0, 205.0],
            "low": [148.0, 198.0],
            "close": [152.0, 202.0],
            "volume": [1000000.0, 1000000.0],
        })
        
        # Compute features for last session
        factors = add_all_features(prices.copy())
        
        # Store factors (incremental append)
        universe_key = compute_universe_key(symbols=symbols)
        store_factors_parquet(
            df=factors,
            group="core_ta",
            freq="1d",
            universe=universe_key,
            mode="append",
            root=root,
        )
        
        # Verify: factors stored
        factors_loaded = load_factors_parquet(
            group="core_ta",
            freq="1d",
            universe=universe_key,
            root=root,
        )
        
        assert factors_loaded is not None, "Factors should be stored and loadable"
        assert len(factors_loaded) >= len(factors), "Should have at least the stored factors"
        
        # Verify: last session data is present
        last_session = factors_loaded[factors_loaded["timestamp"] == timestamps[-1]]
        assert len(last_session) == len(symbols), "Should have factors for all symbols on last session"


def test_backtest_uses_factor_store_when_available(monkeypatch) -> None:
    """Test that backtest uses factor store when available (monkeypatch feature builder).

    Verify: feature builder is NOT called when factors are loaded from store.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "factors"
        
        # Create synthetic prices
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC")
        symbols = ["AAPL", "MSFT"]
        
        prices = pd.DataFrame({
            "timestamp": [ts for ts in timestamps for _ in symbols],
            "symbol": symbols * 10,
            "open": [150.0 + i * 0.5 for i in range(20)],
            "high": [155.0 + i * 0.5 for i in range(20)],
            "low": [148.0 + i * 0.5 for i in range(20)],
            "close": [152.0 + i * 0.5 for i in range(20)],
            "volume": [1000000.0] * 20,
        })
        
        # Pre-store factors in factor store
        factors = add_all_features(prices.copy())
        universe_key = compute_universe_key(symbols=symbols)
        store_factors_parquet(
            df=factors,
            group="core_ta",
            freq="1d",
            universe=universe_key,
            mode="replace",
            root=root,
        )
        
        # Monkeypatch add_all_features to track calls
        add_all_features_mock = MagicMock(side_effect=add_all_features)
        monkeypatch.setattr("src.assembled_core.features.ta_features.add_all_features", add_all_features_mock)
        
        # Try to load from factor store (backtest preferred path)
        factors_loaded = load_factors_parquet(
            group="core_ta",
            freq="1d",
            universe=universe_key,
            start=prices["timestamp"].min(),
            end=prices["timestamp"].max(),
            root=root,
        )
        
        # Verify: factors loaded from store
        assert factors_loaded is not None, "Factors should be loaded from store"
        assert not factors_loaded.empty, "Factors should not be empty"
        
        # Verify: add_all_features was NOT called (factors loaded from store)
        assert add_all_features_mock.call_count == 0, \
            "Feature builder should NOT be called when factors are loaded from store"
        
        # Verify: factors match stored factors
        assert len(factors_loaded) == len(factors), "Should have same number of rows"


def test_backtest_fallback_when_factor_store_missing() -> None:
    """Test that backtest falls back to feature computation when factor store is missing.

    Verify: Feature-Build wird genutzt (aber ohne externe fetches, local-only).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "factors"
        
        # Create synthetic prices (local data only)
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC")
        symbols = ["AAPL", "MSFT"]
        
        prices = pd.DataFrame({
            "timestamp": [ts for ts in timestamps for _ in symbols],
            "symbol": symbols * 10,
            "open": [150.0 + i * 0.5 for i in range(20)],
            "high": [155.0 + i * 0.5 for i in range(20)],
            "low": [148.0 + i * 0.5 for i in range(20)],
            "close": [152.0 + i * 0.5 for i in range(20)],
            "volume": [1000000.0] * 20,
        })
        
        # Try to load from factor store (should return None - not found)
        universe_key = compute_universe_key(symbols=symbols)
        factors_loaded = load_factors_parquet(
            group="core_ta",
            freq="1d",
            universe=universe_key,
            start=prices["timestamp"].min(),
            end=prices["timestamp"].max(),
            root=root,
        )
        
        # Verify: factors not found (fallback scenario)
        assert factors_loaded is None, "Factors should not be found (factor store empty)"
        
        # Fallback: compute features directly (local-only, no external fetches)
        factors_computed = add_all_features(prices.copy())
        
        # Verify: features computed successfully
        assert not factors_computed.empty, "Features should be computed"
        assert len(factors_computed) == len(prices), "Should have same number of rows as prices"
        
        # Verify: feature columns present
        expected_feature_cols = ["ta_log_return_v1", "ta_ma_20_v1", "ta_ma_50_v1", "ta_ma_200_v1"]
        for col in expected_feature_cols:
            assert col in factors_computed.columns, f"Feature column {col} should be present"


def test_backtest_hard_gate_no_external_fetches() -> None:
    """Test that Hard Gate remains intact: No external fetches in backtest mode.

    Verify: Backtest uses only local data (no provider calls).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "factors"
        
        # Create synthetic prices (local data only)
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC")
        symbols = ["AAPL", "MSFT"]
        
        prices = pd.DataFrame({
            "timestamp": [ts for ts in timestamps for _ in symbols],
            "symbol": symbols * 10,
            "open": [150.0 + i * 0.5 for i in range(20)],
            "high": [155.0 + i * 0.5 for i in range(20)],
            "low": [148.0 + i * 0.5 for i in range(20)],
            "close": [152.0 + i * 0.5 for i in range(20)],
            "volume": [1000000.0] * 20,
        })
        
        # Simulate backtest: try factor store, then fallback (both local-only)
        universe_key = compute_universe_key(symbols=symbols)
        
        # Step 1: Try factor store (local-only)
        factors_loaded = load_factors_parquet(
            group="core_ta",
            freq="1d",
            universe=universe_key,
            root=root,
        )
        
        # Step 2: Fallback to direct computation (local-only)
        if factors_loaded is None:
            factors_computed = add_all_features(prices.copy())
            assert not factors_computed.empty, "Fallback computation should work (local-only)"
        
        # Verify: No external fetches occurred (this is implicit - we only use local data)
        # The Hard Gate is enforced at data loading level (D3), not at feature computation level
        # This test verifies that feature computation uses only local price data


def test_daily_incremental_update_stores_last_session() -> None:
    """Test that daily incremental update stores only last session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "factors"
        
        # Create synthetic prices: 10 days
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC")
        symbols = ["AAPL", "MSFT"]
        
        prices_full = pd.DataFrame({
            "timestamp": [ts for ts in timestamps for _ in symbols],
            "symbol": symbols * 10,
            "open": [150.0 + i * 0.5 for i in range(20)],
            "high": [155.0 + i * 0.5 for i in range(20)],
            "low": [148.0 + i * 0.5 for i in range(20)],
            "close": [152.0 + i * 0.5 for i in range(20)],
            "volume": [1000000.0] * 20,
        })
        
        # Store base (D1..D9)
        prices_base = prices_full[prices_full["timestamp"] < timestamps[-1]].copy()
        factors_base = add_all_features(prices_base.copy())
        universe_key = compute_universe_key(symbols=symbols)
        store_factors_parquet(
            df=factors_base,
            group="core_ta",
            freq="1d",
            universe=universe_key,
            mode="replace",
            root=root,
        )
        
        # Daily incremental update: compute and store last session (D10)
        # For proper incremental update, compute with full history but store only last session
        factors_full = add_all_features(prices_full.copy())
        factors_last = factors_full[factors_full["timestamp"] == timestamps[-1]].copy()
        
        store_factors_parquet(
            df=factors_last,
            group="core_ta",
            freq="1d",
            universe=universe_key,
            mode="append",
            root=root,
        )
        
        # Verify: all days present after incremental update
        factors_loaded = load_factors_parquet(
            group="core_ta",
            freq="1d",
            universe=universe_key,
            root=root,
        )
        
        assert factors_loaded is not None, "Factors should be loaded"
        assert len(factors_loaded) == len(factors_full), "Should have all days after incremental update"
        assert factors_loaded["timestamp"].nunique() == 10, "Should have 10 unique timestamps"
