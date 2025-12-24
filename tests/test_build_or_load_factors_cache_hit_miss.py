"""Tests for build_or_load_factors cache hit/miss behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.assembled_core.features.factor_store_integration import build_or_load_factors

pytestmark = pytest.mark.advanced


def test_build_or_load_factors_cache_hit(tmp_path: Path) -> None:
    """Test cache hit: factors are loaded from store."""
    # Create mock prices
    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "open": [100.0] * len(dates),
        "high": [105.0] * len(dates),
        "low": [95.0] * len(dates),
        "close": [102.0] * len(dates),
        "volume": [1000000] * len(dates),
    })

    # Create cached factors DataFrame (simulating store)
    cached_factors = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "date": [d.date().isoformat() for d in dates],
        "ta_ma_20": [100.0] * len(dates),
        "ta_rsi_14": [50.0] * len(dates),
    })

    # Mock load_factors to return cached factors
    with patch("src.assembled_core.features.factor_store_integration.load_factors") as mock_load:
        mock_load.return_value = cached_factors

        with patch("src.assembled_core.features.factor_store_integration.store_factors") as mock_store:
            # Call build_or_load_factors
            result = build_or_load_factors(
                prices=prices,
                factor_group="core_ta",
                freq="1d",
                factors_root=tmp_path / "factors",
            )

            # Verify cache hit: load_factors was called
            mock_load.assert_called_once()
            
            # Verify cache hit: store_factors was NOT called (no computation needed)
            mock_store.assert_not_called()

            # Verify result matches cached factors
            assert result is not None
            assert len(result) == len(cached_factors)
            assert "ta_ma_20" in result.columns
            assert "ta_rsi_14" in result.columns


def test_build_or_load_factors_cache_miss(tmp_path: Path) -> None:
    """Test cache miss: factors are computed and stored."""
    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "open": [100.0] * len(dates),
        "high": [105.0] * len(dates),
        "low": [95.0] * len(dates),
        "close": [102.0] * len(dates),
        "volume": [1000000] * len(dates),
    })

    # Mock load_factors to return None (cache miss)
    with patch("src.assembled_core.features.factor_store_integration.load_factors") as mock_load:
        mock_load.return_value = None

        with patch("src.assembled_core.features.factor_store_integration.store_factors") as mock_store:
            with patch("src.assembled_core.features.factor_store_integration.add_all_features") as mock_builder:
                # Mock builder to return factors
                computed_factors = pd.DataFrame({
                    "timestamp": dates,
                    "symbol": ["AAPL"] * len(dates),
                    "ta_ma_20": [100.0] * len(dates),
                })
                mock_builder.return_value = computed_factors

                # Call build_or_load_factors
                result = build_or_load_factors(
                    prices=prices,
                    factor_group="core_ta",
                    freq="1d",
                    factors_root=tmp_path / "factors",
                )

                # Verify cache miss: load_factors was called
                mock_load.assert_called_once()
                
                # Verify cache miss: builder was called
                mock_builder.assert_called_once()
                
                # Verify cache miss: store_factors was called
                mock_store.assert_called_once()

                # Verify result matches computed factors
                assert result is not None
                assert len(result) == len(computed_factors)


def test_build_or_load_factors_force_rebuild(tmp_path: Path) -> None:
    """Test force_rebuild: even if cache exists, recompute."""
    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "open": [100.0] * len(dates),
        "high": [105.0] * len(dates),
        "low": [95.0] * len(dates),
        "close": [102.0] * len(dates),
        "volume": [1000000] * len(dates),
    })

    cached_factors = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "date": [d.date().isoformat() for d in dates],
        "ta_ma_20": [100.0] * len(dates),
    })

    # Mock load_factors to return cached factors (but force_rebuild=True)
    with patch("src.assembled_core.features.factor_store_integration.load_factors") as mock_load:
        mock_load.return_value = cached_factors

        with patch("src.assembled_core.features.factor_store_integration.store_factors") as mock_store:
            with patch("src.assembled_core.features.factor_store_integration.add_all_features") as mock_builder:
                computed_factors = pd.DataFrame({
                    "timestamp": dates,
                    "symbol": ["AAPL"] * len(dates),
                    "ta_ma_20": [101.0] * len(dates),  # Different value
                })
                mock_builder.return_value = computed_factors

                # Call with force_rebuild=True
                result = build_or_load_factors(
                    prices=prices,
                    factor_group="core_ta",
                    freq="1d",
                    force_rebuild=True,
                    factors_root=tmp_path / "factors",
                )

                # Verify force_rebuild: load_factors was NOT called (skip cache check)
                mock_load.assert_not_called()
                
                # Verify force_rebuild: builder was called
                mock_builder.assert_called_once()
                
                # Verify force_rebuild: store_factors was called with mode="overwrite"
                mock_store.assert_called_once()
                call_kwargs = mock_store.call_args[1]
                assert call_kwargs.get("mode") == "overwrite"

                # Verify result matches computed factors (not cached)
                assert result is not None
                assert result["ta_ma_20"].iloc[0] == 101.0  # Computed value, not cached


def test_build_or_load_factors_universe_key_computation(tmp_path: Path) -> None:
    """Test that universe_key is computed from prices if not provided."""
    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    # Create prices with 2 symbols × 5 days = 10 rows
    rows = []
    for date in dates:
        for symbol in ["AAPL", "MSFT"]:
            rows.append({
                "timestamp": date,
                "symbol": symbol,
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000000,
            })
    prices = pd.DataFrame(rows)

    with patch("src.assembled_core.features.factor_store_integration.load_factors") as mock_load:
        mock_load.return_value = None

        with patch("src.assembled_core.features.factor_store_integration.store_factors") as mock_store:
            with patch("src.assembled_core.features.factor_store_integration.add_all_features") as mock_builder:
                mock_builder.return_value = prices.copy()

                build_or_load_factors(
                    prices=prices,
                    factor_group="core_ta",
                    freq="1d",
                    factors_root=tmp_path / "factors",
                )

                # Verify universe_key was computed and passed to load_factors
                load_call_kwargs = mock_load.call_args[1]
                assert "universe_key" in load_call_kwargs
                assert load_call_kwargs["universe_key"] is not None
                assert "AAPL" in load_call_kwargs["universe_key"] or "MSFT" in load_call_kwargs["universe_key"]

                # Verify universe_key was passed to store_factors
                store_call_kwargs = mock_store.call_args[1]
                assert "universe_key" in store_call_kwargs
                assert store_call_kwargs["universe_key"] == load_call_kwargs["universe_key"]


def test_build_or_load_factors_pit_filtering(tmp_path: Path) -> None:
    """Test that as_of parameter is applied for PIT filtering."""
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "open": [100.0] * len(dates),
        "high": [105.0] * len(dates),
        "low": [95.0] * len(dates),
        "close": [102.0] * len(dates),
        "volume": [1000000] * len(dates),
    })

    # Create cached factors with all dates
    cached_factors = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "date": [d.date().isoformat() for d in dates],
        "ta_ma_20": range(len(dates)),
    })

    with patch("src.assembled_core.features.factor_store_integration.load_factors") as mock_load:
        mock_load.return_value = cached_factors

        as_of_date = pd.Timestamp("2024-01-05", tz="UTC")

        result = build_or_load_factors(
            prices=prices,
            factor_group="core_ta",
            freq="1d",
            as_of=as_of_date,
            factors_root=tmp_path / "factors",
        )

        # Verify as_of was passed to load_factors
        load_call_kwargs = mock_load.call_args[1]
        assert load_call_kwargs["as_of"] == as_of_date

        # Verify result is filtered (even if cached has all dates, as_of filter applies)
        assert result is not None
        assert result["timestamp"].max() <= as_of_date

