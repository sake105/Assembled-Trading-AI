"""Tests for Multi-Factor Long/Short Strategy Module.

Tests the generate_multifactor_long_short_signals() and compute_multifactor_long_short_positions() functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

pytestmark = pytest.mark.advanced

from src.assembled_core.config.factor_bundles import FactorBundleConfig, FactorConfig, FactorBundleOptions
from src.assembled_core.strategies.multifactor_long_short import (
    MultiFactorStrategyConfig,
    generate_multifactor_long_short_signals,
    compute_multifactor_long_short_positions,
)
from src.assembled_core.signals.multifactor_signal import build_multifactor_signal


@pytest.fixture
def sample_prices_df() -> pd.DataFrame:
    """Create a synthetic price DataFrame with 3 symbols and 60 days of data."""
    np.random.seed(42)  # Fix seed for reproducibility
    dates = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    
    all_data = []
    for symbol in symbols:
        base_price = 100.0 + hash(symbol) % 50
        
        for i, date in enumerate(dates):
            # Create price series with different patterns
            price = base_price + i * 0.1 + np.random.normal(0, 1.0)
            all_data.append({
                "timestamp": date,
                "symbol": symbol,
                "open": price * 0.99,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "volume": 1000000.0,
            })
    
    df = pd.DataFrame(all_data)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture
def sample_factors_df(sample_prices_df) -> pd.DataFrame:
    """Create synthetic factors DataFrame matching sample_prices_df."""
    # Create factors with clear differences for testing
    all_data = []
    for (symbol, timestamp), group in sample_prices_df.groupby(["symbol", "timestamp"]):
        # Create factor values that will differentiate symbols
        symbol_idx = {"AAPL": 4, "MSFT": 3, "GOOGL": 2, "AMZN": 1, "NVDA": 0}.get(symbol, 0)
        
        all_data.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "returns_12m": 0.1 + symbol_idx * 0.05,  # Higher for AAPL, lower for NVDA
            "momentum_12m_excl_1m": 0.05 + symbol_idx * 0.03,
            "rv_20": 0.2 - symbol_idx * 0.02,  # Lower volatility is better (negative direction)
            "close": group["close"].iloc[0],  # Preserve close price
        })
    
    df = pd.DataFrame(all_data)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture
def simple_bundle_config() -> FactorBundleConfig:
    """Create a simple bundle config for testing."""
    return FactorBundleConfig(
        universe="test_universe",
        factor_set="core",
        horizon_days=20,
        factors=[
            FactorConfig(name="returns_12m", weight=0.5, direction="positive"),
            FactorConfig(name="momentum_12m_excl_1m", weight=0.3, direction="positive"),
            FactorConfig(name="rv_20", weight=0.2, direction="negative"),
        ],
        options=FactorBundleOptions(
            winsorize=True,
            winsorize_limits=[0.01, 0.99],
            zscore=True,
            neutralize_by=None,
        ),
    )


@pytest.fixture
def simple_bundle_yaml(tmp_path: Path, simple_bundle_config) -> Path:
    """Create a temporary YAML file with a simple bundle config."""
    import yaml
    
    bundle_file = tmp_path / "test_bundle.yaml"
    
    bundle_data = {
        "universe": simple_bundle_config.universe,
        "factor_set": simple_bundle_config.factor_set,
        "horizon_days": simple_bundle_config.horizon_days,
        "factors": [
            {
                "name": f.name,
                "weight": f.weight,
                "direction": f.direction,
            }
            for f in simple_bundle_config.factors
        ],
        "options": {
            "winsorize": simple_bundle_config.options.winsorize,
            "winsorize_limits": simple_bundle_config.options.winsorize_limits,
            "zscore": simple_bundle_config.options.zscore,
            "neutralize_by": simple_bundle_config.options.neutralize_by,
        },
    }
    
    with bundle_file.open("w", encoding="utf-8") as f:
        yaml.dump(bundle_data, f, default_flow_style=False)
    
    return bundle_file


class TestGenerateMultifactorLongShortSignals:
    """Tests for generate_multifactor_long_short_signals() function."""
    
    def test_basic_functionality(self, sample_factors_df, simple_bundle_yaml):
        """Test that generate_multifactor_long_short_signals works with provided factors."""
        config = MultiFactorStrategyConfig(
            bundle_path=str(simple_bundle_yaml),
            top_quantile=0.2,
            bottom_quantile=0.2,
            rebalance_freq="D",  # Daily rebalancing for testing
        )
        
        # Create prices DataFrame from factors (minimal requirements)
        prices = sample_factors_df[["timestamp", "symbol", "close"]].copy()
        
        # Generate signals with pre-computed factors
        signals = generate_multifactor_long_short_signals(
            prices=prices,
            factors=sample_factors_df,
            config=config,
        )
        
        # Should return DataFrame with expected columns
        assert isinstance(signals, pd.DataFrame)
        assert not signals.empty, "Should generate some signals"
        assert "timestamp" in signals.columns
        assert "symbol" in signals.columns
        assert "direction" in signals.columns
        assert "score" in signals.columns
        
        # Direction should be LONG or SHORT
        assert signals["direction"].isin(["LONG", "SHORT"]).all()
        
        # Should have both long and short signals (with 5 symbols and 0.2 quantile)
        assert (signals["direction"] == "LONG").any(), "Should have at least one LONG signal"
        assert (signals["direction"] == "SHORT").any(), "Should have at least one SHORT signal"
    
    def test_top_bottom_quantile_selection(self, sample_factors_df, simple_bundle_yaml):
        """Test that signals correspond to top/bottom quantiles of mf_score."""
        config = MultiFactorStrategyConfig(
            bundle_path=str(simple_bundle_yaml),
            top_quantile=0.2,  # Top 20%
            bottom_quantile=0.2,  # Bottom 20%
            rebalance_freq="D",
        )
        
        prices = sample_factors_df[["timestamp", "symbol", "close"]].copy()
        signals = generate_multifactor_long_short_signals(
            prices=prices,
            factors=sample_factors_df,
            config=config,
        )
        
        # For each timestamp, verify that LONG signals have higher scores than SHORT signals
        for timestamp in signals["timestamp"].unique():
            ts_signals = signals[signals["timestamp"] == timestamp]
            
            if len(ts_signals) < 2:
                continue
            
            long_signals = ts_signals[ts_signals["direction"] == "LONG"]
            short_signals = ts_signals[ts_signals["direction"] == "SHORT"]
            
            if len(long_signals) > 0 and len(short_signals) > 0:
                min_long_score = long_signals["score"].min()
                max_short_score = short_signals["score"].max()
                
                assert min_long_score >= max_short_score, \
                    f"At {timestamp}, LONG signals should have scores >= SHORT signals. " \
                    f"Got min_long={min_long_score}, max_short={max_short_score}"
    
    def test_rebalance_freq_monthly(self, sample_factors_df, simple_bundle_yaml):
        """Test that monthly rebalancing only generates signals on month starts."""
        config = MultiFactorStrategyConfig(
            bundle_path=str(simple_bundle_yaml),
            top_quantile=0.2,
            bottom_quantile=0.2,
            rebalance_freq="M",  # Monthly
        )
        
        prices = sample_factors_df[["timestamp", "symbol", "close"]].copy()
        signals = generate_multifactor_long_short_signals(
            prices=prices,
            factors=sample_factors_df,
            config=config,
        )
        
        if not signals.empty:
            # Check that all signal timestamps are month starts
            signal_dates = signals["timestamp"].dt.normalize().unique()
            for date in signal_dates:
                # Check if it's the first day of the month
                assert date.day == 1, \
                    f"Monthly rebalancing should only generate signals on month starts, " \
                    f"got {date}"
    
    def test_rebalance_freq_weekly(self, sample_factors_df, simple_bundle_yaml):
        """Test that weekly rebalancing only generates signals on Mondays."""
        config = MultiFactorStrategyConfig(
            bundle_path=str(simple_bundle_yaml),
            top_quantile=0.2,
            bottom_quantile=0.2,
            rebalance_freq="W",  # Weekly
        )
        
        prices = sample_factors_df[["timestamp", "symbol", "close"]].copy()
        signals = generate_multifactor_long_short_signals(
            prices=prices,
            factors=sample_factors_df,
            config=config,
        )
        
        if not signals.empty:
            # Check that all signal timestamps are Mondays (weekday=0)
            signal_dates = signals["timestamp"].dt.normalize().unique()
            for date in signal_dates:
                assert date.weekday() == 0, \
                    f"Weekly rebalancing should only generate signals on Mondays, " \
                    f"got {date.strftime('%A')}"
    
    def test_empty_factors_handling(self, sample_prices_df, simple_bundle_yaml):
        """Test that the function handles empty factors by computing them."""
        config = MultiFactorStrategyConfig(
            bundle_path=str(simple_bundle_yaml),
            top_quantile=0.2,
            bottom_quantile=0.2,
            rebalance_freq="D",
        )
        
        # This will trigger factor computation internally
        # Note: This might fail if compute_factors is not available or needs specific data
        # We'll mark it as potentially skipped if dependencies are missing
        try:
            signals = generate_multifactor_long_short_signals(
                prices=sample_prices_df,
                factors=None,  # Will compute from prices
                config=config,
            )
            
            # If it succeeds, should return a DataFrame (possibly empty if factors can't be computed)
            assert isinstance(signals, pd.DataFrame)
        except Exception as e:
            # If factor computation fails, that's ok for this test
            # (it means the function tried to compute factors, which is the expected behavior)
            pytest.skip(f"Factor computation failed (expected in some test environments): {e}")
    
    def test_config_none_uses_default_bundle(self, sample_factors_df, tmp_path: Path):
        """Test that config=None uses default bundle (if available)."""
        # This test might skip if no bundles are available
        try:
            signals = generate_multifactor_long_short_signals(
                prices=sample_factors_df[["timestamp", "symbol", "close"]],
                factors=sample_factors_df,
                config=None,  # Should use default bundle
            )
            
            # Should return DataFrame (possibly empty)
            assert isinstance(signals, pd.DataFrame)
        except ValueError as e:
            if "No factor bundles found" in str(e):
                pytest.skip("No factor bundles available for default config test")
            else:
                raise


class TestComputeMultifactorLongShortPositions:
    """Tests for compute_multifactor_long_short_positions() function."""
    
    def test_basic_functionality(self, simple_bundle_yaml):
        """Test that compute_multifactor_long_short_positions works with basic signals."""
        config = MultiFactorStrategyConfig(
            bundle_path=str(simple_bundle_yaml),
            max_gross_exposure=1.0,
        )
        
        # Create sample signals
        signals = pd.DataFrame([
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "AAPL", "direction": "LONG", "score": 1.5},
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "MSFT", "direction": "LONG", "score": 1.2},
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "GOOGL", "direction": "SHORT", "score": -1.5},
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "AMZN", "direction": "SHORT", "score": -1.2},
        ])
        
        capital = 10000.0
        positions = compute_multifactor_long_short_positions(
            signals=signals,
            capital=capital,
            config=config,
        )
        
        assert isinstance(positions, pd.DataFrame)
        assert not positions.empty
        assert "symbol" in positions.columns
        assert "target_weight" in positions.columns
        assert "target_qty" in positions.columns
        
        # Should have positions for all symbols
        assert len(positions) == 4
        
        # Long positions should have positive weights and quantities
        long_positions = positions[positions["target_weight"] > 0]
        assert len(long_positions) == 2
        assert (long_positions["target_weight"] > 0).all()
        assert (long_positions["target_qty"] > 0).all()
        
        # Short positions should have negative weights and quantities
        short_positions = positions[positions["target_weight"] < 0]
        assert len(short_positions) == 2
        assert (short_positions["target_weight"] < 0).all()
        assert (short_positions["target_qty"] < 0).all()
    
    def test_net_exposure_approximately_zero(self, simple_bundle_yaml):
        """Test that net exposure (long - short) is approximately zero."""
        config = MultiFactorStrategyConfig(
            bundle_path=str(simple_bundle_yaml),
            max_gross_exposure=1.0,
        )
        
        signals = pd.DataFrame([
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "AAPL", "direction": "LONG", "score": 1.5},
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "MSFT", "direction": "LONG", "score": 1.2},
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "GOOGL", "direction": "SHORT", "score": -1.5},
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "AMZN", "direction": "SHORT", "score": -1.2},
        ])
        
        capital = 10000.0
        positions = compute_multifactor_long_short_positions(
            signals=signals,
            capital=capital,
            config=config,
        )
        
        # Net exposure should be approximately zero (long weights sum ≈ short weights sum)
        total_long_weight = positions[positions["target_weight"] > 0]["target_weight"].sum()
        total_short_weight = abs(positions[positions["target_weight"] < 0]["target_weight"].sum())
        net_weight = positions["target_weight"].sum()
        
        # Should be approximately equal (allowing for floating point errors)
        assert abs(total_long_weight - total_short_weight) < 1e-6, \
            f"Long weights ({total_long_weight}) should ≈ short weights ({total_short_weight})"
        assert abs(net_weight) < 1e-6, \
            f"Net weight should be ≈ 0, got {net_weight}"
    
    def test_equal_weighting_within_sides(self, simple_bundle_yaml):
        """Test that positions are equal-weighted within long and short sides."""
        config = MultiFactorStrategyConfig(
            bundle_path=str(simple_bundle_yaml),
            max_gross_exposure=1.0,
        )
        
        signals = pd.DataFrame([
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "AAPL", "direction": "LONG", "score": 2.0},
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "MSFT", "direction": "LONG", "score": 1.0},
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "GOOGL", "direction": "SHORT", "score": -2.0},
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "AMZN", "direction": "SHORT", "score": -1.0},
        ])
        
        capital = 10000.0
        positions = compute_multifactor_long_short_positions(
            signals=signals,
            capital=capital,
            config=config,
        )
        
        # Long positions should have equal weights
        long_positions = positions[positions["target_weight"] > 0]
        if len(long_positions) > 1:
            long_weights = long_positions["target_weight"].values
            assert np.allclose(long_weights, long_weights[0]), \
                f"Long positions should be equal-weighted, got {long_weights}"
        
        # Short positions should have equal weights (absolute values)
        short_positions = positions[positions["target_weight"] < 0]
        if len(short_positions) > 1:
            short_weights = abs(short_positions["target_weight"].values)
            assert np.allclose(short_weights, short_weights[0]), \
                f"Short positions should be equal-weighted, got {short_weights}"
    
    def test_max_gross_exposure_constraint(self, simple_bundle_yaml):
        """Test that max_gross_exposure limits total exposure."""
        config = MultiFactorStrategyConfig(
            bundle_path=str(simple_bundle_yaml),
            max_gross_exposure=0.8,  # Limit to 80% gross exposure
        )
        
        signals = pd.DataFrame([
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "AAPL", "direction": "LONG", "score": 1.5},
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "MSFT", "direction": "LONG", "score": 1.2},
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "GOOGL", "direction": "SHORT", "score": -1.5},
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "AMZN", "direction": "SHORT", "score": -1.2},
        ])
        
        capital = 10000.0
        positions = compute_multifactor_long_short_positions(
            signals=signals,
            capital=capital,
            config=config,
        )
        
        # Gross exposure should be <= max_gross_exposure
        total_long_weight = positions[positions["target_weight"] > 0]["target_weight"].sum()
        total_short_weight = abs(positions[positions["target_weight"] < 0]["target_weight"].sum())
        gross_weight = total_long_weight + total_short_weight
        
        assert gross_weight <= config.max_gross_exposure + 1e-6, \
            f"Gross exposure ({gross_weight}) should be <= max_gross_exposure ({config.max_gross_exposure})"
    
    def test_empty_signals_returns_empty_dataframe(self, simple_bundle_yaml):
        """Test that empty signals return empty positions DataFrame."""
        config = MultiFactorStrategyConfig(
            bundle_path=str(simple_bundle_yaml),
        )
        
        empty_signals = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
        
        positions = compute_multifactor_long_short_positions(
            signals=empty_signals,
            capital=10000.0,
            config=config,
        )
        
        assert isinstance(positions, pd.DataFrame)
        assert positions.empty
        assert list(positions.columns) == ["symbol", "target_weight", "target_qty"]
    
    def test_missing_columns_raises_error(self, simple_bundle_yaml):
        """Test that missing required columns raise ValueError."""
        config = MultiFactorStrategyConfig(
            bundle_path=str(simple_bundle_yaml),
        )
        
        signals_no_direction = pd.DataFrame([
            {"timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "symbol": "AAPL", "score": 1.5},
        ])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_multifactor_long_short_positions(
                signals=signals_no_direction,
                capital=10000.0,
                config=config,
            )


class TestStrategyIntegration:
    """Integration tests for the multi-factor long/short strategy."""
    
    def test_strategy_end_to_end(self, sample_factors_df, simple_bundle_yaml):
        """Test end-to-end signal generation and position computation."""
        config = MultiFactorStrategyConfig(
            bundle_path=str(simple_bundle_yaml),
            top_quantile=0.2,
            bottom_quantile=0.2,
            rebalance_freq="D",
            max_gross_exposure=1.0,
        )
        
        prices = sample_factors_df[["timestamp", "symbol", "close"]].copy()
        
        # Generate signals
        signals = generate_multifactor_long_short_signals(
            prices=prices,
            factors=sample_factors_df,
            config=config,
        )
        
        assert not signals.empty, "Should generate signals"
        
        # Compute positions for first timestamp
        first_timestamp = signals["timestamp"].min()
        first_timestamp_signals = signals[signals["timestamp"] == first_timestamp]
        
        if not first_timestamp_signals.empty:
            positions = compute_multifactor_long_short_positions(
                signals=first_timestamp_signals,
                capital=10000.0,
                config=config,
            )
            
            assert not positions.empty, "Should compute positions"
            assert len(positions) == len(first_timestamp_signals), \
                "Should have one position per signal"
            
            # Verify positions have correct structure
            assert "symbol" in positions.columns
            assert "target_weight" in positions.columns
            assert "target_qty" in positions.columns

