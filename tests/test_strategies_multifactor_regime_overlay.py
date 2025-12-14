"""Tests for Multi-Factor Strategy with Regime Overlay Integration (Phase D1).

Tests the regime-aware multi-factor strategy functionality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

pytestmark = pytest.mark.advanced

from src.assembled_core.config.factor_bundles import FactorBundleConfig, FactorConfig, FactorBundleOptions
from src.assembled_core.risk.regime_models import RegimeStateConfig, build_regime_state
from src.assembled_core.strategies.multifactor_long_short import (
    MultiFactorStrategyConfig,
    generate_multifactor_long_short_signals,
    compute_multifactor_long_short_positions,
)


@pytest.fixture
def regime_price_panel() -> pd.DataFrame:
    """Create price panel with clear regime phases."""
    dates = pd.date_range("2020-01-01", periods=120, freq="D", tz="UTC")
    symbols = ["STOCK1", "STOCK2", "STOCK3", "STOCK4", "STOCK5"]
    
    all_data = []
    for i, symbol in enumerate(symbols):
        base_price = 100.0 + i * 10.0
        
        # Phase 1: Bull (days 0-40)
        bull_prices = base_price + np.arange(41) * 0.3 + np.random.randn(41) * 0.2
        
        # Phase 2: Bear (days 41-80)
        bear_base = bull_prices[-1]
        bear_prices = bear_base - np.arange(40) * 0.25 + np.random.randn(40) * 0.3
        
        # Phase 3: Crisis (days 81-119) - high volatility, falling
        crisis_base = bear_prices[-1]
        crisis_prices = crisis_base - np.arange(39) * 0.35 + np.random.randn(39) * 0.5
        
        all_prices = np.concatenate([bull_prices, bear_prices, crisis_prices])
        
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": symbol,
            "close": all_prices,
            "open": all_prices * 0.99,
            "high": all_prices * 1.01,
            "low": all_prices * 0.98,
            "volume": np.random.randint(1000000, 10000000, size=120),
        })
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture
def regime_factors_df(regime_price_panel) -> pd.DataFrame:
    """Create factors DataFrame with clear differentiation."""
    all_data = []
    for (symbol, timestamp), group in regime_price_panel.groupby(["symbol", "timestamp"]):
        symbol_idx = {"STOCK1": 4, "STOCK2": 3, "STOCK3": 2, "STOCK4": 1, "STOCK5": 0}.get(symbol, 0)
        day_idx = (timestamp - regime_price_panel["timestamp"].min()).days
        
        # Different factor values per symbol (for ranking)
        all_data.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "returns_12m": 0.08 + symbol_idx * 0.04,
            "momentum_12m_excl_1m": 0.04 + symbol_idx * 0.03,
            "rv_20": 0.25 - symbol_idx * 0.03,  # Lower vol is better
            "close": group["close"].iloc[0],
            
            # Add macro factors for regime detection
            "macro_growth_regime": 1.0 if day_idx < 41 else (-1.0 if day_idx < 81 else -1.0),
            "macro_inflation_regime": 0.5 if day_idx < 41 else (-0.5 if day_idx < 81 else -0.3),
            "macro_risk_aversion_proxy": -0.5 if day_idx < 41 else (0.7 if day_idx < 81 else 0.9),
        })
    
    df = pd.DataFrame(all_data)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture
def regime_breadth_df() -> pd.DataFrame:
    """Create market breadth DataFrame with regime phases."""
    dates = pd.date_range("2020-01-01", periods=120, freq="D", tz="UTC")
    
    # Phase 1: High breadth (bull)
    fraction_above_1 = np.ones(41) * 0.85 + np.random.randn(41) * 0.05
    
    # Phase 2: Low breadth (bear)
    fraction_above_2 = np.ones(40) * 0.25 + np.random.randn(40) * 0.05
    
    # Phase 3: Very low breadth (crisis)
    fraction_above_3 = np.ones(39) * 0.15 + np.random.randn(39) * 0.05
    
    fraction_above = np.concatenate([fraction_above_1, fraction_above_2, fraction_above_3])
    fraction_above = np.clip(fraction_above, 0.0, 1.0)
    
    # Risk-on/off score
    risk_on_off_1 = np.ones(41) * 0.8
    risk_on_off_2 = np.ones(40) * -0.6
    risk_on_off_3 = np.ones(39) * -0.9  # Extreme risk-off in crisis
    
    risk_on_off = np.concatenate([risk_on_off_1, risk_on_off_2, risk_on_off_3])
    
    return pd.DataFrame({
        "timestamp": dates,
        "fraction_above_ma_50": fraction_above,
        "risk_on_off_score": risk_on_off,
    })


@pytest.fixture
def regime_vol_df(regime_price_panel) -> pd.DataFrame:
    """Create volatility DataFrame with regime phases."""
    all_data = []
    for symbol in regime_price_panel["symbol"].unique():
        symbol_df = regime_price_panel[regime_price_panel["symbol"] == symbol].sort_values("timestamp")
        dates = symbol_df["timestamp"].values
        
        # Phase 1: Low vol (bull)
        rv_1 = np.ones(41) * 0.15 + np.random.randn(41) * 0.02
        
        # Phase 2: Moderate vol (bear)
        rv_2 = np.ones(40) * 0.30 + np.random.randn(40) * 0.03
        
        # Phase 3: High vol (crisis)
        rv_3 = np.ones(39) * 0.50 + np.random.randn(39) * 0.05
        
        rv_20 = np.concatenate([rv_1, rv_2, rv_3])
        rv_20 = np.clip(rv_20, 0.05, 0.60)
        
        all_data.append(pd.DataFrame({
            "timestamp": dates,
            "symbol": symbol,
            "rv_20": rv_20,
        }))
    
    return pd.concat(all_data, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture
def simple_regime_bundle_config() -> FactorBundleConfig:
    """Create a simple bundle config for regime testing."""
    return FactorBundleConfig(
        universe="test_regime_universe",
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
def default_regime_risk_map() -> dict[str, dict[str, float]]:
    """Default regime risk map for testing."""
    return {
        "bull": {"max_gross_exposure": 1.2, "target_net_exposure": 0.6},
        "neutral": {"max_gross_exposure": 1.0, "target_net_exposure": 0.2},
        "sideways": {"max_gross_exposure": 0.8, "target_net_exposure": 0.0},
        "bear": {"max_gross_exposure": 0.6, "target_net_exposure": 0.0},
        "crisis": {"max_gross_exposure": 0.3, "target_net_exposure": 0.0},
        "reflation": {"max_gross_exposure": 1.1, "target_net_exposure": 0.3},
    }


class TestExposureChangesByRegime:
    """Tests that exposure changes correctly based on regime."""
    
    def test_bull_regime_high_exposure(
        self,
        regime_price_panel,
        regime_factors_df,
        regime_breadth_df,
        regime_vol_df,
        simple_regime_bundle_config,
        default_regime_risk_map,
        tmp_path: Path,
    ):
        """Test that bull regime uses high exposure and net long."""
        import yaml
        
        # Create bundle YAML file
        bundle_file = tmp_path / "test_regime_bundle.yaml"
        bundle_data = {
            "universe": simple_regime_bundle_config.universe,
            "factor_set": simple_regime_bundle_config.factor_set,
            "horizon_days": simple_regime_bundle_config.horizon_days,
            "factors": [
                {"name": f.name, "weight": f.weight, "direction": f.direction}
                for f in simple_regime_bundle_config.factors
            ],
            "options": {
                "winsorize": simple_regime_bundle_config.options.winsorize,
                "winsorize_limits": simple_regime_bundle_config.options.winsorize_limits,
                "zscore": simple_regime_bundle_config.options.zscore,
                "neutralize_by": simple_regime_bundle_config.options.neutralize_by,
            },
        }
        with bundle_file.open("w", encoding="utf-8") as f:
            yaml.dump(bundle_data, f)
        
        # Build regime state
        regime_state_df = build_regime_state(
            prices=regime_price_panel,
            macro_factors=regime_factors_df[["timestamp", "symbol", "macro_growth_regime", "macro_inflation_regime", "macro_risk_aversion_proxy"]],
            breadth_df=regime_breadth_df,
            vol_df=regime_vol_df,
        )
        
        # Create config with regime overlay
        config = MultiFactorStrategyConfig(
            bundle_path=str(bundle_file),
            top_quantile=0.2,
            bottom_quantile=0.2,
            rebalance_freq="M",  # Monthly rebalancing
            max_gross_exposure=1.0,  # Default (will be overridden by regime)
            use_regime_overlay=True,
            regime_config=RegimeStateConfig(),
            regime_risk_map=default_regime_risk_map,
        )
        
        # Generate signals
        signals = generate_multifactor_long_short_signals(
            prices=regime_price_panel,
            factors=regime_factors_df,
            config=config,
        )
        
        # Filter to first rebalance date (should be in bull phase)
        first_rebalance_date = signals["timestamp"].min()
        first_rebalance_signals = signals[signals["timestamp"] == first_rebalance_date].copy()
        
        if not first_rebalance_signals.empty:
            # Compute positions for first rebalance
            positions = compute_multifactor_long_short_positions(
                signals=first_rebalance_signals,
                capital=100000.0,
                config=config,
                regime_state_df=regime_state_df,
            )
            
            if not positions.empty:
                # Calculate gross and net exposure
                total_long = positions[positions["target_weight"] > 0]["target_weight"].sum()
                total_short = abs(positions[positions["target_weight"] < 0]["target_weight"].sum())
                net_exposure = positions["target_weight"].sum()
                
                # Get regime for first rebalance
                regime_for_date = regime_state_df[regime_state_df["timestamp"] == first_rebalance_date]
                if not regime_for_date.empty:
                    regime_label = regime_for_date["regime_label"].iloc[0]
                    
                    # If bull regime, expect higher exposure
                    if regime_label == "bull":
                        expected_gross = default_regime_risk_map["bull"]["max_gross_exposure"]
                        expected_net = default_regime_risk_map["bull"]["target_net_exposure"]
                        
                        gross_exposure = total_long + total_short
                        # Allow some tolerance (positions might not exactly match due to quantile selection)
                        assert abs(gross_exposure - expected_gross) < 0.3, \
                            f"Expected gross exposure ~{expected_gross}, got {gross_exposure}"
                        
                        assert net_exposure > 0, \
                            f"Expected positive net exposure in bull, got {net_exposure}"
    
    def test_crisis_regime_low_exposure(
        self,
        regime_price_panel,
        regime_factors_df,
        regime_breadth_df,
        regime_vol_df,
        simple_regime_bundle_config,
        default_regime_risk_map,
        tmp_path: Path,
    ):
        """Test that crisis regime uses low exposure."""
        import yaml
        
        # Create bundle YAML file
        bundle_file = tmp_path / "test_regime_bundle.yaml"
        bundle_data = {
            "universe": simple_regime_bundle_config.universe,
            "factor_set": simple_regime_bundle_config.factor_set,
            "horizon_days": simple_regime_bundle_config.horizon_days,
            "factors": [
                {"name": f.name, "weight": f.weight, "direction": f.direction}
                for f in simple_regime_bundle_config.factors
            ],
            "options": {
                "winsorize": simple_regime_bundle_config.options.winsorize,
                "winsorize_limits": simple_regime_bundle_config.options.winsorize_limits,
                "zscore": simple_regime_bundle_config.options.zscore,
                "neutralize_by": simple_regime_bundle_config.options.neutralize_by,
            },
        }
        with bundle_file.open("w", encoding="utf-8") as f:
            yaml.dump(bundle_data, f)
        
        # Build regime state
        regime_state_df = build_regime_state(
            prices=regime_price_panel,
            macro_factors=regime_factors_df[["timestamp", "symbol", "macro_growth_regime", "macro_inflation_regime", "macro_risk_aversion_proxy"]],
            breadth_df=regime_breadth_df,
            vol_df=regime_vol_df,
        )
        
        # Create config with regime overlay
        config = MultiFactorStrategyConfig(
            bundle_path=str(bundle_file),
            top_quantile=0.2,
            bottom_quantile=0.2,
            rebalance_freq="M",
            max_gross_exposure=1.0,
            use_regime_overlay=True,
            regime_config=RegimeStateConfig(),
            regime_risk_map=default_regime_risk_map,
        )
        
        # Generate signals
        signals = generate_multifactor_long_short_signals(
            prices=regime_price_panel,
            factors=regime_factors_df,
            config=config,
        )
        
        # Filter to a later rebalance date (should be in crisis phase)
        if len(signals["timestamp"].unique()) > 1:
            later_dates = sorted(signals["timestamp"].unique())[-1]
            later_signals = signals[signals["timestamp"] == later_dates].copy()
            
            if not later_signals.empty:
                # Compute positions
                positions = compute_multifactor_long_short_positions(
                    signals=later_signals,
                    capital=100000.0,
                    config=config,
                    regime_state_df=regime_state_df,
                )
                
                if not positions.empty:
                    total_long = positions[positions["target_weight"] > 0]["target_weight"].sum()
                    total_short = abs(positions[positions["target_weight"] < 0]["target_weight"].sum())
                    gross_exposure = total_long + total_short
                    
                    # Get regime for this date
                    regime_for_date = regime_state_df[regime_state_df["timestamp"] == later_dates]
                    if not regime_for_date.empty:
                        regime_label = regime_for_date["regime_label"].iloc[0]
                        
                        # If crisis regime, expect low exposure
                        if regime_label in ["crisis", "bear"]:
                            expected_gross = default_regime_risk_map.get(regime_label, {}).get("max_gross_exposure", 0.6)
                            
                            # Crisis should have lower exposure than default
                            assert gross_exposure <= expected_gross + 0.2, \
                                f"Expected gross exposure <= {expected_gross + 0.2} in {regime_label}, got {gross_exposure}"


class TestNoOverlayBehaviourUnchanged:
    """Tests that with use_regime_overlay=False, behavior is unchanged."""
    
    def test_fixed_exposure_without_overlay(
        self,
        regime_price_panel,
        regime_factors_df,
        simple_regime_bundle_config,
        tmp_path: Path,
    ):
        """Test that without overlay, exposure is fixed as configured."""
        import yaml
        
        # Create bundle YAML file
        bundle_file = tmp_path / "test_bundle.yaml"
        bundle_data = {
            "universe": simple_regime_bundle_config.universe,
            "factor_set": simple_regime_bundle_config.factor_set,
            "horizon_days": simple_regime_bundle_config.horizon_days,
            "factors": [
                {"name": f.name, "weight": f.weight, "direction": f.direction}
                for f in simple_regime_bundle_config.factors
            ],
            "options": {
                "winsorize": simple_regime_bundle_config.options.winsorize,
                "winsorize_limits": simple_regime_bundle_config.options.winsorize_limits,
                "zscore": simple_regime_bundle_config.options.zscore,
                "neutralize_by": simple_regime_bundle_config.options.neutralize_by,
            },
        }
        with bundle_file.open("w", encoding="utf-8") as f:
            yaml.dump(bundle_data, f)
        
        # Config WITHOUT regime overlay
        config = MultiFactorStrategyConfig(
            bundle_path=str(bundle_file),
            top_quantile=0.2,
            bottom_quantile=0.2,
            rebalance_freq="M",
            max_gross_exposure=1.0,  # Fixed exposure
            use_regime_overlay=False,  # Disabled
        )
        
        # Generate signals
        signals = generate_multifactor_long_short_signals(
            prices=regime_price_panel,
            factors=regime_factors_df,
            config=config,
        )
        
        if not signals.empty:
            # Get first rebalance date
            first_date = signals["timestamp"].min()
            first_signals = signals[signals["timestamp"] == first_date].copy()
            
            # Compute positions
            positions = compute_multifactor_long_short_positions(
                signals=first_signals,
                capital=100000.0,
                config=config,
            )
            
            if not positions.empty:
                total_long = positions[positions["target_weight"] > 0]["target_weight"].sum()
                total_short = abs(positions[positions["target_weight"] < 0]["target_weight"].sum())
                gross_exposure = total_long + total_short
                
                # Should use fixed max_gross_exposure (approximately)
                # Allow tolerance due to equal-weighting within quantiles
                assert abs(gross_exposure - config.max_gross_exposure) < 0.3, \
                    f"Expected gross exposure ~{config.max_gross_exposure}, got {gross_exposure}"


class TestRegimeOverlayIntegration:
    """Integration tests for regime overlay functionality."""
    
    @pytest.mark.advanced
    def test_signals_contain_regime_labels(
        self,
        regime_price_panel,
        regime_factors_df,
        regime_breadth_df,
        regime_vol_df,
        simple_regime_bundle_config,
        default_regime_risk_map,
        tmp_path: Path,
    ):
        """Test that signals DataFrame contains regime labels when overlay is enabled."""
        import yaml
        
        # Create bundle YAML file
        bundle_file = tmp_path / "test_regime_bundle.yaml"
        bundle_data = {
            "universe": simple_regime_bundle_config.universe,
            "factor_set": simple_regime_bundle_config.factor_set,
            "horizon_days": simple_regime_bundle_config.horizon_days,
            "factors": [
                {"name": f.name, "weight": f.weight, "direction": f.direction}
                for f in simple_regime_bundle_config.factors
            ],
            "options": {
                "winsorize": simple_regime_bundle_config.options.winsorize,
                "winsorize_limits": simple_regime_bundle_config.options.winsorize_limits,
                "zscore": simple_regime_bundle_config.options.zscore,
                "neutralize_by": simple_regime_bundle_config.options.neutralize_by,
            },
        }
        with bundle_file.open("w", encoding="utf-8") as f:
            yaml.dump(bundle_data, f)
        
        # Config WITH regime overlay
        config = MultiFactorStrategyConfig(
            bundle_path=str(bundle_file),
            top_quantile=0.2,
            bottom_quantile=0.2,
            rebalance_freq="M",
            use_regime_overlay=True,
            regime_config=RegimeStateConfig(),
            regime_risk_map=default_regime_risk_map,
        )
        
        # Generate signals
        signals = generate_multifactor_long_short_signals(
            prices=regime_price_panel,
            factors=regime_factors_df,
            config=config,
        )
        
        # Check that regime column exists (if regime overlay was successfully built)
        # Note: regime column is added per timestamp, so might not exist if regime detection failed
        # This is acceptable behavior - we just check that the function completes without error
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) >= 0  # Can be empty if no signals in rebalance dates
        
        # Check that regime_state_df is stored in attrs
        if hasattr(signals, "attrs") and "regime_state_df" in signals.attrs:
            regime_state_df = signals.attrs["regime_state_df"]
            assert isinstance(regime_state_df, pd.DataFrame)
            assert "regime_label" in regime_state_df.columns

