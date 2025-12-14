"""Tests for Factor Bundle Configuration Module.

This module tests the factor bundle loading and validation functionality.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.factor_bundles import (
    FactorBundleConfig,
    FactorConfig,
    FactorBundleOptions,
    load_factor_bundle,
)

pytestmark = pytest.mark.advanced


@pytest.mark.advanced
def test_load_factor_bundle_core():
    """Test loading existing core bundle."""
    bundle = load_factor_bundle("config/factor_bundles/macro_world_etfs_core_bundle.yaml")
    
    assert isinstance(bundle, FactorBundleConfig)
    assert bundle.universe == "macro_world_etfs"
    assert bundle.factor_set == "core+vol_liquidity"
    assert bundle.horizon_days == 20
    
    # Check weights sum to 1.0
    total_weight = sum(f.weight for f in bundle.factors)
    assert abs(total_weight - 1.0) < 0.01, f"Weights should sum to 1.0, got {total_weight:.4f}"
    
    # Check all factors have valid directions
    for factor in bundle.factors:
        assert factor.direction in ("positive", "negative"), f"Invalid direction: {factor.direction}"
        assert isinstance(factor.name, str), "Factor name should be string"


@pytest.mark.advanced
def test_load_factor_bundle_ai_tech_core_alt():
    """Test loading AI Tech core+alt bundle."""
    bundle = load_factor_bundle("config/factor_bundles/ai_tech_core_alt_bundle.yaml")
    
    assert isinstance(bundle, FactorBundleConfig)
    assert bundle.universe == "universe_ai_tech"
    assert bundle.factor_set == "core+alt_full"
    assert bundle.horizon_days == 20
    
    # Check weights sum to 1.0
    total_weight = sum(f.weight for f in bundle.factors)
    assert abs(total_weight - 1.0) < 0.01, f"Weights should sum to 1.0, got {total_weight:.4f}"


@pytest.mark.advanced
def test_load_factor_bundle_ml_alpha_only():
    """Test loading ML alpha only bundle."""
    bundle = load_factor_bundle("config/factor_bundles/ai_tech_ml_alpha_bundle.yaml")
    
    assert isinstance(bundle, FactorBundleConfig)
    assert bundle.universe == "universe_ai_tech"
    assert bundle.factor_set == "ml_alpha_only"
    assert bundle.horizon_days == 20
    
    # Check weights sum to 1.0
    total_weight = sum(f.weight for f in bundle.factors)
    assert abs(total_weight - 1.0) < 0.01, f"Weights should sum to 1.0, got {total_weight:.4f}"
    
    # Check ML alpha factor is present
    ml_alpha_factors = [f for f in bundle.factors if f.name.startswith("ml_alpha_")]
    assert len(ml_alpha_factors) > 0, "ML alpha factor should be present"
    assert ml_alpha_factors[0].name == "ml_alpha_ridge_20d"
    assert ml_alpha_factors[0].direction == "positive"
    assert ml_alpha_factors[0].weight == 1.0
    
    # Check all factors have valid directions
    for factor in bundle.factors:
        assert factor.direction in ("positive", "negative"), f"Invalid direction: {factor.direction}"
        assert isinstance(factor.name, str), "Factor name should be string"


@pytest.mark.advanced
def test_load_factor_bundle_core_ml_mixed():
    """Test loading mixed core+ML bundle."""
    bundle = load_factor_bundle("config/factor_bundles/ai_tech_core_ml_bundle.yaml")
    
    assert isinstance(bundle, FactorBundleConfig)
    assert bundle.universe == "universe_ai_tech"
    assert bundle.factor_set == "core+alt_full+ml"
    assert bundle.horizon_days == 20
    
    # Check weights sum to 1.0
    total_weight = sum(f.weight for f in bundle.factors)
    assert abs(total_weight - 1.0) < 0.01, f"Weights should sum to 1.0, got {total_weight:.4f}"
    
    # Check ML alpha factor is present
    ml_alpha_factors = [f for f in bundle.factors if f.name.startswith("ml_alpha_")]
    assert len(ml_alpha_factors) > 0, "ML alpha factor should be present"
    assert ml_alpha_factors[0].name == "ml_alpha_ridge_20d"
    assert ml_alpha_factors[0].weight == 0.40
    
    # Check traditional factors are also present
    traditional_factors = [f for f in bundle.factors if not f.name.startswith("ml_alpha_")]
    assert len(traditional_factors) > 0, "Traditional factors should be present"
    
    # Check all factors have valid directions
    for factor in bundle.factors:
        assert factor.direction in ("positive", "negative"), f"Invalid direction: {factor.direction}"
        assert isinstance(factor.name, str), "Factor name should be string"
    
    # Verify specific factors
    factor_names = {f.name: f for f in bundle.factors}
    assert "momentum_12m_excl_1m" in factor_names
    assert "trend_strength_50" in factor_names
    assert "rv_20" in factor_names
    assert factor_names["rv_20"].direction == "negative"  # Volatility should be negative


@pytest.mark.advanced
def test_bundle_options_validation():
    """Test that bundle options are correctly loaded and validated."""
    bundle = load_factor_bundle("config/factor_bundles/ai_tech_ml_alpha_bundle.yaml")
    
    assert isinstance(bundle.options, FactorBundleOptions)
    
    # Check ML alpha bundle has winsorize: false (as specified)
    assert bundle.options.winsorize is False
    
    # Check mixed bundle has winsorize: true
    mixed_bundle = load_factor_bundle("config/factor_bundles/ai_tech_core_ml_bundle.yaml")
    assert mixed_bundle.options.winsorize is True
    
    # Both should have zscore: true
    assert bundle.options.zscore is True
    assert mixed_bundle.options.zscore is True


@pytest.mark.advanced
def test_all_factor_names_are_strings():
    """Test that all factor names in bundles are strings."""
    bundles_to_test = [
        "config/factor_bundles/macro_world_etfs_core_bundle.yaml",
        "config/factor_bundles/ai_tech_core_alt_bundle.yaml",
        "config/factor_bundles/ai_tech_ml_alpha_bundle.yaml",
        "config/factor_bundles/ai_tech_core_ml_bundle.yaml",
    ]
    
    for bundle_path in bundles_to_test:
        bundle = load_factor_bundle(bundle_path)
        for factor in bundle.factors:
            assert isinstance(factor.name, str), f"Factor name should be string, got {type(factor.name)} for {factor.name}"


@pytest.mark.advanced
def test_all_directions_are_valid():
    """Test that all factor directions are valid (positive or negative)."""
    bundles_to_test = [
        "config/factor_bundles/macro_world_etfs_core_bundle.yaml",
        "config/factor_bundles/ai_tech_core_alt_bundle.yaml",
        "config/factor_bundles/ai_tech_ml_alpha_bundle.yaml",
        "config/factor_bundles/ai_tech_core_ml_bundle.yaml",
    ]
    
    for bundle_path in bundles_to_test:
        bundle = load_factor_bundle(bundle_path)
        for factor in bundle.factors:
            assert factor.direction in ("positive", "negative"), (
                f"Invalid direction '{factor.direction}' for factor '{factor.name}' in bundle {bundle_path}"
            )

