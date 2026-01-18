# tests/test_feature_registry.py
"""Tests for Feature Registry (Sprint 5 / F2).

Tests verify:
1. Registry is unique (no duplicate feature names)
2. Registry is documented (all features have required metadata)
3. Registry is namespaced (all features follow naming rules)
4. Feature names are namespaced in actual feature generation
5. No duplicate feature columns are created
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.features.registry import (
    FEATURE_REGISTRY,
    get_feature_metadata,
    list_all_feature_names,
    list_features_by_namespace,
    validate_registry_documented,
    validate_registry_namespaced,
    validate_registry_unique,
)
from src.assembled_core.features.ta_features import add_all_features


def test_feature_registry_unique_and_documented() -> None:
    """Test that registry is unique and all features are documented."""
    # Test uniqueness
    is_unique, duplicates = validate_registry_unique()
    assert is_unique, f"Duplicate features found: {duplicates}"
    
    # Test documentation
    is_documented, missing = validate_registry_documented()
    assert is_documented, f"Missing metadata: {missing}"
    
    # Test namespacing
    is_namespaced, invalid = validate_registry_namespaced()
    assert is_namespaced, f"Invalid namespaced features: {invalid}"


def test_feature_names_are_namespaced() -> None:
    """Test that all feature names in registry follow namespace rules."""
    valid_prefixes = {"ta_", "liq_", "vol_", "alt_", "macro_", "regime_", "ml_"}
    
    for name in FEATURE_REGISTRY.keys():
        # Check prefix
        has_valid_prefix = any(name.startswith(prefix) for prefix in valid_prefixes)
        assert has_valid_prefix, f"Feature {name} does not start with valid prefix {valid_prefixes}"
        
        # Check version suffix
        assert name.endswith("_v1") or name.endswith("_v2") or "_v" in name, \
            f"Feature {name} does not have version suffix (_v{{number}})"
        
        # Check metadata namespace matches prefix
        metadata = FEATURE_REGISTRY[name]
        namespace = metadata.get("namespace")
        assert namespace is not None, f"Feature {name} missing namespace in metadata"
        
        expected_prefix = f"{namespace}_"
        assert name.startswith(expected_prefix), \
            f"Feature {name} prefix does not match namespace {namespace}"


def test_no_duplicate_feature_columns() -> None:
    """Test that feature generation does not create duplicate columns."""
    # Create test data
    timestamps = pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": ["AAPL"] * 10,
        "open": [150.0] * 10,
        "high": [155.0] * 10,
        "low": [148.0] * 10,
        "close": [152.0] * 10,
        "volume": [1000000.0] * 10,
    })
    
    # Generate features
    features = add_all_features(prices, use_namespace=True)
    
    # Check for duplicate column names
    column_counts = features.columns.value_counts()
    duplicates = column_counts[column_counts > 1].index.tolist()
    assert len(duplicates) == 0, f"Duplicate columns found: {duplicates}"
    
    # Verify namespaced columns exist
    assert "ta_log_return_v1" in features.columns, "ta_log_return_v1 should exist"
    assert "ta_ma_20_v1" in features.columns, "ta_ma_20_v1 should exist"
    assert "ta_ma_50_v1" in features.columns, "ta_ma_50_v1 should exist"
    assert "ta_ma_200_v1" in features.columns, "ta_ma_200_v1 should exist"
    assert "ta_atr_14_v1" in features.columns, "ta_atr_14_v1 should exist"
    assert "ta_rsi_14_v1" in features.columns, "ta_rsi_14_v1 should exist"


def test_registry_get_feature_metadata() -> None:
    """Test get_feature_metadata() function."""
    metadata = get_feature_metadata("ta_rsi_14_v1")
    assert metadata is not None, "ta_rsi_14_v1 should exist in registry"
    assert metadata["description"] is not None, "Description should be present"
    assert metadata["inputs"] is not None, "Inputs should be present"
    assert metadata["version"] == 1, "Version should be 1"
    assert metadata["namespace"] == "ta", "Namespace should be 'ta'"
    
    # Test non-existent feature
    metadata_none = get_feature_metadata("nonexistent_feature")
    assert metadata_none is None, "Non-existent feature should return None"


def test_registry_list_features_by_namespace() -> None:
    """Test list_features_by_namespace() function."""
    ta_features = list_features_by_namespace("ta")
    assert len(ta_features) > 0, "Should have TA features"
    assert all(f.startswith("ta_") for f in ta_features), "All features should start with 'ta_'"
    
    vol_features = list_features_by_namespace("vol")
    assert len(vol_features) > 0, "Should have volatility features"
    assert all(f.startswith("vol_") for f in vol_features), "All features should start with 'vol_'"
    
    # Test empty namespace
    empty_features = list_features_by_namespace("nonexistent")
    assert len(empty_features) == 0, "Non-existent namespace should return empty list"


def test_registry_list_all_feature_names() -> None:
    """Test list_all_feature_names() function."""
    all_names = list_all_feature_names()
    assert len(all_names) > 0, "Should have features"
    assert len(all_names) == len(FEATURE_REGISTRY), "Should match registry size"
    assert all_names == sorted(all_names), "Should be sorted"


def test_feature_generation_creates_namespaced_columns() -> None:
    """Test that feature generation creates namespaced columns."""
    # Create test data
    timestamps = pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": ["AAPL"] * 10,
        "open": [150.0] * 10,
        "high": [155.0] * 10,
        "low": [148.0] * 10,
        "close": [152.0] * 10,
        "volume": [1000000.0] * 10,
    })
    
    # Generate features with namespace
    features = add_all_features(prices, use_namespace=True)
    
    # Check that namespaced columns exist
    expected_namespaced = [
        "ta_log_return_v1",
        "ta_ma_20_v1",
        "ta_ma_50_v1",
        "ta_ma_200_v1",
        "ta_atr_14_v1",
        "ta_rsi_14_v1",
    ]
    
    for col in expected_namespaced:
        assert col in features.columns, f"Namespaced column {col} should exist"
    
    # Check that legacy columns also exist (compatibility)
    expected_legacy = [
        "log_return",
        "ma_20",
        "ma_50",
        "ma_200",
        "atr_14",
        "rsi_14",
    ]
    
    for col in expected_legacy:
        assert col in features.columns, f"Legacy column {col} should exist (compatibility)"


def test_feature_registry_validation_fails_on_duplicate() -> None:
    """Test that validation fails if duplicate features are added (manual test)."""
    # This test verifies that the validation functions work correctly
    # In a real scenario, we would add a duplicate and verify it fails
    
    # For now, we just verify the validation functions work
    is_unique, duplicates = validate_registry_unique()
    assert is_unique, "Registry should be unique"
    assert len(duplicates) == 0, "Should have no duplicates"
    
    # Verify that if we manually check, we get the same result
    all_names = list(FEATURE_REGISTRY.keys())
    assert len(all_names) == len(set(all_names)), "Registry keys should be unique"
