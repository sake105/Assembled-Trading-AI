# src/assembled_core/features/registry.py
"""Feature Registry (Sprint 5 / F2).

This module provides a registry system for feature names, ensuring:
- Unique, namespaced feature names
- Versioning support (e.g., ta_rsi_v1, ta_ma_20_v1)
- Documentation of feature metadata (description, inputs, version, layer)

Feature Naming Rules:
- Prefixes: ta_ (technical analysis), liq_ (liquidity), vol_ (volatility),
  alt_ (alternative data), macro_ (macroeconomic), regime_ (regime detection),
  ml_ (machine learning features)
- Versioning: {prefix}_{feature_name}_v{version} (e.g., ta_rsi_v1)
- No collisions: each feature column must be unique and stable

Registry Structure:
FEATURE_REGISTRY: dict[str, dict]
  - key: feature name (e.g., "ta_rsi_14_v1")
  - value: metadata dict with keys:
    - description: str (human-readable description)
    - inputs: list[str] (required input columns, e.g., ["close"])
    - version: int (version number, default: 1)
    - layer: str (feature layer: "ta", "liq", "vol", "alt", "macro", "regime", "ml")
    - namespace: str (namespace prefix, e.g., "ta")
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Feature Registry: name -> metadata
FEATURE_REGISTRY: dict[str, dict[str, Any]] = {
    # Technical Analysis Features (ta_)
    "ta_log_return_v1": {
        "description": "Logarithmic return (log(close[t] / close[t-1]))",
        "inputs": ["close"],
        "version": 1,
        "layer": "ta",
        "namespace": "ta",
    },
    "ta_ma_20_v1": {
        "description": "Simple Moving Average with window 20",
        "inputs": ["close"],
        "version": 1,
        "layer": "ta",
        "namespace": "ta",
    },
    "ta_ma_50_v1": {
        "description": "Simple Moving Average with window 50",
        "inputs": ["close"],
        "version": 1,
        "layer": "ta",
        "namespace": "ta",
    },
    "ta_ma_200_v1": {
        "description": "Simple Moving Average with window 200",
        "inputs": ["close"],
        "version": 1,
        "layer": "ta",
        "namespace": "ta",
    },
    "ta_atr_14_v1": {
        "description": "Average True Range with window 14",
        "inputs": ["high", "low", "close"],
        "version": 1,
        "layer": "ta",
        "namespace": "ta",
    },
    "ta_rsi_14_v1": {
        "description": "Relative Strength Index (Wilder) with window 14",
        "inputs": ["close"],
        "version": 1,
        "layer": "ta",
        "namespace": "ta",
    },
    # Volatility Features (vol_)
    "vol_rv_20_v1": {
        "description": "Realized Volatility (annualized) with window 20",
        "inputs": ["close"],
        "version": 1,
        "layer": "vol",
        "namespace": "vol",
    },
    "vol_vov_20_60_v1": {
        "description": "Volatility of Volatility (rv_20 / rv_60)",
        "inputs": ["close"],
        "version": 1,
        "layer": "vol",
        "namespace": "vol",
    },
    # Liquidity Features (liq_)
    "liq_turnover_v1": {
        "description": "Turnover (volume * price / market_cap)",
        "inputs": ["volume", "close"],
        "version": 1,
        "layer": "liq",
        "namespace": "liq",
    },
    "liq_volume_zscore_v1": {
        "description": "Volume Z-score (normalized volume)",
        "inputs": ["volume"],
        "version": 1,
        "layer": "liq",
        "namespace": "liq",
    },
    "liq_spread_proxy_v1": {
        "description": "Spread proxy ((high - low) / close)",
        "inputs": ["high", "low", "close"],
        "version": 1,
        "layer": "liq",
        "namespace": "liq",
    },
    # Alternative Data Features (alt_)
    "alt_insider_net_buy_20d_v1": {
        "description": "Insider net buy (20-day rolling sum)",
        "inputs": ["insider_events"],
        "version": 1,
        "layer": "alt",
        "namespace": "alt",
    },
    "alt_insider_trade_count_20d_v1": {
        "description": "Insider trade count (20-day rolling sum)",
        "inputs": ["insider_events"],
        "version": 1,
        "layer": "alt",
        "namespace": "alt",
    },
    "alt_congress_trade_count_60d_v1": {
        "description": "Congress trade count (60-day rolling sum)",
        "inputs": ["congress_events"],
        "version": 1,
        "layer": "alt",
        "namespace": "alt",
    },
    "alt_news_sentiment_7d_v1": {
        "description": "News sentiment (7-day rolling mean)",
        "inputs": ["news_events"],
        "version": 1,
        "layer": "alt",
        "namespace": "alt",
    },
    "alt_shipping_congestion_score_v1": {
        "description": "Shipping congestion score (latest)",
        "inputs": ["shipping_events"],
        "version": 1,
        "layer": "alt",
        "namespace": "alt",
    },
    # Regime Features (regime_)
    "regime_trend_strength_20_v1": {
        "description": "Trend strength ((price - MA_20) / ATR_20)",
        "inputs": ["close", "ta_ma_20_v1", "ta_atr_14_v1"],
        "version": 1,
        "layer": "regime",
        "namespace": "regime",
    },
    "regime_fraction_above_ma_50_v1": {
        "description": "Fraction of symbols above MA_50 (market breadth)",
        "inputs": ["close", "ta_ma_50_v1"],
        "version": 1,
        "layer": "regime",
        "namespace": "regime",
    },
}


def validate_registry_unique() -> tuple[bool, list[str]]:
    """Validate that all feature names in registry are unique.

    Returns:
        Tuple of (is_valid, duplicate_names)
        - is_valid: True if all names are unique
        - duplicate_names: List of duplicate feature names (empty if valid)
    """
    names = list(FEATURE_REGISTRY.keys())
    seen = set()
    duplicates = []
    
    for name in names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    
    is_valid = len(duplicates) == 0
    return is_valid, duplicates


def validate_registry_documented() -> tuple[bool, list[str]]:
    """Validate that all features in registry have required metadata.

    Required metadata keys: description, inputs, version, layer, namespace

    Returns:
        Tuple of (is_valid, missing_metadata)
        - is_valid: True if all features have required metadata
        - missing_metadata: List of feature names with missing metadata (empty if valid)
    """
    required_keys = {"description", "inputs", "version", "layer", "namespace"}
    missing = []
    
    for name, metadata in FEATURE_REGISTRY.items():
        missing_keys = required_keys - set(metadata.keys())
        if missing_keys:
            missing.append(f"{name}: missing {missing_keys}")
    
    is_valid = len(missing) == 0
    return is_valid, missing


def validate_registry_namespaced() -> tuple[bool, list[str]]:
    """Validate that all feature names follow namespace rules.

    Rules:
    - Must start with valid prefix (ta_, liq_, vol_, alt_, macro_, regime_, ml_)
    - Must end with _v{version} (e.g., _v1)

    Returns:
        Tuple of (is_valid, invalid_names)
        - is_valid: True if all names follow namespace rules
        - invalid_names: List of feature names that violate rules (empty if valid)
    """
    valid_prefixes = {"ta_", "liq_", "vol_", "alt_", "macro_", "regime_", "ml_"}
    invalid = []
    
    for name in FEATURE_REGISTRY.keys():
        # Check prefix
        has_valid_prefix = any(name.startswith(prefix) for prefix in valid_prefixes)
        if not has_valid_prefix:
            invalid.append(f"{name}: invalid prefix (must start with {valid_prefixes})")
            continue
        
        # Check version suffix
        if not name.endswith("_v1") and not name.endswith("_v2"):
            # Allow _v{number} pattern
            parts = name.rsplit("_v", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                invalid.append(f"{name}: invalid version suffix (must end with _v{{number}})")
    
    is_valid = len(invalid) == 0
    return is_valid, invalid


def get_feature_metadata(feature_name: str) -> dict[str, Any] | None:
    """Get metadata for a feature by name.

    Args:
        feature_name: Feature name (e.g., "ta_rsi_14_v1")

    Returns:
        Metadata dict or None if feature not found
    """
    return FEATURE_REGISTRY.get(feature_name)


def list_features_by_namespace(namespace: str) -> list[str]:
    """List all features in a given namespace.

    Args:
        namespace: Namespace prefix (e.g., "ta", "liq", "vol")

    Returns:
        List of feature names in namespace
    """
    prefix = f"{namespace}_"
    return [name for name in FEATURE_REGISTRY.keys() if name.startswith(prefix)]


def list_all_feature_names() -> list[str]:
    """List all registered feature names.

    Returns:
        List of all feature names (sorted)
    """
    return sorted(FEATURE_REGISTRY.keys())
