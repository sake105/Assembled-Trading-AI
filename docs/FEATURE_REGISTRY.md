# Feature Registry - Naming Rules & Versioning (Sprint 5 / F2)

**Status:** Implementiert  
**Last Updated:** 2025-01-04  
**ASCII-only:** Yes

---

## Overview

Das **Feature Registry** System stellt sicher, dass alle Feature-Spalten eindeutig, namespaced und versioniert sind. Dies verhindert Kollisionen und ermoeglicht stabile, reproduzierbare Feature-Namen.

### Ziele

- **Eindeutigkeit**: Jede Feature-Spalte muss eindeutig sein (keine Kollisionen)
- **Namespace**: Features sind nach Kategorie organisiert (ta_, liq_, vol_, alt_, etc.)
- **Versionierung**: Features sind versioniert (z.B. ta_rsi_v1, ta_rsi_v2)
- **Dokumentation**: Jedes Feature hat Metadaten (description, inputs, version, layer)

---

## Naming Rules

### Prefixes (Namespaces)

| Prefix | Beschreibung | Beispiele |
|--------|--------------|-----------|
| `ta_` | Technical Analysis | `ta_rsi_14_v1`, `ta_ma_20_v1` |
| `liq_` | Liquidity | `liq_turnover_v1`, `liq_volume_zscore_v1` |
| `vol_` | Volatility | `vol_rv_20_v1`, `vol_vov_20_60_v1` |
| `alt_` | Alternative Data | `alt_insider_net_buy_20d_v1`, `alt_news_sentiment_7d_v1` |
| `macro_` | Macroeconomic | `macro_gdp_growth_v1` (zukunft) |
| `regime_` | Regime Detection | `regime_trend_strength_20_v1` |
| `ml_` | Machine Learning | `ml_embedding_128_v1` (zukunft) |

### Versionierung

Format: `{prefix}_{feature_name}_v{version}`

Beispiele:
- `ta_rsi_14_v1`: RSI mit window 14, Version 1
- `ta_ma_20_v1`: Moving Average mit window 20, Version 1
- `ta_rsi_14_v2`: RSI mit window 14, Version 2 (verbesserte Berechnung)

**Regeln:**
- Version muss numerisch sein (v1, v2, v3, ...)
- Neue Versionen muessen explizit registriert werden
- Alte Versionen bleiben stabil (keine Breaking Changes)

### Feature Name Structure

```
{namespace}_{feature_type}_{parameters}_v{version}
```

Beispiele:
- `ta_rsi_14_v1`: Technical Analysis, RSI, window=14, version 1
- `ta_ma_20_v1`: Technical Analysis, Moving Average, window=20, version 1
- `alt_insider_net_buy_20d_v1`: Alternative Data, Insider Net Buy, 20-day window, version 1
- `vol_rv_20_v1`: Volatility, Realized Volatility, window=20, version 1

---

## Registry Structure

Das Registry-System ist in `src/assembled_core/features/registry.py` implementiert.

### FEATURE_REGISTRY

```python
FEATURE_REGISTRY: dict[str, dict[str, Any]] = {
    "ta_rsi_14_v1": {
        "description": "Relative Strength Index (Wilder) with window 14",
        "inputs": ["close"],
        "version": 1,
        "layer": "ta",
        "namespace": "ta",
    },
    # ... weitere Features
}
```

### Metadata Schema

Jedes Feature hat folgende Metadaten:

- **description**: str - Human-readable Beschreibung
- **inputs**: list[str] - Erforderliche Input-Spalten (z.B. ["close"], ["high", "low", "close"])
- **version**: int - Versionsnummer (default: 1)
- **layer**: str - Feature-Layer ("ta", "liq", "vol", "alt", "macro", "regime", "ml")
- **namespace**: str - Namespace-Prefix (z.B. "ta")

---

## Validation Functions

### validate_registry_unique()

Validiert, dass alle Feature-Namen eindeutig sind.

```python
from src.assembled_core.features.registry import validate_registry_unique

is_valid, duplicates = validate_registry_unique()
if not is_valid:
    print(f"Duplicate features: {duplicates}")
```

### validate_registry_documented()

Validiert, dass alle Features erforderliche Metadaten haben.

```python
from src.assembled_core.features.registry import validate_registry_documented

is_valid, missing = validate_registry_documented()
if not is_valid:
    print(f"Missing metadata: {missing}")
```

### validate_registry_namespaced()

Validiert, dass alle Feature-Namen den Namespace-Regeln folgen.

```python
from src.assembled_core.features.registry import validate_registry_namespaced

is_valid, invalid = validate_registry_namespaced()
if not is_valid:
    print(f"Invalid names: {invalid}")
```

---

## Registered Features

### Technical Analysis (ta_)

| Feature Name | Description | Inputs |
|--------------|-------------|--------|
| `ta_log_return_v1` | Logarithmic return | `["close"]` |
| `ta_ma_20_v1` | Simple Moving Average (window 20) | `["close"]` |
| `ta_ma_50_v1` | Simple Moving Average (window 50) | `["close"]` |
| `ta_ma_200_v1` | Simple Moving Average (window 200) | `["close"]` |
| `ta_atr_14_v1` | Average True Range (window 14) | `["high", "low", "close"]` |
| `ta_rsi_14_v1` | Relative Strength Index (window 14) | `["close"]` |

### Volatility (vol_)

| Feature Name | Description | Inputs |
|--------------|-------------|--------|
| `vol_rv_20_v1` | Realized Volatility (window 20) | `["close"]` |
| `vol_vov_20_60_v1` | Volatility of Volatility | `["close"]` |

### Liquidity (liq_)

| Feature Name | Description | Inputs |
|--------------|-------------|--------|
| `liq_turnover_v1` | Turnover (volume * price / market_cap) | `["volume", "close"]` |
| `liq_volume_zscore_v1` | Volume Z-score | `["volume"]` |
| `liq_spread_proxy_v1` | Spread proxy ((high - low) / close) | `["high", "low", "close"]` |

### Alternative Data (alt_)

| Feature Name | Description | Inputs |
|--------------|-------------|--------|
| `alt_insider_net_buy_20d_v1` | Insider net buy (20-day) | `["insider_events"]` |
| `alt_insider_trade_count_20d_v1` | Insider trade count (20-day) | `["insider_events"]` |
| `alt_congress_trade_count_60d_v1` | Congress trade count (60-day) | `["congress_events"]` |
| `alt_news_sentiment_7d_v1` | News sentiment (7-day) | `["news_events"]` |
| `alt_shipping_congestion_score_v1` | Shipping congestion score | `["shipping_events"]` |

### Regime (regime_)

| Feature Name | Description | Inputs |
|--------------|-------------|--------|
| `regime_trend_strength_20_v1` | Trend strength ((price - MA_20) / ATR_20) | `["close", "ta_ma_20_v1", "ta_atr_14_v1"]` |
| `regime_fraction_above_ma_50_v1` | Fraction of symbols above MA_50 | `["close", "ta_ma_50_v1"]` |

---

## Migration from Legacy Names

Aktuelle Feature-Namen (ohne Namespace) werden schrittweise migriert:

**Legacy -> New:**
- `log_return` -> `ta_log_return_v1`
- `ma_20` -> `ta_ma_20_v1`
- `ma_50` -> `ta_ma_50_v1`
- `ma_200` -> `ta_ma_200_v1`
- `atr_14` -> `ta_atr_14_v1`
- `rsi_14` -> `ta_rsi_14_v1`

**Kompatibilitaet:**
- Legacy-Namen werden zunaechst als Alias unterstuetzt (deprecation)
- Neue Features muessen namespaced sein
- Alte Features werden schrittweise migriert

---

## Usage

### Register New Feature

```python
from src.assembled_core.features.registry import FEATURE_REGISTRY

FEATURE_REGISTRY["ta_macd_v1"] = {
    "description": "MACD (Moving Average Convergence Divergence)",
    "inputs": ["close"],
    "version": 1,
    "layer": "ta",
    "namespace": "ta",
}
```

### Get Feature Metadata

```python
from src.assembled_core.features.registry import get_feature_metadata

metadata = get_feature_metadata("ta_rsi_14_v1")
print(metadata["description"])  # "Relative Strength Index (Wilder) with window 14"
```

### List Features by Namespace

```python
from src.assembled_core.features.registry import list_features_by_namespace

ta_features = list_features_by_namespace("ta")
# ["ta_log_return_v1", "ta_ma_20_v1", "ta_ma_50_v1", ...]
```

---

## Validation in Tests

Tests validieren automatisch:
- Eindeutigkeit (keine Duplikate)
- Dokumentation (alle Metadaten vorhanden)
- Namespace-Regeln (gueltige Prefixe, Versionierung)

```python
def test_feature_registry_unique_and_documented():
    is_unique, duplicates = validate_registry_unique()
    assert is_unique, f"Duplicate features: {duplicates}"
    
    is_documented, missing = validate_registry_documented()
    assert is_documented, f"Missing metadata: {missing}"
    
    is_namespaced, invalid = validate_registry_namespaced()
    assert is_namespaced, f"Invalid names: {invalid}"
```

---

## References

- Implementation: `src/assembled_core/features/registry.py`
- Tests: `tests/test_feature_registry.py`
- Feature Generation: `src/assembled_core/features/ta_features.py`
