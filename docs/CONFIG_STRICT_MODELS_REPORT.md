# Strict Config Models - Implementation Report

**Status:** Implementiert (Phase 0.3)
**Datum:** 2025-01-04

## Zusammenfassung

Alle zentralen `*_config`-Dicts wurden durch Pydantic BaseModels ersetzt und abgesichert. Die Implementierung ist **backward-compatible**: Call-Sites können weiterhin Dicts übergeben, die werden sofort validiert.

---

## Implementierte Models

### 1. FeatureConfig

**Zweck:** Konfiguration für Technical Analysis Feature-Computation.

**Keys:**
- `ma_windows`: `tuple[int, ...]` = `(20, 50, 200)` (Default)
- `atr_window`: `int` = `14` (Default, >= 1)
- `rsi_window`: `int` = `14` (Default, >= 1)
- `include_rsi`: `bool` = `True` (Default)

**Constraints:**
- Alle MA-Windows müssen > 0 sein
- `ma_windows` darf nicht leer sein
- `atr_window` und `rsi_window` müssen >= 1 sein

**Extra Policy:** `extra="forbid"` (unbekannte Keys werden abgelehnt)

**Verwendung:**
```python
from src.assembled_core.config.models import FeatureConfig, ensure_feature_config

# Als Dict (backward compatible)
config = ensure_feature_config({"ma_windows": (10, 30), "atr_window": 20})

# Als Model
config = FeatureConfig(ma_windows=(10, 30), atr_window=20)
```

---

### 2. SignalConfig

**Zweck:** Konfiguration für Signal-Generierung (flexibel für verschiedene Strategien).

**Keys:**
- `ma_fast`: `int | None` = `None` (Default, >= 1)
- `ma_slow`: `int | None` = `None` (Default, >= 1)

**Constraints:**
- Wenn beide `ma_fast` und `ma_slow` gesetzt sind: `ma_slow > ma_fast`

**Extra Policy:** `extra="allow"` (erlaubt strategy-spezifische Parameter)

**Verwendung:**
```python
from src.assembled_core.config.models import SignalConfig, ensure_signal_config

# Als Dict (backward compatible)
config = ensure_signal_config({"ma_fast": 20, "ma_slow": 50})

# Als Model
config = SignalConfig(ma_fast=20, ma_slow=50)

# Mit extra keys (für strategy-spezifische Parameter)
config = SignalConfig(ma_fast=20, ma_slow=50, threshold=0.5, custom_param="value")
```

---

### 3. RiskConfig

**Zweck:** Konfiguration für Risk-Controls.

**Keys:**
- `enable_kill_switch`: `bool` = `True` (Default)
- `enable_pre_trade_checks`: `bool` = `True` (Default)

**Constraints:**
- Keine speziellen Constraints (nur bool-Werte)

**Extra Policy:** `extra="allow"` (erlaubt zukünftige Risk-Parameter)

**Verwendung:**
```python
from src.assembled_core.config.models import RiskConfig, ensure_risk_config

# Als Dict (backward compatible)
config = ensure_risk_config({"enable_kill_switch": False})

# Als Model
config = RiskConfig(enable_kill_switch=False)
```

---

### 4. GateConfig

**Zweck:** Konfiguration für QA-Gates-Evaluierung.

**Keys:**
- `sharpe`: `GateThresholdConfig` (Default: `min=1.0`, `warning=0.5`)
- `max_drawdown`: `GateThresholdConfig` (Default: `max=-20.0`, `warning=-15.0`)
- `turnover`: `GateThresholdConfig` (Default: `max=50.0`, `warning=30.0`)
- `cagr`: `GateThresholdConfig` (Default: `min=0.05`, `warning=0.0`)
- `volatility`: `GateThresholdConfig` (Default: `max=0.30`, `warning=0.25`)
- `hit_rate`: `GateThresholdConfig` (Default: `min=0.50`, `warning=0.40`)
- `profit_factor`: `GateThresholdConfig` (Default: `min=1.5`, `warning=1.2`)

**GateThresholdConfig:**
- `min`: `float | None` (Minimum-Threshold)
- `max`: `float | None` (Maximum-Threshold)
- `warning`: `float | None` (Warning-Threshold)

**Constraints:**
- Keine speziellen Constraints (nur float-Werte)

**Extra Policy:** `extra="forbid"` (unbekannte Gates werden abgelehnt)

**Verwendung:**
```python
from src.assembled_core.config.models import GateConfig, GateThresholdConfig, ensure_gate_config

# Als Dict (backward compatible)
config_dict = {
    "sharpe": {"min": 2.0, "warning": 1.0},
    "max_drawdown": {"max": -30.0, "warning": -25.0},
}
config = ensure_gate_config(config_dict)

# Als Model
config = GateConfig(
    sharpe=GateThresholdConfig(min=2.0, warning=1.0),
    max_drawdown=GateThresholdConfig(max=-30.0, warning=-25.0),
)

# Konvertierung zu Dict-Format (für evaluate_all_gates)
gate_dict = config.to_dict()
```

---

## Helper-Funktionen

Alle Helper-Funktionen akzeptieren `dict | BaseModel | None` und validieren/konvertieren:

- `ensure_feature_config()`: Konvertiert zu `FeatureConfig | None`
- `ensure_signal_config()`: Konvertiert zu `SignalConfig` (nie None)
- `ensure_risk_config()`: Konvertiert zu `RiskConfig` (nie None)
- `ensure_gate_config()`: Konvertiert zu `GateConfig` (nie None)

**Backward Compatibility:** Alle Funktionen akzeptieren Dicts und validieren sie sofort.

---

## Integration

### TradingContext

**Änderungen:**
- `feature_config`: `dict[str, Any] | FeatureConfig | None` (akzeptiert beide)
- `signal_config`: `dict[str, Any] | SignalConfig` (akzeptiert beide)
- `risk_config`: `dict[str, Any] | RiskConfig` (akzeptiert beide)

**Validierung:**
- Configs werden in `run_trading_cycle()` validiert (via `ensure_*_config()`)
- Dicts werden zu Models konvertiert, Models werden direkt verwendet
- Kein Behavior-Change: Defaults bleiben gleich

### Entry-Points

**Status:** Noch nicht vollständig integriert (kann schrittweise erfolgen)

**Empfohlene Integration:**
- `scripts/run_backtest_strategy.py`: Validierung bei CLI-Argument-Parsing
- `scripts/run_daily.py`: Validierung bei Config-Erstellung
- `scripts/batch_runner.py`: Validierung bei Batch-Config-Laden

**Hinweis:** Da TradingContext bereits validiert, funktionieren bestehende Entry-Points weiterhin.

---

## Tests

**Datei:** `tests/test_config_strict_models.py`

**Coverage:**
- ✅ Defaults stimmen (erwarteter Wert)
- ✅ Ranges greifen (invalid -> ValidationError)
- ✅ Unknown keys werfen Fehler (extra forbid)
- ✅ Helper-Funktionen akzeptieren dict | BaseModel | None
- ✅ TradingContext backward compatibility
- ✅ TradingContext mit Models

**Ergebnis:** 26 Tests, alle passed ✅

---

## Bestehende Runs

**Status:** Unverändert (backward-compatible)

**Grund:**
- TradingContext akzeptiert weiterhin Dicts
- Dicts werden intern validiert und zu Models konvertiert
- Defaults bleiben gleich
- Keine Breaking Changes

**Beispiel:**
```python
# Alt (funktioniert weiterhin)
ctx = TradingContext(
    prices=prices,
    feature_config={"ma_windows": (20, 50, 200)},
    signal_config={"ma_fast": 20, "ma_slow": 50},
)

# Neu (auch möglich)
ctx = TradingContext(
    prices=prices,
    feature_config=FeatureConfig(ma_windows=(20, 50, 200)),
    signal_config=SignalConfig(ma_fast=20, ma_slow=50),
)
```

---

## Code-Qualität

**Linting:**
- ✅ `ruff check` passed
- ✅ `py_compile` passed

**Tests:**
- ✅ `pytest tests/test_config_strict_models.py` passed (26/26)
- ✅ `pytest tests/test_pipeline_trading_cycle_contract.py` passed (bestehende Tests)

---

## Nächste Schritte (Optional)

1. **Entry-Point Integration:** Validierung in CLI/Scripts hinzufügen (frühe Fehlererkennung)
2. **Dokumentation:** API-Docs für Config-Models aktualisieren
3. **Migration Guide:** Anleitung für Migration von Dicts zu Models (optional)

---

## Dateien

**Neu erstellt:**
- `src/assembled_core/config/models.py` - Pydantic Models
- `tests/test_config_strict_models.py` - Tests

**Geändert:**
- `src/assembled_core/config/__init__.py` - Exports hinzugefügt
- `src/assembled_core/pipeline/trading_cycle.py` - Integration in TradingContext

---

## Zusammenfassung

✅ **Alle Config-Dicts durch Pydantic Models ersetzt**
✅ **Backward-compatible: Dicts werden weiterhin akzeptiert**
✅ **Validierung: Unknown keys werden abgelehnt (extra="forbid")**
✅ **Defaults: Unverändert (bestehende Runs funktionieren weiterhin)**
✅ **Tests: 26 Tests, alle passed**
✅ **Code-Qualität: Linting passed**

**Status:** ✅ DoD erfüllt
