# Phase 9 – Model Governance & Validation

## Übersicht

Phase 9 führt Model Governance und strukturierte Validierungsprozesse ein. Ziel ist es, Modelle systematisch zu dokumentieren, zu validieren und ihre Qualität zu überwachen – im Stil von Bank- und Hedgefonds-Standards.

**Komponenten:**
- **Model Inventory**: Zentrale Übersicht aller Modelle/Strategien (siehe `docs/MODEL_INVENTORY.md`)
- **Model Cards**: Detaillierte Dokumentation pro Modell (Template: `docs/models/MODEL_CARD_TEMPLATE.md`)
- **Validation Engine**: Strukturierte Validierung von Performance, Overfitting und Data Quality (`src/assembled_core/qa/validation.py`)

---

## Validation Engine

### Modul: `src/assembled_core/qa/validation.py`

Das Validation-Modul stellt strukturierte Validierungsfunktionen bereit, die Modelle auf Basis von Performance-Metriken, Overfitting-Indikatoren und Datenqualität prüfen.

### Datenstruktur

**`ModelValidationResult`** (Dataclass):
- `model_name`: Name/ID des validierten Modells
- `is_ok`: `True` wenn Validierung bestanden, `False` bei kritischen Fehlern
- `errors`: Liste von Fehlermeldungen (kritische Probleme, die Validierung scheitern lassen)
- `warnings`: Liste von Warnmeldungen (nicht-kritische Probleme)
- `metadata`: Optionales Dictionary mit zusätzlichen Validierungsdetails

### Validierungsfunktionen

#### 1. `validate_performance()`

Validiert Performance-Metriken gegen Schwellwerte:

**Eingabe:**
- `metrics`: Dictionary mit Performance-Metriken
  - `sharpe_ratio`: Sharpe Ratio (annualisiert, optional)
  - `max_drawdown_pct`: Maximaler Drawdown in Prozent (negativer Wert, z.B. -15.0)
  - `total_trades`: Gesamtanzahl Trades (optional)
  - Weitere Keys werden ignoriert
- `min_sharpe`: Minimum Sharpe Ratio Threshold (default: 1.0)
- `max_drawdown`: Maximaler Drawdown als positive Fraction (default: 0.25 = 25%)
- `min_trades`: Minimale Anzahl Trades (default: 30)

**Ausgabe:**
- `ModelValidationResult` mit `is_ok=True` wenn alle Checks bestehen

**Beispiel:**
```python
from src.assembled_core.qa.validation import validate_performance

metrics = {
    "sharpe_ratio": 1.5,
    "max_drawdown_pct": -15.0,
    "total_trades": 100
}

result = validate_performance(metrics, min_sharpe=1.0, max_drawdown=0.20)
if result.is_ok:
    print("Performance validation passed")
else:
    print(f"Errors: {result.errors}")
```

#### 2. `validate_overfitting()`

Validiert, dass ein Modell nicht overfitted ist, basierend auf dem Deflated Sharpe Ratio.

**Eingabe:**
- `deflated_sharpe`: Deflated Sharpe Ratio (adjustiert für Anzahl Trials/Parameter)
- `threshold`: Minimum Deflated Sharpe Threshold (default: 0.5)

**Ausgabe:**
- `ModelValidationResult`:
  - `is_ok=True` wenn `deflated_sharpe >= threshold` oder `deflated_sharpe is None` (nur Warning)
  - `is_ok=False` wenn `deflated_sharpe < threshold` (indiziert Overfitting)

**Beispiel:**
```python
from src.assembled_core.qa.validation import validate_overfitting

# Good: Deflated Sharpe above threshold
result = validate_overfitting(deflated_sharpe=0.8, threshold=0.5)
assert result.is_ok is True

# Overfitted: Deflated Sharpe below threshold
result = validate_overfitting(deflated_sharpe=0.3, threshold=0.5)
assert result.is_ok is False
```

#### 3. `validate_data_quality()`

Validiert Datenqualität eines Feature-DataFrames.

**Eingabe:**
- `feature_df`: DataFrame mit Feature-Spalten
- `max_missing_fraction`: Maximaler erlaubter Anteil fehlender Werte pro Spalte (default: 0.05 = 5%)

**Ausgabe:**
- `ModelValidationResult`:
  - `is_ok=True` wenn alle Spalten <= `max_missing_fraction` fehlende Werte haben
  - `is_ok=False` wenn eine Spalte den Threshold überschreitet

**Beispiel:**
```python
import pandas as pd
import numpy as np
from src.assembled_core.qa.validation import validate_data_quality

# Clean DataFrame
df = pd.DataFrame({
    "feature1": [1, 2, 3],
    "feature2": [4, 5, 6]
})
result = validate_data_quality(df)
assert result.is_ok is True

# DataFrame with too many missing values
df_dirty = pd.DataFrame({
    "feature1": [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan],
    "feature2": [4, 5, 6, 7, 8, 9, 10]
})
result = validate_data_quality(df_dirty, max_missing_fraction=0.05)
assert result.is_ok is False
```

#### 4. `run_full_model_validation()`

Führt vollständige Modell-Validierung durch, aggregiert alle Checks.

**Eingabe:**
- `model_name`: Name/ID des zu validierenden Modells
- `metrics`: Dictionary mit Performance-Metriken
- `feature_df`: Optional DataFrame mit Features für Data Quality Validation
- `deflated_sharpe`: Optional Deflated Sharpe Ratio
- `config`: Optional Konfigurations-Dictionary mit Validierungsparametern:
  - `min_sharpe`: Minimum Sharpe Ratio (default: 1.0)
  - `max_drawdown`: Maximum Drawdown als Fraction (default: 0.25)
  - `min_trades`: Minimum Anzahl Trades (default: 30)
  - `overfitting_threshold`: Deflated Sharpe Threshold (default: 0.5)
  - `max_missing_fraction`: Maximum Missing Fraction für Data Quality (default: 0.05)

**Ausgabe:**
- `ModelValidationResult` mit aggregierten Errors und Warnings aus allen Checks

**Beispiel:**
```python
from src.assembled_core.qa.validation import run_full_model_validation
from src.assembled_core.qa.metrics import compute_all_metrics
from src.assembled_core.qa.backtest_engine import run_portfolio_backtest
import pandas as pd

# Nach einem Backtest:
# 1. Backtest ausführen
backtest_result = run_portfolio_backtest(...)

# 2. Metriken berechnen
metrics_dict = {
    "sharpe_ratio": backtest_result.metrics.sharpe_ratio,
    "max_drawdown_pct": backtest_result.metrics.max_drawdown_pct,
    "total_trades": backtest_result.metrics.total_trades,
    # ... weitere Metriken
}

# 3. Features laden (falls vorhanden)
features_df = pd.read_parquet("output/features.parquet")

# 4. Vollständige Validierung durchführen
validation_result = run_full_model_validation(
    model_name="trend_baseline",
    metrics=metrics_dict,
    feature_df=features_df,
    deflated_sharpe=0.8,  # Falls berechnet
    config={
        "min_sharpe": 1.0,
        "max_drawdown": 0.20,
        "min_trades": 30
    }
)

# 5. Ergebnis prüfen
if validation_result.is_ok:
    print(f"✅ Model {validation_result.model_name} validation passed")
else:
    print(f"❌ Validation failed:")
    for error in validation_result.errors:
        print(f"  - {error}")
    
if validation_result.warnings:
    print(f"⚠️ Warnings:")
    for warning in validation_result.warnings:
        print(f"  - {warning}")
```

---

## Integration in Backtest-Workflow

### Typischer Workflow nach Backtest

1. **Backtest ausführen** (z.B. via `run_backtest_strategy.py` oder `qa.backtest_engine`)
2. **Metriken extrahieren** (aus `BacktestResult.metrics` oder `PerformanceMetrics`)
3. **Features laden** (falls für Data Quality Check benötigt)
4. **Validierung durchführen** (`run_full_model_validation()`)
5. **Ergebnis evaluieren** (`validation_result.is_ok`, Errors/Warnings prüfen)

### Beispiel: Integration in Script

```python
# In einem Backtest-Script:
from src.assembled_core.qa.backtest_engine import run_portfolio_backtest, BacktestResult
from src.assembled_core.qa.validation import run_full_model_validation

# ... Backtest durchführen ...
result: BacktestResult = run_portfolio_backtest(...)

# Metriken extrahieren
metrics = {
    "sharpe_ratio": result.metrics.sharpe_ratio,
    "max_drawdown_pct": result.metrics.max_drawdown_pct,
    "total_trades": result.metrics.total_trades,
    "cagr": result.metrics.cagr,
    "start_capital": result.metrics.start_capital,
}

# Validierung
validation = run_full_model_validation(
    model_name="trend_baseline",
    metrics=metrics,
    feature_df=features_df if "features_df" in locals() else None,
    deflated_sharpe=None,  # Optional: Falls berechnet
)

# Ergebnis
if not validation.is_ok:
    logger.error(f"Model validation failed: {validation.errors}")
    sys.exit(1)
elif validation.warnings:
    logger.warning(f"Validation warnings: {validation.warnings}")
else:
    logger.info("Model validation passed")
```

---

## Tests

**Test-Datei:** `tests/test_qa_validation.py`  
**Marker:** `@pytest.mark.phase9`

**Test-Kategorien:**
- Performance Validation Tests (OK, Bad Sharpe, Bad Drawdown, Bad Trade Count, Missing Metrics)
- Overfitting Validation Tests (OK, Fail, None, At Threshold)
- Data Quality Validation Tests (OK, Fail, Empty, Warning)
- Full Model Validation Tests (All OK, Multiple Failures, No Feature DF, Custom Config, Aggregates Warnings)

**Testlauf:**
```bash
# Nur Phase 9
pytest -m "phase9" -q

# Alle Phasen inkl. Phase 9
pytest -m "phase4 or phase6 or phase7 or phase8 or phase9" --maxfail=1 -q
```

---

## Nächste Schritte

- **Deflated Sharpe Berechnung**: Implementierung einer Funktion zur Berechnung des Deflated Sharpe Ratios
- **Integration in Reports**: Validierungsergebnisse in QA-Reports aufnehmen
- **Automatisierung**: Validierung automatisch nach jedem Backtest ausführen
- **Model Cards ausfüllen**: Für jedes Modell im Inventory eine spezifische Model Card erstellen

---

## Referenzen

- **Model Inventory**: `docs/MODEL_INVENTORY.md`
- **Model Card Template**: `docs/models/MODEL_CARD_TEMPLATE.md`
- **Validation Module**: `src/assembled_core/qa/validation.py`
- **Validation Tests**: `tests/test_qa_validation.py`

