# Testing Commands - Quick Reference

## Empfohlene Test-Kommandos

### Phase-4-Tests (Backend Core)

**Komplette Phase-4-Suite:**
```powershell
cd F:\Python_Projekt\Aktiengerüst
.\.venv\Scripts\python.exe -m pytest -m phase4
```

**Mit Dauer-Informationen:**
```powershell
.\.venv\Scripts\python.exe -m pytest -m phase4 --durations=5
```

**Erwartete Ausgabe:**
- ~117 Tests in ~13-17 Sekunden
- Alle Tests sollten grün sein

---

### Phase-6-Tests (Event Features)

**Phase-6-Suite:**
```powershell
.\.venv\Scripts\python.exe -m pytest -m phase6
```

**Erwartete Ausgabe:**
- ~29 Tests in < 2 Sekunden
- Alle Tests sollten grün sein

**Test-Dateien:**
- `tests/test_features_events_phase6.py` - Event-Feature-Tests
- `tests/test_signals_event_phase6.py` - Event-Signal-Tests
- `tests/test_run_backtest_strategy.py` - Backtest-Integration (Event-Strategie)
- `tests/test_compare_strategies_trend_vs_event.py` - Strategie-Vergleichs-Tests

---

### Phase-7-Tests (Labeling & ML Dataset Builder)

**Phase-7-Suite:**
```powershell
.\.venv\Scripts\python.exe -m pytest -m phase7
```

**Erwartete Ausgabe:**
- ~22 Tests in < 5 Sekunden
- Alle Tests sollten grün sein

**Test-Dateien:**
- `tests/test_qa_labeling.py` - Labeling-Funktionen (11 Tests)
- `tests/test_qa_dataset_builder.py` - Dataset-Builder (9 Tests)
- `tests/test_cli_ml_dataset.py` - CLI-Integration (2 Tests)

**Gezielte Tests:**
```powershell
# Nur Labeling
.\.venv\Scripts\python.exe -m pytest tests/test_qa_labeling.py -q

# Nur Dataset-Builder
.\.venv\Scripts\python.exe -m pytest tests/test_qa_dataset_builder.py -q

# Nur CLI-Tests
.\.venv\Scripts\python.exe -m pytest tests/test_cli_ml_dataset.py -q
```

---

### Phase-8-Tests (Risk Engine & Scenario Analysis)

**Phase-8-Suite:**
```powershell
.\.venv\Scripts\python.exe -m pytest -m phase8
```

**Erwartete Ausgabe:**
- ~39 Tests in < 2 Sekunden
- Alle Tests sollten grün sein

**Test-Dateien:**
- `tests/test_qa_risk_metrics.py` - Portfolio Risk Metrics (13 Tests)
- `tests/test_qa_scenario_engine.py` - Scenario Engine (13 Tests)
- `tests/test_qa_shipping_risk.py` - Shipping Risk (13 Tests)

**Gezielte Tests:**
```powershell
# Nur Risk Metrics
.\.venv\Scripts\python.exe -m pytest tests/test_qa_risk_metrics.py -q

# Nur Scenario Engine
.\.venv\Scripts\python.exe -m pytest tests/test_qa_scenario_engine.py -q

# Nur Shipping Risk
.\.venv\Scripts\python.exe -m pytest tests/test_qa_shipping_risk.py -q
```

---

### Phase-9-Tests (Model Governance & Validation)

**Phase-9-Suite:**
```powershell
.\.venv\Scripts\python.exe -m pytest -m phase9
```

**Erwartete Ausgabe:**
- ~41 Tests in < 2 Sekunden
- Alle Tests sollten grün sein

**Test-Dateien:**
- `tests/test_qa_validation.py` - Model Validation (22 Tests)
- `tests/test_qa_drift_detection.py` - Drift Detection (19 Tests)

**Gezielte Tests:**
```powershell
# Nur Validation
.\.venv\Scripts\python.exe -m pytest tests/test_qa_validation.py -q

# Nur Drift Detection
.\.venv\Scripts\python.exe -m pytest tests/test_qa_drift_detection.py -q
```

---

### Gezielte Test-Dateien

**Backtest & EOD-Pipeline:**
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_run_backtest_strategy.py tests/test_run_eod_pipeline.py -q
```

**TA-Features:**
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_features_ta.py -q
```

**QA-Metriken & Gates:**
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_qa_metrics.py tests/test_qa_gates.py -q
```

---

### Alle Tests (ohne externe Dependencies)

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not external"
```

---

## Standard-Test-Profile (QoL)

### 🚀 Schnell (für tägliche Arbeit)

**Phase 4 + 6:** Backend Core + Event Features (~146 Tests, ~18s)
```powershell
.\.venv\Scripts\python.exe -m pytest -m "phase4 or phase6" -q
```

**Mit Warnings-Check:**
```powershell
.\.venv\Scripts\python.exe -m pytest -m "phase4 or phase6" -q -W error
```

### 🔬 Vollständig (inkl. Meta & Risk)

**Alle Phasen:** Phase 4 + 6 + 7 + 8 (~206 Tests, ~19s)
```powershell
.\.venv\Scripts\python.exe -m pytest -m "phase4 or phase6 or phase7 or phase8" -q
```

**Mit Performance-Analyse:**
```powershell
.\.venv\Scripts\python.exe -m pytest -m "phase4 or phase6 or phase7 or phase8" -q --durations=10
```

**Mit harten Warnings-Check (0 Warnings erzwungen):**
```powershell
.\.venv\Scripts\python.exe -m pytest -m "phase4 or phase6 or phase7 or phase8" -W error --maxfail=1
```

---

## Wichtige Hinweise

### ❌ Nicht verwenden

**Vermeide PowerShell-Pipes mit Select-String/Select-Object:**
```powershell
# ❌ FALSCH - wirkt wie "hängt", obwohl Tests laufen
pytest ... 2>&1 | Select-String -Pattern "..." | Select-Object -Last 2
```

**Problem:**
- PowerShell wartet auf vollständigen Stream
- Keine Zwischen-Ausgabe sichtbar
- Wirkt wie Freeze, obwohl Tests normal laufen

### ✅ Richtig

**Einfache pytest-Kommandos:**
```powershell
# ✅ RICHTIG - direkte Ausgabe, sofort sichtbar
.\.venv\Scripts\python.exe -m pytest -m phase4
```

---

## Performance-Übersicht

| Test-Suite | Anzahl Tests | Dauer | Status |
|------------|--------------|-------|--------|
| Phase-4 (Backend Core) | ~117 | ~13-17s | ✅ Grün |
| Phase-6 (Event Features) | ~29 | < 2s | ✅ Grün |
| Phase-7 (Labeling & ML Dataset) | ~22 | < 5s | ✅ Grün |
| Phase-8 (Risk Engine) | ~39 | < 2s | ✅ Grün |
| Phase-9 (Model Governance) | ~41 | < 2s | ✅ Grün |
| test_run_backtest_strategy.py | 6 | ~1.4s | ✅ Grün |
| test_run_eod_pipeline.py | 2 | < 0.1s | ✅ Grün |

**Langsamste Tests (Phase-4):**
- `test_backtest_engine_optional_outputs`: ~1.6s
- `test_run_backtest_strategy_smoke`: ~1.4s
- `test_backtest_engine_with_costs`: ~1.3s

---

## Troubleshooting

### Tests hängen / keine Ausgabe

**Problem:** PowerShell-Pipe blockiert Ausgabe

**Lösung:** Direktes pytest-Kommando verwenden (siehe oben)

### Tests fehlgeschlagen

**Prüfen:**
1. Venv aktiviert? → `.\.venv\Scripts\Activate.ps1`
2. Dependencies installiert? → `pip install -e .[dev]`
3. Parquet-Engine vorhanden? → `pip install pyarrow fastparquet`

### Langsame Tests

**Langsame Tests sind mit `@pytest.mark.slow` markiert:**
```powershell
# Nur schnelle Tests
.\.venv\Scripts\python.exe -m pytest -m "phase4 and not slow"
```

---

## Integration in CI/CD

**GitHub Actions CI (.github/workflows/backend-ci.yml):**

Die CI testet alle stabilen Phasen mit harten Warnings-Check:

```yaml
- name: Run backend core tests (all phases)
  run: |
    pytest -m "phase4 or phase6 or phase7 or phase8 or phase9" -q --maxfail=1 -W error
```

**Getestete Phasen:**

- ✅ Phase 4: Backend Core (TA, QA, Backtest, Reports) – stabil und getestet
- ✅ Phase 6: Event Features (Insider, Shipping, etc.) – stabil und getestet
- ✅ Phase 7: Labeling & ML Dataset Builder – stabil und getestet
- ✅ Phase 8: Risk Engine & Scenario Analysis – stabil und getestet
- ✅ Phase 9: Model Governance & Validation – stabil und getestet
- 🔒 **-W error**: Erzwingt 0 Warnings im CI (keine neuen Warnings durchrutschen lassen)

**Lokal alle Tests ausführen:**

```powershell
# Alle Tests (inkl. Legacy)
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

**Nur Kern-Backend (wie CI):**

```powershell
.\.venv\Scripts\python.exe -m pytest -m "phase4 or phase6" -q
```

