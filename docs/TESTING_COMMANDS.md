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
- ~11 Tests in < 1 Sekunde
- Alle Tests sollten grün sein

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
| Phase-6 (Event Features) | ~11 | < 1s | ✅ Grün |
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

Die CI testet nur die stabilen Kern-Suites (Phase 4 + Phase 6):

```yaml
- name: Run backend core tests (phase4 + phase6)
  run: |
    pytest -m "phase4 or phase6" -q --maxfail=1 --disable-warnings
```

**Warum nur Phase 4 + Phase 6?**

- ✅ Phase 4: Backend Core (TA, QA, Backtest, Reports) – stabil und getestet
- ✅ Phase 6: Event Features (Insider, Shipping, etc.) – stabil und getestet
- ⚠️ Legacy-Tests (API-Smoke, alte Portfolio-Tests) sind noch nicht auf den neuen Stand gehoben und würden CI blockieren

**Lokal alle Tests ausführen:**

```powershell
# Alle Tests (inkl. Legacy)
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

**Nur Kern-Backend (wie CI):**

```powershell
.\.venv\Scripts\python.exe -m pytest -m "phase4 or phase6" -q
```

