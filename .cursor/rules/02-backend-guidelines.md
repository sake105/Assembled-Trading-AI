# Backend Guidelines - Assembled Trading AI

## Coding-Guidelines

### Code-Qualität

**Type Hints:**
- Alle Funktionen müssen vollständige Type-Annotations haben
- Rückgabetypen explizit angeben: `-> pd.DataFrame`, `-> dict[str, float]`, etc.
- Optional-Typen verwenden: `str | None`, `Path | None`

**Docstrings:**
- Alle Funktionen müssen Docstrings haben (Google-Style oder NumPy-Style)
- Beschreibe: Args, Returns, Side Effects, Raises
- Beispiel:
  ```python
  def load_prices(freq: str, price_file: Path | None = None) -> pd.DataFrame:
      """Load price data for a given frequency.
      
      Args:
          freq: Frequency string ("1d" or "5min")
          price_file: Optional explicit path to price file
      
      Returns:
          DataFrame with columns: timestamp (UTC), symbol, close
      
      Raises:
          FileNotFoundError: If price file not found
          ValueError: If schema is invalid
      """
  ```

**Funktions-Größe:**
- Kleine, testbare Funktionen (max. ~50 Zeilen)
- Keine globalen Variablen (außer Konstanten in `config.py`)
- Pure Functions bevorzugen (keine Seiteneffekte, außer explizit dokumentiert)

**Imports:**
- Standard-Imports: `import pandas as pd`, `import numpy as np`
- Path-Handling: `from pathlib import Path`
- Type Hints: `from typing import Any, Literal`
- Relative Imports innerhalb `src/assembled_core/`: `from .config import OUTPUT_DIR`

---

### Tests

**Test-Pflicht:**
- Neue Module müssen Tests haben (`tests/test_*.py`)
- Smoke-Tests für kritische Funktionen
- Integration-Tests für Pipeline-Schritte

**Test-Struktur:**
- `tests/test_io_smoke.py` - I/O-Funktionen
- `tests/test_signals_ema.py` - Signal-Generierung
- `tests/test_backtest_portfolio_smoke.py` - Backtest/Portfolio
- `tests/test_api_smoke.py` - FastAPI-Endpoints
- `tests/test_qa_health.py` - QA-Checks

**Test-Pattern:**
- Verwende `tmp_path` Fixture für temporäre Dateien
- Verwende `monkeypatch` für Config-Overrides
- Synthetische Daten für schnelle Tests

---

### Logging

**Logging-Stil:**
- Knapp und präzise: `[TAG] message`
- Tags: `[EXEC]`, `[BT9]`, `[PF10]`, `[EOD]`, `[API]`
- Keine GUI-Logs, nur stdout/stderr
- Keine Debug-Logs in Produktionscode (nur in Dev-Scripts)

**Beispiel:**
```python
print(f"[EXEC] START Execution | freq={freq}")
print(f"[EXEC] [OK] written: {out_path} | rows={len(orders)}")
print("[EXEC] DONE Execution")
```

---

## Umgang mit Daten

### Öffentliche Daten Only

**Erlaubt:**
- Marktdaten (Preise, Volumen) von öffentlichen APIs (Yahoo Finance, Alpha Vantage)
- Öffentlich verfügbare Fundamentaldaten (SEC Filings)
- Öffentliche News-Feeds
- Öffentliche Insider-Transaktions-Daten (SEC Form 4)

**Verboten:**
- **MNPI (Material Non-Public Information)** - Niemals verwenden oder speichern
- Insider-Informationen, die nicht öffentlich sind
- Vertrauliche Daten von Dritten
- Persönliche Daten ohne Einwilligung

**Regel:** Wenn unsicher, ob Daten öffentlich sind → **NICHT verwenden**.

---

### Daten-Speicherung

**Lokale Dateien:**
- Rohdaten: `data/raw/1min/*.parquet` (nicht in Git)
- Aggregierte Daten: `output/aggregates/*.parquet` (nicht in Git)
- Orders: `output/orders_{freq}.csv` (nicht in Git)
- Reports: `output/*.md` (nicht in Git)

**Git-Ignore:**
- `data/` - Alle Rohdaten
- `output/` - Alle Pipeline-Outputs
- `logs/` - Alle Log-Dateien
- `*.zip`, `*.log` - Große/Artefakt-Dateien

**Regel:** Keine großen Daten-Dateien in Git committen.

---

### Secrets & API-Keys

**Verboten:**
- API-Keys im Code hardcoden
- Secrets in Git committen
- Passwörter in Konfigurationsdateien (außer `.gitignore`)

**Erlaubt:**
- API-Keys als Umgebungsvariablen: `$env:ALPHAVANTAGE_API_KEY`
- Konfigurationsdateien mit Platzhaltern: `config/datasource.psd1`
- `.env`-Dateien (in `.gitignore`)

**Regel:** Niemals Secrets im Code oder in versionierten Dateien.

---

## Research vs. Production

### Production Code

**Ort:** `src/assembled_core/`

**Anforderungen:**
- Vollständige Type-Annotations
- Docstrings
- Tests
- Keine Experimente
- Stabile APIs

**Regel:** Nur getestete, dokumentierte Code in `src/assembled_core/`.

---

### Experimental Code

**Orte:**
- `scripts/dev/` - Entwicklungs-Skripte
- `scripts/experiments/` - Experimente (falls vorhanden)
- `notes/` - Notizen und Skizzen
- `tmp_*.py` - Temporäre Skripte (nicht in Git)

**Regel:** Experimenteller Code gehört **nicht** in `src/assembled_core/`.

---

## Architektur-Respekt

### Bestehende Architektur

**Respektieren:**
- Module-Struktur in `src/assembled_core/`
- Datenfluss (siehe `ARCHITECTURE_BACKEND.md`)
- API-Endpoints (siehe `backend_api.md`)
- Pipeline-Schritte (siehe `eod_pipeline.md`)

**Nicht ändern ohne Grund:**
- Bestehende Funktions-Signaturen
- Output-Dateiformate
- API-Response-Modelle
- Konfigurations-Pfade

**Regel:** Bei Unsicherheit → Architektur-Docs lesen, dann fragen.

---

### Breaking Changes

**Vermeiden:**
- Änderungen, die bestehende Scripts brechen
- Änderungen an Output-Schemas ohne Migration
- Änderungen an API-Endpoints ohne Versionierung

**Wenn nötig:**
- Deprecation-Warnings hinzufügen
- Migration-Pfad dokumentieren
- Backwards-Kompatibilität wahren (wenn möglich)

**Regel:** Breaking Changes nur mit expliziter Anweisung.

---

## Code-Review-Checkliste

**Vor jedem Commit prüfen:**
- [ ] Type-Annotations vollständig
- [ ] Docstrings vorhanden
- [ ] Tests vorhanden (für neue Module)
- [ ] Keine Secrets im Code
- [ ] Keine großen Daten-Dateien
- [ ] Architektur-Docs respektiert
- [ ] Bestehende Funktionalität nicht gebrochen

---

## Verwendung in Cursor

**Diese Regel referenzieren:**
```
@02-backend-guidelines
```

**Wann verwenden:**
- Bei Code-Änderungen
- Bei Unsicherheit über Coding-Standards
- Bei Fragen zu Daten-Handling
- Bei Code-Reviews

**Weiterführende Regeln:**
- `@01-backend-overview` - Projekt-Übersicht und Architektur

