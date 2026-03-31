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
- Keine globalen Variablen (außer Konstanten in `config.py` / `config/`)
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

**Test-Pattern:**
- Verwende `tmp_path` Fixture für temporäre Dateien
- Verwende `monkeypatch` für Config-Overrides
- Synthetische Daten für schnelle Unit-Tests

---

### Logging

Das Projekt verfügt über eine **zentrale Logging-Infrastruktur**:

- `src/assembled_core/logging_config.py` — vollständiges Setup mit Run-IDs, Console-Handler, File-Handler (`logs/`)
- `src/assembled_core/logging_utils.py` — vereinfachtes Setup für leichtere Scripts (kein Run-ID, kein File-Handler)

**Pflichtregeln für produktive Module (`src/assembled_core/**`):**

- Produktiver Code nutzt **ausschließlich** Python-`logging`, nie `print()` für operative Statusmeldungen
- Jedes Modul holt sich einen Logger via `logging.getLogger(__name__)`
- `print()` ist nur zulässig für: temporäre lokale Diagnose, explizit nicht-produktiven Code oder einmalige CLI-Ausgaben ohne operativen Informationswert
- Es darf keine Doppelstruktur aus `logger.*` + `print()` entstehen

**Standard-Muster für Core-Module:**
```python
import logging

logger = logging.getLogger(__name__)

def some_function(...):
    logger.info("Pipeline started for freq=%s", freq)
    logger.warning("Missing data for symbol %s", symbol)
    logger.error("Failed to load prices: %s", path)
```

**Standard-Muster für Entry-Point-Scripts (`scripts/`):**
```python
from src.assembled_core.logging_config import generate_run_id, setup_logging
import logging

run_id = generate_run_id(prefix="eod")
setup_logging(run_id=run_id, level="INFO")
logger = logging.getLogger(__name__)
```

**Log-Formate (durch `logging_config.py` vorgegeben):**
- Console: `[LEVEL] message`
- File (`logs/<run_id>.log`): `timestamp | level | module | [run_id] | message`

**Was zu vermeiden ist:**
- `print(f"[EXEC] START ...")` in produktiven Modulen
- Freitext-Prints ohne Kontext als Laufzeit-Status
- Debug-`print()`-Calls dauerhaft im Produktivpfad belassen
- Secrets, API-Keys oder sensible Werte in Log-Messages

---

## Umgang mit Daten

### Öffentliche Daten Only

**Erlaubt:**
- Marktdaten (Preise, Volumen) von öffentlichen APIs (Yahoo Finance, Alpha Vantage, Twelve Data)
- Öffentlich verfügbare Fundamentaldaten (SEC Filings)
- Öffentliche News-Feeds, öffentliche Insider-Transaktionsdaten (SEC Form 4)
- Öffentliche Kongresshandel-Daten (House Stock Watcher, Senate Disclosure)

**Verboten:**
- **MNPI (Material Non-Public Information)** — niemals verwenden oder speichern
- Insider-Informationen, die nicht öffentlich sind
- Vertrauliche Daten von Dritten
- Persönliche Daten ohne Einwilligung

**Regel:** Wenn unsicher, ob Daten öffentlich sind → **NICHT verwenden**.

---

### Daten-Speicherung

**Lokale Dateien:**
- Rohdaten: `data/raw/` (nicht in Git)
- Aggregierte Daten: `output/aggregates/` (nicht in Git)
- Orders, Reports, Equity-Curves: `output/` (nicht in Git)
- Factor-Cache: `data/factors/` (nicht in Git)

**Git-Ignore:**
- `data/` — alle Rohdaten (Hinweis: matcht auch `src/assembled_core/data/`, daher ggf. `git add -f` nötig)
- `output/` — alle Pipeline-Outputs
- `logs/` — alle Log-Dateien

**Regel:** Keine großen Daten-Dateien in Git committen.

---

### Secrets & API-Keys

**Verboten:**
- API-Keys im Code hardcoden
- Secrets in Git committen (auch nicht in `.env`, die versehentlich committed wird)
- Passwörter in Konfigurationsdateien

**Erlaubt:**
- API-Keys als Umgebungsvariablen (z. B. `ASSEMBLED_FINNHUB_API_KEY`)
- `.env`-Dateien — müssen in `.gitignore` stehen und dürfen nie committed werden
- Konfigurationsdateien mit Platzhaltern

**Regel:** Niemals Secrets im Code oder in versionierten Dateien.

---

## Research vs. Production

### Production Code

**Ort:** `src/assembled_core/`

**Anforderungen:**
- Vollständige Type-Annotations
- Docstrings
- Tests
- Logging via `logging`-Modul (kein `print()`)
- Stabile APIs ohne unkommentierte Breaking Changes

**Regel:** Nur getestete, dokumentierte Code in `src/assembled_core/`.

---

### Experimental Code

**Orte:**
- `scripts/dev/` — Entwicklungs-Scripts
- `notes/` — Notizen und Skizzen
- `tmp_*.py` — temporäre Scripts (nicht in Git)

**Regel:** Experimenteller Code gehört **nicht** in `src/assembled_core/`.

---

## Architektur-Respekt

### Bestehende Architektur

**Respektieren:**
- Modul-Struktur in `src/assembled_core/`
- Datenfluss (siehe `docs/ARCHITECTURE_BACKEND.md`)
- API-Endpoints (siehe `docs/backend_api.md`)
- Pipeline-Schritte (siehe `docs/eod_pipeline.md`)
- CLI-Referenz (siehe `docs/CLI_REFERENCE.md`)

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
- [ ] Logging via `logging`-Modul, kein operatives `print()` in produktivem Code
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
- Bei Fragen zu Daten-Handling oder Logging
- Bei Code-Reviews

**Weiterführende Regeln:**
- `@01-backend-overview` — Projekt-Übersicht und Architektur
