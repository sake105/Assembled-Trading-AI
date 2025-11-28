# Backend Core - Configuration & Testing

## Übersicht

Dieses Dokument beschreibt die zentrale Konfiguration und Test-Suite für den Sprint-9/10 Pipeline-Backend.

---

## Konfiguration

### `src/assembled_core/config.py`

Das zentrale Konfigurationsmodul stellt folgende Konstanten und Funktionen bereit:

#### Konstanten

- **`BASE_DIR`**: Pfad zum Repository-Root (wird automatisch aus der Dateiposition berechnet)
- **`OUTPUT_DIR`**: Pfad zum `output/` Verzeichnis (`BASE_DIR / "output"`)
- **`SUPPORTED_FREQS`**: Tuple mit unterstützten Frequenzen `("1d", "5min")`

#### Funktionen

- **`get_output_path(*parts: str) -> Path`**: Erstellt einen Pfad innerhalb des Output-Verzeichnisses
  ```python
  from src.assembled_core.config import get_output_path
  
  # Beispiel:
  path = get_output_path("aggregates", "5min.parquet")
  # → Path("output/aggregates/5min.parquet")
  ```

- **`get_base_dir() -> Path`**: Gibt das Repository-Root-Verzeichnis zurück

### OUTPUT_DIR Bestimmung

`OUTPUT_DIR` wird automatisch aus der Position von `config.py` berechnet:

```python
# config.py liegt in: src/assembled_core/config.py
# BASE_DIR = config.py.parents[2]  # Geht 2 Ebenen hoch: src/assembled_core -> src -> repo root
OUTPUT_DIR = BASE_DIR / "output"
```

**Wichtig:** Wenn `config.py` verschoben wird, muss die Berechnung von `BASE_DIR` angepasst werden.

### Verwendung in Pipeline-Modulen

Alle Pipeline-Module nutzen `config.OUTPUT_DIR` als Standard für Output-Pfade:

- **`pipeline.io`**: `load_prices()`, `load_orders()`, etc. nutzen `OUTPUT_DIR` als Default
- **`pipeline.backtest`**: `write_backtest_report()` nutzt `OUTPUT_DIR` als Default
- **`pipeline.portfolio`**: `write_portfolio_report()` nutzt `OUTPUT_DIR` als Default
- **`pipeline.orders`**: `write_orders()` nutzt `OUTPUT_DIR` als Default

**CLI-Kompatibilität:** Alle Scripts (`sprint9_execute.py`, `sprint9_backtest.py`, `sprint10_portfolio.py`) nutzen `OUTPUT_DIR` als Default, aber erlauben weiterhin `--out` Override.

---

## Test-Suite

### Struktur

Tests befinden sich im `tests/` Verzeichnis:

```
tests/
├── __init__.py
├── test_io_smoke.py          # I/O Smoke Tests
├── test_signals_ema.py        # EMA Signal Generation Tests
├── test_backtest_portfolio_smoke.py  # Backtest/Portfolio Simulation Tests
└── test_qa_health.py         # QA Health Check Tests
```

### Test-Module

#### 1. `test_io_smoke.py`

**Zweck:** Smoke Tests für I/O-Funktionen mit existierenden Output-Dateien.

**Tests:**
- `test_load_prices_1d_if_exists()`: Lädt tägliche Preise, wenn Datei existiert
- `test_load_prices_5min_if_exists()`: Lädt 5-Minuten-Preise, wenn Datei existiert
- `test_load_prices_with_fallback_5min()`: Testet Fallback-Pfade für 5-Minuten-Daten
- `test_load_orders_1d_if_exists()`: Lädt tägliche Orders, wenn Datei existiert
- `test_load_orders_5min_if_exists()`: Lädt 5-Minuten-Orders, wenn Datei existiert

**Verhalten:** Tests werden übersprungen (skip), wenn die erwarteten Dateien nicht existieren.

#### 2. `test_signals_ema.py`

**Zweck:** Tests für EMA-Signal-Generierung mit synthetischen Daten.

**Tests:**
- `test_compute_ema_signals_produces_signals()`: Prüft, dass Signale generiert werden
- `test_compute_ema_signals_no_nans()`: Prüft, dass keine NaNs in numerischen Spalten sind
- `test_compute_ema_signals_crossovers()`: Prüft, dass Crossovers korrekt erkannt werden

**Daten:** Verwendet synthetische Preis-Daten mit 2 Symbolen (AAPL, MSFT) und 10 Timestamps.

#### 3. `test_backtest_portfolio_smoke.py`

**Zweck:** Smoke Tests für Backtest- und Portfolio-Simulation mit synthetischen Daten.

**Tests:**
- `test_simulate_equity_basic()`: Grundlegende Equity-Simulation
- `test_simulate_equity_with_trades()`: Equity-Simulation mit Trades
- `test_simulate_with_costs_basic()`: Portfolio-Simulation mit Kosten
- `test_simulate_with_costs_equity_positive()`: Prüft, dass Equity immer positiv bleibt

**Daten:** Verwendet synthetische Preis- und Order-Daten.

#### 4. `test_qa_health.py`

**Zweck:** Tests für QA Health-Check-Funktionen.

**Tests:**
- `test_check_prices_*()`: Prüft `check_prices()` mit verschiedenen Szenarien (Datei fehlt, leer, fehlende Spalten, OK)
- `test_check_orders_*()`: Prüft `check_orders()` mit verschiedenen Szenarien
- `test_check_portfolio_*()`: Prüft `check_portfolio()` mit verschiedenen Szenarien
- `test_aggregate_qa_status_*()`: Prüft `aggregate_qa_status()` mit fehlenden und vollständigen Dateien

**Daten:** Verwendet `tmp_path` Fixture für isolierte Tests mit temporären Dateien.

---

## Tests ausführen

### Voraussetzungen

1. **pytest installieren:**
   ```bash
   pip install pytest
   ```

2. **Repository-Root als Working Directory:**
   ```bash
   cd F:\Python_Projekt\Aktiengerüst
   ```

### Test-Ausführung

#### Alle Tests ausführen:
```bash
pytest tests/
```

#### Einzelne Test-Datei:
```bash
pytest tests/test_io_smoke.py
pytest tests/test_signals_ema.py
pytest tests/test_backtest_portfolio_smoke.py
pytest tests/test_qa_health.py
```

#### Einzelner Test:
```bash
pytest tests/test_signals_ema.py::test_compute_ema_signals_produces_signals
```

#### Mit ausführlicher Ausgabe:
```bash
pytest tests/ -v
```

#### Mit Coverage (optional):
```bash
pip install pytest-cov
pytest tests/ --cov=src/assembled_core --cov-report=html
```

### Erwartete Ausgabe

**Erfolgreiche Ausführung:**
```
======================== test session starts ========================
platform win32 -- Python 3.13.x
collected 12 items

tests/test_io_smoke.py ........                              [ 66%]
tests/test_signals_ema.py ...                                [ 91%]
tests/test_backtest_portfolio_smoke.py ...                   [100%]

======================== 12 passed in 2.34s ========================
```

**Mit Skips (wenn Output-Dateien fehlen):**
```
tests/test_io_smoke.py sssss                                  [ 41%]
tests/test_signals_ema.py ...                                [ 66%]
tests/test_backtest_portfolio_smoke.py ...                   [100%]

======================== 6 passed, 5 skipped in 1.23s ========================
```

### Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'src.assembled_core'`
- **Lösung:** Stelle sicher, dass du im Repository-Root-Verzeichnis bist und `sys.path` korrekt gesetzt ist (wird in Tests automatisch gemacht).

**Problem:** Tests schlagen fehl wegen fehlender Output-Dateien
- **Lösung:** Das ist normal. Tests mit `_if_exists` im Namen werden übersprungen, wenn Dateien fehlen. Führe zuerst die Pipeline aus, um Output-Dateien zu generieren.

**Problem:** `pytest` nicht gefunden
- **Lösung:** Installiere pytest: `pip install pytest` oder `python -m pip install pytest`

---

---

## QA / Health Checks

### `src/assembled_core/qa/health.py`

Das QA-Modul stellt Health-Check-Funktionen bereit, um die Qualität und Vollständigkeit der Pipeline-Outputs zu überprüfen.

#### QaCheckResult

Die `QaCheckResult` Dataclass repräsentiert das Ergebnis einer einzelnen QA-Prüfung:

```python
@dataclass
class QaCheckResult:
    name: str                    # Name der Prüfung (z.B. "prices", "orders", "portfolio")
    status: Literal["ok", "warning", "error"]  # Status der Prüfung
    message: str                 # Kurze Beschreibung des Ergebnisses
    details: dict[str, Any] | None = None  # Zusätzliche Details (optional)
```

#### Prüfungen

**1. `check_prices(freq: str, output_dir: Path | None = None) -> QaCheckResult`**

Prüft die Preis-Datei für eine gegebene Frequenz:
- Datei existiert (z.B. `output/aggregates/{freq}.parquet`)
- DataFrame ist nicht leer
- Erforderliche Spalten vorhanden: `["timestamp", "symbol", "close"]`
- Keine NaNs in kritischen Spalten

**Status:**
- `"error"`: Datei fehlt oder Spalten fehlen
- `"warning"`: Datei leer oder NaNs gefunden
- `"ok"`: Alle Prüfungen bestanden

**2. `check_orders(freq: str, output_dir: Path | None = None) -> QaCheckResult`**

Prüft die Orders-Datei für eine gegebene Frequenz:
- Datei existiert (`output/orders_{freq}.csv`)
- DataFrame ist nicht leer
- Erforderliche Spalten vorhanden: `["timestamp", "symbol", "side", "qty", "price"]`
- Gültige `side`-Werte ("BUY" oder "SELL")

**Status:**
- `"error"`: Datei fehlt oder Spalten fehlen
- `"warning"`: Datei leer oder ungültige `side`-Werte
- `"ok"`: Alle Prüfungen bestanden

**3. `check_portfolio(freq: str, output_dir: Path | None = None) -> QaCheckResult`**

Prüft die Portfolio-Equity-Datei für eine gegebene Frequenz:
- Datei existiert (`output/portfolio_equity_{freq}.csv`)
- Mindestens 5 Zeilen vorhanden
- Keine NaNs in der `equity`-Spalte
- Keine negativen oder Null-Equity-Werte

**Status:**
- `"error"`: Datei fehlt, NaNs gefunden, oder `equity`-Spalte fehlt
- `"warning"`: Zu wenige Zeilen oder nicht-positive Equity-Werte
- `"ok"`: Alle Prüfungen bestanden

#### Aggregation

**`aggregate_qa_status(freq: str, output_dir: Path | None = None) -> dict[str, Any]`**

Führt alle drei Prüfungen aus und aggregiert die Ergebnisse:

```python
{
    "freq": "1d",
    "overall_status": "ok",  # oder "warning" oder "error"
    "checks": [
        {"name": "prices", "status": "ok", "message": "...", "details": {...}},
        {"name": "orders", "status": "ok", "message": "...", "details": {...}},
        {"name": "portfolio", "status": "ok", "message": "...", "details": {...}}
    ]
}
```

**Overall Status:**
- `"error"`: Mindestens eine Prüfung hat Status `"error"`
- `"warning"`: Keine Fehler, aber mindestens eine Warnung
- `"ok"`: Alle Prüfungen bestanden

### Verwendung

**Manuelle QA-Prüfung in Python:**

```python
>>> from src.assembled_core.qa.health import aggregate_qa_status
>>> result = aggregate_qa_status("1d")
>>> print(result["overall_status"])
'ok'
>>> for check in result["checks"]:
...     print(f"{check['name']}: {check['status']} - {check['message']}")
prices: ok - Price file OK: 100 rows, 5 symbols
orders: ok - Orders file OK: 10 orders
portfolio: ok - Portfolio file OK: 50 rows, equity range [10000.00, 10050.00]
```

**Einzelne Prüfung:**

```python
>>> from src.assembled_core.qa.health import check_prices
>>> result = check_prices("5min")
>>> print(result.status)
'ok'
>>> print(result.message)
'Price file OK: 200 rows, 3 symbols'
```

**Mit custom output directory:**

```python
>>> from pathlib import Path
>>> result = aggregate_qa_status("1d", output_dir=Path("/custom/path"))
```

### Integration in Pipeline

Die QA-Checks können nach jedem Pipeline-Schritt ausgeführt werden:

```python
# Nach Execute
from src.assembled_core.qa.health import check_orders
result = check_orders("5min")
if result.status == "error":
    print(f"ERROR: {result.message}")

# Nach Portfolio
from src.assembled_core.qa.health import aggregate_qa_status
status = aggregate_qa_status("5min")
if status["overall_status"] != "ok":
    print(f"QA failed: {status['overall_status']}")
```

---

## Integration mit FastAPI (Zukunft)

Die zentrale Konfiguration wird auch in der zukünftigen FastAPI-Implementierung genutzt:

- **API-Endpoints** können `config.OUTPUT_DIR` verwenden, um Output-Dateien zu lesen
- **Dependency Injection** kann `config.OUTPUT_DIR` als Default-Parameter verwenden
- **Tests** können `config.OUTPUT_DIR` mocken, um isolierte Tests zu ermöglichen
- **QA-Endpoints** können `aggregate_qa_status()` verwenden, um Health-Checks via API bereitzustellen

---

## Weiterführende Dokumentation

- [Backend Architecture](ARCHITECTURE_BACKEND.md) - Gesamtarchitektur und Datenfluss
- [Backend Modules](BACKEND_MODULES.md) - Detaillierte Modulübersicht
- [Backend API](backend_api.md) - FastAPI-Endpoints
- [EOD Pipeline](eod_pipeline.md) - Pipeline-Orchestrierung
- [Data Sources](DATA_SOURCES_BACKEND.md) - Datenquellen-Übersicht
- [Backend Roadmap](BACKEND_ROADMAP.md) - Entwicklungs-Roadmap
- [Security & Secrets](SECURITY_SECRETS.md) - Secrets-Management und Best Practices

---

## Secrets & .env

**Wichtig:** Alle Produktions-Skripte nutzen Umgebungsvariablen für API-Keys und Secrets, niemals hardcodierte Werte.

**Details:**
- Siehe `docs/SECURITY_SECRETS.md` für vollständige Dokumentation
- API-Keys werden über Umgebungsvariablen gesetzt (z. B. `$env:ALPHAVANTAGE_API_KEY`)
- `.env`-Dateien sind lokal und werden nicht in Git getrackt (siehe `.gitignore`)
- Konfigurationsdateien (`config/datasource.psd1`) nutzen Platzhalter: `$env:VARIABLE_NAME`

**Beispiel:**
```powershell
# PowerShell: Umgebungsvariable setzen
$env:ALPHAVANTAGE_API_KEY = "your-key-here"

# Python: Aus Umgebungsvariable lesen
import os
api_key = os.getenv("ALPHAVANTAGE_API_KEY")
```

**Grundregel:** Niemals Secrets im Code oder in versionierten Dateien speichern.

---

## Test-Layer

**Wichtig:** Alle Tests sind offline-sicher und rufen keine externen APIs oder Netzwerkdienste auf.

### Test-Kategorien

**Unit-Tests (`tests/test_*.py`):**
- Schnelle, isolierte Tests einzelner Funktionen/Module
- Laufzeit: < 1 Sekunde pro Test
- Marker: `@pytest.mark.unit` (optional)
- Beispiele: `test_data_prices_ingest.py`, `test_features_ta.py`, `test_signals_ema.py`

**Smoke-Tests:**
- End-to-End-Tests für vollständige Pipeline-Läufe
- Laufzeit: Kann länger sein (mehrere Sekunden)
- Marker: `@pytest.mark.smoke`
- Beispiele: `test_run_daily_smoke.py`, `test_run_eod_pipeline.py`, `test_api_smoke.py`

**Integration-Tests:**
- Tests, die mehrere Module zusammen testen
- Marker: `@pytest.mark.integration`
- Beispiele: `test_backtest_portfolio_smoke.py`, `test_io_smoke.py`

**Externe Tests (ausgeschlossen):**
- Tests, die Netzwerkzugriffe oder externe Services benötigen
- Marker: `@pytest.mark.external`
- Standardmäßig ausgeschlossen (siehe `pytest.ini`)

### Pytest-Konfiguration

**Datei:** `pytest.ini` im Repo-Root

**Konfiguration:**
- `testpaths = tests` - Test-Verzeichnis
- `python_files = test_*.py` - Test-Datei-Muster
- Marker: `slow`, `smoke`, `unit`, `integration`, `external`
- Standard: Externe Tests ausgeschlossen (`-m "not external"`)

**Beispiel-Commands:**
```bash
# Alle Tests (außer externe)
pytest

# Nur Smoke-Tests
pytest -m smoke

# Nur schnelle Tests (keine slow/external)
pytest -m "not slow and not external"

# Alle Tests inkl. externe
pytest -m ""

# Mit Coverage (falls pytest-cov installiert)
pytest --cov=src/assembled_core --cov-report=term-missing
```

### Shared Fixtures

**Datei:** `tests/conftest.py`

**Verfügbare Fixtures:**
- `tmp_output_dir`: Temporäres Output-Verzeichnis für Tests
- `sample_universe`: Beispiel-Universe-Datei
- `sample_price_data`: Beispiel-Preis-Daten (Parquet)

**Verwendung:**
```python
def test_something(tmp_output_dir, monkeypatch):
    # Use tmp_output_dir as OUTPUT_DIR
    monkeypatch.setattr("src.assembled_core.config.OUTPUT_DIR", tmp_output_dir)
    # ... test code ...
```

### Offline-Garantie

**Regel:** Keine Tests rufen externe APIs oder Netzwerkdienste auf.

**Prüfung:**
- Keine Imports von `yfinance`, `requests`, `httpx`, `aiohttp` in Tests
- Keine HTTP/HTTPS-Requests in Tests
- Alle Daten sind synthetisch oder lokal (Parquet/CSV)

**Falls Netzwerkzugriffe nötig:**
- Tests mit `@pytest.mark.external` markieren
- Standardmäßig ausgeschlossen (siehe `pytest.ini`)
- Mocking bevorzugen statt echter Netzwerkzugriffe

---

## Code-Style & Static Checks

**Wichtig:** Code-Style und Static-Analysis-Tools helfen, Code-Qualität und Konsistenz zu gewährleisten.

### Tools

**Ruff:**
- Schneller Linter für Python (ersetzt flake8, isort, etc.)
- Prüft Syntax-Fehler, unbenutzte Imports, Code-Style

**Black:**
- Code-Formatter für konsistentes Formatting
- Automatische Formatierung nach PEP 8

**mypy:**
- Static Type Checker für Python
- Prüft Type-Annotations auf Konsistenz

### Installation

**Dev-Dependencies installieren:**
```bash
pip install -e ".[dev]"
```

Oder einzeln:
```bash
pip install ruff black mypy
```

### Verwendung

**Ruff (Linting):**
```bash
# Alle Dateien prüfen
ruff check src tests scripts

# Automatisch fixen (wo möglich)
ruff check --fix src tests scripts
```

**Black (Formatting):**
```bash
# Code formatieren
black src tests scripts

# Nur prüfen (keine Änderungen)
black --check src tests scripts
```

**mypy (Type Checking):**
```bash
# Kernmodule prüfen
mypy src/assembled_core/data src/assembled_core/features src/assembled_core/signals src/assembled_core/execution src/assembled_core/portfolio

# Alle Module (wenn gewünscht)
mypy src/assembled_core
```

### Konfiguration

**Datei:** `pyproject.toml`

**Ruff:**
- `target-version = "py310"`
- `select = ["E", "F", "I"]` - Errors, pyflakes, isort
- `line-length = 88`

**Black:**
- `target-version = ["py310"]`
- `line-length = 88`

**mypy:**
- `python_version = "3.10"`
- `strict = False` - Moderater Modus (nicht zu strikt)
- `mypy_path = "src"`

### Empfehlungen

**Vor Commits:**
- `ruff check src tests` - Offensichtliche Fehler prüfen
- `black src tests scripts` - Code formatieren
- `mypy` auf Kernmodulen - Type-Checks (optional)

**CI/CD (Zukunft):**
- Diese Checks können später als Pflicht in CI/CD-Pipelines integriert werden
- Aktuell sind sie Empfehlungen, keine Hard-Requirements

---

## Weiterführende Dokumente

