# Packaging & Installation

## Package-Struktur

Das Projekt verwendet ein `src/`-Layout mit dem Package `assembled_core`:

```
src/
  assembled_core/
    __init__.py          # Package-Initialisierung
    config.py            # Zentrale Konfiguration
    costs.py             # Cost-Model-Konfiguration
    ema_config.py        # EMA-Parameter-Konfiguration
    pipeline/            # Trading-Pipeline-Module
    api/                 # FastAPI-Endpoints
    qa/                  # QA/Health-Checks
```

## Installation

### Lokale Entwicklung (Editable Install)

Für die Entwicklung sollte das Package im "editable" Modus installiert werden:

```bash
pip install -e .
```

Dies macht das Package importierbar, ohne dass Änderungen am Code neu installiert werden müssen.

### Verifizierung

Nach der Installation kann das Package importiert werden:

```bash
python -c "import assembled_core; print(assembled_core.__file__)"
```

Erwartete Ausgabe:
```
F:\Python_Projekt\Aktiengerüst\src\assembled_core\__init__.py
```

### Package-Informationen

```bash
python -c "import assembled_core; print(assembled_core.__version__)"
```

Erwartete Ausgabe:
```
0.0.1
```

## Abhängigkeiten

Die Abhängigkeiten sind in `pyproject.toml` definiert. Hauptdependencies:

- **pandas, numpy**: Datenverarbeitung
- **pyarrow, fastparquet**: Parquet-Dateiformat
- **fastapi, uvicorn, pydantic**: API-Backend
- **yfinance**: Marktdaten-Download
- **matplotlib**: Visualisierung

Vollständige Liste siehe `pyproject.toml` → `[project]` → `dependencies`.

## Build & Distribution

### Build (für Distribution)

```bash
python -m build
```

Erstellt `dist/assembled_trading_core-0.0.1-py3-none-any.whl` und Source-Distribution.

### Installation aus Wheel

```bash
pip install dist/assembled_trading_core-0.0.1-py3-none-any.whl
```

## Entwicklung

### Dev-Dependencies

Für Entwicklung (Tests, Linting):

```bash
pip install -e ".[dev]"
```

Installiert zusätzlich:
- `pytest`: Test-Framework
- `pytest-cov`: Coverage-Reports
- `black`: Code-Formatierung
- `ruff`: Linting

### Tests ausführen

```bash
pytest tests/
```

### Code-Formatierung

```bash
black src/ tests/
```

### Linting

```bash
ruff check src/ tests/
```

## Projekt-Metadaten

- **Name**: `assembled-trading-core`
- **Version**: `0.0.1`
- **Python**: `>=3.10`
- **Build-System**: `setuptools` (via `pyproject.toml`)

## Hinweise

- Das Package verwendet ein `src/`-Layout für bessere Test-Isolation
- Alle Imports sollten `from assembled_core import ...` verwenden (nach Installation)
- Für lokale Entwicklung ohne Installation: `sys.path.insert(0, str(ROOT))` in Scripts (wie in `scripts/*.py`)

