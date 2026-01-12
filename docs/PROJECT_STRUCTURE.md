# Project Structure - Assembled Trading AI

**Status:** Verbindlich (muss bei Code-Aenderungen befolgt werden)
**Letzte Aktualisierung:** 2025-01-04
**Ergaenzt:** `docs/ARCHITECTURE_LAYERING.md` (Layer-Architektur)

## Zweck

Dieses Dokument definiert die **verbindliche Repo-Struktur** fuer das Assembled Trading AI Projekt. Es legt fest, welche Dateien/Module in welche Ordner gehoeren und welche Hygiene-Regeln gelten.

**Wichtig:** Dieses Dokument ergaenzt `docs/ARCHITECTURE_LAYERING.md`, das die Layer-Architektur (Import-Regeln) definiert. Beide Dokumente muessen zusammen befolgt werden.

---

## Hauptordner-Struktur

```
Aktiengeruest/
 src/assembled_core/     # Core Backend (Python Package)
 scripts/                 # CLI Tools, PowerShell Wrappers, Dev Scripts
 tests/                   # Test Suite
 docs/                    # Dokumentation
 configs/                 # Konfigurationsdateien (YAML/JSON)
 data/                   # Rohdaten (nicht in Git)
 output/                 # Generierte Artefakte (nicht in Git)
 logs/                   # Log-Dateien (nicht in Git)
 .venv/                  # Python Virtual Environment (nicht in Git)
```

---

## 1. `src/assembled_core/` - Core Backend

### Zweck
Haupt-Python-Package mit allen Backend-Modulen. Organisiert nach Layer-Architektur (siehe `docs/ARCHITECTURE_LAYERING.md`).

### Struktur
```
src/assembled_core/
 __init__.py
 config/          # Layer 0: Shared Configuration
 utils/           # Layer 0: Shared Utilities
 data/            # Layer 1: Data Ingestion
 features/        # Layer 2: Feature Engineering
 signals/         # Layer 3: Signal Generation
 portfolio/       # Layer 4: Position Sizing & Portfolio
 execution/       # Layer 5: Order Generation & Risk Controls
 pipeline/        # Layer 6: Pipeline Orchestration
 qa/              # Sidecar: Quality Assurance & Backtesting
 api/             # Layer 7: FastAPI Backend
 reports/         # Layer 7: Report Generation
 costs.py         # Shared: Cost Model
 ema_config.py    # Shared: EMA Configuration
 logging_utils.py # Shared: Logging Utilities
```

### Was rein darf
- Python-Module (`.py` Dateien)
- Package-Initialisierungen (`__init__.py`)
- Type Hints, Docstrings
- Unit-Tests fuer einzelne Module (nur wenn sehr spezifisch, sonst in `tests/`)

### Was NICHT rein darf
- CLI-Scripts (gehoeren nach `scripts/`)
- PowerShell-Scripts (gehoeren nach `scripts/`)
- Konfigurationsdateien (gehoeren nach `configs/`)
- Test-Dateien (gehoeren nach `tests/`, ausser sehr spezifische Unit-Tests)
- Temporaere Dateien (`tmp_*`, gehoeren nach `scripts/dev/`)
- Dokumentation (gehoert nach `docs/`)
- Daten-Dateien (gehoeren nach `data/`)

### Layer-Mapping
Siehe `docs/ARCHITECTURE_LAYERING.md` fuer detaillierte Import-Regeln.

| Ordner | Layer | Beschreibung |
|--------|-------|--------------|
| `config/` | 0 (Shared) | Zentrale Konfiguration (Settings, Models, Constants) |
| `utils/` | 0 (Shared) | Shared Utilities (DataFrame, Paths, Timing, Random) |
| `data/` | 1 | Daten-Ingestion (Preise, Alt-Daten, Factor Store) |
| `features/` | 2 | Feature-Engineering (TA-Features, Factor Integration) |
| `signals/` | 3 | Signal-Generierung (Trend, Event, Multi-Factor) |
| `portfolio/` | 4 | Position Sizing & Portfolio-Konstruktion |
| `execution/` | 5 | Order-Generierung & Risk-Controls |
| `pipeline/` | 6 | Pipeline-Orchestrierung (Trading Cycle) |
| `qa/` | Sidecar | Quality Assurance, Backtesting, Metrics |
| `api/` | 7 | FastAPI Backend (REST-API) |
| `reports/` | 7 | Report-Generierung (Markdown, CSV) |

---

## 2. `scripts/` - CLI Tools & Scripts

### Zweck
Alle ausfuehrbaren Scripts: CLI-Tools, PowerShell-Wrappers, Dev-Scripts, Pipeline-Scripts.

### Struktur
```
scripts/
 cli.py                    # Haupt-CLI (Entry Point)
 run_daily.py              # Daily Pipeline Runner
 run_backtest_strategy.py  # Strategy Backtest Runner
 batch_runner.py           # Batch Backtest Runner
 batch_backtest.py         # Batch Backtest Entry Point (Wrapper)
 run_eod_pipeline.py       # EOD Pipeline Runner
 run_paper_track.py        # Paper Trading Runner
 leaderboard.py            # Leaderboard Generator
 generate_*.py             # Report Generators
 dev/                      # Dev-Scripts (tmp_*, experiments)
 live/                     # Live Trading Scripts
 ps/                       # PowerShell Utilities
 tools/                    # Utility Scripts
 data/                     # Data Ingestion Scripts
```

### Was rein darf
- Python CLI-Scripts (`.py`)
- PowerShell-Scripts (`.ps1`)
- Dev-Scripts (in `scripts/dev/`)
- Experiment-Scripts (in `scripts/dev/`)
- Temporaeere Scripts (in `scripts/dev/`, z.B. `tmp_*.py`)

### Was NICHT rein darf
- Core-Backend-Module (gehoeren nach `src/assembled_core/`)
- Test-Dateien (gehoeren nach `tests/`)
- Konfigurationsdateien (gehoeren nach `configs/`)
- Dokumentation (gehoert nach `docs/`)

### Beispiele

**Neue CLI-Subcommands:**
- Hinzufuegen zu `scripts/cli.py` (nicht neue Datei)
- Oder neue Datei in `scripts/` mit `main()` Funktion

**Neue Dev-Scripts:**
- In `scripts/dev/` ablegen
- Beispiel: `scripts/dev/tmp_check.py`, `scripts/dev/quick_test.py`

**Neue PowerShell-Wrappers:**
- In `scripts/` oder `scripts/ps/` ablegen
- Beispiel: `scripts/run_all_sprint10.ps1`

---

## 3. `tests/` - Test Suite

### Zweck
Alle Test-Dateien fuer das Backend.

### Struktur
```
tests/
 test_*.py                 # Unit-Tests
 conftest.py               # Pytest Configuration
 test_config_*.py          # Config-Model Tests
 test_contracts_*.py       # Contract Tests
 test_pipeline_*.py        # Pipeline Tests
 test_qa_*.py              # QA Tests
```

### Was rein darf
- Test-Dateien (`.py` mit `test_` Praefix)
- Test-Fixtures (`conftest.py`)
- Test-Daten (kleine Sample-Dateien, z.B. `data/sample/`)

### Was NICHT rein darf
- Production-Code (gehoert nach `src/assembled_core/`)
- CLI-Scripts (gehoeren nach `scripts/`)
- Konfigurationsdateien (gehoeren nach `configs/`)

### Naming Convention
- Test-Dateien: `test_<module_name>.py`
- Test-Funktionen: `def test_<functionality>()`
- Test-Klassen: `class Test<ClassName>`

---

## 4. `docs/` - Dokumentation

### Zweck
Alle Projekt-Dokumentation (Markdown, ASCII-only).

### Struktur
```
docs/
 ARCHITECTURE_*.md         # Architektur-Dokumentation
 CONTRACTS.md              # Data Contracts
 PROJECT_STRUCTURE.md      # Repo-Struktur (dieses Dokument)
 BATCH_RUNNER_*.md         # Batch Runner Dokumentation
 ROADMAP_*.md              # Roadmap & Status
 *.md                      # Weitere Dokumentation
```

### Was rein darf
- Markdown-Dateien (`.md`)
- ASCII-only (keine Unicode-Zeichen ausser in Code-Beispielen)
- Architektur-Diagramme (ASCII-Art)
- API-Dokumentation

### Was NICHT rein darf
- Code-Dateien (gehoeren nach `src/assembled_core/` oder `scripts/`)
- Konfigurationsdateien (gehoeren nach `configs/`)
- Test-Dateien (gehoeren nach `tests/`)
- Daten-Dateien (gehoeren nach `data/`)

---

## 5. `configs/` - Konfigurationsdateien

### Zweck
YAML/JSON-Konfigurationsdateien fuer Batch-Runner, Paper-Track, etc.

### Struktur
```
configs/
 batch_*.yaml              # Batch Backtest Configs
 paper_track/               # Paper Trading Configs
 *.yaml                     # Weitere Configs
```

### Was rein darf
- YAML-Dateien (`.yaml`, `.yml`)
- JSON-Dateien (`.json`)
- Konfigurations-Templates

### Was NICHT rein darf
- Python-Code (gehoert nach `src/assembled_core/` oder `scripts/`)
- Test-Dateien (gehoeren nach `tests/`)
- Dokumentation (gehoert nach `docs/`)

---

## 6. `data/` - Rohdaten

### Zweck
Rohdaten (Preise, Alt-Daten, etc.). **NICHT in Git.**

### Struktur
```
data/
 raw/                      # Rohdaten (1min, EOD)
 sample/                   # Sample-Daten (kann in Git)
 ...
```

### Was rein darf
- Parquet-Dateien (`.parquet`)
- CSV-Dateien (`.csv`)
- Sample-Daten (kleine Dateien fuer Tests)

### Was NICHT rein darf
- Code-Dateien
- Konfigurationsdateien
- Dokumentation

### Git-Policy
- `data/raw/` ist in `.gitignore` (nicht in Git)
- `data/sample/` kann in Git (kleine Sample-Dateien)

---

## 7. `output/` - Generierte Artefakte

### Zweck
Alle generierten Artefakte (Orders, Equity Curves, Reports). **NICHT in Git.**

### Struktur
```
output/
 aggregates/               # Aggregierte Preise (5min, daily)
 orders_*.csv              # Generierte Orders
 equity_curve_*.csv        # Equity Curves
 portfolio_equity_*.csv    # Portfolio Equity
 performance_report_*.md   # Performance Reports
 ...
```

### Was rein darf
- Generierte CSV-Dateien
- Generierte Parquet-Dateien
- Generierte Markdown-Reports
- Batch-Backtest-Outputs

### Was NICHT rein darf
- Code-Dateien
- Konfigurationsdateien
- Dokumentation

### Git-Policy
- `output/` ist in `.gitignore` (nicht in Git)

---

## 8. `logs/` - Log-Dateien

### Zweck
Log-Dateien. **NICHT in Git.**

### Was rein darf
- Log-Dateien (`.log`, `.txt`)

### Was NICHT rein darf
- Code-Dateien
- Konfigurationsdateien

### Git-Policy
- `logs/` ist in `.gitignore` (nicht in Git)

---

## Verbindung zum Layering

### Mapping: Repo-Struktur  Layer-Architektur

Siehe `docs/ARCHITECTURE_LAYERING.md` fuer detaillierte Import-Regeln.

| Repo-Ordner | Layer | Erlaubte Imports |
|-------------|-------|------------------|
| `src/assembled_core/config/` | 0 (Shared) | Standard-Libs, `pandas`, `numpy`, `pathlib` |
| `src/assembled_core/utils/` | 0 (Shared) | Standard-Libs, `pandas`, `numpy`, `pathlib`, `config` |
| `src/assembled_core/data/` | 1 | `config`, `utils`, Standard-Libs, `pandas`, `numpy` |
| `src/assembled_core/features/` | 2 | `data`, `config`, `utils`, Standard-Libs, `pandas`, `numpy` |
| `src/assembled_core/signals/` | 3 | `data`, `features`, `config`, `utils`, Standard-Libs, `pandas`, `numpy` |
| `src/assembled_core/portfolio/` | 4 | `data`, `features`, `signals`, `config`, `utils`, Standard-Libs, `pandas`, `numpy` |
| `src/assembled_core/execution/` | 5 | `data`, `features`, `signals`, `portfolio`, `config`, `utils`, Standard-Libs, `pandas`, `numpy` |
| `src/assembled_core/pipeline/` | 6 | `data`, `features`, `signals`, `portfolio`, `execution`, `config`, `utils`, Standard-Libs, `pandas`, `numpy` |
| `src/assembled_core/qa/` | Sidecar | Alle anderen Layer (darf alle importieren) |
| `src/assembled_core/api/` | 7 | Alle Layer (darf alle importieren) |
| `src/assembled_core/reports/` | 7 | Alle Layer (darf alle importieren) |

### Do / Don't Beispiele

**Neue Signale hinzufuegen:**
-  `src/assembled_core/signals/<strategy_name>.py`
-  `src/assembled_core/pipeline/signals.py` (nur Orchestrierung)
-  `scripts/<strategy_name>.py` (nur CLI-Wrapper)

**Neue Feature Builder hinzufuegen:**
-  `src/assembled_core/features/<feature_type>.py`
-  `src/assembled_core/data/<feature_type>.py` (nur Ingestion)
-  `scripts/<feature_type>.py` (nur CLI-Wrapper)

**Neue CLI-Subcommands hinzufuegen:**
-  `scripts/cli.py` (Haupt-CLI erweitern)
-  `scripts/<command_name>.py` (neue Datei mit `main()`)
-  `src/assembled_core/api/<command_name>.py` (nur REST-API)

**Dev-Scripts hinzufuegen:**
-  `scripts/dev/tmp_<name>.py`
-  `scripts/dev/quick_<name>.py`
-  `tmp_<name>.py` im Root (nicht erlaubt)
-  `src/assembled_core/<name>.py` (nur Production-Code)

---

## Hygiene-Regeln

### 1. Keine `tmp_*` im Root

**Regel:** Alle temporaeren Dateien gehoeren nach `scripts/dev/`.

**Beispiele:**
-  `scripts/dev/tmp_check.py`
-  `scripts/dev/tmp_peek_ec.py`
-  `tmp_check.py` (im Root)
-  `tmp_peek_ec.py` (im Root)

**Begruendung:** Root-Verzeichnis soll sauber bleiben (NR3 DoD).

### 2. Keine grossen Artefakte in Git

**Regel:** `output/`, `data/raw/`, `logs/` sind in `.gitignore`.

**Was NICHT in Git:**
- `output/*` (generierte Artefakte)
- `data/raw/*` (Rohdaten)
- `logs/*` (Log-Dateien)
- `.venv/` (Virtual Environment)
- `__pycache__/` (Python Cache)

**Was KANN in Git:**
- `data/sample/*` (kleine Sample-Dateien fuer Tests)
- `configs/*` (Konfigurationsdateien)
- `docs/*` (Dokumentation)

**Begruendung:** Git-Repository soll klein und schnell bleiben.

### 3. Einheitliche Dateinamen/Module

**Regel:** Kurze, eindeutige Namen ohne Sonderzeichen.

**Naming Convention:**
- Python-Module: `snake_case.py`
- PowerShell-Scripts: `snake_case.ps1`
- Test-Dateien: `test_<module_name>.py`
- Config-Dateien: `snake_case.yaml`

**Beispiele:**
-  `src/assembled_core/pipeline/trading_cycle.py`
-  `scripts/run_backtest_strategy.py`
-  `tests/test_pipeline_trading_cycle.py`
-  `src/assembled_core/pipeline/TradingCycle.py` (PascalCase)
-  `scripts/run-backtest-strategy.py` (Kebab-Case)
-  `tests/testPipelineTradingCycle.py` (CamelCase)

**Begruendung:** Konsistenz erleichtert Navigation und Wartung.

### 4. ASCII-only in Dokumentation

**Regel:** Alle Dokumentation in `docs/` muss ASCII-only sein.

**Erlaubt:**
- ASCII-Zeichen (0x20-0x7E)
- Newlines, Tabs
- ASCII-Art-Diagramme

**Nicht erlaubt:**
- Unicode-Zeichen (z.B. , , )
- Emojis
- Sonderzeichen ausserhalb ASCII

**Begruendung:** ASCII-only ist portabel und funktioniert ueberall.

### 5. Keine Code-Duplikation

**Regel:** Code-Duplikation vermeiden, Shared-Module nutzen.

**Beispiele:**
-  `src/assembled_core/utils/dataframe.py` (shared utilities)
-  `src/assembled_core/config/models.py` (shared config models)
-  Duplizierte Funktionen in mehreren Modulen
-  Copy-Paste zwischen Scripts

**Begruendung:** DRY-Prinzip (Don't Repeat Yourself) erleichtert Wartung.

---

## Checkliste fuer neue Dateien

Vor dem Hinzufuegen einer neuen Datei pruefen:

- [ ] **Ort:** Gehoert die Datei wirklich in diesen Ordner?
- [ ] **Naming:** Folgt der Dateiname der Naming Convention?
- [ ] **Layer:** Verletzt die Datei die Layer-Architektur (siehe `docs/ARCHITECTURE_LAYERING.md`)?
- [ ] **Hygiene:** Verletzt die Datei Hygiene-Regeln (tmp_*, grosse Artefakte, etc.)?
- [ ] **Git:** Ist die Datei in `.gitignore` wenn noetig (output/, data/raw/, logs/)?
- [ ] **ASCII:** Ist die Datei ASCII-only (wenn Dokumentation)?

---

## Referenzen

- `docs/ARCHITECTURE_LAYERING.md` - Layer-Architektur & Import-Regeln
- `docs/CONTRACTS.md` - Data Contracts
- `.gitignore` - Git-Ignore-Regeln

---

## Aenderungen an der Struktur

**Wichtig:** Die Repo-Struktur ist **verbindlich** und sollte nicht ohne triftigen Grund geaendert werden.

**Prozess fuer Aenderungen:**
1. Aenderung in `docs/PROJECT_STRUCTURE.md` dokumentieren
2. Alle betroffenen Dateien verschieben/anpassen
3. `.gitignore` aktualisieren (falls noetig)
4. Integrationstests ausfuehren
5. Dokumentation aktualisieren
