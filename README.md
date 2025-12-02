# Assembled Trading AI - Backend

**Ein modulares Trading-Core-System für Daten-Ingest, Signal-Generierung, Backtesting und Portfolio-Simulation.**

---

## Schnellstart

### 1. Repository klonen

```bash
git clone <repository-url>
cd Aktiengerüst
```

### 2. Virtual Environment erstellen und aktivieren

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Dependencies installieren

```bash
pip install -e .[dev]
```

Dies installiert:
- Core-Dependencies (pandas, numpy, fastapi, etc.)
- Dev-Dependencies (pytest, ruff, black, mypy)

### 4. Phase-4-Tests ausführen

**Über CLI:**
```bash
python scripts/cli.py run_phase4_tests
```

**Oder direkt mit pytest:**
```bash
pytest -m phase4
```

**Erwartete Ausgabe:**
- ~117 Tests in ~13-17 Sekunden
- Alle Tests sollten grün sein ✅

---

## CLI (Command Line Interface)

Das zentrale CLI (`scripts/cli.py`) bietet eine einheitliche Schnittstelle für die wichtigsten Backend-Operationen.

### Verfügbare Subcommands

#### 1. Projekt-Informationen

```bash
# Zeige Projekt-Informationen und verfügbare Subcommands
python scripts/cli.py info

# Zeige Version
python scripts/cli.py --version
```

#### 2. Phase-4-Test-Suite

```bash
# Standard-Testlauf
python scripts/cli.py run_phase4_tests

# Mit detaillierter Ausgabe
python scripts/cli.py run_phase4_tests --verbose

# Mit Dauer-Informationen
python scripts/cli.py run_phase4_tests --durations 5
```

**PowerShell-Wrapper:**
```powershell
.\scripts\run_phase4_tests.ps1
.\scripts\run_phase4_tests.ps1 -Verbose -Durations
```

#### 3. Strategy-Backtest

**Verfügbare Strategien:**
- `trend_baseline`: EMA-basierte Trend-Strategie (Standard)
- `event_insider_shipping`: Event-basierte Strategie mit Insider-Trading und Shipping-Daten (Phase 6)

```bash
# Standard-Backtest (Trend-Baseline)
python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt

# Mit QA-Report
python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt --generate-report

# Ohne Transaktionskosten
python scripts/cli.py run_backtest --freq 1d --no-costs

# Event-Strategie (Phase 6)
python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt --strategy event_insider_shipping --start-capital 10000 --with-costs
```

**Event-Strategie (`event_insider_shipping`):**
- Nutzt Phase-6-Features: Insider-Trading-Daten und Shipping-Route-Congestion
- Aktuell implementiert als Proof-of-Concept mit einfacher Regel-Logik:
  - **LONG**: Starker Insider-Netto-Kauf + niedrige Shipping-Congestion
  - **SHORT**: Starker Insider-Netto-Verkauf + hohe Shipping-Congestion
  - **FLAT**: Sonst
- Verwendet Dummy-Daten für Insider- und Shipping-Events (echte Datenquellen geplant)

#### 4. EOD-Pipeline

```bash
# Vollständiger Pipeline-Lauf (execute, backtest, portfolio, QA)
python scripts/cli.py run_daily --freq 1d

# Mit angepasstem Startkapital
python scripts/cli.py run_daily --freq 1d --start-capital 50000

# Mit expliziter Preis-Datei
python scripts/cli.py run_daily --freq 5min --price-file data/sample/eod_sample.parquet
```

### Hilfe

```bash
# Allgemeine Hilfe
python scripts/cli.py --help

# Hilfe zu einem Subcommand
python scripts/cli.py <subcommand> --help
```

**Weitere Details:** Siehe `docs/CLI_REFERENCE.md`

---

## Tests

### Phase-4-Tests (Backend Core)

Die Phase-4-Test-Suite umfasst:
- TA-Features (Technical Analysis)
- QA-Metriken (Performance-Metriken)
- QA-Gates (Quality Assurance Checks)
- Backtest-Engine
- Reports (Daily QA Reports)
- Pipelines (EOD-Pipeline, Backtest-Pipeline)

**Ausführen:**

```bash
# Über CLI (empfohlen)
python scripts/cli.py run_phase4_tests

# Direkt mit pytest
pytest -m phase4

# Mit Details
pytest -m phase4 --durations=5
```

**Erwartete Dauer:** ~13-17 Sekunden für ~117 Tests

### Phase-6-Tests (Event Features)

Phase-6-Tests für neue Event-basierte Features (Insider, Congress, Shipping, News):

```bash
pytest -m phase6
```

**Erwartete Dauer:** < 1 Sekunde für ~11 Tests

### Langsame Backtest-Tests

Backtest-Tests mit größeren Datensätzen sind mit `@pytest.mark.slow` markiert:

```bash
# Nur langsame Tests
pytest tests/test_qa_backtest_engine.py -m "slow"

# Alle Backtest-Tests (inkl. schnelle)
pytest tests/test_qa_backtest_engine.py
```

### Vollständige Test-Suite

```bash
# Alle Tests (ohne externe Dependencies)
pytest -m "not external"

# Alle Tests inkl. externe (falls konfiguriert)
pytest
```

### Test-Marker

Verfügbare Marker:
- `phase4`: Phase-4-Backend-Tests (TA, QA-Metrics, Gates, Backtest, Reports, Pipelines)
- `phase6`: Phase-6-Event-Features (Insider, Congress, Shipping, News)
- `slow`: Langsame Tests (> 1 Sekunde)
- `unit`: Schnelle Unit-Tests (< 1 Sekunde)
- `integration`: Integration-Tests (mehrere Module)
- `smoke`: End-to-End Smoke-Tests
- `external`: Tests, die externe Services benötigen (standardmäßig ausgeschlossen)

**Weitere Details:** Siehe `docs/TESTING_COMMANDS.md`

---

## Projekt-Struktur

```
Aktiengerüst/
├── src/
│   └── assembled_core/          # Core-Backend-Package
│       ├── data/                # Daten-Ingest
│       ├── features/            # Feature-Engineering
│       ├── signals/             # Signal-Generierung
│       ├── portfolio/           # Position Sizing
│       ├── execution/           # Order-Generierung
│       ├── qa/                  # Quality Assurance
│       ├── pipeline/            # Pipeline-Orchestrierung
│       └── api/                 # FastAPI-Endpoints
├── scripts/
│   ├── cli.py                   # Zentrales CLI
│   ├── run_backtest_strategy.py
│   ├── run_eod_pipeline.py
│   └── run_phase4_tests.ps1     # PowerShell-Wrapper
├── tests/                       # Test-Suite
├── docs/                        # Dokumentation
│   ├── ARCHITECTURE_BACKEND.md
│   ├── CLI_REFERENCE.md
│   ├── TESTING_COMMANDS.md
│   └── ...
└── output/                      # Pipeline-Outputs
```

---

## Dokumentation

- **Backend-Architektur:** `docs/ARCHITECTURE_BACKEND.md`
- **CLI-Referenz:** `docs/CLI_REFERENCE.md`
- **Testing-Commands:** `docs/TESTING_COMMANDS.md`
- **Phase 6 Events:** `docs/PHASE6_EVENTS.md`
- **Legacy-Übersicht:** `docs/LEGACY_OVERVIEW.md`
- **Legacy-Mapping:** `docs/LEGACY_TO_CORE_MAPPING.md`
- **PowerShell-Wrapper:** `docs/POWERSHELL_WRAPPERS.md`

---

## Status

- ✅ **Phase 4:** Backend Core stabil (110+ Tests, ~17s)
- ✅ **Phase 5:** Dokumentation & Legacy-Mapping
- ✅ **Phase 6:** Event-Features Skeletons (Insider, Congress, Shipping, News)

---

## Entwicklung

### Code-Style

- **Python:** PEP 8, Type Hints, Docstrings
- **Linting:** `ruff check`
- **Formatting:** `black`
- **Type Checking:** `mypy` (optional)

### Git-Workflow

- Feature-Branches für neue Features
- Commits mit klaren Messages
- Tests müssen grün sein vor Merge

---

## Lizenz

[Lizenz-Informationen hier einfügen]

---

## Kontakt

[Kontakt-Informationen hier einfügen]
