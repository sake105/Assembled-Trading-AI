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

## Usage & Workflows

The `assembled-cli` command-line interface is the preferred entry point for all backend operations. All workflows support optional experiment tracking via `--track-experiment` flags.

### Quick Links

- **EOD Pipeline & QA** → [Workflows – Daily EOD Pipeline & QA](docs/WORKFLOWS_EOD_AND_QA.md)
  - Run daily end-of-day pipeline
  - Support for both local (historical) and live (Yahoo Finance) data sources
  - Generate signals/orders (SAFE CSV)
  - Review QA reports
  - Logging and experiment tracking

- **Backtests & Meta-Model Ensemble** → [Workflows – Backtests & Meta-Model Ensemble](docs/WORKFLOWS_BACKTEST_AND_ENSEMBLE.md)
  - Run strategy backtests
  - Use ML meta-model ensemble (filter or scaling mode)
  - Track experiments for research
  - Compare performance metrics

- **ML Datasets, Meta-Models & Experiments** → [Workflows – ML Meta-Models & Experiments](docs/WORKFLOWS_ML_AND_EXPERIMENTS.md)
  - Build ML-ready datasets from backtests

- **ML Validation & Model Comparison** → [Workflows – ML Validation & Model Comparison](docs/WORKFLOWS_ML_VALIDATION_AND_MODEL_COMPARISON.md)
  - Validate ML models on factor panels
  - **ML Model Zoo** – Compare multiple ML models automatically on same panel
  - Compare model performance (Linear, Ridge, Lasso, Random Forest)
  - Time-series cross-validation with IC/Rank-IC and portfolio metrics
  - **Feature Importance & Explainability (E2)** – Understand which factors drive ML predictions, compare with classical factor rankings
  - Train meta-models for setup success prediction
  - Evaluate model performance (ROC-AUC, Brier, Log Loss)
  - Connect to research notebooks

- **Multi-Factor Long/Short Strategy** → [Workflows – Multi-Factor Long/Short Strategy](docs/WORKFLOWS_STRATEGIES_MULTIFACTOR.md)
  - Factor-based long/short trading strategy
  - Uses factor bundles and local alt-data snapshots
  - Quantile-based position selection (top/bottom quantiles)
  - Configurable rebalancing frequencies (daily, weekly, monthly)

- **Regime Models & Risk Overlay** → [Workflows – Regime Models & Risk Overlay](docs/WORKFLOWS_REGIME_MODELS_AND_RISK.md)
  - Market regime detection (bull, bear, sideways, crisis, reflation)
  - Adaptive risk management with regime-based exposure control
  - Factor performance evaluation by regime
  - Integration with multi-factor strategies

- **Risk Metrics & Attribution** → [Workflows – Risk Metrics & Attribution](docs/WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md)
  - Extended risk metrics (Sharpe, Sortino, VaR, ES, Skewness, Kurtosis)
  - Exposure analysis (Gross/Net Exposure, HHI Concentration)
  - Risk segmentation by market regime
  - Performance attribution by factor groups
  - Factor exposure analysis (optional: rolling regression of strategy returns vs. factor returns)
  - Comprehensive risk reports from backtest results

- **Batch Backtests & Parallelization (P4)** → [Workflows – Batch Backtests & Parallelization](docs/WORKFLOWS_BATCH_BACKTESTS_AND_PARALLELIZATION.md)
  - Run multiple backtests from a single YAML/JSON config
  - Parameter sweeps (risk parameters, rebalancing, costs)
  - Strategy comparisons (Core vs. ML vs. ML-only)
  - Clean output structure with batch summaries (CSV/Markdown)

- **Walk-Forward & Regime Analysis** → [Factor Analysis Workflows](docs/WORKFLOWS_FACTOR_ANALYSIS.md) and [Walk-Forward & Regime B3 Design](docs/WALK_FORWARD_AND_REGIME_B3_DESIGN.md)
  - Walk-Forward analysis for out-of-sample validation (rolling/expanding windows)
  - Regime-based performance evaluation (Bull, Bear, Crisis, Sideways, Reflation)
  - Integration with risk reports for regime-aware performance analysis
  - Research tool for strategy stability testing

- **Use Cases & Roles** → [Use Cases & Roles](docs/USE_CASES_AND_ROLES_A1.md)
  - Overview of backend capabilities by role (Quant PM, Quant Researcher, Quant Dev/Backend, Data Engineer)
  - Use cases per role with CLI commands and expected outputs
  - Component map linking roles to workflows, scripts, and documentation

- **Operations & Monitoring** → [Operations Backend Runbook](docs/OPERATIONS_BACKEND.md)
  - Daily and weekly health check procedures
  - Health check CLI usage and status interpretation
  - Troubleshooting guide for common operational issues
  - Automation recommendations for daily health checks

- **Paper-Track Playbook** → [Paper Track Playbook](docs/PAPER_TRACK_PLAYBOOK.md)
  - Kriterien & Checklisten fur Backtest → Paper → (spater) Live
  - Gate-Kriterien (Mindest-Backtestdauer, Regime-Coverage, Deflated Sharpe, Max Drawdown, PIT-Sicherheit)
  - Paper-Track-Ablauf (Dauer, Metriken, akzeptable Abweichungen vs. Backtest)
  - Go/No-Go-Kriterien fur Live-Trading-Vorbereitung

### Getting Started

**Basic EOD Pipeline (with local data):**
```bash
python scripts/cli.py run_daily --freq 1d
```

**EOD Pipeline with live data (Yahoo Finance):**
```bash
python scripts/cli.py run_daily --freq 1d --data-source yahoo --symbols AAPL MSFT GOOGL --end-date today
```

**Basic Backtest:**
```bash
python scripts/cli.py run_backtest --freq 1d --strategy trend_baseline --generate-report
```

**Build ML Dataset:**
```bash
python scripts/cli.py build_ml_dataset --strategy trend_baseline --freq 1d
```

For detailed workflows, examples, and troubleshooting, see the workflow documentation linked above.

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

#### 4. Strategie-Vergleich

Vergleiche Trend-Baseline vs Event-Insider-Shipping-Strategie auf denselben Preisdaten:

```bash
# Standard-Vergleich
python scripts/compare_strategies_trend_vs_event.py --freq 1d --price-file data/sample/eod_sample.parquet

# Ohne Transaktionskosten
python scripts/compare_strategies_trend_vs_event.py --freq 1d --price-file data/sample/eod_sample.parquet --no-costs

# Mit angepasstem Startkapital
python scripts/compare_strategies_trend_vs_event.py --freq 1d --price-file data/sample/eod_sample.parquet --start-capital 50000
```

**Output-Dateien** (unter `output/strategy_compare/trend_vs_event/`):
- `comparison_summary.md`: Markdown-Tabelle mit Performance-Metriken beider Strategien
- `comparison_summary.csv`: CSV-Datei mit allen Metriken für weitere Analysen

#### 5. EOD-Pipeline

```bash
# Vollständiger Pipeline-Lauf (execute, backtest, portfolio, QA)
python scripts/cli.py run_daily --freq 1d

# Mit angepasstem Startkapital
python scripts/cli.py run_daily --freq 1d --start-capital 50000

# Mit expliziter Preis-Datei
python scripts/cli.py run_daily --freq 5min --price-file data/sample/eod_sample.parquet
```

**Hinweis:** Alle generierten Orders durchlaufen automatisch Pre-Trade-Checks und Kill-Switch-Validierung (Phase 10). Siehe `docs/PHASE10_PAPER_OMS.md` für Details.

#### 6. ML-Dataset-Export

Exportiere Backtest-Ergebnisse als ML-ready Dataset mit Features und Labels:

```bash
# Standard-Dataset (Trend-Baseline, 10 Tage Horizon, 2% Threshold)
python scripts/cli.py build_ml_dataset --strategy trend_baseline --freq 1d --price-file data/sample/eod_sample.parquet

# Event-Strategie mit angepassten Parametern
python scripts/cli.py build_ml_dataset --strategy event_insider_shipping --freq 1d --label-horizon-days 5 --success-threshold 0.03

# Mit explizitem Output-Pfad
python scripts/cli.py build_ml_dataset --strategy trend_baseline --freq 1d --out output/ml_datasets/my_dataset.parquet
```

**Output:**
- Dataset wird als Parquet-Datei gespeichert (Standard: `output/ml_datasets/<strategy>_<freq>.parquet`)
- Enthält: Labels (0/1), Trade-Metadaten (symbol, open_time, open_price), Feature-Spalten (ta_*, insider_*, shipping_*, etc.)
- Labels basieren auf P&L-Performance innerhalb des angegebenen Horizonts

**Weitere Details:** Siehe `docs/PHASE7_META_LAYER.md`

#### 7. Paper-Trading-API

Die Paper-Trading-API bietet REST-Endpoints für Paper-Trading-Orders mit in-memory Engine.

**Verfügbare Endpoints:**
- `POST /api/v1/paper/orders`: Orders einreichen
- `GET /api/v1/paper/orders`: Orders auflisten
- `GET /api/v1/paper/positions`: Aktuelle Positionen abrufen
- `POST /api/v1/paper/reset`: Engine zurücksetzen (für Tests/Dev)

**Python-Client-Beispiel:**
```python
import requests

BASE_URL = "http://localhost:8000/api/v1/paper"

# Orders einreichen
orders = [
    {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 10.0,
        "price": 150.0
    }
]
response = requests.post(f"{BASE_URL}/orders", json=orders)
print(response.json())

# Positionen abrufen
response = requests.get(f"{BASE_URL}/positions")
print(response.json())
```

**Hinweise:**
- Alle Orders durchlaufen automatisch Pre-Trade-Checks und Kill-Switch (wie reguläre Pipeline)
- Orders werden sofort als "FILLED" oder "REJECTED" zurückgegeben
- Positionen werden automatisch aggregiert (BUY = +, SELL = -)

**Weitere Details:** Siehe `docs/PHASE10_PAPER_OMS.md`

**Use-Case: Vom Signal zur Order im Paper-OMS**

Ein vollständiger Flow-Beispiel zeigt, wie Orders von der Signal-Generierung bis zur Sichtbarkeit im OMS-Blotter gelangen:

1. **Signal-Generierung**: Aus Backtest oder Price-Daten
2. **Position-Sizing**: Ableitung von Target-Positionen
3. **Order-Generierung**: Umwandlung in Orders (symbol, side, qty, price)
4. **Risk Controls**: Pre-Trade-Checks + Kill-Switch
5. **Paper-API Submission**: POST `/api/v1/paper/orders` mit `source` und `route`
6. **OMS-Blotter**: GET `/api/v1/oms/blotter` zur Überprüfung

**Beispiel-JSON-Payload:**

```json
[
  {
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 10.0,
    "price": 150.25,
    "source": "CLI_BACKTEST",
    "route": "PAPER",
    "client_order_id": "BACKTEST_AAPL_2025-01-15"
  }
]
```

**Source- und Route-Tagging:**
- **Source**: Identifiziert Ursprung (`CLI_BACKTEST`, `CLI_EOD`, `API`, `DASHBOARD`)
- **Route**: Identifiziert Destination (`PAPER`, `PAPER_ALT`, zukünftig `IBKR`, `ALPACA`)

**Vollständige Dokumentation:** Siehe `docs/PHASE10_PAPER_OMS.md` → "Use-Case: Vom Signal zur Order im Paper-OMS"

#### 8. OMS-Light (Blotter & Routing)

OMS-Light bietet eine minimale Order-Management-Schicht für Dashboard/Operator-Views über der Paper-Trading-Engine.

**Zweck:** Operative Übersicht über alle Orders mit Filterung und Sortierung.

**Verfügbare Endpoints:**
- `GET /api/v1/oms/blotter`: Blotter-View aller Orders (Filter: symbol, status, route, limit)
- `GET /api/v1/oms/executions`: Execution-View (Fills) mit Filterung
- `GET /api/v1/oms/routes`: Liste verfügbarer Routen

**Beispiel-Usage:**

```bash
# Blotter abfragen
curl "http://localhost:8000/api/v1/oms/blotter?limit=20"

# Blotter nach Symbol und Status filtern
curl "http://localhost:8000/api/v1/oms/blotter?symbol=AAPL&status=FILLED"

# Executions abfragen
curl "http://localhost:8000/api/v1/oms/executions?symbol=MSFT"
```

**Hinweise:**
- OMS-Light spiegelt den Zustand der Paper-Trading-Engine wider (keine Duplikation)
- Geblockte Orders (Pre-Trade-Checks/Kill-Switch) erscheinen als `REJECTED` im Blotter
- Nur `FILLED` Orders erscheinen in der Executions-View
- Orders können mit `source` (z.B. "CLI_EOD", "API") und `route` (z.B. "PAPER") versehen werden

**Weitere Details:** Siehe `docs/PHASE10_PAPER_OMS.md`

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

### Phase-8-Tests (Risk Engine)

Phase-8-Tests für Risk-Engine-Komponenten (Portfolio Risk Metrics, Scenario Engine, Shipping Risk):

```bash
pytest -m phase8
```

**Erwartete Dauer:** < 2 Sekunden für ~39 Tests

**Weitere Details:** Siehe `docs/PHASE8_RISK_ENGINE.md`

### Phase-9-Tests (Model Governance & Validation)

Phase-9-Tests für Model Governance (Validation, Drift Detection):

```bash
pytest -m phase9
```

**Erwartete Dauer:** < 2 Sekunden für ~41 Tests

**Weitere Details:** Siehe `docs/PHASE9_MODEL_GOVERNANCE.md`

### Phase-10-Tests (Pre-Trade Checks, Kill-Switch & Paper-Trading-API)

Phase-10-Tests für Pre-Trade-Kontrollen, Kill-Switch und Paper-Trading-API:

```bash
# Alle Phase-10-Tests
pytest -m phase10

# Nur Paper-Trading-API-Tests
pytest -m phase10 tests/test_api_paper_trading.py
```

**Erwartete Dauer:** < 1 Sekunde für ~35 Tests (inkl. Paper-Trading-API)

**Weitere Details:** Siehe `docs/PHASE10_PAPER_OMS.md`

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

## Research & Experimente

Das Projekt verfügt über eine strukturierte Research-Infrastruktur für die systematische Exploration neuer Trading-Ideen, Strategien und Datenquellen.

**Research-Roadmap:** `docs/RESEARCH_ROADMAP.md`
- Zielbild und aktueller Stand
- Fokus 3–6 Monate (neue Strategien, Alt-Daten, ML-Experimente)
- Konkreter Research-Backlog mit Prioritäten
- Arbeitsweise für Research-Experimente

**Advanced Analytics & Factor Labs:** `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md`
- Extended roadmap for advanced factor development (Phases A–E)
- Technical analysis factor library (Phase A) ✅
- Alternative data factors 2.0 (Phase B)
- Factor analysis & event study engine (Phase C) ✅
- **Factor Analysis Workflows:** `docs/WORKFLOWS_FACTOR_ANALYSIS.md` - Comprehensive guide for factor evaluation using IC/IR (C1) and portfolio-based metrics (C2)
- Regime models & risk 2.0 (Phase D)
- ML validation, explainability & TCA (Phase E)

**Research-Ordner:** `research/`
- Strukturierte Experimente in Kategorien (trend/, meta/, altdata/, risk/)
- Notebook-Templates für reproduzierbare Experimente
- Best Practices für Research-Workflow

**Typischer Research-Workflow:**
1. **Hypothese formulieren**: Klare Frage, erwartetes Ergebnis
2. **Setup definieren**: Daten, Strategie-Parameter, Backtest-Konfiguration
3. **Experiment durchführen**: Code in `research/`, Backtest ausführen
4. **Auswertung**: Performance-Metriken, Visualisierung, Vergleich mit Baseline
5. **Dokumentation**: Ergebnisse, Fazit, nächste Schritte

**Verfügbare Tools für Research:**
- Backtest-Engine mit konfigurierbaren Kostenmodellen
- ML-Dataset-Export für Meta-Model-Experimente
- QA-Reports für Performance-Analysen
- Monitoring-API für QA/Risk/Drift-Status
- **Experiment-Tracking** (Sprint 12.2): Strukturiertes Tracking von Runs mit Config, Metriken und Artefakten

**Experiment-Tracking:**
- Runs werden in `experiments/` gespeichert (kein externer Service nötig)
- Jeder Run enthält: `run.json` (Metadaten), `metrics.csv` (Zeitreihen-Metriken), `artifacts/` (Dateien)
- Integration in CLI: `--track-experiment --experiment-name "..." --experiment-tags "tag1,tag2"`
- Verfügbar für: Backtests (`run_backtest`), Meta-Model-Training (`train_meta_model`)

**Weitere Details:** Siehe `docs/RESEARCH_ROADMAP.md` und `research/README.md`

---

## Reviews & Known Issues

Das Backend ist **review-fähig** und bereit für externe Reviews.

**Review Guide:** `docs/REVIEW_GUIDE_BACKEND.md`
- Empfohlene Reihenfolge für Reviews
- Praktische Einstiegspunkte (CLI-Kommandos, wichtige Module)
- Checkliste für Reviews (Architektur, Tests, Logging, ML, Research)
- Konkrete Review-Fragen
- Feedback-Struktur

**Known Issues:** `KNOWN_ISSUES.md`
- Funktionale Open Points (Labeling, Trade-Metriken, Pre-Trade-Checks)
- Technische Schulden (Legacy-Migration, Validation-Split)
- Performance & Skalierung (Parallelisierung, Caching)
- Nice-to-Haves (erweiterte Strategien, Alt-Daten, ML-Experimente)

**Feedback abgeben:**
- GitHub Issues mit Template: `.github/ISSUE_TEMPLATE/review_feedback.md`
- Strukturiert nach Bereich, Schweregrad, Referenzen

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
│   ├── RESEARCH_ROADMAP.md      # Research-Roadmap (Phase 12)
│   └── ...
├── research/                    # Research-Experimente
│   ├── README.md                # Research-Workflow & Best Practices
│   ├── trend/                   # Trend-Strategie-Experimente
│   ├── meta/                    # ML-Meta-Layer-Experimente
│   ├── altdata/                 # Alternative-Daten-Experimente
│   └── risk/                    # Risk-Engine-Experimente
└── output/                      # Pipeline-Outputs
```

---

## Dokumentation

- **Backend-Architektur:** `docs/ARCHITECTURE_BACKEND.md`
- **CLI-Referenz:** `docs/CLI_REFERENCE.md`
- **Testing-Commands:** `docs/TESTING_COMMANDS.md`
- **Phase 6 Events:** `docs/PHASE6_EVENTS.md`
- **Phase 8 Risk Engine:** `docs/PHASE8_RISK_ENGINE.md` (Portfolio Risk, Scenarios, Shipping Risk)
- **Phase 9 Model Governance:** `docs/PHASE9_MODEL_GOVERNANCE.md` (Model Validation, Drift Detection)
- **Phase 10 Paper-Trading & OMS:** `docs/PHASE10_PAPER_OMS.md` (Pre-Trade Checks, Kill-Switch)
- **Phase 12 Research Roadmap:** `docs/RESEARCH_ROADMAP.md` (Research-Prozess, Backlog, Fokus)
- **Review Guide:** `docs/REVIEW_GUIDE_BACKEND.md` (Anleitung für externe Reviewer)
- **Known Issues:** `KNOWN_ISSUES.md` (Offene Punkte, technische Schulden, Nice-to-Haves)
- **Legacy-Übersicht:** `docs/LEGACY_OVERVIEW.md`
- **Legacy-Mapping:** `docs/LEGACY_TO_CORE_MAPPING.md`
- **PowerShell-Wrapper:** `docs/POWERSHELL_WRAPPERS.md`

---

## CI / Continuous Integration

[![Backend CI](https://github.com/sake105/Assembled-Trading-AI/actions/workflows/backend-ci.yml/badge.svg)](https://github.com/sake105/Assembled-Trading-AI/actions/workflows/backend-ci.yml)

Die CI prüft bei jedem Push/PR auf `main` und `feature/*` Branches:

- **Tests**: Alle Backend-Tests (Phase 4-11) mit `pytest -W error`
- **Linting**: Code-Qualität mit `ruff check`
- **Formatting**: Code-Format mit `black --check`
- **Type Checking**: Statische Typprüfung mit `mypy` (optional, non-blocking)

**Unterstützte Python-Versionen**: 3.10, 3.11

**Workflow-Details**: Siehe [`.github/workflows/backend-ci.yml`](.github/workflows/backend-ci.yml)

---

## Status

- ✅ **Phase 4:** Backend Core stabil (110+ Tests, ~17s)
- ✅ **Phase 5:** Dokumentation & Legacy-Mapping
- ✅ **Phase 6:** Event-Features Skeletons (Insider, Congress, Shipping, News)
- ✅ **Phase 7:** ML-Meta-Layer
  - ✅ 7.1 – Labeling & ML-Dataset: fertig
  - ✅ 7.2 – Meta-Modelle (Confidence-Scores, ROC/AUC, Brier, Calibration): fertig
  - ✅ 7.3 – Ensemble-Layer (Kombination Regel-Signale + Meta-Model): fertig
- ✅ **Phase 8:** Risk Engine & Scenario Analysis (39 Tests, <2s)
- ✅ **Phase 9:** Model Governance & Validation (41 Tests, <2s)
- ✅ **Phase 10:** Paper-Trading & OMS-Light
  - ✅ 10.1 – Pre-Trade Checks & Kill-Switch: fertig
  - ✅ 10.2 – Paper-Trading-API: fertig (inkl. Pre-Trade & Kill-Switch)
  - ✅ 10.3 – OMS-Light (Blotter & Routing): fertig
- ✅ **Phase 12:** "God-Level" Research & Evolution
  - ✅ 12.1 – Research-Prozess & Roadmap: fertig (Research-Roadmap vorhanden)
  - ✅ 12.2 – Experiment-Tracking: fertig (leichtgewichtiges Tracking ohne externe Services)
  - ✅ 12.3 – Review-Vorbereitung: fertig (Known Issues, Review Guide, Issue-Template)

---

## Logging

Das Backend verwendet ein zentrales Logging-System (`src/assembled_core/logging_config.py`):

- **Log-Dateien**: Alle Logs werden in `logs/` geschrieben (Dateiname enthält Run-ID)
- **Run-IDs**: Jede Ausführung erhält eine eindeutige Run-ID (z. B. `backtest_20250115_143022_abc12345`)
- **Formate**:
  - **Console**: `[LEVEL] message` (einfach, für CLI)
  - **File**: `timestamp | level | logger | [run_id] message` (detailliert)
- **Integration**: Automatisch in `scripts/cli.py`, `scripts/run_backtest_strategy.py`, `scripts/run_eod_pipeline.py`

**Beispiel:**
```python
from src.assembled_core.logging_config import setup_logging, generate_run_id
import logging

run_id = generate_run_id(prefix="backtest")
setup_logging(run_id=run_id, level="INFO")
logger = logging.getLogger(__name__)
logger.info("Pipeline started")
```

**Log-Dateien**: `logs/{run_id}.log` (z. B. `logs/backtest_20250115_143022_abc12345.log`)

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
