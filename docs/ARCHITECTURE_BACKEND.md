# Backend Architecture - Assembled Trading AI

**Letzte Aktualisierung:** 2025-01-15

## Überblick

Das Backend von Assembled Trading AI ist ein modulares Trading-Core-System, das folgende Hauptfunktionen bereitstellt:

- **Daten-Ingest**: Laden und Verarbeitung von Preis-Daten (EOD, Intraday)
- **Feature-Engineering**: Technische Indikatoren (TA-Features)
- **Signal-Generierung**: Trading-Signale basierend auf Strategien (z.B. Trend-Baseline)
- **Backtesting**: Portfolio-Simulation mit Kostenmodellen
- **QA & Metriken**: Performance-Metriken, QA-Gates, Health-Checks
- **Reports**: Automatische QA-Reports und Performance-Analysen
- **API**: FastAPI-basierte REST-API für Lese-Zugriffe

## High-Level-Komponenten

### 1. `assembled_core.data` - Daten-Ingest

**Zweck**: Laden und Verarbeitung von Preis-Daten aus verschiedenen Quellen.

**Module**:
- `prices_ingest.py`: 
  - `load_eod_prices()`: Lädt EOD-Preise aus Parquet-Dateien
  - `load_eod_prices_for_universe()`: Lädt Preise für ein Universe (z.B. watchlist.txt)

**Unterstützte Frequenzen**: `1d` (täglich), `5min` (5-Minuten)

**Datenquellen** (aktuell):
- Lokale Parquet-Dateien (`output/aggregates/daily.parquet`, `output/aggregates/5min.parquet`)
- Universe-Dateien (z.B. `watchlist.txt`)

**Phase 6 Erweiterungen** (geplant):
- Insider-Trading-Daten
- Congress-Trading-Daten
- Shipping-Daten
- News-Feeds

---

### 2. `assembled_core.features` - Feature-Engineering

**Zweck**: Berechnung technischer Indikatoren und Event-Features für Trading-Strategien.

**TA-Features** (Technical Analysis):
- `ta_features.py`:
  - `add_log_returns()`: Logarithmische Renditen
  - `add_atr()`: Average True Range
  - `add_rsi()`: Relative Strength Index
  - `add_moving_averages()`: Gleitende Durchschnitte
  - `add_all_features()`: Alle TA-Features auf einmal

**Event-Features** (Phase 6):
- `insider_features.py`:
  - `add_insider_features()`: Insider-Trading-Features (Net-Buy, Trade-Count über 20d/60d)
- `congress_features.py`:
  - `add_congress_features()`: Congressional-Trading-Features (Trade-Count, Total-Amount über 60d/90d)
- `shipping_features.py`:
  - `add_shipping_features()`: Shipping-Route-Features (Congestion-Score, Ships-Count über 7d)
- `news_features.py`:
  - `add_news_features()`: News-Sentiment-Features (Sentiment-Score, News-Count über 7d/30d)

**Verwendung**: Features werden typischerweise vor der Signal-Generierung berechnet. Event-Features erfordern Event-Daten aus den entsprechenden Ingest-Modulen.

---

### 3. `assembled_core.signals` - Signal-Generierung

**Zweck**: Generierung von Trading-Signalen basierend auf Strategien.

**Trend-Strategien**:
- `rules_trend.py`:
  - `generate_trend_signals_from_prices()`: Trend-Baseline-Strategie (EMA-basiert)
  - `generate_trend_signals()`: Trend-Signale mit konfigurierbaren Moving-Average-Parametern

**Event-Strategien** (Phase 6):
- `rules_event_insider_shipping.py`:
  - `generate_event_signals()`: Event-basierte Strategie mit Insider-Trading und Shipping-Daten
  - **Strategie-Logik** (Proof-of-Concept):
    - **LONG**: Starker Insider-Netto-Kauf (net_buy_20d > threshold) + niedrige Shipping-Congestion (< threshold)
    - **SHORT**: Starker Insider-Netto-Verkauf (net_buy_20d < threshold) + hohe Shipping-Congestion (> threshold)
    - **FLAT**: Sonst
  - Nutzt Phase-6-Features: `insider_net_buy_20d`, `shipping_congestion_score_7d`
  - Unterstützt konfigurierbare Gewichte und Thresholds

**Signal-Typen**:
- `LONG`: Kaufsignal
- `FLAT`: Keine Position
- `SHORT`: Verkaufssignal (in Event-Strategien unterstützt)

---

### 4. `assembled_core.portfolio` - Position Sizing

**Zweck**: Berechnung von Ziel-Positionen basierend auf Signalen.

**Module**:
- `position_sizing.py`:
  - `compute_target_positions_equal_weight()`: Gleichgewichtete Positionen
  - `compute_target_positions_top_n()`: Top-N-Positionen
  - `compute_target_positions_from_trend_signals()`: Position Sizing für Trend-Signale

---

### 5. `assembled_core.execution` - Order-Generierung

**Zweck**: Generierung von Orders aus Ziel-Positionen.

**Module**:
- `order_generation.py`: Order-Generierung
- `safe_bridge.py`: Safe-Bridge-Modus (keine echten Orders, nur CSV-Output)

**Output**: Orders werden als CSV-Dateien geschrieben (`output/orders_1d.csv`, `output/orders_5min.csv`)

---

### 6. `assembled_core.qa` - Quality Assurance

**Zweck**: Backtesting, Performance-Metriken, QA-Gates und Health-Checks.

**Module**:

#### `backtest_engine.py`
- `run_portfolio_backtest()`: Portfolio-Level-Backtest-Engine
- `BacktestResult`: Ergebnis-Dataclass

#### `metrics.py`
- `compute_all_metrics()`: Alle Performance-Metriken
- `compute_sharpe_ratio()`: Sharpe-Ratio
- `compute_sortino_ratio()`: Sortino-Ratio
- `compute_equity_metrics()`: Equity-Kurven-Metriken
- `compute_trade_metrics()`: Trade-basierte Metriken

#### `qa_gates.py`
- `evaluate_all_gates()`: Evaluierung aller QA-Gates
- `QAResult`: OK / WARNING / BLOCK
- Gate-Typen: Sharpe, Sortino, Max Drawdown, Turnover, CAGR, etc.

#### `walk_forward.py`
- Walk-Forward-Analyse (geplant)

#### `health.py`
- `check_prices()`: Prüfung von Preis-Dateien
- `check_orders()`: Prüfung von Order-Dateien
- `check_portfolio()`: Prüfung von Portfolio-Dateien
- `aggregate_qa_status()`: Aggregation aller Health-Checks

---

### 7. `assembled_core.pipeline` - Pipeline-Orchestrierung

**Zweck**: Orchestrierung der gesamten Trading-Pipeline.

**Module**:

#### `orchestrator.py`
- `run_eod_pipeline()`: Vollständige EOD-Pipeline (Execute → Backtest → Portfolio → QA)

#### `signals.py`
- Signal-Generierung innerhalb der Pipeline

#### `orders.py`
- Order-Generierung innerhalb der Pipeline

#### `backtest.py`
- Backtest-Simulation innerhalb der Pipeline

#### `portfolio.py`
- Portfolio-Simulation innerhalb der Pipeline

#### `io.py`
- I/O-Funktionen (Laden/Speichern von Daten)

---

### 8. `assembled_core.reports` - Report-Generierung

**Zweck**: Automatische Generierung von QA-Reports und Performance-Analysen.

**Module**:
- `daily_qa_report.py`:
  - `generate_qa_report()`: Generiert QA-Report aus Metriken
  - `generate_qa_report_from_files()`: Generiert QA-Report aus Dateien

**Output**: Markdown-Reports (`output/reports/qa_report_*.md`)

---

### 9. `assembled_core.api` - FastAPI Backend

**Zweck**: REST-API für Lese-Zugriffe auf Pipeline-Outputs.

**Module**:
- `app.py`: FastAPI-App-Factory (`create_app()`)
- `models.py`: Pydantic-Models für API-Requests/Responses
- `routers/`:
  - `signals.py`: Signal-Endpoints
  - `orders.py`: Order-Endpoints
  - `portfolio.py`: Portfolio-Endpoints
  - `performance.py`: Performance-Metriken-Endpoints
  - `risk.py`: Risk-Metriken-Endpoints
  - `qa.py`: QA-Status-Endpoints

**Verwendung**: 
```bash
python scripts/run_api.py
# API läuft auf http://localhost:8000
```

---

### 10. Command Line Interface (CLI)

**Zweck**: Einheitliche Schnittstelle für die wichtigsten Backend-Operationen.

**Modul**: `scripts/cli.py`

**Subcommands**:
- `info`: Zeigt Projekt-Informationen und verfügbare Subcommands
- `run_daily`: Führt die vollständige EOD-Pipeline aus (Execute → Backtest → Portfolio → QA)
- `run_backtest`: Führt einen Strategy-Backtest mit dem Portfolio-Level-Backtest-Engine aus
- `run_phase4_tests`: Führt die Phase-4-Regression-Test-Suite aus (~13s, 110 Tests)

**Architektur-Prinzip**:
- Das CLI ist eine **dünne Orchestrierungsschicht**
- Die eigentliche Business-Logik lebt in den Modulen unter `src/assembled_core/`
- Das CLI ruft Funktionen aus den bestehenden Scripts auf:
  - `run_daily` → `scripts/run_eod_pipeline.run_eod_from_args()`
  - `run_backtest` → `scripts/run_backtest_strategy.run_backtest_from_args()`
  - `run_phase4_tests` → `pytest -m phase4` (via subprocess)

**PowerShell-Wrapper**:
- `scripts/run_phase4_tests.ps1` ist ein dünner Wrapper, der `python scripts/cli.py run_phase4_tests` aufruft
- Weitere PowerShell-Wrapper können analog erstellt werden

**Verwendung**:
```bash
# Projekt-Informationen
python scripts/cli.py info

# EOD-Pipeline
python scripts/cli.py run_daily --freq 1d

# Strategy-Backtest
python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt

# Phase-4-Tests
python scripts/cli.py run_phase4_tests
```

**Weitere Details**: Siehe `docs/CLI_REFERENCE.md`

---

## Architektur-Diagramm (Text-basiert)

```
┌─────────────────────────────────────────────────────────────────┐
│                    User-Facing Scripts                          │
├─────────────────────────────────────────────────────────────────┤
│  cli.py (Central CLI)  │  run_backtest_strategy.py  │  run_api.py│
│  run_eod_pipeline.py   │  run_phase4_tests.ps1      │            │
└──────────────┬──────────────┴──────────┬───────────┴────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                        │
│                    (orchestrator.py)                            │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Data       │  │  Features    │  │   Signals    │          │
│  │  (Ingest)    │→ │  (TA)       │→ │  (Trend)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                 │                                │
│                                 ▼                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Portfolio   │  │  Execution   │  │   Orders     │          │
│  │ (Sizing)     │→ │  (Generate)  │→ │  (CSV)       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                 │                                │
│                                 ▼                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Backtest   │  │     QA       │  │   Reports    │          │
│  │  (Engine)    │→ │ (Metrics/Gates)│→ │  (Markdown)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                              │
│              (Read-only API Access)                             │
└─────────────────────────────────────────────────────────────────┘
```

## Datenfluss

1. **Daten-Ingest**: `data/prices_ingest.py` lädt Preise aus Parquet-Dateien
2. **Feature-Engineering**: `features/ta_features.py` berechnet TA-Indikatoren
3. **Signal-Generierung**: `signals/rules_trend.py` generiert Trading-Signale
4. **Position Sizing**: `portfolio/position_sizing.py` berechnet Ziel-Positionen
5. **Order-Generierung**: `execution/order_generation.py` erzeugt Orders
6. **Backtesting**: `qa/backtest_engine.py` simuliert Portfolio-Performance
7. **QA & Metriken**: `qa/metrics.py` und `qa/qa_gates.py` evaluieren Performance
8. **Reports**: `reports/daily_qa_report.py` generiert QA-Reports
9. **API**: `api/` stellt REST-Endpoints für Lese-Zugriffe bereit

## Konfiguration

- **Zentrale Config**: `config.py` (OUTPUT_DIR, SUPPORTED_FREQS, etc.)
- **EMA-Config**: `ema_config.py` (Moving-Average-Parameter)
- **Cost-Model**: `costs.py` (Commission, Spread, Impact)

## Test-Phasen

Das Projekt verwendet eine strukturierte Test-Phasen-Architektur:

### Phase 4: Backend Core Tests

**Umfang**: ~117 Tests in ~13-17 Sekunden

**Abgedeckte Bereiche**:
- TA-Features (Technical Analysis)
- QA-Metriken (Performance-Metriken)
- QA-Gates (Quality Assurance Checks)
- Backtest-Engine (Portfolio-Level)
- Reports (Daily QA Reports)
- Pipelines (EOD-Pipeline, Backtest-Pipeline)

**Ausführung**:
```bash
# Über CLI (empfohlen)
python scripts/cli.py run_phase4_tests

# Direkt mit pytest
pytest -m phase4
```

**Marker**: `@pytest.mark.phase4`

### Phase 6: Event Features Tests

**Umfang**: ~11 Tests in < 1 Sekunde

**Abgedeckte Bereiche**:
- Insider-Trading-Features
- Congress-Trading-Features
- Shipping-Route-Features
- News-Sentiment-Features

**Ausführung**:
```bash
pytest -m phase6
```

**Marker**: `@pytest.mark.phase6`

**Weitere Details**: Siehe `docs/TESTING_COMMANDS.md`

---

## Phase 6 Erweiterungen

Phase 6 fügt zusätzliche Datenquellen hinzu:

- **Insider-Trading-Daten**: `assembled_core.data.insider_ingest`
- **Congress-Trading-Daten**: `assembled_core.data.congress_trades_ingest`
- **Shipping-Daten**: `assembled_core.data.shipping_routes_ingest`
- **News-Feeds**: `assembled_core.data.news_ingest`

**Feature-Module**:
- `assembled_core.features.insider_features`
- `assembled_core.features.congress_features`
- `assembled_core.features.shipping_features`
- `assembled_core.features.news_features`

Diese werden in die bestehende Pipeline-Architektur integriert.
