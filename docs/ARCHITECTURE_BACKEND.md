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

**Zweck**: Berechnung technischer Indikatoren für Trading-Strategien.

**Module**:
- `ta_features.py`:
  - `add_log_returns()`: Logarithmische Renditen
  - `add_atr()`: Average True Range
  - `add_rsi()`: Relative Strength Index
  - `add_moving_averages()`: Gleitende Durchschnitte
  - `add_all_features()`: Alle TA-Features auf einmal

**Verwendung**: Features werden typischerweise vor der Signal-Generierung berechnet.

---

### 3. `assembled_core.signals` - Signal-Generierung

**Zweck**: Generierung von Trading-Signalen basierend auf Strategien.

**Module**:
- `rules_trend.py`:
  - `generate_trend_signals_from_prices()`: Trend-Baseline-Strategie (EMA-basiert)

**Signal-Typen**:
- `LONG`: Kaufsignal
- `FLAT`: Keine Position
- `SHORT`: Verkaufssignal (geplant)

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

## Architektur-Diagramm (Text-basiert)

```
┌─────────────────────────────────────────────────────────────────┐
│                    User-Facing Scripts                          │
├─────────────────────────────────────────────────────────────────┤
│  run_backtest_strategy.py  │  run_eod_pipeline.py  │  run_api.py│
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

## Phase 6 Erweiterungen (geplant)

Phase 6 wird zusätzliche Datenquellen hinzufügen:

- **Insider-Trading-Daten**: `assembled_core.data.insider`
- **Congress-Trading-Daten**: `assembled_core.data.congress`
- **Shipping-Daten**: `assembled_core.data.shipping`
- **News-Feeds**: `assembled_core.data.news`

Diese werden in die bestehende Pipeline-Architektur integriert.
