# Backend Architecture - Assembled Trading AI

## Übersicht

Das Backend von Assembled Trading AI ist eine **file-based Trading-Pipeline** mit einem **read-only FastAPI-Server**. Es verarbeitet Marktdaten, generiert Trading-Signale, simuliert Backtests und Portfolio-Performance, und stellt die Ergebnisse über eine REST-API bereit.

**Kernprinzipien:**
- **Single Source of Truth:** `src/assembled_core/` enthält alle Kernlogik
- **File-based:** Keine Datenbank, alle Daten in CSV/Parquet-Dateien
- **SAFE-Bridge:** Keine Live-Trading-Anbindung, nur Simulation via `orders_*.csv`
- **Offline-first:** Lokale Daten bevorzugt, Netz-Calls nur in Pull-Skripten

---

## Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                      │
│  (scripts/live/pull_intraday.py, pull_intraday_av.py)       │
│  → data/raw/1min/*.parquet                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Resampling Layer                          │
│  (scripts/run_all_sprint10.ps1 - inline Python)             │
│  → output/aggregates/5min.parquet, daily.parquet            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Core Pipeline (src/assembled_core/)            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Signals     │→ │   Orders     │→ │  Backtest   │       │
│  │  (EMA)       │  │  Generation  │  │  Simulation │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Portfolio   │→ │     QA       │→ │   Reports    │       │
│  │  (Costs)     │  │  Health      │  │  Generation  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  → output/orders_{freq}.csv                                  │
│  → output/equity_curve_{freq}.csv                           │
│  → output/portfolio_equity_{freq}.csv                       │
│  → output/performance_report_{freq}.md                      │
│  → output/portfolio_report_{freq}.md                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend (src/assembled_core/api/)      │
│                                                              │
│  GET /api/v1/signals/{freq}                                 │
│  GET /api/v1/orders/{freq}                                  │
│  GET /api/v1/portfolio/{freq}/current                       │
│  GET /api/v1/performance/{freq}/metrics                     │
│  GET /api/v1/risk/{freq}/summary                            │
│  GET /api/v1/qa/status                                      │
│                                                              │
│  → Liest direkt aus output/* Dateien                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Datenfluss

### 1. Data Ingestion

**Scripts:**
- `scripts/live/pull_intraday.py` - Yahoo Finance (1-Minuten-Daten)
- `scripts/live/pull_intraday_av.py` - Alpha Vantage (Fallback)

**Output:**
- `data/raw/1min/{SYMBOL}.parquet` - Rohdaten pro Symbol

**Format:**
- Spalten: `timestamp` (UTC), `symbol`, `open`, `high`, `low`, `close`, `volume`

### 2. Resampling

**Script:**
- `scripts/run_all_sprint10.ps1` - Inline Python für 1m → 5m Resampling

**Output:**
- `output/aggregates/5min.parquet` - Aggregierte 5-Minuten-Daten
- `output/aggregates/daily.parquet` - Tägliche Daten (falls vorhanden)

**Format:**
- Spalten: `symbol`, `timestamp` (UTC), `close`
- Sortiert: `timestamp`, `symbol` aufsteigend

### 3. Signal Generation

**Module:** `src/assembled_core/pipeline/signals.py`

**Funktion:** `compute_ema_signals(prices, fast, slow)`

**Strategie:** EMA-Crossover (Fast EMA kreuzt Slow EMA)

**Output:** Signale mit `sig` (-1, 0, +1) für BUY/NEUTRAL/SELL

### 4. Order Generation

**Module:** `src/assembled_core/pipeline/orders.py`

**Funktion:** `signals_to_orders(signals)`

**Output:**
- `output/orders_{freq}.csv`
- Spalten: `timestamp`, `symbol`, `side` (BUY/SELL), `qty`, `price`

### 5. Backtest Simulation

**Module:** `src/assembled_core/pipeline/backtest.py`

**Funktion:** `simulate_equity(prices, orders, start_capital)`

**Output:**
- `output/equity_curve_{freq}.csv` - Equity-Kurve ohne Kosten
- `output/performance_report_{freq}.md` - Performance-Metriken

### 6. Portfolio Simulation

**Module:** `src/assembled_core/pipeline/portfolio.py`

**Funktion:** `simulate_with_costs(orders, start_capital, commission_bps, spread_w, impact_w, freq)`

**Kostenmodell:**
- Commission (Basis-Punkte)
- Spread (Bid/Ask-Spread)
- Market Impact (Preisabschlag)

**Output:**
- `output/portfolio_equity_{freq}.csv` - Equity-Kurve mit Kosten
- `output/portfolio_report_{freq}.md` - Portfolio-Metriken (PF, Sharpe, Trades)

### 7. QA/Health Checks

**Module:** `src/assembled_core/qa/health.py`

**Funktion:** `aggregate_qa_status(freq)`

**Checks:**
- Prices: Datei-Existenz, Schema, Datenqualität
- Orders: Datei-Existenz, Schema, Validierung
- Portfolio: Datei-Existenz, Equity-Kurve, Metriken

**Output:** JSON-Status mit `overall_status` (ok/warning/error)

---

## EOD Pipeline Orchestration

**Script:** `scripts/run_eod_pipeline.py`

**Funktion:** Führt alle Pipeline-Schritte in einem Lauf aus:
1. Preis-Daten-Prüfung
2. Execute (Signale → Orders)
3. Backtest (Equity ohne Kosten)
4. Portfolio (Equity mit Kosten)
5. QA (Health-Checks)

**Output:**
- Alle Pipeline-Outputs (siehe oben)
- `output/run_manifest_{freq}.json` - Maschinenlesbares Run-Manifest

**Verwendung:**
```bash
python scripts/run_eod_pipeline.py --freq 1d --start-capital 10000
```

Siehe auch: [EOD Pipeline Dokumentation](eod_pipeline.md)

---

## FastAPI Backend

**Module:** `src/assembled_core/api/`

**App Factory:** `api.app.create_app()`

**Routers:**
- `api/routers/signals.py` - Trading-Signale
- `api/routers/orders.py` - Generierte Orders
- `api/routers/portfolio.py` - Portfolio-Zustand und Equity-Kurve
- `api/routers/performance.py` - Backtest-Performance-Metriken
- `api/routers/risk.py` - Risk-Metriken (Sharpe, Drawdown, VaR)
- `api/routers/qa.py` - QA/Health-Status

**Start:**
```bash
python scripts/run_api.py
```

**URLs:**
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs` (Swagger UI)
- ReDoc: `http://localhost:8000/redoc`

Siehe auch: [Backend API Dokumentation](backend_api.md)

---

## Module-Struktur

**Kern-Package:** `src/assembled_core/`

**Hauptmodule:**
- `pipeline/` - Core Trading-Pipeline (I/O, Signale, Orders, Backtest, Portfolio)
- `api/` - FastAPI Backend (App, Models, Routers)
- `qa/` - QA/Health-Checks
- `config.py` - Zentrale Konfiguration (OUTPUT_DIR, SUPPORTED_FREQS)
- `costs.py` - Cost-Model-Konfiguration
- `ema_config.py` - EMA-Parameter-Konfiguration

**Zukünftige Module (Skelett vorhanden):**
- `data/` - Data Ingestion (Multi-Source)
- `features/` - Technical Analysis Features
- `signals/` - Signal-Generation-Framework
- `portfolio/` - Portfolio-Management
- `execution/` - Order-Execution-Logik
- `reports/` - Report-Generierung

Siehe auch: [Backend Modules Dokumentation](BACKEND_MODULES.md)

---

## Konfiguration

**Zentrale Config:** `src/assembled_core/config.py`

**Konstanten:**
- `OUTPUT_DIR` - Pfad zum `output/` Verzeichnis
- `SUPPORTED_FREQS` - Unterstützte Frequenzen (`("1d", "5min")`)

**Cost Model:** `src/assembled_core/costs.py`
- `get_default_cost_model()` - Standard-Kosten-Parameter

**EMA Config:** `src/assembled_core/ema_config.py`
- `get_default_ema_config(freq)` - Frequenz-spezifische EMA-Parameter

Siehe auch: [Backend Core Dokumentation](backend_core.md)

---

## Datenquellen

**Aktuell:**
- **Yahoo Finance** (`yfinance`) - Primärquelle für Intraday-Daten
- **Alpha Vantage** - Fallback bei Rate-Limits
- **Lokale Dateien** - Parquet/CSV für Offline-Betrieb

**Geplant:**
- Fundamentals (z. B. SEC Filings)
- Insider-Transaktionen
- Congress Trading
- Shipping-Daten
- News-Feeds

Siehe auch: [Data Sources Dokumentation](DATA_SOURCES_BACKEND.md)

---

## Scripts & Entry Points

**Pipeline-Scripts:**
- `scripts/sprint9_execute.py` - Signal-Generierung und Order-Erstellung
- `scripts/sprint9_backtest.py` - Backtest-Simulation
- `scripts/sprint10_portfolio.py` - Portfolio-Simulation mit Kosten
- `scripts/run_eod_pipeline.py` - Vollständige EOD-Pipeline

**API:**
- `scripts/run_api.py` - FastAPI-Server starten

**Data Ingestion:**
- `scripts/live/pull_intraday.py` - Yahoo Finance Pull
- `scripts/live/pull_intraday_av.py` - Alpha Vantage Pull

---

## Weiterführende Dokumentation

- [Backend Modules](BACKEND_MODULES.md) - Detaillierte Modulübersicht
- [Backend API](backend_api.md) - FastAPI-Endpoints
- [Backend Core](backend_core.md) - Konfiguration & Testing
- [EOD Pipeline](eod_pipeline.md) - Pipeline-Orchestrierung
- [Data Sources](DATA_SOURCES_BACKEND.md) - Datenquellen-Übersicht
- [Backend Roadmap](BACKEND_ROADMAP.md) - Entwicklungs-Roadmap

---

## Design-Prinzipien

1. **Separation of Concerns:** Jedes Modul hat eine klare Verantwortlichkeit
2. **Pure Functions:** Pipeline-Funktionen sind möglichst pure (keine Seiteneffekte)
3. **File-based I/O:** Alle Daten in CSV/Parquet, keine Datenbank
4. **Type Hints:** Vollständige Type-Annotations für bessere IDE-Unterstützung
5. **Testability:** Module sind testbar ohne externe Abhängigkeiten
6. **Backwards Compatibility:** Bestehende Scripts/APIs bleiben funktionsfähig

---

## Nächste Schritte

Siehe [Backend Roadmap](BACKEND_ROADMAP.md) für geplante Erweiterungen und Phasen.

