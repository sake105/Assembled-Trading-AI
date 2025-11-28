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

**Data Layer:**
- `src/assembled_core/data/prices_ingest.py` - EOD-Preis-Ingestion mit OHLCV
  - `load_eod_prices()` - Lädt EOD-Preise mit vollständigen OHLCV-Daten
  - `load_eod_prices_for_universe()` - Lädt Preise für Symbol-Universum (z. B. watchlist.txt)
  - `validate_price_data()` - Validiert Datenqualität (OHLC-Relationen, NaNs, etc.)

**Features Layer:**
- `src/assembled_core/features/ta_features.py` - Technische Indikatoren
  - `add_log_returns()` - Logarithmische Returns
  - `add_moving_averages()` - Simple Moving Averages (SMA)
  - `add_atr()` - Average True Range (Volatilität)
  - `add_rsi()` - Relative Strength Index (Momentum)
  - `add_all_features()` - Alle Features auf einmal

**Signals Layer:**
- `src/assembled_core/signals/rules_trend.py` - Trend-Following-Signale
  - `generate_trend_signals()` - Generiert LONG/FLAT Signale basierend auf MA-Crossover
  - Signal-Logik: LONG wenn ma_fast > ma_slow AND Volumen über Schwellenwert
  - Score: Signal-Stärke (0.0 bis 1.0) basierend auf MA-Spread und Volumen

**Portfolio Layer:**
- `src/assembled_core/portfolio/position_sizing.py` - Positionsgrößen-Bestimmung
  - `compute_target_positions()` - Berechnet Zielpositionen aus Signalen
  - Strategien: Equal Weight (1/N) oder Score-basiert
  - Top-N Selektion: Wählt beste Signale nach Score
  - Output: DataFrame mit symbol, target_weight, target_qty

**Execution Layer:**
- `src/assembled_core/execution/order_generation.py` - Order-Generierung
  - `generate_orders_from_targets()` - Vergleicht aktuelle vs. Zielpositionen
  - `generate_orders_from_signals()` - Convenience: Signale → Orders in einem Schritt
  - Output: DataFrame mit timestamp, symbol, side, qty, price

**SAFE-Bridge:**
- `src/assembled_core/execution/safe_bridge.py` - SAFE-kompatible Order-Dateien
  - `write_safe_orders_csv()` - Erstellt SAFE-Bridge CSV-Dateien
  - Format: `orders_YYYYMMDD.csv` mit Spalten: Ticker, Side, Quantity, PriceType, Comment
  - Human-in-the-Loop: Alle Orders müssen manuell geprüft werden

**Output:**
- `data/raw/1min/{SYMBOL}.parquet` - Rohdaten pro Symbol
- `data/sample/eod_sample.parquet` - Beispiel-Daten für Tests (2-3 Ticker, ~30 Tage)

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

**Scripts:**
- `scripts/run_eod_pipeline.py` - Vollständige EOD-Pipeline (Execute, Backtest, Portfolio, QA)
- `scripts/run_daily.py` - EOD-MVP Runner (fokussiert auf SAFE-Order-Generierung)

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

---

## Secrets & .env

**Wichtig:** Alle Produktions-Skripte nutzen Umgebungsvariablen für API-Keys und Secrets, niemals hardcodierte Werte.

**Details:**
- Siehe `docs/SECURITY_SECRETS.md` für vollständige Dokumentation
- API-Keys werden über Umgebungsvariablen gesetzt (z. B. `$env:ALPHAVANTAGE_API_KEY`)
- `.env`-Dateien sind lokal und werden nicht in Git getrackt (siehe `.gitignore`)
- Konfigurationsdateien (`config/datasource.psd1`) nutzen Platzhalter: `$env:VARIABLE_NAME`

**Grundregel:** Niemals Secrets im Code oder in versionierten Dateien speichern.

---

## Logging & Error Handling

**Zentrale Logging-Konfiguration:**
- `src/assembled_core/logging_utils.py` - Logging-Helper-Modul
  - `setup_logging(level="INFO")` - Richtet einheitliches Logging ein
  - Format: `[LEVEL] message`
  - Levels: INFO (normal), WARNING (soft problems), ERROR (hard errors)

**Verwendung in CLI-Scripts:**
```python
from src.assembled_core.logging_utils import setup_logging

logger = setup_logging(level="INFO")
logger.info("Starting pipeline")
logger.error("Error occurred")
sys.exit(1)  # Bei Fehlern
```

**Fehlerbehandlung:**
- Klare ERROR-Log-Meldungen
- `sys.exit(1)` für Fehler, `sys.exit(0)` für Erfolg
- Keine unformatierten Tracebacks im Normalfall

**Betroffene Scripts:**
- `scripts/run_daily.py` - EOD-MVP Runner
- `scripts/run_eod_pipeline.py` - Vollständige EOD-Pipeline

Siehe auch: [Backend Core - Logging](backend_core.md#logging--error-handling-in-cli-scripts)

---

## CI / GitHub Actions

**Wichtig:** Alle Pull Requests müssen die CI-Checks bestehen, bevor sie gemerged werden.

### Workflow

**Datei:** `.github/workflows/backend-ci.yml`

**Trigger:**
- Push zu `main` oder `develop` Branch
- Pull Requests zu `main` oder `develop` Branch

**Python-Versionen:**
- Aktuell: Python 3.10
- Zukünftig erweiterbar auf 3.11, 3.12

### CI-Checks

**1. Tests:**
- Führt `pytest tests/` aus
- Prüft Unit-Tests und Smoke-Tests
- Muss erfolgreich sein (Exit-Code 0)

**2. Ruff (Linting):**
- Führt `ruff check src tests scripts` aus
- Prüft Code-Style, Syntax-Fehler, unbenutzte Imports
- Muss erfolgreich sein (Exit-Code 0)

**3. mypy (Type Checking):**
- Führt `mypy` auf Kernmodulen aus:
  - `src/assembled_core/data`
  - `src/assembled_core/features`
  - `src/assembled_core/signals`
  - `src/assembled_core/execution`
  - `src/assembled_core/portfolio`
- Aktuell optional (`continue-on-error: true`)
- Zukünftig kann dies zu einem Hard-Requirement werden

### Lokales Testen

**Vor dem Push:**
```bash
# Tests lokal ausführen
pytest tests/

# Ruff lokal prüfen
ruff check src tests scripts

# mypy lokal prüfen
mypy src/assembled_core/data src/assembled_core/features src/assembled_core/signals src/assembled_core/execution src/assembled_core/portfolio
```

**Empfehlung:** Führe diese Checks lokal aus, bevor du einen PR erstellst.

### PR-Merge-Policy

**Regel:** PRs sollten nur gemerged werden, wenn:
- Alle CI-Checks grün sind (✓)
- Tests erfolgreich
- Ruff erfolgreich
- (Optional) mypy erfolgreich oder akzeptable Warnungen

**Ausnahmen:**
- Nur in Ausnahmefällen (z. B. Dokumentations-Only-Änderungen) können Checks übersprungen werden
- Im Normalfall: CI muss grün sein

---

## Weiterführende Dokumente

- [BACKEND_MODULES.md](BACKEND_MODULES.md): Detaillierte Beschreibung der einzelnen Module.
- [BACKEND_ROADMAP.md](BACKEND_ROADMAP.md): Die Entwicklungs-Roadmap des Backends.
- [DATA_SOURCES_BACKEND.md](DATA_SOURCES_BACKEND.md): Übersicht über alle Datenquellen.
- [backend_core.md](backend_core.md): Details zur Kernkonfiguration und Test-Suite.
- [backend_api.md](backend_api.md): Detaillierte API-Dokumentation.
- [eod_pipeline.md](eod_pipeline.md): Detaillierte Beschreibung der EOD-Pipeline.
- [SECURITY_SECRETS.md](SECURITY_SECRETS.md): Secrets-Management und Best Practices.

