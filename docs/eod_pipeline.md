# EOD Pipeline Orchestration

## Übersicht

Das EOD-Pipeline-Skript (`scripts/run_eod_pipeline.py`) führt die gesamte Pipeline in einem einzigen Lauf aus und erstellt ein maschinenlesbares Run-Manifest.

**Zweck:** Automatisierte End-to-End-Ausführung der Trading-Pipeline mit konsistenter Fehlerbehandlung und Manifest-Generierung.

---

## Verwendung

### Basis-Kommando

```bash
python scripts/run_eod_pipeline.py --freq 1d --start-capital 10000
```

### Optionen

**Erforderlich:**
- `--freq`: Trading-Frequenz (`"1d"` oder `"5min"`)

**Optional:**
- `--start-capital`: Startkapital (default: 10000.0)
- `--skip-backtest`: Backtest-Schritt überspringen
- `--skip-portfolio`: Portfolio-Schritt überspringen
- `--skip-qa`: QA-Schritt überspringen
- `--price-file`: Expliziter Pfad zur Preis-Datei
- `--commission-bps`: Commission in Basis-Punkten (default: aus Cost-Model)
- `--spread-w`: Spread-Weight (default: aus Cost-Model)
- `--impact-w`: Impact-Weight (default: aus Cost-Model)
- `--out`: Output-Verzeichnis (default: aus config.OUTPUT_DIR)

### Beispiele

**Vollständiger Lauf (1d):**
```bash
python scripts/run_eod_pipeline.py --freq 1d --start-capital 10000
```

**5-Minuten-Pipeline:**
```bash
python scripts/run_eod_pipeline.py --freq 5min --start-capital 10000
```

**Ohne Backtest:**
```bash
python scripts/run_eod_pipeline.py --freq 1d --skip-backtest
```

**Mit custom Cost-Parametern:**
```bash
python scripts/run_eod_pipeline.py --freq 1d --commission-bps 0.5 --spread-w 0.3 --impact-w 0.6
```

---

## Pipeline-Schritte

Die Pipeline führt folgende Schritte in dieser Reihenfolge aus:

### 1. Preis-Daten-Prüfung

- Prüft, ob Preis-Daten für die gegebene Frequenz existieren
- Bei Fehlern: Setzt `failure_flag`, fährt aber fort (schreibt Manifest)

### 2. Execute

- Lädt Preise
- Berechnet EMA-Signale
- Generiert Orders
- Schreibt `output/orders_{freq}.csv`

### 3. Backtest

- Simuliert Equity-Kurve ohne Kosten
- Berechnet Performance-Metriken
- Schreibt:
  - `output/equity_curve_{freq}.csv`
  - `output/performance_report_{freq}.md`

### 4. Portfolio

- Simuliert Portfolio mit Transaktionskosten
- Schreibt:
  - `output/portfolio_equity_{freq}.csv`
  - `output/portfolio_report_{freq}.md`

### 5. QA

- Führt alle QA-Checks aus (prices, orders, portfolio)
- Aggregiert Status

---

## Run-Manifest

Nach jedem Lauf wird ein Run-Manifest als JSON geschrieben:

**Pfad:** `output/run_manifest_{freq}.json`

**Beispiel-Inhalt:**
```json
{
  "freq": "1d",
  "start_capital": 10000.0,
  "completed_steps": [
    "execute",
    "backtest",
    "portfolio",
    "qa"
  ],
  "qa_overall_status": "ok",
  "qa_checks": [
    {
      "name": "prices",
      "status": "ok",
      "message": "Price file OK: 100 rows, 5 symbols",
      "details": {
        "file": "output/aggregates/daily.parquet",
        "rows": 100,
        "symbols": 5
      }
    },
    {
      "name": "orders",
      "status": "ok",
      "message": "Orders file OK: 10 orders",
      "details": {
        "file": "output/orders_1d.csv",
        "rows": 10
      }
    },
    {
      "name": "portfolio",
      "status": "ok",
      "message": "Portfolio file OK: 50 rows",
      "details": {
        "file": "output/portfolio_equity_1d.csv",
        "rows": 50
      }
    }
  ],
  "timestamps": {
    "started": "2025-11-28T16:00:00Z",
    "finished": "2025-11-28T16:05:00Z"
  },
  "failure": false
}
```

**Felder:**
- `freq`: Trading-Frequenz
- `start_capital`: Verwendetes Startkapital
- `completed_steps`: Liste der erfolgreich abgeschlossenen Schritte
- `qa_overall_status`: QA-Overall-Status ("ok", "warning", "error")
- `qa_checks`: Liste der QA-Check-Ergebnisse
- `timestamps`: Start- und End-Zeitstempel (ISO 8601)
- `failure`: Boolean, ob Fehler aufgetreten sind

---

## Exit-Codes

- `0`: Erfolgreich (alle Schritte abgeschlossen, keine Fehler)
- `1`: Fehler aufgetreten (ein oder mehrere Schritte fehlgeschlagen)

---

## Integration mit FastAPI

Das Run-Manifest kann von der FastAPI-API gelesen werden, um den Status des letzten Pipeline-Laufs zu ermitteln:

```python
import json
from pathlib import Path

manifest_path = Path("output/run_manifest_1d.json")
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Last run: {manifest['timestamps']['finished']}")
    print(f"QA status: {manifest['qa_overall_status']}")
```

---

## Orchestrator-Modul

Die Pipeline-Logik ist in `src/assembled_core/pipeline/orchestrator.py` implementiert:

**Funktionen:**
- `run_execute_step()`: Führt Execute-Schritt aus
- `run_backtest_step()`: Führt Backtest-Schritt aus
- `run_portfolio_step()`: Führt Portfolio-Schritt aus
- `run_eod_pipeline()`: Führt gesamte Pipeline aus

Diese Funktionen können auch programmatisch verwendet werden (z.B. in Tests oder anderen Skripten).

---

## Portfolio-Level Backtest Engine

### Übersicht

Die Backtest-Engine (`src/assembled_core/qa/backtest_engine.py`) bietet eine generische, flexible Backtest-Infrastruktur für Portfolio-Level-Strategien.

**Zweck:**
- Orchestriert den kompletten Backtest-Workflow (Features → Signale → Position-Sizing → Orders → Equity)
- Unterstützt custom Signal- und Position-Sizing-Funktionen
- Ermöglicht Strategie-Experimente ohne Code-Änderungen

**Integration mit EOD-Pipeline:**

**1. Integration in `run_eod_pipeline.py`:**
- Die Backtest-Engine kann als Alternative zu `run_backtest_step()` verwendet werden
- Ermöglicht flexiblere Backtest-Szenarien mit custom Signal-Funktionen
- Beispiel: Strategie-Vergleich mit verschiedenen Signal-Parametern

**2. Integration in `run_daily.py`:**
- Optional: Nach Order-Generierung einen schnellen Backtest durchführen
- Validierung der erwarteten Performance vor manueller Prüfung
- Beispiel: "Wie würde diese Order-Liste performen?"

### Verwendung

**Basis-Beispiel:**
```python
from src.assembled_core.qa.backtest_engine import run_portfolio_backtest
from src.assembled_core.data.prices_ingest import load_eod_prices
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices
from src.assembled_core.portfolio.position_sizing import compute_target_positions

# Preise laden
prices = load_eod_prices(freq="1d")

# Signal-Funktion definieren
def signal_fn(prices_df):
    return generate_trend_signals_from_prices(prices_df, ma_fast=20, ma_slow=50)

# Position-Sizing-Funktion definieren
def sizing_fn(signals_df, capital):
    return compute_target_positions(signals_df, total_capital=capital, equal_weight=True)

# Backtest ausführen
result = run_portfolio_backtest(
    prices=prices,
    signal_fn=signal_fn,
    position_sizing_fn=sizing_fn,
    start_capital=10000.0,
    include_costs=True,
    include_trades=True
)

# Ergebnisse auswerten
print(f"Final PF: {result.metrics['final_pf']:.4f}")
print(f"Sharpe: {result.metrics['sharpe']:.4f}")
print(f"Trades: {result.metrics['trades']}")
```

**Mit custom Features:**
```python
result = run_portfolio_backtest(
    prices=prices,
    signal_fn=signal_fn,
    position_sizing_fn=sizing_fn,
    start_capital=10000.0,
    compute_features=True,
    feature_config={
        "ma_windows": (10, 30, 100),
        "atr_window": 20,
        "rsi_window": 14,
        "include_rsi": True
    }
)
```

**Ohne Kosten (kostenfreie Simulation):**
```python
result = run_portfolio_backtest(
    prices=prices,
    signal_fn=signal_fn,
    position_sizing_fn=sizing_fn,
    start_capital=10000.0,
    include_costs=False  # Nutzt pipeline.backtest.simulate_equity
)
```

### Eingaben

**Erforderlich:**
- `prices`: DataFrame mit OHLCV-Daten (timestamp, symbol, close, ...)
- `signal_fn`: Callable, das Preise in Signale umwandelt
- `position_sizing_fn`: Callable, das Signale + Kapital in Zielpositionen umwandelt

**Optional:**
- `start_capital`: Startkapital (default: 10000.0)
- `commission_bps`, `spread_w`, `impact_w`: Kosten-Parameter
- `cost_model`: CostModel-Instanz (alternativ zu einzelnen Parametern)
- `include_costs`: Ob Kosten berücksichtigt werden sollen (default: True)
- `include_trades`: Ob Trade-Liste im Result enthalten sein soll (default: False)
- `include_signals`: Ob Signal-Liste im Result enthalten sein soll (default: False)
- `include_targets`: Ob Zielpositionen im Result enthalten sein sollen (default: False)
- `compute_features`: Ob TA-Features berechnet werden sollen (default: True)
- `feature_config`: Konfiguration für Feature-Computation

### Ausgaben

**BacktestResult:**
- `equity`: DataFrame mit Spalten: `date`, `timestamp`, `equity`, `daily_return`
  - `date`: Date-Objekt (date)
  - `timestamp`: pd.Timestamp (UTC)
  - `equity`: Portfolio-Equity-Wert
  - `daily_return`: Tägliche Rendite (pct_change)
- `metrics`: Dictionary mit Performance-Metriken
  - `final_pf`: Final Performance Factor (equity[-1] / equity[0])
  - `sharpe`: Sharpe Ratio
  - `trades`: Anzahl Trades
- `trades`: Optional DataFrame (timestamp, symbol, side, qty, price)
- `signals`: Optional DataFrame (timestamp, symbol, direction, score)
- `target_positions`: Optional DataFrame (symbol, target_weight, target_qty)

### Vorteile

**Flexibilität:**
- Custom Signal- und Sizing-Funktionen ermöglichen schnelle Strategie-Experimente
- Keine Code-Änderungen nötig für neue Strategien

**Komposabilität:**
- Nutzt bestehende Module (keine Duplikation)
- Kann mit jedem Signal- oder Sizing-Algorithmus kombiniert werden

**Vollständigkeit:**
- Equity-Kurve mit täglichen Returns
- Performance-Metriken
- Optionale Details (Trades, Signale, Zielpositionen)

**Testbarkeit:**
- Kann mit synthetischen Daten getestet werden
- Offline-first (keine Netzwerkzugriffe)

### Integration in Pipeline-Scripts

**Zukünftige Erweiterung von `run_eod_pipeline.py`:**
```python
# Optional: Backtest-Engine für flexible Strategie-Tests
from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

# Custom Signal-Funktion
def custom_signal_fn(prices_df):
    # Eigene Signal-Logik
    return signals

# Backtest mit custom Strategie
result = run_portfolio_backtest(
    prices=prices,
    signal_fn=custom_signal_fn,
    position_sizing_fn=sizing_fn,
    start_capital=10000.0
)
```

**Zukünftige Erweiterung von `run_daily.py`:**
```python
# Optional: Schneller Backtest nach Order-Generierung
result = run_portfolio_backtest(
    prices=filtered_prices,
    signal_fn=signal_fn,
    position_sizing_fn=sizing_fn,
    start_capital=10000.0,
    include_costs=True,
    include_trades=True
)

# Logge erwartete Performance
logger.info(f"Expected PF: {result.metrics['final_pf']:.4f}")
logger.info(f"Expected Sharpe: {result.metrics['sharpe']:.4f}")
```

---

## Fehlerbehandlung

- **Preis-Daten fehlen:** Fehler wird geloggt, `failure_flag` gesetzt, Pipeline fährt fort
- **Execute-Fehler:** Fehler wird geloggt, `failure_flag` gesetzt, Pipeline fährt fort
- **Backtest-Fehler:** Fehler wird geloggt, `failure_flag` gesetzt, Portfolio wird trotzdem versucht
- **Portfolio-Fehler:** Fehler wird geloggt, `failure_flag` gesetzt, QA wird trotzdem versucht
- **QA-Fehler:** Fehler wird geloggt, `failure_flag` gesetzt, Manifest wird trotzdem geschrieben

**Prinzip:** Pipeline versucht so viele Schritte wie möglich auszuführen, auch wenn frühere Schritte fehlschlagen. Das Manifest dokumentiert, welche Schritte erfolgreich waren.

---

## Tests

Tests befinden sich in `tests/test_run_eod_pipeline.py`:

- `test_run_eod_pipeline_smoke()`: Vollständiger Pipeline-Lauf mit synthetischen Daten
- `test_run_eod_pipeline_skip_steps()`: Pipeline mit übersprungenen Schritten

**Ausführen:**
```bash
pytest tests/test_run_eod_pipeline.py
```

---

## Beispiel-Workflow

### 1. Pipeline ausführen

```bash
python scripts/run_eod_pipeline.py --freq 1d --start-capital 10000
```

### 2. Manifest prüfen

```bash
cat output/run_manifest_1d.json
```

### 3. API abfragen

```bash
# QA-Status
curl http://localhost:8000/api/v1/qa/status?freq=1d

# Portfolio-Zustand
curl http://localhost:8000/api/v1/portfolio/1d/current
```

---

## EOD-MVP (run_daily.py)

### Übersicht

Das EOD-MVP-Skript (`scripts/run_daily.py`) ist ein fokussierter Runner für tägliche Order-Generierung, der die neuen modularen Layer aus Phase 3 nutzt:

- **data.prices_ingest**: Lädt EOD-Preise mit OHLCV
- **features.ta_features**: Berechnet technische Indikatoren
- **signals.rules_trend**: Generiert Trend-Following-Signale
- **portfolio.position_sizing**: Bestimmt Zielpositionen
- **execution.order_generation**: Generiert Orders aus Zielpositionen
- **execution.safe_bridge**: Schreibt SAFE-Bridge-kompatible CSV-Dateien

**Unterschied zu `run_eod_pipeline.py`:**
- `run_eod_pipeline.py`: Vollständige Pipeline mit Backtest, Portfolio-Simulation, QA
- `run_daily.py`: Fokussiertes EOD-MVP, das nur SAFE-Orders generiert (ohne Backtest/Portfolio-Equity)

**Zukünftige Integration mit Backtest-Engine:**
- Die Backtest-Engine (`qa.backtest_engine`) kann optional in `run_daily.py` integriert werden, um nach der Order-Generierung einen schnellen Backtest durchzuführen
- Dies würde es ermöglichen, die erwartete Performance der generierten Orders zu validieren, bevor sie manuell geprüft werden

### Verwendung

**Basis-Kommando:**
```bash
python scripts/run_daily.py --date 2025-01-15
```

**Mit Optionen:**
```bash
python scripts/run_daily.py --date 2025-01-15 --top-n 5 --ma-fast 20 --ma-slow 50
```

**Ohne Datum (heute):**
```bash
python scripts/run_daily.py
```

### Optionen

**Optional:**
- `--date`: Datum (YYYY-MM-DD), default: heute
- `--universe`: Pfad zur Universe-Datei (default: watchlist.txt)
- `--price-file`: Expliziter Pfad zur Preis-Datei
- `--out`: Output-Verzeichnis (default: config.OUTPUT_DIR)
- `--total-capital`: Gesamtkapital für Position-Sizing (default: 1.0)
- `--top-n`: Maximale Anzahl Positionen (default: None = alle)
- `--ma-fast`: Fast Moving Average Fenster (default: 20)
- `--ma-slow`: Slow Moving Average Fenster (default: 50)
- `--min-score`: Minimum Signal-Score-Schwellenwert (default: 0.0)

### Ablauf

1. **EOD-Preise laden**: `data.prices_ingest.load_eod_prices_for_universe()`
2. **TA-Features berechnen**: `features.ta_features.add_all_features()`
3. **Trend-Signale generieren**: `signals.rules_trend.generate_trend_signals_from_prices()`
4. **Zielpositionen bestimmen**: `portfolio.position_sizing.compute_target_positions_from_trend_signals()`
5. **Orders generieren**: `execution.order_generation.generate_orders_from_signals()`
6. **SAFE-Orders schreiben**: `execution.safe_bridge.write_safe_orders_csv()`

### Output

- **SAFE-Orders CSV**: `output/orders_YYYYMMDD.csv`
  - Format: `Ticker`, `Side`, `Quantity`, `PriceType`, `Comment`
  - Human-in-the-Loop: Alle Orders müssen manuell geprüft werden

### Beispiele

**Standard-EOD-MVP:**
```bash
python scripts/run_daily.py --date 2025-01-15
```

**Mit Top-5 Selektion:**
```bash
python scripts/run_daily.py --date 2025-01-15 --top-n 5
```

**Mit benutzerdefinierten MA-Parametern:**
```bash
python scripts/run_daily.py --date 2025-01-15 --ma-fast 10 --ma-slow 30
```

**Mit expliziter Preis-Datei:**
```bash
python scripts/run_daily.py --date 2025-01-15 --price-file data/sample/eod_sample.parquet
```

### Datums-Handling

**`--date` Parameter:**
- Format: `YYYY-MM-DD` (z. B. `2025-01-15`)
- Default: Heute (UTC)
- Bedeutung: Handelstag, für den Orders erstellt werden sollen

**Daten-Filterung:**
- Das Script verwendet die **letzten verfügbaren Daten <= target_date** (pro Symbol)
- Falls für den exakten target_date keine Daten vorhanden sind, wird der letzte verfügbare Tag verwendet
- Dies stellt sicher, dass auch bei fehlenden Tagesdaten die neuesten verfügbaren Daten verwendet werden

**Beispiel:**
```bash
# Orders für 2025-01-15 generieren
# Verwendet letzte verfügbare Daten <= 2025-01-15 pro Symbol
python scripts/run_daily.py --date 2025-01-15
```

### Fehlerverhalten

**Universe-Symbole ohne Daten:**
- Wenn Symbole im Universe-File sind, für die keine Preis-Daten existieren:
  - **WARNUNG** wird geloggt mit Liste der betroffenen Symbole
  - Diese Symbole werden aus dem Flow entfernt (keine Orders generiert)
  - Script fährt mit verbleibenden Symbolen fort

**Kein Symbol übrig nach Filtering:**
- Wenn nach dem Filtering keine Symbole mehr übrig bleiben:
  - Script bricht sauber ab mit Exit-Code 1
  - Klare Fehlermeldung: "No valid symbols with price data remain after filtering"
  - Keine SAFE-Orders-Datei wird erstellt

**Weitere Fehlerszenarien:**
- **Universe-File nicht gefunden:** Exit-Code 1, klare Fehlermeldung
- **Preis-File nicht gefunden (wenn `--price-file` gesetzt):** Exit-Code 1, klare Fehlermeldung
- **Leeres DataFrame nach Laden:** Exit-Code 1, Fehlermeldung "Price data is empty"
- **Ungültiges Datum:** Exit-Code 1, Fehlermeldung "Invalid date format"
- **Keine Orders generiert:** Leere SAFE-Datei wird erstellt, Script beendet erfolgreich (Exit-Code 0)

**Exit-Codes:**
- `0`: Erfolgreich (Orders generiert oder leere Datei erstellt)
- `1`: Fehler (keine Daten, keine Symbole, Datei nicht gefunden, etc.)

---

## Nächste Schritte

- **Scheduling:** Cron-Job oder Task-Scheduler für automatische EOD-Läufe
- **Manifest-Historie:** Mehrere Manifeste pro Frequenz (mit Timestamp im Dateinamen)
- **Notifications:** E-Mail/Slack-Benachrichtigungen bei Fehlern
- **Retry-Logik:** Automatische Wiederholung fehlgeschlagener Schritte

---

## Weiterführende Dokumentation

- [Backend Architecture](ARCHITECTURE_BACKEND.md) - Gesamtarchitektur und Datenfluss
- [Backend Modules](BACKEND_MODULES.md) - Detaillierte Modulübersicht
- [Backend API](backend_api.md) - FastAPI-Endpoints
- [Backend Core](backend_core.md) - Konfiguration & Testing
- [Data Sources](DATA_SOURCES_BACKEND.md) - Datenquellen-Übersicht
- [Backend Roadmap](BACKEND_ROADMAP.md) - Entwicklungs-Roadmap

