# Backend API - FastAPI Server

## Übersicht

Das FastAPI-Backend stellt eine read-only API für die Pipeline-Outputs bereit. Es liest direkt aus den existierenden CSV/Parquet-Dateien im `output/` Verzeichnis.

**Prinzip:** File-based, keine Datenbank. Alle Endpoints lesen aus den Pipeline-Outputs.

---

## Server starten

### Voraussetzungen

1. **Dependencies installieren:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Dies installiert:
   - `fastapi>=0.100.0`
   - `uvicorn[standard]>=0.23.0`
   - `pydantic>=2.0.0`

2. **Pipeline-Outputs generieren:**
   ```bash
   # Führe die Pipeline aus, um Output-Dateien zu erstellen
   python scripts/sprint9_execute.py --freq 1d
   python scripts/sprint9_backtest.py --freq 1d
   python scripts/sprint10_portfolio.py --freq 1d
   ```

### Server starten

```bash
python scripts/run_api.py
```

Der Server läuft dann auf:
- **URL:** `http://localhost:8000`
- **API Docs:** `http://localhost:8000/docs` (Swagger UI)
- **ReDoc:** `http://localhost:8000/redoc`

### Server stoppen

Drücke `Ctrl+C` im Terminal.

---

## API-Endpoints

### Orders

#### `GET /api/v1/orders/{freq}`

Gibt alle Orders für eine gegebene Frequenz zurück.

**Path Parameter:**
- `freq`: Trading-Frequenz (`"1d"` oder `"5min"`)

**Response:** `OrdersResponse`
```json
{
  "frequency": "1d",
  "orders": [
    {
      "timestamp": "2025-11-28T14:00:00Z",
      "symbol": "AAPL",
      "side": "BUY",
      "qty": 1.0,
      "price": 100.0,
      "notional": 100.0
    }
  ],
  "count": 1,
  "total_notional": 100.0,
  "first_timestamp": "2025-11-28T14:00:00Z",
  "last_timestamp": "2025-11-28T14:00:00Z"
}
```

**Beispiel:**
```bash
curl http://localhost:8000/api/v1/orders/1d
```

**Mit HTTPie:**
```bash
http GET http://localhost:8000/api/v1/orders/1d
```

---

### Performance

#### `GET /api/v1/performance/{freq}/backtest-curve`

Gibt die Backtest-Equity-Kurve zurück.

**Path Parameter:**
- `freq`: Trading-Frequenz (`"1d"` oder `"5min"`)

**Response:** `EquityCurveResponse`
```json
{
  "frequency": "1d",
  "points": [
    {
      "timestamp": "2025-11-28T14:00:00Z",
      "equity": 10000.0
    }
  ],
  "count": 10,
  "start_equity": 10000.0,
  "end_equity": 10090.0
}
```

**Beispiel:**
```bash
curl http://localhost:8000/api/v1/performance/1d/backtest-curve
```

#### `GET /api/v1/performance/{freq}/metrics`

Gibt Performance-Metriken zurück.

**Path Parameter:**
- `freq`: Trading-Frequenz (`"1d"` oder `"5min"`)

**Response:** Dictionary mit Metriken
```json
{
  "freq": "1d",
  "final_pf": 1.0007,
  "sharpe": 0.1566,
  "rows": 10,
  "first": "2025-11-28T14:00:00Z",
  "last": "2025-11-28T23:00:00Z"
}
```

**Beispiel:**
```bash
curl http://localhost:8000/api/v1/performance/1d/metrics
```

---

### Risk

#### `GET /api/v1/risk/{freq}/summary`

Gibt Risiko-Kennzahlen zurück.

**Path Parameter:**
- `freq`: Trading-Frequenz (`"1d"` oder `"5min"`)

**Response:** `RiskMetrics`
```json
{
  "sharpe_ratio": 0.1566,
  "max_drawdown": -50.25,
  "max_drawdown_pct": -0.5025,
  "volatility": 0.15,
  "current_drawdown": -25.0,
  "var_95": -100.0
}
```

**Beispiel:**
```bash
curl http://localhost:8000/api/v1/risk/1d/summary
```

---

### Signals

#### `GET /api/v1/signals/{freq}`

Gibt alle EMA-Crossover-Signale für eine gegebene Frequenz zurück.

**Path Parameter:**
- `freq`: Trading-Frequenz (`"1d"` oder `"5min"`)

**Response:** `SignalsResponse`
```json
{
  "frequency": "1d",
  "signals": [
    {
      "timestamp": "2025-11-28T14:00:00Z",
      "symbol": "AAPL",
      "signal_type": "BUY",
      "price": 278.58,
      "ema_fast": null,
      "ema_slow": null
    }
  ],
  "count": 194,
  "first_timestamp": "2025-11-26T14:30:00Z",
  "last_timestamp": "2025-11-28T16:00:00Z"
}
```

**Beispiel:**
```bash
curl http://localhost:8000/api/v1/signals/1d
```

**Hinweis:** Signale werden on-the-fly aus Preis-Daten berechnet. Die EMA-Konfiguration wird automatisch basierend auf der Frequenz gewählt (Standard: fast=20, slow=60).

#### `GET /api/v1/signals/{freq}/latest`

Gibt das neueste Signal pro Symbol zurück.

**Path Parameter:**
- `freq`: Trading-Frequenz (`"1d"` oder `"5min"`)

**Response:** `list[Signal]`
```json
[
  {
    "timestamp": "2025-11-28T16:00:00Z",
    "symbol": "AAPL",
    "signal_type": "BUY",
    "price": 278.58,
    "ema_fast": null,
    "ema_slow": null
  },
  {
    "timestamp": "2025-11-28T16:00:00Z",
    "symbol": "MSFT",
    "signal_type": "SELL",
    "price": 450.25,
    "ema_fast": null,
    "ema_slow": null
  }
]
```

**Beispiel:**
```bash
curl http://localhost:8000/api/v1/signals/1d/latest
```

---

### Portfolio

#### `GET /api/v1/portfolio/{freq}/current`

Gibt den aktuellen Portfolio-Zustand für eine gegebene Frequenz zurück.

**Path Parameter:**
- `freq`: Trading-Frequenz (`"1d"` oder `"5min"`)

**Response:** `PortfolioSnapshot`
```json
{
  "timestamp": "2025-11-28T16:00:00Z",
  "equity": 10050.25,
  "cash": 9721.67,
  "positions": {
    "AAPL": 1.0,
    "MSFT": 0.0
  },
  "pf": 1.0050,
  "sharpe": 0.1566,
  "total_trades": 2,
  "start_capital": 10000.0
}
```

**Beispiel:**
```bash
curl http://localhost:8000/api/v1/portfolio/1d/current
```

**Hinweis:** 
- `equity` und `timestamp` stammen aus der letzten Zeile von `portfolio_equity_{freq}.csv`
- `pf`, `sharpe`, `total_trades` werden aus `portfolio_report_{freq}.md` geparst (mit Fallback auf `portfolio_report.md` für Backwards-Kompatibilität)
- `positions` werden aus `orders_{freq}.csv` kumulativ berechnet
- `cash` wird geschätzt als `equity - position_value`

#### `GET /api/v1/portfolio/{freq}/equity-curve`

Gibt die Portfolio-Equity-Kurve zurück (mit Kosten).

**Path Parameter:**
- `freq`: Trading-Frequenz (`"1d"` oder `"5min"`)

**Response:** `EquityCurveResponse`
```json
{
  "frequency": "1d",
  "points": [
    {
      "timestamp": "2025-11-28T14:00:00Z",
      "equity": 10000.0
    }
  ],
  "count": 10,
  "start_equity": 10000.0,
  "end_equity": 10090.0
}
```

**Beispiel:**
```bash
curl http://localhost:8000/api/v1/portfolio/1d/equity-curve
```

**Hinweis:** Diese Kurve enthält Transaktionskosten (Commission, Spread, Impact), im Gegensatz zur Backtest-Equity-Kurve.

---

### QA / Health

#### `GET /api/v1/qa/status`

Gibt den QA/Health-Check-Status für eine gegebene Frequenz zurück.

**Query Parameter:**
- `freq`: Trading-Frequenz (`"1d"` oder `"5min"`), default `"1d"`

**Response:** `QaStatus`
```json
{
  "overall_status": "ok",
  "timestamp": "2025-11-28T16:00:00Z",
  "checks": [
    {
      "check_name": "prices",
      "status": "ok",
      "message": "Price file OK: 100 rows, 5 symbols",
      "details": {
        "file": "output/aggregates/daily.parquet",
        "rows": 100,
        "symbols": 5
      }
    },
    {
      "check_name": "orders",
      "status": "ok",
      "message": "Orders file OK: 10 orders",
      "details": {
        "file": "output/orders_1d.csv",
        "rows": 10
      }
    },
    {
      "check_name": "portfolio",
      "status": "ok",
      "message": "Portfolio file OK: 50 rows, equity range [10000.00, 10050.00]",
      "details": {
        "file": "output/portfolio_equity_1d.csv",
        "rows": 50
      }
    }
  ],
  "summary": {
    "ok": 3,
    "warning": 0,
    "error": 0
  }
}
```

**Beispiel (OK):**
```bash
curl http://localhost:8000/api/v1/qa/status?freq=1d
```

**Beispiel (Error - wenn Dateien fehlen):**
```json
{
  "overall_status": "error",
  "timestamp": "2025-11-28T16:00:00Z",
  "checks": [
    {
      "check_name": "prices",
      "status": "error",
      "message": "Price file not found: output/aggregates/daily.parquet",
      "details": {
        "file": "output/aggregates/daily.parquet",
        "freq": "1d"
      }
    },
    {
      "check_name": "orders",
      "status": "error",
      "message": "Orders file not found: output/orders_1d.csv",
      "details": {
        "file": "output/orders_1d.csv",
        "freq": "1d"
      }
    },
    {
      "check_name": "portfolio",
      "status": "error",
      "message": "Portfolio equity file not found: output/portfolio_equity_1d.csv",
      "details": {
        "file": "output/portfolio_equity_1d.csv",
        "freq": "1d"
      }
    }
  ],
  "summary": {
    "ok": 0,
    "warning": 0,
    "error": 3
  }
}
```

**Hinweis:** Der Endpoint gibt immer HTTP 200 zurück, auch wenn `overall_status` "error" oder "warning" ist. Nur unerwartete Fehler (Exceptions) führen zu HTTP 5xx.

---

## Weiterführende Dokumentation

- [Backend Architecture](ARCHITECTURE_BACKEND.md) - Gesamtarchitektur und Datenfluss
- [Backend Modules](BACKEND_MODULES.md) - Detaillierte Modulübersicht
- [Backend Core](backend_core.md) - Konfiguration & Testing
- [EOD Pipeline](eod_pipeline.md) - Pipeline-Orchestrierung
- [Data Sources](DATA_SOURCES_BACKEND.md) - Datenquellen-Übersicht
- [Backend Roadmap](BACKEND_ROADMAP.md) - Entwicklungs-Roadmap

---

## Error Handling

### 404 Not Found

Wenn eine erforderliche Datei nicht existiert:

```json
{
  "detail": "Orders file not found: output/orders_1d.csv. Run sprint9_execute.py first."
}
```

**Beispiel:**
```bash
curl http://localhost:8000/api/v1/orders/1d
# → 404 wenn orders_1d.csv nicht existiert
```

### 500 Internal Server Error

Wenn Daten fehlerhaft sind:

```json
{
  "detail": "Malformed orders data: Missing required columns: ['timestamp']"
}
```

---

## API-Dokumentation

### Swagger UI

Interaktive API-Dokumentation:
```
http://localhost:8000/docs
```

### ReDoc

Alternative API-Dokumentation:
```
http://localhost:8000/redoc
```

---

## Tests

### API-Tests ausführen

```bash
pytest tests/test_api_smoke.py
```

### Mit ausführlicher Ausgabe

```bash
pytest tests/test_api_smoke.py -v
```

### Erwartete Ausgabe

```
======================== test session starts ========================
platform win32 -- Python 3.13.x
collected 5 items

tests/test_api_smoke.py .....                                      [100%]

======================== 5 passed in 2.34s ========================
```

---

## Beispiel-Workflow

### 1. Pipeline ausführen

```bash
# Daten ziehen
python scripts/live/pull_intraday.py --symbols AAPL,MSFT --days 10

# Resample
python scripts/run_all_sprint10.ps1  # oder manuell resample

# Execute
python scripts/sprint9_execute.py --freq 1d

# Backtest
python scripts/sprint9_backtest.py --freq 1d

# Portfolio
python scripts/sprint10_portfolio.py --freq 1d
```

### 2. API starten

```bash
python scripts/run_api.py
```

### 3. API abfragen

```bash
# Orders abrufen
curl http://localhost:8000/api/v1/orders/1d

# Performance-Metriken
curl http://localhost:8000/api/v1/performance/1d/metrics

# Equity-Kurve
curl http://localhost:8000/api/v1/performance/1d/backtest-curve

# Risiko-Kennzahlen
curl http://localhost:8000/api/v1/risk/1d/summary
```

---

## Architektur

### Dateistruktur

```
src/assembled_core/api/
├── app.py                    # FastAPI app factory
├── models.py                 # Pydantic models
└── routers/
    ├── __init__.py
    ├── orders.py             # Orders endpoints
    ├── performance.py        # Performance endpoints
    ├── risk.py               # Risk endpoints
    ├── signals.py            # Signals endpoints (placeholder)
    └── portfolio.py           # Portfolio endpoints (placeholder)
```

### Datenfluss

1. **Client Request** → FastAPI Router
2. **Router** → Pipeline-Module (`pipeline.io`, `pipeline.backtest`)
3. **Pipeline-Module** → Lesen aus `output/` (via `config.OUTPUT_DIR`)
4. **Daten** → Pydantic Models
5. **Response** → JSON

### Konfiguration

- **Output-Verzeichnis:** `config.OUTPUT_DIR` (standardmäßig `output/`)
- **Port:** 8000 (hardcoded in `scripts/run_api.py`)
- **Host:** 0.0.0.0 (alle Interfaces)

---

## Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'fastapi'`
- **Lösung:** `pip install -r requirements.txt`

**Problem:** `404 Not Found` für alle Endpoints
- **Lösung:** Stelle sicher, dass die Pipeline-Outputs existieren. Führe die Pipeline-Schritte aus.

**Problem:** `500 Internal Server Error`
- **Lösung:** Prüfe die Logs im Terminal. Oft liegt es an fehlerhaften Daten in den CSV/Parquet-Dateien.

**Problem:** Port 8000 bereits belegt
- **Lösung:** Ändere den Port in `scripts/run_api.py`:
  ```python
  uvicorn.run(app, host="0.0.0.0", port=8001)
  ```

---

## Nächste Schritte

- **Signals-Endpoint:** Implementierung, wenn Signale persistiert werden
- **Portfolio-Endpoint:** Implementierung für aktuelle Portfolio-Snapshots
- **QA-Endpoint:** Integration der QA-Health-Checks
- **Caching:** Optionales Caching für statische Daten
- **Authentication:** Optional für Produktionsumgebung

Siehe auch: `docs/fastapi_endpoints_design.md` für das vollständige API-Design.

