# FastAPI Endpoints Design - Sprint-9 Pipeline

## Übersicht

Design der FastAPI-Endpoints basierend auf den refactorierten Pipeline-Modulen und aktuellen Output-Dateien.

**Prinzip:** Alle Endpoints lesen aus den existierenden Pipeline-Outputs (keine Datenbank, file-based).

**Konfiguration:** Siehe auch `docs/backend_core.md` für Details zur zentralen Konfiguration (`src/assembled_core/config.py`).

---

## Datenmodelle

Siehe: `src/assembled_core/api/models.py`

**Hauptmodelle:**
- `Signal`: Einzelnes Trading-Signal (aus EMA-Crossover)
- `OrderPreview`: SAFE-Bridge Order (aus orders_{freq}.csv)
- `PortfolioSnapshot`: Portfolio-Zustand (aus portfolio_equity + portfolio_report)
- `RiskMetrics`: Risiko-Kennzahlen (aus equity curves + reports)
- `QaStatus`: QA/QC-Status (aus QC-Checks)

---

## API-Endpoints

### 1. Signals Endpoints

#### `GET /api/v1/signals/{freq}`
**Zweck:** Aktuelle Trading-Signale abrufen.

**Input:**
- Path Parameter: `freq` (enum: "5min" | "1d")
- Query Parameter (optional):
  - `symbol`: Filter nach Symbol
  - `since`: Timestamp (nur Signale danach)
  - `limit`: Maximale Anzahl

**Output:** `SignalsResponse`
```json
{
  "frequency": "5min",
  "signals": [...],
  "count": 194,
  "first_timestamp": "2025-11-26T14:30:00Z",
  "last_timestamp": "2025-11-28T16:00:00Z"
}
```

**Datenquelle:**
- **Aktuell:** Wird nicht gespeichert, müsste aus `pipeline.signals.compute_ema_signals()` generiert werden
- **Zukünftig:** `output/signals_{freq}.parquet` (wenn implementiert)
- **Fallback:** Aus `output/aggregates/{freq}.parquet` + EMA-Config neu berechnen

**Implementierung:**
```python
# Pseudo-Code
prices = load_prices(freq)
ema_config = get_default_ema_config(freq)
signals = compute_ema_signals(prices, ema_config.fast, ema_config.slow)
```

---

#### `GET /api/v1/signals/{freq}/latest`
**Zweck:** Neuestes Signal pro Symbol.

**Input:**
- Path Parameter: `freq` (enum: "5min" | "1d")
- Query Parameter (optional): `symbol`: Filter nach Symbol

**Output:** `list[Signal]` (ein Signal pro Symbol)

**Datenquelle:** Wie oben, aber nur letzte Zeile pro Symbol.

---

### 2. Orders Endpoints

#### `GET /api/v1/orders/{freq}`
**Zweck:** Generierte Orders abrufen (SAFE-Bridge kompatibel).

**Input:**
- Path Parameter: `freq` (enum: "5min" | "1d")
- Query Parameter (optional):
  - `symbol`: Filter nach Symbol
  - `since`: Timestamp (nur Orders danach)
  - `side`: Filter nach BUY/SELL
  - `limit`: Maximale Anzahl

**Output:** `OrdersResponse`
```json
{
  "frequency": "5min",
  "orders": [...],
  "count": 2,
  "total_notional": 557.08,
  "first_timestamp": "2025-11-26T19:25:00Z",
  "last_timestamp": "2025-11-26T19:25:00Z"
}
```

**Datenquelle:**
- **Datei:** `output/orders_{freq}.csv`
- **Schema:** timestamp, symbol, side, qty, price
- **Implementierung:** `pipeline.io.load_orders(freq, strict=True)`

---

#### `GET /api/v1/orders/{freq}/today`
**Zweck:** Orders des heutigen Tages.

**Input:**
- Path Parameter: `freq` (enum: "5min" | "1d")
- Query Parameter (optional): `symbol`: Filter nach Symbol

**Output:** `OrdersResponse`

**Datenquelle:** Wie oben, aber gefiltert nach `timestamp >= today 00:00 UTC`.

---

#### `GET /api/v1/orders/{freq}/pending`
**Zweck:** Noch nicht ausgeführte Orders (für SAFE-Bridge).

**Input:**
- Path Parameter: `freq` (enum: "5min" | "1d")

**Output:** `OrdersResponse`

**Datenquelle:** Wie oben, aber gefiltert nach Status (wenn Status-Spalte existiert, sonst alle).

---

### 3. Portfolio Endpoints

#### `GET /api/v1/portfolio/{freq}/current`
**Zweck:** Aktueller Portfolio-Zustand.

**Input:**
- Path Parameter: `freq` (enum: "5min" | "1d")

**Output:** `PortfolioSnapshot`
```json
{
  "timestamp": "2025-11-28T16:00:00Z",
  "equity": 10050.25,
  "cash": 9721.67,
  "positions": {"AAPL": 1.0, "MSFT": 0.0},
  "pf": 1.0050,
  "sharpe": 0.1566,
  "total_trades": 2,
  "start_capital": 10000.0
}
```

**Datenquelle:**
- **Equity:** `output/portfolio_equity_{freq}.csv` (letzte Zeile)
- **Metriken:** `output/portfolio_report.md` (parsed)
- **Positions:** Aus `output/orders_{freq}.csv` berechnet (kumulative qty pro Symbol)

**Implementierung:**
```python
# Pseudo-Code
equity_df = pd.read_csv(f"output/portfolio_equity_{freq}.csv")
latest_equity = equity_df.iloc[-1]

orders = load_orders(freq)
positions = calculate_positions(orders)  # kumulative qty

metrics = parse_portfolio_report(f"output/portfolio_report.md")

return PortfolioSnapshot(
    timestamp=latest_equity["timestamp"],
    equity=latest_equity["equity"],
    cash=...,  # aus orders berechnen
    positions=positions,
    pf=metrics["final_pf"],
    sharpe=metrics["sharpe"],
    total_trades=metrics["trades"],
    start_capital=...
)
```

---

#### `GET /api/v1/portfolio/{freq}/equity-curve`
**Zweck:** Vollständige Equity-Kurve.

**Input:**
- Path Parameter: `freq` (enum: "5min" | "1d")
- Query Parameter (optional):
  - `since`: Timestamp (nur Punkte danach)
  - `limit`: Maximale Anzahl Punkte

**Output:** `EquityCurveResponse`
```json
{
  "frequency": "5min",
  "points": [...],
  "count": 97,
  "start_equity": 10000.0,
  "end_equity": 10007.0
}
```

**Datenquelle:**
- **Datei:** `output/portfolio_equity_{freq}.csv`
- **Schema:** timestamp, equity
- **Implementierung:** `pd.read_csv()` → `EquityPoint`-Liste

---

#### `GET /api/v1/portfolio/{freq}/history`
**Zweck:** Portfolio-Historie mit Metriken.

**Input:**
- Path Parameter: `freq` (enum: "5min" | "1d")
- Query Parameter (optional):
  - `since`: Timestamp
  - `limit`: Maximale Anzahl

**Output:** `list[PortfolioSnapshot]` (zeitlich sortiert)

**Datenquelle:**
- **Equity:** `output/portfolio_equity_{freq}.csv`
- **Orders:** `output/orders_{freq}.csv` (für Positions-Berechnung)
- **Metriken:** `output/portfolio_report.md` (für PF, Sharpe, Trades)

---

### 4. Risk Endpoints

#### `GET /api/v1/risk/{freq}/summary`
**Zweck:** Risiko-Kennzahlen-Zusammenfassung.

**Input:**
- Path Parameter: `freq` (enum: "5min" | "1d")

**Output:** `RiskMetrics`
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

**Datenquelle:**
- **Sharpe:** `output/performance_report_{freq}.md` oder `output/portfolio_report.md`
- **Drawdown:** Aus `output/portfolio_equity_{freq}.csv` berechnet (rolling max, dann drawdown)
- **Volatility:** Aus Equity-Returns berechnet (annualisiert)
- **VaR:** Aus Equity-Verteilung berechnet (95% Quantil)

**Implementierung:**
```python
# Pseudo-Code
equity_df = pd.read_csv(f"output/portfolio_equity_{freq}.csv")
equity_series = equity_df["equity"]

# Drawdown
rolling_max = equity_series.expanding().max()
drawdown = equity_series - rolling_max
max_dd = drawdown.min()

# Volatility
returns = equity_series.pct_change().dropna()
vol = returns.std() * np.sqrt(252 if freq == "1d" else 252 * 78)

# VaR
var_95 = np.percentile(returns, 5) * equity_series.iloc[-1]

metrics = parse_performance_report(f"output/performance_report_{freq}.md")

return RiskMetrics(
    sharpe_ratio=metrics["sharpe"],
    max_drawdown=max_dd,
    max_drawdown_pct=(max_dd / rolling_max.max()) * 100,
    volatility=vol,
    current_drawdown=drawdown.iloc[-1],
    var_95=var_95
)
```

---

#### `GET /api/v1/risk/{freq}/drawdown`
**Zweck:** Drawdown-Kurve.

**Input:**
- Path Parameter: `freq` (enum: "5min" | "1d")

**Output:** `list[EquityPoint]` (aber equity = drawdown)

**Datenquelle:** Wie oben, aber drawdown statt equity.

---

### 5. QA/QC Endpoints

#### `GET /api/v1/qa/status`
**Zweck:** Gesamter QA/QC-Status.

**Input:**
- Query Parameter (optional): `freq`: Filter nach Frequenz

**Output:** `QaStatus`
```json
{
  "overall_status": "ok",
  "timestamp": "2025-11-28T16:00:00Z",
  "checks": [
    {
      "check_name": "schema_validation",
      "status": "ok",
      "message": "Schema correct: symbol, timestamp, close",
      "details": {"columns": ["symbol", "timestamp", "close"]}
    },
    {
      "check_name": "duplicate_check",
      "status": "ok",
      "message": "No duplicates found",
      "details": {"duplicate_count": 0}
    }
  ],
  "summary": {"ok": 5, "warning": 0, "error": 0}
}
```

**Datenquelle:**
- **QC-Checks auf:**
  - `output/aggregates/{freq}.parquet` (Schema, Duplikate, Timestamps)
  - `output/orders_{freq}.csv` (Schema, Validierung)
  - `output/equity_curve_{freq}.csv` (Kontinuität)
  - `output/portfolio_equity_{freq}.csv` (Kontinuität)

**Implementierung:**
```python
# Pseudo-Code
checks = []

# Schema-Check für aggregates
df = pd.read_parquet(f"output/aggregates/{freq}.parquet")
if list(df.columns) == ["symbol", "timestamp", "close"]:
    checks.append(QaCheck(
        check_name="schema_validation",
        status=QaStatus.OK,
        message="Schema correct"
    ))
else:
    checks.append(QaCheck(
        check_name="schema_validation",
        status=QaStatus.ERROR,
        message=f"Schema incorrect: {list(df.columns)}"
    ))

# Duplikate-Check
dupes = df.duplicated(subset=["timestamp", "symbol"]).sum()
if dupes == 0:
    checks.append(QaCheck(
        check_name="duplicate_check",
        status=QaStatus.OK,
        message="No duplicates found"
    ))
else:
    checks.append(QaCheck(
        check_name="duplicate_check",
        status=QaStatus.ERROR,
        message=f"Found {dupes} duplicates"
    ))

# ... weitere Checks ...

summary = {
    "ok": sum(1 for c in checks if c.status == QaStatus.OK),
    "warning": sum(1 for c in checks if c.status == QaStatus.WARNING),
    "error": sum(1 for c in checks if c.status == QaStatus.ERROR)
}

overall = QaStatus.ERROR if summary["error"] > 0 else (QaStatus.WARNING if summary["warning"] > 0 else QaStatus.OK)

return QaStatus(
    overall_status=overall,
    timestamp=datetime.utcnow(),
    checks=checks,
    summary=summary
)
```

---

#### `GET /api/v1/qa/checks/{check_name}`
**Zweck:** Einzelner QA-Check.

**Input:**
- Path Parameter: `check_name` (z.B. "schema_validation", "duplicate_check")

**Output:** `QaCheck`

**Datenquelle:** Wie oben, aber nur ein Check.

---

### 6. Performance Endpoints

#### `GET /api/v1/performance/{freq}/metrics`
**Zweck:** Performance-Metriken.

**Input:**
- Path Parameter: `freq` (enum: "5min" | "1d")

**Output:** `dict` mit Metriken
```json
{
  "final_pf": 1.0007,
  "sharpe": 0.1566,
  "rows": 97,
  "first": "2025-11-26T14:30:00Z",
  "last": "2025-11-28T16:00:00Z"
}
```

**Datenquelle:**
- **Datei:** `output/performance_report_{freq}.md`
- **Implementierung:** Markdown parsen oder direkt aus `pipeline.backtest.compute_metrics()` lesen

---

#### `GET /api/v1/performance/{freq}/backtest-curve`
**Zweck:** Backtest Equity-Kurve (ohne Kosten).

**Input:**
- Path Parameter: `freq` (enum: "5min" | "1d")

**Output:** `EquityCurveResponse`

**Datenquelle:**
- **Datei:** `output/equity_curve_{freq}.csv`
- **Schema:** timestamp, equity

---

## Endpoint-Zusammenfassung

| Endpoint | Methode | Input | Output | Datenquelle |
|----------|---------|-------|--------|-------------|
| `/api/v1/signals/{freq}` | GET | freq, symbol?, since?, limit? | SignalsResponse | `output/aggregates/{freq}.parquet` + EMA-Config |
| `/api/v1/signals/{freq}/latest` | GET | freq, symbol? | list[Signal] | Wie oben |
| `/api/v1/orders/{freq}` | GET | freq, symbol?, since?, side?, limit? | OrdersResponse | `output/orders_{freq}.csv` |
| `/api/v1/orders/{freq}/today` | GET | freq, symbol? | OrdersResponse | `output/orders_{freq}.csv` (gefiltert) |
| `/api/v1/orders/{freq}/pending` | GET | freq | OrdersResponse | `output/orders_{freq}.csv` (gefiltert) |
| `/api/v1/portfolio/{freq}/current` | GET | freq | PortfolioSnapshot | `output/portfolio_equity_{freq}.csv` + `output/portfolio_report.md` + `output/orders_{freq}.csv` |
| `/api/v1/portfolio/{freq}/equity-curve` | GET | freq, since?, limit? | EquityCurveResponse | `output/portfolio_equity_{freq}.csv` |
| `/api/v1/portfolio/{freq}/history` | GET | freq, since?, limit? | list[PortfolioSnapshot] | Wie current, aber alle Zeilen |
| `/api/v1/risk/{freq}/summary` | GET | freq | RiskMetrics | `output/portfolio_equity_{freq}.csv` + `output/portfolio_report.md` |
| `/api/v1/risk/{freq}/drawdown` | GET | freq | list[EquityPoint] | `output/portfolio_equity_{freq}.csv` (berechnet) |
| `/api/v1/qa/status` | GET | freq? | QaStatus | `output/aggregates/{freq}.parquet` + `output/orders_{freq}.csv` + ... |
| `/api/v1/qa/checks/{check_name}` | GET | check_name | QaCheck | Wie oben, aber ein Check |
| `/api/v1/performance/{freq}/metrics` | GET | freq | dict | `output/performance_report_{freq}.md` |
| `/api/v1/performance/{freq}/backtest-curve` | GET | freq | EquityCurveResponse | `output/equity_curve_{freq}.csv` |

---

## Implementierungs-Hinweise

### 1. File-basierte Datenquelle
- Alle Endpoints lesen aus existierenden CSV/Parquet/Markdown-Dateien
- Keine Datenbank nötig (kann später hinzugefügt werden)
- Pipeline-Module (`pipeline.io`, `pipeline.backtest`, etc.) werden direkt genutzt

### 2. Caching
- Equity-Curves können gecacht werden (ändern sich nur bei neuem Backtest)
- Orders können gecacht werden (ändern sich nur bei neuem Execute)
- QA-Status sollte bei jedem Request neu berechnet werden (kann sich ändern)

### 3. Fehlerbehandlung
- `404 Not Found`: Datei existiert nicht
- `422 Unprocessable Entity`: Ungültige Parameter
- `500 Internal Server Error`: Fehler beim Lesen/Parsen

### 4. Performance
- Für große Equity-Curves: Pagination (`limit`, `offset`)
- Für viele Orders: Filterung nach `since` oder `symbol`
- Streaming für große Responses (optional)

---

## Nächste Schritte (nach Implementierung)

1. **FastAPI-App erstellen:** `src/assembled_core/api/app.py`
2. **Router-Module:** Separate Router für Signals, Orders, Portfolio, Risk, QA
3. **Dependency Injection:** Pipeline-Module als Dependencies
4. **OpenAPI-Dokumentation:** Automatisch generiert aus Pydantic-Modellen
5. **Tests:** Unit-Tests für jeden Endpoint mit Mock-Daten

