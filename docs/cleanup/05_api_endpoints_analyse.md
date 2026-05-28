# Paket 5 — Fehlende API-Endpoints (GO_LIVE F2)

## Schritt 1 — Ist-Analyse

### Bestehendes Endpoint-Pattern (Beispiel)

`GET /api/v1/performance/{freq}/backtest-curve` in `src/assembled_core/api/routers/performance.py`:

```python
# 1. Router
router = APIRouter()

# 2. Pydantic Response-Modell (in api/models.py):
class EquityPoint(BaseModel):
    timestamp: datetime = Field(...)
    equity: float = Field(...)

class EquityCurveResponse(BaseModel):
    frequency: str
    points: list[EquityPoint]
    count: int
    start_equity: float
    end_equity: float
    model_config = ConfigDict(json_schema_extra={...})

# 3. Endpoint
@router.get("/performance/{freq}/backtest-curve", response_model=EquityCurveResponse)
def get_backtest_curve(freq: Frequency) -> EquityCurveResponse:
    curve_file = OUTPUT_DIR / f"equity_curve_{freq.value}.csv"
    if not curve_file.exists():
        raise HTTPException(status_code=404, detail="...")
    try:
        df = pd.read_csv(curve_file, ...)
        ...
        return EquityCurveResponse(frequency=freq, points=points, ...)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=...)
```

**Regel:** Kein `Depends(require_api_key)` auf Read-only GET — nur POST/Command-Endpoints sind auth-gesperrt.
**Regel:** `try/except HTTPException: raise` gefolgt von breitem `except Exception` → nie ungefangene 500er.
**Regel:** Kein Schreibzugriff, keine Side Effects.

---

### Datenquellen der bestehenden Endpoints

| Endpoint | Datenquelle |
|----------|-------------|
| `GET /api/v1/performance/{freq}/backtest-curve` | `OUTPUT_DIR/equity_curve_{freq}.csv` (CSV) |
| `GET /api/v1/monitoring/portfolio` | `LedgerStore` (SQLite `data/paper_ledger.db`) |
| `GET /api/v1/monitoring/qa_status` | `OUTPUT_DIR/run_manifest_{freq}.json`, dann `equity_curve_{freq}.csv` fallback |
| `GET /api/v1/monitoring/risk_status` | `OUTPUT_DIR/portfolio_equity_{freq}.csv` oder `equity_curve_{freq}.csv` |

---

### Paper-Ledger zur Laufzeit

**JSON-Ledger** (`ops/paper_ledger.py`):
- Pfad: `output/runs/_paper_ledger/ledger_state.json` (Default in `paper_runner.py:42`)
- Schema: `{schema_version, updated_utc, cash, positions: {sym: {qty, avg_price, hwm}}, equity_curve: [{utc, equity}]}`
- Schreib-API: `save_ledger_state()` mit FileLock + Backup-Rotation (3 Generationen)
- Snapshot-Artefakt: `ledger_snapshot.json` im Run-Output-Dir (enthält equity + cash + positions)
- Verwendet von: `paper_runner.py`, `run_live_paper()`

**SQLite-Ledger** (`data/ledger_store.py`):
- Pfad: `data/paper_ledger.db` (SQLite via LedgerStore)
- Verwendet von: `monitoring/portfolio`, `reconcile_worker`

Die neuen Endpoints nutzen das JSON-Ledger (direkter Zugriff, kein SQLite-Overhead, vollständige equity_curve).

---

### Health-Check-Infrastruktur

**`ops/daily_scheduler._health_check_worker`** prüft:
1. Output-Dir beschreibbar (Schreibprobe)
2. Preisdaten-Frische (`prices_{date}.parquet` Alter ≤ 26h)
3. Kritische Module verfügbar (pandas, numpy)

**`execution/broker_adapter.BrokerAdapter.health_check()`**:
- Ruft `get_account()` auf
- Gibt `{"ok": bool, "message": str, "account_equity": ...}` zurück
- Wirft keine Ausnahmen (intern try/except)

**Bestehende `/health`** in `app.py:48-51`: Nur Liveness-Check (`{"status": "ok", "uptime_s": ...}`), keine echten Checks.  
→ Wird durch reichhaltigen `/health`-Router ersetzt (alter Thin-Check wird entfernt).

---

## Schritt 2 — Implementierung

### Neue Dateien

| Datei | Zweck |
|-------|-------|
| `src/assembled_core/api/routers/health.py` | `GET /health` — reich, maschinenlesbar |
| `src/assembled_core/api/routers/ledger.py` | `GET /api/v1/ledger` — Paper-Ledger-Stand |

### Geänderte Dateien

| Datei | Änderung |
|-------|----------|
| `src/assembled_core/api/models.py` | `HealthCheckItem`, `HealthResponse`, `LedgerPosition`, `LedgerResponse` |
| `src/assembled_core/api/routers/performance.py` | `GET /api/v1/performance/{freq}/live-curve` |
| `src/assembled_core/api/app.py` | Inline-`/health` entfernt; `health.router` + `ledger.router` eingebunden |

---

## Schritt 3 — Testergebnis

| Test | Beschreibung | Ergebnis |
|------|-------------|---------|
| `test_health_200_valid_structure` | 200 + {status, timestamp_utc, checks} vorhanden | PASS |
| `test_health_503_when_dir_unwritable` | Probe-Dir ist Verzeichnis → write_text schlägt fehl → 503 | PASS |
| `test_health_checks_dict_present` | checks enthält output_dir-Key | PASS |
| `test_ledger_no_file_empty_valid` | Kein Ledger → status=no_ledger, cash=0, positions=[] | PASS |
| `test_ledger_with_data` | Ledger mit AAPL-Position → cash/equity/positions korrekt | PASS |
| `test_ledger_with_date_param` | ?date=2026-05-20 → equity aus equity_curve gefiltert | PASS |
| `test_ledger_date_no_match_graceful` | ?date nicht in equity_curve → equity=0, kein 500 | PASS |
| `test_live_curve_no_pilot_data` | Kein Ledger → leere valide EquityCurveResponse (kein 500) | PASS |
| `test_live_curve_same_schema_as_backtest` | Gleiche Felder wie backtest-curve, count+equity korrekt | PASS |

---

## Schritt 4 — Frontend-Vertrag

| Endpoint | Methode | Response-Felder | Beispiel-JSON |
|----------|---------|----------------|---------------|
| `/health` | GET | `status` (healthy/unhealthy), `timestamp_utc` (ISO), `checks` (dict: name→{ok,detail}) | `{"status":"healthy","timestamp_utc":"2026-05-28T19:30:00+00:00","checks":{"output_dir":{"ok":true,"detail":null},"data_freshness":{"ok":true,"detail":"latest: prices_2026-05-28.parquet, age: 0.3h"},"broker":{"ok":false,"detail":"check skipped: no API key"},"kill_switch":{"ok":true,"detail":"state=INACTIVE"}}}` |
| `/api/v1/ledger` | GET | `status` (ok/no_ledger), `as_of` (ISO/null), `cash` (float), `equity` (float), `n_positions` (int), `positions` (list[{symbol,qty,avg_price,cost_basis}]), `unrealized_pnl_approx` (float/null), `date_requested` (str/null) | `{"status":"ok","as_of":"2026-05-28T21:30:00+00:00","cash":8234.12,"equity":9987.45,"n_positions":2,"positions":[{"symbol":"AAPL","qty":10.0,"avg_price":150.0,"cost_basis":1500.0}],"unrealized_pnl_approx":253.33,"date_requested":null}` |
| `/api/v1/performance/{freq}/live-curve` | GET | `frequency` (str), `points` (list[{timestamp,equity}]), `count` (int), `start_equity` (float), `end_equity` (float) — **identisch mit backtest-curve** | `{"frequency":"1d","points":[{"timestamp":"2026-05-20T21:30:00Z","equity":10000.0},{"timestamp":"2026-05-21T21:30:00Z","equity":10120.0}],"count":2,"start_equity":10000.0,"end_equity":10120.0}` |

### Query-Parameter

| Endpoint | Parameter | Default | Beschreibung |
|----------|-----------|---------|--------------|
| `/health` | `output_dir` | `"output"` | Output-Verzeichnis für Schreibprobe + Datenfrische |
| `/api/v1/ledger` | `date` | `null` | Historische Abfrage (YYYY-MM-DD) |
| `/api/v1/ledger` | `ledger_path` | `"output/runs/_paper_ledger/ledger_state.json"` | Pfad zum JSON-Ledger |
| `/api/v1/performance/{freq}/live-curve` | `ledger_path` | `"output/runs/_paper_ledger/ledger_state.json"` | Pfad zum JSON-Ledger |

### Verhalten bei fehlenden Laufzeitdaten

| Endpoint | Kein Pilot aktiv | Verhalten |
|----------|-----------------|-----------|
| `/health` | Preisdaten fehlen | `checks.data_freshness.ok = false`; HTTP 200 (non-critical) |
| `/api/v1/ledger` | Ledger nicht vorhanden | `status=no_ledger`, alle Werte 0/leer; HTTP 200 |
| `/api/v1/performance/{freq}/live-curve` | Ledger nicht vorhanden | `count=0, points=[]`; HTTP 200 |
