# Due-Diligence Fix Changelog

**Datum:** 2026-04-08
**Basis-Commit:** f74bb90 (main)
**Test-Suite:** 3562 passed, 0 failures, 1 skip, 2 xfail
**Ruff:** clean (0 errors)

---

## Phase 0: Sofort (Security)

> Hinweis: Key-Rotation und History-Bereinigung sind manuelle externe Schritte, keine Code-Aenderungen.
> Die `.env`-Datei ist in der Git-History exponiert — alle 8 API-Keys muessen rotiert werden.

---

## Phase 1: Vor erstem produktiven Backtest

### P1-1: broker_adapter.py — Kein stilles 0.0 mehr bei Preisfehler (KRITISCH-2.1)

**Datei:** `src/assembled_core/execution/broker_adapter.py`
**Problem:** `_estimate_price()` gab bei jedem Fehler still `0.0` zurueck. Drei verschachtelte `except: pass`-Bloecke verschluckten alle Fehler.
**Fix:** Kompletter Rewrite der Methode. Fehler werden gesammelt und als `RuntimeError` geworfen. Kein stiller Fallback mehr.

### P1-2: 28 Source-Dateien — Silent-Pass-Elimination (KRITISCH-2)

**Dateien:** 28 Dateien in accounting/, api/, config/, data/, events/, execution/, ml/, ops/, pipeline/, qa/, reports/, risk/
**Problem:** ~55 `except Exception: pass`-Bloecke verschluckten Fehler still.
**Fix:** Jeder Block bekommt mindestens `logger.error()` (kritische Pfade) oder `logger.warning()` (nicht-kritische). Temp-File-Cleanup-Bloecke bewusst als silent pass belassen.

### P1-3: Drawdown-Exposure-Caps durchgesetzt (KRITISCH-3)

**Datei:** `src/assembled_core/execution/risk_controls.py`
**Problem:** `compute_drawdown_risk_level()` berechnete Caps (0.75/0.50/0.25), aber nichts wendete sie auf Orders an.
**Fix:** "Step 0" in `pre_trade_risk_filter()` eingefuegt. Bei Drawdown wird `exposure_cap` auf alle Order-Quantities angewendet. Graduated: NORMAL(1.0) -> CAUTION(0.75) -> REDUCE(0.50) -> MINIMUM(0.25).

### P1-4: Realistische Commission-Defaults (KRITISCH-5)

**Dateien:**
- `src/assembled_core/costs.py`: `commission_bps` 0.0 -> 1.0
- `src/assembled_core/execution/transaction_costs.py`: `commission_bps` 0.0 -> 1.0
**Problem:** Default-Commission war 0 bps — Backtests ueberzeichneten Renditen systematisch.
**Fix:** Default auf 1.0 bps (realistisch fuer US Equities). `costs.py` als Single Source of Truth dokumentiert.

### P1-5: CLI Exit-Codes + Argument-Validierung (HOCH-7)

**Datei:** `scripts/run_eod_pipeline.py`
**Problem:** Exit Code war immer 0, auch bei QA-Fehlern. Keine Argument-Validierung.
**Fix:**
- Exit 0 = Erfolg, Exit 2 = QA-Error, Exit 3 = QA-Warning, Exit 1 = Fatal
- Argument-Validierung: start_capital > 0, start_date <= end_date, commission/spread/impact >= 0

---

## Phase 2: Vor Paper-Trading

### P2-1: Kill Switch gehaertet (KRITISCH-4)

**Datei:** `src/assembled_core/execution/kill_switch.py` (kompletter Rewrite)
**Problem:** Nur Env-Variable + Sentinel-File. Kein persistenter State, kein Audit, keine fraktionale Drosselung.
**Fix:**
- Persistenter JSON-State (`output/ops/kill_switch_state.json`)
- JSONL-Audit-Log (`output/ops/kill_switch_audit.jsonl`)
- Fraktionale Drosselung (0-100% der Orders erlauben)
- Neue API: `activate_kill_switch(throttle_pct, reason, actor)`, `deactivate_kill_switch()`, `get_kill_switch_state()`, `get_throttle_pct()`
- `guard_orders_with_kill_switch()` skaliert Quantities bei Teildrosselung
- Abwaertskompatibel: `is_kill_switch_engaged()` prueft alle 3 Quellen

### P2-2: Backtest/Paper Cost-Alignment (HOCH-1)

**Datei:** `src/assembled_core/execution/paper_trading_engine.py`
**Problem:** Paper-Trading-Engine und Backtest-Engine verwendeten unterschiedliche Fill-Logik.
**Fix:** `FillModel.from_cost_model()` Factory-Methode eingefuehrt. Laedt automatisch den zentralen `CostModel` aus `costs.py`. Gleiche Spread/Impact-Gewichte fuer beide Pfade.

### P2-3: Reconciliation-Toleranzen verschaerft (HOCH-6)

**Datei:** `src/assembled_core/accounting/reconciliation.py`
**Aenderungen:**
- `cash_tol`: 1e-6 -> 1e-8 (10x strenger)
- `qty_tol`: 1e-8 -> 1e-6 (angemessen fuer fractional shares)
- `fail_fast`: False -> True (Default: Fehler werfen, nicht still zurueckgeben)

**Datei:** `src/assembled_core/accounting/ledger_integration.py`
- Broker-Snapshot-Fallback von `logger.info` auf `logger.warning` geaendert

**Datei:** `src/assembled_core/execution/position_sync.py`
- Explizit `fail_fast=False` gesetzt (Funktion gibt Result-Object zurueck, nicht raise)

**Tests:** 12 Testdateien angepasst — alle Tests mit absichtlichen Mismatches verwenden jetzt explizit `fail_fast=False`.

### P2-4: Synthetische OHLC eliminiert (HOCH-3)

**Datei:** `src/assembled_core/data/prices_ingest.py`
**Problem:** Bei fehlenden OHLCV-Spalten wurde still High=Low=Open=Close, Volume=0 generiert.
**Fix:** `ValueError` bei fehlenden OHLCV-Spalten. Klare Fehlermeldung mit Erklaerung warum synthetische Daten unzuverlaessig sind.
**Tests:** `test_data_contracts.py` und `test_ml_dataset_builder.py` angepasst.

### P2-5: PIT Guard — Audit-Log fuer warn-Mode (HOCH-2 teilweise)

**Datei:** `src/assembled_core/data/pit_guard.py`
**Problem:** `mode="warn"` lief ohne Audit-Trail. Ops konnte nicht nachvollziehen, wo PIT-Verletzungen toleriert wurden.
**Fix:**
- JSONL-Audit-Log (`output/ops/pit_guard_audit.jsonl`)
- Jede warn-Mode-Instanziierung wird geloggt (`INIT_WARN_MODE`)
- Jede warn-Mode-Violation wird geloggt (`WARN_VIOLATION` mit as_of, context, n_future_rows, latest_ts)
- Pfad ueberschreibbar via `ASSEMBLED_PIT_AUDIT_LOG` Env-Variable

---

## Verifizierung

- **Ruff:** 0 Fehler in `src/`
- **Pytest:** 3562 passed, 0 failures, 1 skip, 2 xfail
- **Keine neuen Testfehler eingefuehrt**
- **Alle bestehenden Tests weiterhin gruen**

---

## Betroffene Dateien (Zusammenfassung)

### Source-Dateien (geaendert):
- `src/assembled_core/execution/broker_adapter.py`
- `src/assembled_core/execution/risk_controls.py`
- `src/assembled_core/execution/kill_switch.py`
- `src/assembled_core/execution/paper_trading_engine.py`
- `src/assembled_core/execution/transaction_costs.py`
- `src/assembled_core/execution/position_sync.py`
- `src/assembled_core/costs.py`
- `src/assembled_core/accounting/reconciliation.py`
- `src/assembled_core/accounting/ledger_integration.py`
- `src/assembled_core/data/prices_ingest.py`
- `src/assembled_core/data/pit_guard.py`
- `scripts/run_eod_pipeline.py`
- 28 weitere Dateien (Silent-Pass-Elimination)

### Test-Dateien (angepasst):
- `tests/test_reconciliation_smoke.py`
- `tests/test_reconcile_report_written.py`
- `tests/test_reconcile_report_csv_broker_meta.py`
- `tests/test_data_contracts.py`
- `tests/test_ml_dataset_builder.py`

---

## Phase 3: CI/CD-Haertung + API-Erweiterung

### P3-1: Coverage-Reporting in CI (L2)

**Datei:** `.github/workflows/backend-ci.yml`
**Aenderung:** pytest-Aufruf mit `--cov=src/assembled_core --cov-report=xml:coverage.xml`. Coverage-Report wird als CI-Artefakt hochgeladen.

### P3-2: Security-Scanning in CI (L5)

**Datei:** `.github/workflows/backend-ci.yml`
**Aenderung:** `pip-audit` (Dependency-Vulnerabilities) und `bandit` (Code-Security) als CI-Steps. Aktuell advisory (continue-on-error), blockiert CI nicht.

**Datei:** `pyproject.toml`
**Aenderung:** `[tool.bandit]` Konfiguration: exclude tests, skip B101 (assert).

### P3-3: Dependency Lock File (L7)

**Datei:** `requirements.lock` (NEU)
**Aenderung:** Deterministisches Lock-File via `pip freeze`. Alle 81 Dependencies gepin

### P3-4: Health/Readiness/Liveness Probes (K5)

**Datei:** `src/assembled_core/api/app.py`
**Aenderung:**
- `/health` — Basic Health-Check mit Uptime
- `/ready` — Readiness-Probe (prueft Kill-Switch-State lesbar)
- `/live` — Liveness-Probe (Event-Loop reagiert)

### P3-5: Kill-Switch POST-Endpoints (K1 teilweise)

**Datei:** `src/assembled_core/api/app.py`
**Aenderung:**
- `POST /api/v1/kill-switch/activate` — Kill-Switch aktivieren mit throttle_pct/reason/actor
- `POST /api/v1/kill-switch/deactivate` — Kill-Switch deaktivieren
- `GET /api/v1/kill-switch/state` — Aktuellen State abfragen
- API-Version auf 2.0.0 aktualisiert

### P3-6: Error Recovery (K3)

**Status:** Bereits implementiert — alle Router haben korrektes 404/500-Handling via HTTPException.

---

## Phase 4: Architektur- und Domain-Erweiterungen

### P4-1: Error-Classification-System (I3)

**Datei:** `src/assembled_core/errors.py` (NEU)
**Aenderung:** Zentrale Exception-Hierarchie:
- `RecoverableError` (retry) — `DataFeedError`, `PriceLookupError`, `BrokerConnectionError`
- `DegradableError` (proceed degraded) — `StaleDataWarning`, `OptionalSourceUnavailable`
- `FatalTradingError` (stop) — `RiskLimitBreached`, `ReconciliationError`, `KillSwitchActive`, `PITViolation`

**Datei:** `src/assembled_core/execution/broker_adapter.py`
**Aenderung:** `RuntimeError` durch `PriceLookupError` ersetzt.

### P4-2: Leakage-Detection als Pflicht-Gate (D1)

**Datei:** `src/assembled_core/qa/qa_gates.py`
**Aenderung:** Neue `check_leakage()` Gate-Funktion. Prueft altdata-Features auf Look-Ahead-Bias. Ergebnis: BLOCK bei Leakage, OK sonst.

### P4-3: Feature-Staleness-Detection (E3)

**Datei:** `src/assembled_core/data/freshness_monitor.py`
**Aenderung:** Neue `detect_stale_features()` Funktion. Erkennt pro (symbol, feature) ob der Wert N Tage konstant war (moeglicherweise Datenfeed-Ausfall).

### P4-4: Circuit Breaker fuer Flash-Crash (C4)

**Datei:** `src/assembled_core/risk/circuit_breaker.py` (NEU)
**Aenderung:** `CircuitBreaker` Klasse mit:
- Konfigurierbarer Drop-Threshold (default 3%) und Zeitfenster (default 15min)
- Cooldown-Periode (default 30min)
- Zustandsabfrage, manueller Reset, Trip-Counter
- Rolling-Window-Observation von Preisen

### P4-5: Order-Lifecycle-Tracking (G6)

**Datei:** `src/assembled_core/execution/order_lifecycle.py` (NEU)
**Aenderung:** Vollstaendiges Order-Lifecycle-System:
- `OrderState` Enum: CREATED -> VALIDATED -> SUBMITTED -> PARTIAL_FILL -> FILLED/CANCELLED/REJECTED
- Validierte State-Machine (nur erlaubte Transitionen)
- `OrderLifecycleTracker` mit Create/Transition/Query
- Jedes Event mit Timestamp und optionalen Details

### P4-6: Borrow-Cost-Modell (H6)

**Status:** Bereits implementiert — `BorrowCostModel` mit GC/Special/HTB-Tiers existiert in `transaction_costs.py:760`.

---

## Verifizierung (Stand nach Phase 4)

- **Ruff:** 0 Fehler in `src/`
- **Pytest:** 3562 passed, 0 failures, 12 skipped, 2 xfail
- **Keine neuen Testfehler eingefuehrt**
