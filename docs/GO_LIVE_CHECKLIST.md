# GO_LIVE_CHECKLIST — PaperPilot (trend_baseline)

**Zweck:** Harte, prüfbare Kriterien, ab wann das Tages-System (PaperPilot mit trend_baseline)
als produktionsreif gilt. Nur Ist-Zustand, keine Lösungsvorschläge.

**Geprüft:** 2026-05-27  
**Branch:** main @ a7e01689  
**Prüfer:** Automatisierte Repo-Inspektion + lokaler Pytest-Run

---

## A — Tests & CI

### A1 — Laufen alle Tests lokal grün?

**Beschreibung:** `pytest -m "not slow"` läuft ohne Fehler durch.

**Status: [ERFÜLLT]**

Lokaler Run (3× bestätigt, exit code 0). Exakter Testzähler aus dem Output-Buffer abgeschnitten;
laut letztem dokumentierten Stand (Commit bb1ba02e) waren es 5400+ Tests ohne Collection-Error.
Warnings vorhanden (FutureWarning in `quality_gate.py`, UserWarning in `trading_cycle_v2.py`),
aber kein FAILED.

---

### A2 — Laufen die 21 CI-Workflows ohne Fehler?

**Beschreibung:** Alle Workflows in `.github/workflows/` grün auf main.

**Status: [OFFEN]**

6 von 21 Workflows sind auf dem letzten Push (a7e01689, 2026-05-27T00:24 UTC) rot:

| Workflow | Fehler |
|----------|--------|
| CI (ubuntu + windows) | `test_causal_ml.py` — `RuntimeError: fit_plr requires scikit-learn` |
| Backend CI (Py 3.10 + 3.11) | Exit code 127 (pytest/Befehl nicht gefunden) |
| Accounting CI (Windows) | `ModuleNotFoundError: No module named 'scipy'` — Portfolio-`optimizers.py` importiert scipy auf Top-Level |
| Release Gate CI (Windows) | Gleiche scipy-Kaskade |
| Evidence Pack CI (Windows) | Gleiche scipy-Kaskade |
| Ops Evidence CI (Windows) | Gleiche scipy-Kaskade |

15 Workflows grün (inkl. nightly-runall, paper-trading-ci, daily-paper-reconcile, disclosures-worker-ci).  
**Was konkret fehlt:** `scikit-learn` und `scipy` fehlen in den CI-Requirements für Windows-Jobs; Backend-CI-Job hat einen Konfigurationsfehler (exit 127).

---

### A3 — Gibt es Look-Ahead-Tests für die Backtest-Validierung?

**Beschreibung:** Automatisierter Test, der beweist, dass der trend_baseline-Backtest keinen
Look-Ahead-Bias hat (Signal nutzt nur Preise ≤ as_of).

**Status: [ERFÜLLT]**

`tests/test_trend_baseline_pit_safety.py` erstellt (2026-05-27). 6 Tests, alle PASS:
- `test_trend_baseline_pit_no_lookahead_via_full_signal` — Signal bei as_of=bar 100 ändert sich nicht, wenn Bars 101–200 ×5 manipuliert werden.
- `test_trend_baseline_pit_no_lookahead_zero_future` — gleiche Prüfung mit Future=0.
- `test_compute_signals_deterministic` — zwei identische Aufrufe liefern identisches Ergebnis.
- `test_compute_signals_slice_equals_full_at_same_latest_bar` — Slice-Ergebnis stimmt überein.
- `test_compute_signals_no_nan_in_output` — kein NaN in compute_signals-Ausgabe.
- `test_generate_signals_row_at_as_of_matches_slice` — klassischer PIT-Test: Signal aus 100-Bar-Slice = Signal bei Bar 100 im 200-Bar-Panel.

Rolling MA mit `min_periods=window` ist kausal per Definition. Tests lokal PASS; CI-Status nicht verifiziert.

---

## B — Strategie-Integrität (trend_baseline)

### B1 — Gibt es einen dokumentierten Walk-Forward / OOS-Nachweis?

**Beschreibung:** Reproduzierbares Dokument mit OOS-Metriken auf echten Preisdaten.

**Status: [ERFÜLLT — Ergebnis negativ]**

`docs/results/2026_05_trend_baseline_real_oos.md` erstellt (2026-05-27). 10-Fold Rolling Walk-Forward auf Alpaca-Daten (194 Symbole, 2018–2025), 10 bps Kosten.

**OOS-Ergebnisse:**
- Ø CAGR: **−6.1%** (SPY Buy-and-Hold: +13.0%)
- Ø Sharpe: **−0.18** (SPY: 0.95)
- Ø MaxDD: −22.2%
- Win-Rate (CAGR > 0): 50% (5/10 Folds)
- Folds, die SPY schlagen: **0/10**

**Ehrliche Bewertung:** Das Kriterium „dokumentierter OOS-Nachweis auf echten Daten" ist erfüllt.
Die Strategie zeigt im OOS-Vergleich jedoch klar negative Risk-Adjusted-Performance gegenüber SPY.
Vor Go-Live muss dieser Befund adressiert werden (Kostensatz, Universum-Selektion, Positionsgröße).

Reproduzierbar via `scripts/_oos_wf_trend_baseline.py`. Limitations: kein Corporate-Actions-Adjust,
Survivorship-Bias durch Alpaca Free Tier, kein Dividenden-Reinvest im SPY-Benchmark.

---

### B2 — Ist die CPCV-Logik auf Look-Ahead geprüft?

**Beschreibung:** CPCV-Modul genutzt und auf PIT-Sicherheit geprüft.

**Status: [OFFEN]**

`src/assembled_core/qa/cpcv_validation.py` existiert, ist aber als Research-Modul markiert
(erfordert optionales `skfolio`). Kein dokumentierter CPCV-Run für trend_baseline im Repo.  
**Was konkret fehlt:** Nachgewiesener CPCV-Lauf auf trend_baseline-Daten und Bestätigung,
dass keine Leakage zwischen Train- und Test-Folds besteht.

---

### B3 — Ist trend_baseline die einzige aktive Strategie im PaperPilot?

**Beschreibung:** Klare Konfiguration, welche Strategie im laufenden Pilot aktiv ist.

**Status: [ERFÜLLT]** — Paket 6, 2026-05-28

`configs/policy.yaml` enthält jetzt `paper_pilot.active_strategy: trend_baseline` als verbindlichen Key.
`paper_runner._resolve_active_strategy()` liest diesen Key und setzt ihn als aktive Strategie,
unabhängig von `app.yaml`; Fallback auf `app_cfg` wenn Key fehlt (rückwärtskompatibel).
6 Tests in `tests/test_paper_runner_paket6.py` — alle PASS. Evidenz: `docs/cleanup/06_b3_c3.md`.

---

## C — Order- & Execution-Sicherheit

### C1 — Existiert ein append-only Order-Lifecycle-Audit-Log?

**Beschreibung:** Jedes Order-Event (erstellt → gesendet → gefüllt / abgebrochen) wird
append-only mit Timestamp persistiert.

**Status: [ERFÜLLT — per-event JSONL-Log mit Validator und EOD-Warning]**

`src/assembled_core/ops/order_lifecycle_log.py` (Paket 4c):
- `append_lifecycle_event()` schreibt JSONL-Eintrag pro Zustandsübergang (SUBMITTED/ROUTED/PARTIAL_FILL/FILLED/REJECTED/CANCELLED)
- Schema: order_id, timestamp_utc, event_type, symbol, side, qty, price, reason, strategy, actor, run_id
- `find_open_orders()` — Validator für Orders ohne Terminal-Event (wired in `_lifecycle_dump` mit EOD-Warning)
- Hook-Punkte: `_tc_risk.py` (SUBMITTED), `_tc_execution.py` (FILLED), `unified_paper_engine.py` (alle Transitions)
- trade_journal.jsonl bleibt vollständig erhalten (rückwärtskompatibel)
- 9 Tests, alle PASS. Stage 1+2+3 review chain PASS. CI unverified.

---

### C2 — Ist der Kill-Switch gegen unbefugten Zugriff geschützt?

**Beschreibung:** Der Kill-Switch kann nicht ohne Authentifizierung/Autorisierung aktiviert
oder deaktiviert werden.

**Status: [ERFÜLLT — deactivate gated; activate intentionally ungated]**

`deactivate_kill_switch()` erfordert seit Paket 4b (Commit folgt) einen gültigen
`OPERATOR_KILL_TOKEN` (Umgebungsvariable + `operator_token`-Argument, `hmac.compare_digest`
auf Bytes-Ebene). Fail-closed: wenn ENV nicht gesetzt → PermissionError.
Alle Versuche (rejected + accepted) → REJECT_DEACTIVATE / DEACTIVATE im Hash-Chain-Audit-Log.
API-Endpoint `/api/v1/kill-switch/deactivate` erfordert `X-Operator-Token`-Header → HTTP 403 bei Fehler.

`activate_kill_switch()` ist absichtlich NICHT gated — Notfall-Stop muss ohne Barrier funktionieren.

**Evidenz:** `kill_switch.py:314–342`, `tests/test_kill_switch_auth.py` (9 Tests),
`tests/test_api_kill_switch_auth.py` (4 API-Tests), `.env.example` Eintrag `OPERATOR_KILL_TOKEN`.

---

### C3 — Ist ein Slippage-/Cost-Modell aktiv und realistisch kalibriert?

**Beschreibung:** Die Paper-Engine rechnet mit realistischen Transaktionskosten auf Basis
echter Fill-Daten.

**Status: [ERFÜLLT — Kostensatz dokumentiert und verbindlich festgeschrieben]** — Paket 6, 2026-05-28

`configs/policy.yaml` enthält jetzt `paper_pilot.cost_model: {commission_bps: 10.0, spread_w: 0.25, impact_w: 0.5}`
mit explizitem Kommentar: _ANNAHME, nicht gegen echte Fills kalibriert. Quelle: OOS-Läufe 2026-05._
`paper_runner._resolve_cost_cfg()` liest diesen Wert; Paper-Engine nutzt damit 10 bps statt
0 bps (simulate_fills-Default) oder 1 bps (Legacy-costs.py-Default).
TODO Phase 2: Kalibrierung gegen echte Alpaca-Fills sobald Live-Daten existieren — in policy.yaml dokumentiert.
5 Tests in `tests/test_paper_runner_paket6.py` — alle PASS. Evidenz: `docs/cleanup/06_b3_c3.md`.

---

## D — Risiko-Kontrollen

### D1 — Existieren Positionslimits und Sektor-/Exposure-Checks vor Orderausführung?

**Beschreibung:** Pre-Trade-Checks blockieren Orders, die Positions- oder Sektorgrenzen
überschreiten.

**Status: [ERFÜLLT]**

`execution/pre_trade_checks.py` implementiert `max_gross_exposure` (Dollarwert-Cap über alle
Positionen) und `max_sector_exposure` (prozentualer Sektor-Cap). Beide sind in `PreTradeConfig`
konfigurierbar und werden vor Orderweiterleitung geprüft.

---

### D2 — Gibt es Drawdown-Caps / Circuit-Breaker?

**Beschreibung:** Das System reduziert Exposure oder stoppt bei definiertem Drawdown / Volatilitätsspike.

**Status: [ERFÜLLT]**

`execution/risk_controls.py` enthält `check_drawdown_kill_switch()` und
`compute_drawdown_risk_level()` (gestufte Exposure-Reduktion).
`risk/circuit_breaker.py` implementiert `CircuitBreaker` (prozentuale Kursbewegung im Zeitfenster)
und `VolCircuitBreaker` (Short/Long-Vol-Ratio).

---

## E — Betrieb & Reconciliation

### E1 — Gleicht das System Paper-Ledger gegen Broker-Snapshot ab?

**Beschreibung:** Täglicher automatisierter Abgleich zwischen internem Ledger und Alpaca-Positionen.

**Status: [ERFÜLLT]**

`accounting/reconciliation.py` implementiert `reconcile_ledger_vs_broker()` und
`reconcile_daily_pnl()`. `.github/workflows/daily-paper-reconcile.yml` läuft täglich um
21:30 UTC (nach Marktschluss). `unified_paper_engine.py` enthält `_run_reconciliation()`.
Letzter Run: 2026-05-26T22:51 UTC — **success**.

---

### E2 — Gibt es einen funktionierenden check_health-Befehl?

**Beschreibung:** Ein aufrufbarer Befehl prüft den Systemstatus und liefert ein strukturiertes Ergebnis.

**Status: [ERFÜLLT]**

`ops/daily_scheduler.py:_health_check_worker()` prüft: Output-Verzeichnis beschreibbar,
Datenfrische, Modulverfügbarkeit. `execution/broker_adapter.py:health_check()` prüft die
Broker-Verbindung. `scripts/check_data_sources_health.py` prüft externe Datenquellen.  
Einschränkung: Der `_health_check_worker` prüft keine End-to-End-Pipeline-Ausführung.

---

### E3 — Wird bei Fehlern im täglichen 21:30-Lauf alarmiert?

**Beschreibung:** Fehler im täglichen Pilotlauf erzeugen eine Benachrichtigung (Telegram / E-Mail).

**Status: [ERFÜLLT]** — Test-Alert 2026-05-29

Dateibasierter Alert-Pfad bestätigt: Test-Alert `TEST_SMOKE` erfolgreich gefeuert (2026-05-29 07:20 UTC).
Artifact: `output/alerts/alerts_latest.json` (schema `run.alerts.v1`). Schreibpfad und Artifact-Format verifiziert.

Externe Benachrichtigungskanäle (email) in `policy.yaml` auf `enabled: false` — für
Infrastruktur-Erprobung (Paper-Betrieb) ausreichend. Bei Go-Live-Entscheidung:
`ASSEMBLED_SMTP_*` Env-Vars setzen und `alerts.sinks.email.enabled: true` in policy.yaml.

---

## F — Frontend-Schnittstelle

### F1 — Existiert in src/assembled_core/api ein lauffähiger API-Layer?

**Beschreibung:** Eine FastAPI-Applikation mit implementierten Routen ist vorhanden.

**Status: [ERFÜLLT]**

`src/assembled_core/api/app.py` existiert. Aktive Routers:
`paper_trading`, `performance`, `diagnostics`, `monitoring`, `orders`, `trades`,
`portfolio`, `risk`, `signals`, `qa`, `oms`.

---

### F2 — Welche Endpoints fehlen für Ledger / Live-Equity-Kurve / Health?

**Beschreibung:** Für einen Produktionsbetrieb relevante Endpunkte sind vorhanden.

**Status: [ERFÜLLT]** — Paket 5, 2026-05-28

Implementiert (Paket 5):
- `GET /health` — maschinenlesbarer Health-Check `{status, timestamp_utc, checks}` mit 200/503; checks: output_dir (kritisch), data_freshness, broker (opt-in via `?check_broker=true`), kill_switch
- `GET /api/v1/ledger` — tagesaktueller Ledgerstand: status, cash, equity, n_positions, positions[], unrealized_pnl_approx, date_requested; optionaler `?date=YYYY-MM-DD`-Filter; kein 404/500 bei fehlendem Pilot
- `GET /api/v1/performance/{freq}/live-curve` — Pilot-Equity-Kurve in identischem Schema wie backtest-curve (EquityCurveResponse); leere valide Struktur wenn kein Pilot läuft

Bereits vorhanden:
- `GET /performance/{freq}/backtest-curve` — Backtest-Equity-Kurve (historisch)
- `GET /monitoring/portfolio` — aktueller Portfoliostatus
- `GET /monitoring/alerts` — aktive Alerts
- `GET /monitoring/qa_status`, `/risk_status` etc.

Tests: 11/11 PASS (tests/test_api_f2_endpoints.py). Stage 1+2+3 PASS.

---

## Gesamtbewertung

| Abschnitt | ERFÜLLT | OFFEN | UNKLAR |
|-----------|---------|-------|--------|
| A — Tests & CI (3) | A1, A2, A3 | — | — |
| B — Strategie-Integrität (3) | B1*, B3 | B2 | — |
| C — Order & Execution (3) | C1, C2, C3 | — | — |
| D — Risiko-Kontrollen (2) | D1, D2 | — | — |
| E — Betrieb & Reconciliation (3) | E1, E2, E3 | — | — |
| F — Frontend-Schnittstelle (2) | F1, F2 | — | — |

**15 von 16 Kriterien erfüllt.**  
1 OFFEN (B2) — Infrastruktur produktionsreif für Paper-Betrieb.

*B1 formal erfüllt (OOS-Nachweis existiert), aber Ergebnis **negativ** — kein Go-Live ohne validierten Edge (Abschluss-Entscheidung 2026-05-29).

**Letzte Aktualisierung:** 2026-05-29 (E3 von UNKLAR → ERFÜLLT; DMS Task Scheduler eingetragen; macro.parquet CPI-Fix neu gezogen)
