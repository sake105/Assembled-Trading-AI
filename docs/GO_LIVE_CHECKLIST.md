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

**Status: [UNKLAR]**

`paper_runner.py` und `paper_track.py` unterstützen `trend_baseline`, `multifactor_v1`,
`multifactor_v2` und `ema_trend_v0` gleichzeitig. In `paper_runner.py:353` ist dokumentiert:
„§9.6(b) Phase 2 — promoted from shadow to primary-eligible." `configs/policy.yaml` enthält
keinen `strategy_type`-Key; die Strategie wird als Parameter übergeben.  
**Was konkret fehlt:** Ein einziger, offizieller Konfigurationseintrag in `policy.yaml` oder
`OPERATING.md`, der den aktiven Pilot-Strategienamen verbindlich festlegt.

---

## C — Order- & Execution-Sicherheit

### C1 — Existiert ein append-only Order-Lifecycle-Audit-Log?

**Beschreibung:** Jedes Order-Event (erstellt → gesendet → gefüllt / abgebrochen) wird
append-only mit Timestamp persistiert.

**Status: [OFFEN]**

`trade_journal.py` schreibt JSONL-Einträge pro Fill (append-only durch JSONL-Natur).
`execution/order_lifecycle.py` und `execution/order_management.py` existieren, aber kein
zentrales, append-only Audit-Log, das jede Lifecycle-Phase eines Orders lückenlos festhält.
Kill-Switch-Audit hat fsync+Hash-Chain (sichergestellt in Commit aus Session 2026-05-12),
aber das deckt nur Kill-Switch-Events ab.  
**Was konkret fehlt:** Ein dediziertes, unveränderliches Lifecycle-Log (submitted, routed,
filled, rejected, cancelled) pro Order mit strukturiertem Schema.

---

### C2 — Ist der Kill-Switch gegen unbefugten Zugriff geschützt?

**Beschreibung:** Der Kill-Switch kann nicht ohne Authentifizierung/Autorisierung aktiviert
oder deaktiviert werden.

**Status: [OFFEN]**

`kill_switch.py` existiert mit fsync-Durabilität und Hash-Chain für Audit-Integrität.
Dead-Man-Switch (`ops/dead_man_switch.py`) existiert ebenfalls (Commit 86468b0c).
Kein Authentifizierungs- oder Autorisierungsmechanismus in `kill_switch.py` gefunden —
`activate_kill_switch()` und `deactivate_kill_switch()` sind ohne Zugriffskontrolle aufrufbar.  
**Was konkret fehlt:** Eine explizite Access-Control-Schicht oder Betreiber-Bestätigung
(z. B. Actor-Whitelist, Token-Check oder OS-Berechtigungsschutz) für Aktivierung/Deaktivierung.

---

### C3 — Ist ein Slippage-/Cost-Modell aktiv und realistisch kalibriert?

**Beschreibung:** Die Paper-Engine rechnet mit realistischen Transaktionskosten auf Basis
echter Fill-Daten.

**Status: [OFFEN]**

`fill_model.py` nutzt `CostModel(commission_bps=1.0, spread_w=0.25, impact_w=0.5)` als Default
(aus `costs.py`). Ein `cost_model_calibrator.py` existiert. Der 5-Jahres-Backtest in
`docs/results/2026_04_trend_baseline_5y.md` nutzte **10 bps** — nicht das API-Default von 1 bps.
Keine Evidenz, dass der Kalibrator gegen echte Alpaca-Fills gelaufen ist.  
**Was konkret fehlt:** Ein dokumentierter Kalibrierungslauf gegen reale Fills und ein
festgehaltener, begründeter Kostensatz in `policy.yaml` oder einem Kalibrierungsartefakt.

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

**Status: [UNKLAR]**

`ops/alerting.py` implementiert Telegram- und E-Mail-Dispatch. `policy.yaml` enthält
`alerts: enabled: true`. Die Kanäle werden ausschließlich über Umgebungsvariablen konfiguriert
(`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `ALERT_EMAIL_TO`). Ob diese Variablen auf dem
Task-Scheduler-Host tatsächlich gesetzt und getestet sind, ist aus dem Repo nicht verifizierbar.  
**Was konkret fehlt:** Dokumentierter Beweis (z. B. Test-Alert-Log oder `.env`-Konfig-Hinweis),
dass Alerting auf dem Produktionshost funktioniert.

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

**Status: [OFFEN]**

Vorhanden:
- `GET /performance/{freq}/backtest-curve` — Backtest-Equity-Kurve (historisch)
- `GET /monitoring/portfolio` — aktueller Portfoliostatus
- `GET /monitoring/alerts` — aktive Alerts
- `GET /monitoring/qa_status`, `/risk_status` etc.

**Was konkret fehlt:**
- Kein `GET /health`-Endpoint (dediziert, maschinenlesbar für Monitoring-Tools)
- Kein `GET /performance/{freq}/live-curve` — Equity-Kurve aus dem laufenden Paper-Pilot
- Kein `GET /ledger` — tagesaktueller Ledgerstand (Transaktionen, Cash, PnL)

---

## Gesamtbewertung

| Abschnitt | ERFÜLLT | OFFEN | UNKLAR |
|-----------|---------|-------|--------|
| A — Tests & CI (3) | A1, A3 | A2 | — |
| B — Strategie-Integrität (3) | B1* | B2 | B3 |
| C — Order & Execution (3) | — | C1, C2, C3 | — |
| D — Risiko-Kontrollen (2) | D1, D2 | — | — |
| E — Betrieb & Reconciliation (3) | E1, E2 | — | E3 |
| F — Frontend-Schnittstelle (2) | F1 | F2 | — |

**8 von 16 Kriterien erfüllt.**  
6 OFFEN, 2 UNKLAR — Produktionsreife nicht gegeben.

*B1 formal erfüllt (OOS-Nachweis existiert), aber Ergebnis **negativ** — Strategie muss vor Go-Live überarbeitet werden.

**Letzte Aktualisierung:** 2026-05-27 (A3 + B1 von OFFEN → ERFÜLLT)
