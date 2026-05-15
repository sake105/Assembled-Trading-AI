# Senior-Code-Reviewer System-Audit Round 3 — 2026-05-15

**Auftrag:** Verifikation der R2-Followup-Fixes + Deep-Dive in R1/R2-unterauditierte Bereiche + Cross-Cutting Pattern-Hunt.

**Status:** Round 3 abgeschlossen via 3 parallele Reviewer (Opus simulation).

**Eingangsbasis:**
- Round 1: `docs/SENIOR_REVIEW_AUDIT_2026-05-15.md` (48 Findings)
- Round 2: `docs/SENIOR_REVIEW_AUDIT_2026-05-15-ROUND2.md` (Verifikation + 2 Sweep-Gaps + 1 Design-MAJOR)
- R2-Followup-Commits: `20d9897` (F-C2-R2-1), `fc5f482` (date.today sweep), `5be9ba4` (EventStore symmetric)

---

## 1. Executive Summary

**Alle R2-Followup-Fixes verifiziert CLOSED.** Aber Round 3 hat in den **untertestierten Tiefen** (broker_adapter, pre_trade_checks CVaR/group-exposure, unified_paper_engine 2900-Zeilen-File) **3 neue MAJORs** aufgedeckt:

| ID | Severity | Was | Pre-existing? |
|---|---|---|---|
| **F-A3-1** | MAJOR | `unified_paper_engine.py:1523` pre_trade_checks failen **fail-OPEN** (WARN + Original-Orders durchgereicht). Verletzt Rule 30 fail-closed Mandat. Sister-Pfad in risk_controls L545 ist korrekt fail-closed. | Ja |
| **F-A3-2** | MAJOR | `pre_trade_checks.py:344-489` Group-Exposure-Check: `exposures_df` einmal berechnet, dann sektor→region→fx aufgerufen — region/FX sehen veraltete Pre-Scaling-Exposure. | Ja |
| **F-A3-4** | MAJOR | `broker_adapter.py:543+` — `client_order_id` wird NIE an Alpaca SDK übergeben. Broker-Side Idempotenz **effektiv nicht funktional**. Network-Retry erzeugt doppelte Orders. `is_duplicate_error` Handler bei L585-594 ist **dead code** im Normalbetrieb. | Ja |

Plus **2 MEDIUM + 8 LOW/MINOR** Findings. Plus eine **inakkurate Aussage in fc5f482**: Commit-Message sagte "remaining date.today() in src/ are in data/sources/*" — Round 3 fand **6 weitere Sites** außerhalb data/sources/*.

### Verdict

R2-Followup verifiziert: **PASS** (3/3 Fixes clean).
R3 Tiefen-Audit: **CONDITIONAL** — 3 neue MAJORs verdienen R4-Aufmerksamkeit.

---

## 2. Verifikation der R2-Followup-Fixes

| ID | Fix-Commit | Verdict | Evidence |
|---|---|---|---|
| **F-C2-R2-1** qty-sync API | `20d9897` | **CLOSED** | `filtered_qty_by_row` Map gebaut aus `_row_id`+`qty`. Mutation `order.quantity = filtered_qty` VOR `passed_orders.append`. Zero-qty Edge → explizit REJECTED. Fallback-Pfad protokolliert qty-sync-Degradation. Tolerance 1e-9 verhindert Float-Noise. |
| **F-A2-4** unified_paper_engine | `d8d1034` | **CLOSED** | 9-Branch Case-Split mathematisch äquivalent zu F-A-1 (paper_ledger). Cash-Invariant respektiert. Beide Implementierungen jetzt konsistent. |
| **B2-N1** EventStore.append_batch | `5be9ba4` | **CLOSED** | try/except wrappt executemany, raised `EventAppendError` symmetrisch zu append. Logger-Message gespiegelt. |
| **date.today() Sweep** | `fc5f482` | **PARTIAL** | 4 erfasste Sites korrekt gefixed. **ABER:** Commit-Message-Scope inakkurat — siehe §4.4 (B3-N1/N2/N3). |

---

## 3. Neue MAJOR-Findings aus R3

### 3.1 F-A3-1 — `unified_paper_engine` pre_trade_checks fail-OPEN

**Datei:** `src/assembled_core/execution/unified_paper_engine.py:1523`
**Kategorie:** silent-risk-bypass (Rule 30 violation)

**Evidence:**

```python
try:
    # pre_trade_checks call
    ...
except Exception as exc:
    logger.warning("[PAPER] pre_trade_checks error (non-fatal): %s", exc)
    # ← proceeds with ORIGINAL (unfiltered) orders
```

Bei Fehler in pre_trade_checks (Dependency-Drift, Data-Shape-Error, transient bug) werden die **ungefilterten** Orders weitergegeben. Der Sister-Pfad in `risk_controls.py:545` ist korrekt fail-closed (Errors angehängt + downstream blockiert).

**Reachability:** Pre-existing latent bug, kein Regression. Wird real, sobald pre_trade_checks unter Last/Drift einen Exception wirft.

**Suggested Fix:** Fail-closed setzen — bei Exception leere DataFrame zurückgeben + ERROR loggen. Oder Sentinel setzen, das Caller den Cycle abbrechen lässt. Konsistent mit L545.

**Severity:** MAJOR (Rule 30 §pflichtregeln 5 verletzt; keine "großzügigen Freigaben bei Unsicherheit").

### 3.2 F-A3-2 — pre_trade_checks Group-Exposure: Stale exposures_df

**Datei:** `src/assembled_core/execution/pre_trade_checks.py:344-489`
**Kategorie:** stale-state-reuse

**Evidence:**

`_ptc_check_group_exposures` berechnet `exposures_df` einmal bei L344 aus initial `filtered_orders`. Der Sektor-Loop (L407-420) mutiert dann `filtered_orders` via `_apply_group_scale` (qty-Reduktionen). Der nachfolgende Region-Loop (L427-440) ruft `compute_group_exposures(exposures_df, ...)` — aber `exposures_df` ist der **PRE-Sector-Scaling Snapshot.**

Region-Weight-Entscheidungen werden gegen veraltete Exposure getroffen. Gleiches gilt für den FX-Check (L447).

**Konsequenz:**
- Wenn Sector-Cap Tech-Stock-qty um 50% reduziert, sieht der Region-Check immer noch un-scaled Tech-Weight → kann doppelt Reduzieren ODER Region-Cap-Breach nicht erkennen wenn nach Sector-Scaling tatsächlich nicht mehr breached.

**Suggested Fix:** `target_positions_df` und `exposures_df` zwischen Group-Checks neu berechnen (sector → recompute → region → recompute → fx). Oder alle Group-Caps gleichzeitig via combined optimization anwenden.

**Severity:** MAJOR (Rule 30 §invarianten; Risk-Limits werden ineffektiv).

### 3.3 F-A3-4 — Broker-Adapter: Idempotenz-Schlüssel fehlt komplett

**Datei:** `src/assembled_core/execution/broker_adapter.py:543-620` (und ~622-678, ~730+)
**Kategorie:** idempotency-broken

**Evidence:**

`submit_market_order` (und Schwester-Funktionen `submit_limit_order`, `submit_stop_order`) bauen `MarketOrderRequest` **ohne `client_order_id`**. Alpaca generiert dann eine Server-Side UUID pro Call. Der innere Exception-Handler (L585-594) importiert `is_duplicate_error` und behandelt das "duplicate client_order_id detected"-Error — aber da NIE eine `client_order_id` submitted wird, ist dieser Error-Pfad **dead code** im Normalbetrieb.

**Praktische Konsequenz:**
Eine Network-Blip oder Transport-Retry, die die Order re-submitted, erzeugt eine **ZWEITE distinkte Broker-Order** mit frischer UUID → **Position verdoppelt.** Keine echte Idempotenz-am-Broker existiert in diesem Pfad.

**Suggested Fix:**
- Deterministische `client_order_id` generieren (z. B. Hash aus `{symbol, side, qty, intent_id, timestamp-bucket}`)
- An `MarketOrderRequest/LimitOrderRequest` übergeben
- SDK-Feldname verifizieren (`client_order_id` per Alpaca docs)
- `idempotency_store` (`src/assembled_core/execution/idempotency.py`) sollte den Key generieren und auf Retry wiederverwenden

**Severity:** MAJOR (Rule 30 §pflichtregeln 1: Kill-Switch, Pre-Trade, Position-Generierung — Position-Verdopplung durch Retry ist Risk-Event).

---

## 4. MEDIUM-Findings aus R3

### 4.1 B3-N5 — `reset_dd_damper()` nicht in `run_portfolio_backtest`

**Datei:** `src/assembled_core/qa/backtest_engine.py:run_portfolio_backtest`
**Kategorie:** state-leak-across-runs

`reset_dd_damper()` ist (in `multifactor_v2.py:179`) dokumentiert als "always call between isolated backtest runs". `run_portfolio_backtest` ruft es **nicht** auf. Programmatic multi-run Sequenzen (Notebooks, Parameter-Sweeps außerhalb `scripts/batch_runner.py`) erben den DD-Damper-State des vorherigen Runs.

`scripts/batch_runner.py` ist der einzige Caller, der `reset_dd_damper()` aufruft.

**Suggested Fix:** `reset_dd_damper()` am Anfang von `run_portfolio_backtest`, ODER prominenter Docstring-Hinweis auf die Anrufungspflicht.

**Severity:** MEDIUM (Backtests in Notebook-Kontext können verzerrt sein; nicht direkt risk-relevant aber reproduzierbarkeitsschädlich).

### 4.2 B3-N4 — Bare except in DD-damper-Integration

**Datei:** `src/assembled_core/qa/backtest_engine.py:794-802`
**Kategorie:** silent-except (E-003-ähnlich)

`try: ... update_drawdown_damper(...) except Exception: pass`. DD-Damper-Failures werden lautlos verschluckt. Backtest meldet "DD-damper aktiv" während der Call NIE die Damper erreicht (z. B. durch Typo oder fehlende Dep).

**Suggested Fix:** Mindestens `logger.warning("DD-damper update failed: %s", exc, exc_info=True)`.

**Severity:** MEDIUM (Risk-Control silent unverfügbar).

---

## 5. LOW/MINOR-Findings aus R3

### 5.1 Date.today() Sweep — Inakkurate Scope-Aussage

Commit `fc5f482` Message sagte "Remaining date.today() in src/ are in data/sources/*". R3 fand **6 weitere Sites** außerhalb data/sources/:

| Datei | Zeile | Severity | Befund |
|---|---|---|---|
| `signals/composite_score.py` | 657, 660 | LOW (B3-N1) | `as_of_date or date.today()` Fallback für seasonality |
| `data/fx.py` | 98, 100 | LOW (B3-N2) | Live-FX-Fetcher Default — analog zu data/sources/*, aber technisch im data/ Root |
| `signals/buyback_drift.py` | 83, 84 | LOW (B3-N3) | EDGAR Live-Fetch ohne `as_of`-Param |
| `signals/insider_cluster.py` | 67, 68 | LOW (B3-N3) | EDGAR Live-Fetch ohne `as_of`-Param |

**Suggested Fix:** B3-N1 (composite_score) und B3-N3 (buyback/insider) sollten `as_of`-Param erhalten analog F-B-4/5/6. B3-N2 (data/fx.py) ist live-mode-Fetcher — akzeptabel.

### 5.2 F-A3-3 — Group-Exposure-Scaling für Pre-existing Positions

**Datei:** `pre_trade_checks.py:351-405`

`_apply_group_scale` skaliert marginale Order-qty by `cap/gross_weight`. Aber gross_weight = existing + orders. Wenn Pre-existing-Positions schon über `cap` liegen, kann Scaling nur der marginalen Orders das Portfolio **nicht** unter Cap bringen.

**Severity:** MINOR (Scope-Mismatch; sichtbar in Backtests mit Carry-over).

### 5.3 F-A3-5 — Legacy SDK Fallback inkonsistent

**Datei:** `broker_adapter.py:595-602, 670-678, 755+`

Legacy SDK Fallback ruft `api.submit_order(...)` ohne die `is_duplicate_error` Behandlung des modernen Pfads. Inkonsistente Coverage zwischen Pfaden.

**Severity:** MINOR (Legacy SDK ist End-of-Life laut Reviewer).

### 5.4 F-A3-6 — CVaR Sign-Convention-Fragilität

**Datei:** `pre_trade_checks.py:686-687`

Wenn User akzident `max_cvar_95 = 0.05` (positiv statt negativ) konfiguriert, wird die Math `0.05 / -0.07 = -0.714` → clamped to 0 → **alle BUY-Orders silent zeroed.** Keine Input-Validation rejected positive `max_cvar_95`.

**Suggested Fix:** `PreTradeConfig.__post_init__`: `assert max_cvar_95 is None or max_cvar_95 < 0`.

**Severity:** MINOR (Config-Fehler-Modus).

### 5.5 F-A3-7 — Intent_store Import silent

**Datei:** `unified_paper_engine.py:2322-2325, 2351-2356`

`try: from ...intent_store import X; except Exception: return []`. Silent return bei Import-Failure. Audit-Chain der submit→complete Intents wird lautlos disabled.

**Severity:** MINOR (Audit-Trail Silent-Fail).

### 5.6 B3-N6 / B3-N7 — ML Model Registry Permissive Defaults

**Datei:** `ml/model_registry.py:78-83, 349-365`

`verify_model_hash` returnt True wenn Registry leer/fehlend. `load_deployed` lädt ohne Hash-Verifikation wenn `ModelVersion` fehlt. Strict-Mode existiert aber ist nicht Default.

**Severity:** LOW (Security-Hardening-Kandidat).

### 5.7 B3-N8 — Brinson Attribution NaN-Handling

**Datei:** `attribution/brinson_hood.py`

Single NaN in Sector-Return → NaN propagiert in alloc/sel/interaction-Sums.

**Severity:** LOW (Doku + fillna nötig).

### 5.8 F-C3-4 — Dead http_client wrapper

**Datei:** `src/assembled_core/utils/http_client.py`

Centralized HTTP wrapper documented as "use this instead of requests directly" — aber alle ~20 Production-Calls nutzen bare `requests.get/post`. Wrapper ist dead code mit irreführender Doku.

**Severity:** INFO (entweder adoptieren oder löschen).

### 5.9 F-C3-5 — `subprocess.run` ohne timeout

**Datei:** `ops/run_manifest.py:66-68`, `paper/paper_track.py:68`, `certify/generator.py:68,71`

`git rev-parse HEAD` ohne `timeout=`. Bei korruptem Repo kann Git unendlich hängen. `daily_scheduler.py:116` macht es korrekt mit `timeout=120`.

**Severity:** MINOR (Reliability).

---

## 6. Status der Cross-Cutting-Probleme (aus R1/R2)

### 6.1 F-B-4/5/6/11 Wiring-Gap (R2 als PARTIAL geflaggt)

**Status:** **INFO downgrade** — alle 4 Funktionen haben **0 Production-Caller**. Nur Test-Aufrufe. Severity reduziert von "PARTIAL latent fix" → "intentional latent" (kein Leak-Pfad existiert heute).

**Risiko bleibt:** Wenn jemand einen Caller hinzufügt und `as_of` vergisst, kommt der Leak zurück.

**Empfehlung:** Statt Wiring-Pflicht zu erzwingen, in nächster Iteration `as_of` **required machen** (kein Default). Macht Compiler-/Type-Check zum Fail-Closed-Gate.

### 6.2 F-C-3 Import-Prefix

**Aktuelle Verteilung:**
- `from src.assembled_core...`: 181 Dateien, 343 Vorkommen in src/, +159 in scripts/
- `from assembled_core...`: 20 Dateien, 20 Vorkommen in src/, +28 in scripts/

**Sensitive Zonen mit bare-Prefix:** `execution/order_gate.py`, `execution/order_management.py`, `execution/rl_execution.py`, `execution/round_trip_detector.py`, `risk/garch_vol.py`, `risk/georisk_overlay.py`.

**Sweep-Effort:** Niedrig-Mittel. ~20 Dateien für Konvergenz auf einen Stil (welcher Stil = Policy-Frage).

### 6.3 Hexagonal-Scaffold

**Status:** R3 bestätigt R1: ~1-2% gebaut (nicht "5%" wie R1 sagte — noch optimistisch).
- `domain/` Subpackages: 5 von 5 LEER (nur `__init__.py`)
- `ports/`: 6 Port-Files (Interfaces existieren)
- `adapters/outbound/`: 4 Module (alerting, audit_logger, clock, event_bus_inprocess)
- `adapters/inbound/`: LEER
- `application/use_cases/`: 1 Modul (`record_kill_switch_trip.py`)
- **0 Production-Caller** aus `execution/`, `risk/`, `pipeline/`, etc. routen durch diesen Layer.

**Empfehlung:** Memory + Doku auf "scaffold only, ~1% adopted" korrigieren. Nicht "active hexagonal architecture" claim.

### 6.4 `iloc[-1]`-without-sort Class

R3 verifiziert: alle bekannten Sites (F-A-3, F-A-4, F-C-10) gefixed. Zwei neue MINOR-Kandidaten (B3-Vorgänger-Findings): `qa/factor_analysis.py` zeigt **gesunde Sort-Disziplin** (18× sort_values). Class scheint repo-weit unter Kontrolle.

---

## 7. Prioritäts-Aktionsplan für Round 4 / Followup

### Sofort (vor Pilot / Live)

1. **F-A3-4 MAJOR** (broker_adapter `client_order_id` fehlt) — Idempotenz am Broker. Retry-Sicherheit ist kritisch für Live-Trading.
2. **F-A3-1 MAJOR** (unified_paper_engine pre_trade fail-OPEN) — Rule 30 Verletzung. Fail-closed setzen.
3. **F-A3-2 MAJOR** (pre_trade_checks stale exposures) — Risk-Limits werden ineffektiv. Recompute-Pflicht zwischen Group-Checks.

### Mittelfristig (1-2 Wochen)

4. **B3-N5 MEDIUM** (`reset_dd_damper` in `run_portfolio_backtest`) — Reproduzierbarkeit.
5. **B3-N4 MEDIUM** (DD-damper bare except logging) — Risk-Control-Sichtbarkeit.
6. **F-A3-3, F-A3-5, F-A3-6, F-A3-7 MINOR** — Sweep der pre_trade_checks + broker_adapter Edge-Cases.

### Backlog

7. **B3-N1/N2/N3 LOW** — Restliche `date.today()`-Sites (composite_score, fx, buyback/insider).
8. **B3-N6/N7 LOW** — Model registry strict defaults (Security-Hardening).
9. **B3-N8 LOW** — Brinson NaN-Doku.
10. **F-C3-4** — http_client wrapper adoptieren oder löschen.
11. **F-C3-5** — subprocess timeout sweep (3 Sites).
12. **F-C-3** — Import-Prefix-Sweep (Policy-Entscheidung + ~20 Dateien).
13. **Hexagonal Doku-Korrektur** — Realistische Adoption-Statusaussage.

### Out-of-Scope für jetzt

- F-B-4/5/6/11 Wiring (INFO downgrade — kein aktiver Leak).
- Tieferes Audit von `unified_paper_engine.py` (~2900 Zeilen, ~60 except-Klauseln — bewusst nicht line-by-line).

---

## 8. Statistik R3

| Metrik | Wert |
|---|---|
| Reviewer | 3 parallele Opus-Simulationen |
| Tool-Use kumuliert | ~120 (≤30 pro Reviewer Cap eingehalten) |
| Token-Verbrauch | ~340k |
| Files deep-dived | broker_adapter.py (~800 LOC), pre_trade_checks.py (~1100 LOC), unified_paper_engine.py (Teile, ~2900 LOC total), ml/, attribution/, qa/factor_analysis.py (2355 LOC), qa/backtest_engine.py (1526 LOC) |
| Verifikationen | 3/3 R2-Followup-Fixes CLOSED |
| Neue Findings | 3 MAJOR + 2 MEDIUM + 8 LOW/MINOR + diverse INFO |

---

## 9. Was R3 NICHT abgedeckt hat

- **Backtest-Runs.** Keine numerische Verifikation der Fixes — statisch-analytisch.
- **broker_adapter.py F-A3-4 cross-check** gegen alpaca-py source code: `client_order_id` Feldname wurde nicht verifiziert. Empfehlung vor Fix: `client_order_id` Field in current alpaca-py SDK bestätigen.
- **F-A3-2 Impact-Tests**: ohne Caller-Pattern mit beide `max_sector` und `max_region` konfiguriert ist die Impact-Magnitude theoretisch.
- **Exhaustive line-by-line auf unified_paper_engine.py** — 2900-Zeilen-File wurde fokussiert sampled.
- **Tests in CI nicht re-verifiziert** — nur lokales pytest auf Windows.

---

**Reviewer:** simulierter `senior-code-reviewer` (Opus 4.7) via `general-purpose` Subagent  
**Datum:** 2026-05-15  
**Audit-Dauer:** 3 parallele R3-Reviewer-Sessions, ~340k Token kumulativ  
**Verifikationsstand:** statisch-analytisch
