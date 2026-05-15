# Senior-Code-Reviewer System-Audit Round 2 — 2026-05-15

**Auftrag:** Verifikation der Fixes aus Round 1 + Suche nach Regressionen + Erkennung neuer Findings.

**Status:** Round 2 abgeschlossen. Drei parallele Reviewer (Opus) gegen die drei Fix-Bereiche.

**Eingangsbasis:** Round 1 Befund (`docs/SENIOR_REVIEW_AUDIT_2026-05-15.md`) — 48 Findings (4 BLOCKER, 14 MAJOR, 21 MINOR, 9 INFO).

---

## 1. Executive Summary

**Alle 4 Round-1 BLOCKER und alle 14 Round-1 MAJORs sind in der Implementierung CLOSED.** Aber Round 2 hat zwei **Sweep-Gaps** aufgedeckt — Fälle, in denen die Original-Fixes nur EINE von zwei parallelen Implementierungen erfasst haben:

| Neues Finding | Severity | Beschreibung | Status |
|---|---|---|---|
| **F-A2-4** | BLOCKER | `unified_paper_engine._update_positions` hat denselben SELL-on-zero Bug wie F-A-1 (paper_ledger). Sister-Modul, im F-A-1 Fix nicht erfasst. | **FIXED in d8d1034** |
| **F-C2-R2-2** | MAJOR | `risk/pdt_counter.py` hat denselben `date.today()`-Bug wie F-C-4 (compliance/pdt.py). Sister-Modul. | **FIXED in d8d1034** |
| F-C2-R2-1 | MAJOR | API-Pfad `paper_trading.py` verwirft qty-scaling der Risk-Controls (drawdown cap, pre-trade clip). Submit_orders erhält Original-Pydantic-Order mit voller qty. Pre-existing, beim F-C-1 Review entdeckt. | OFFEN (Design-Frage) |
| B2-N1 / F-A2-2 | MINOR | `EventStore.append_batch` raised raw `sqlite3.Error`, asymmetrisch zum F-B-12 Fix in `append`. | OFFEN (1-min fix) |
| F-A2-1 | MINOR | `paper_ledger.mark_to_market_equity` HWM-Semantik undefiniert für Shorts. Dormant. | OFFEN (folge-Audit) |
| F-C2-R2-3 | MINOR | `ops/experience_log.py:147-148` `iloc[-1]` ohne Sort-Guard. | OFFEN |
| F-C2-R2-4 | MINOR | `intel/wild_card_detector.py:36-37` Series `iloc[-1]` ohne Sort. | OFFEN |
| B2-N2 | MINOR (latent) | `_update_dd_damper`: `today = as_of or _dt.date.today()` Defensive Default. | OFFEN |
| B2-N3 | INFO | F-B-4/5/6/11 sind PARTIAL geclosed: Implementation korrekt, **aber keine Production-Caller passen `as_of`**. Latente Korrektheit. | OFFEN (Wiring) |

**Verdict nach R2-Sweep:**
- **2 von 2 R2-Sister-Bugs sofort gefixed** (Commit d8d1034).
- **18 von 18 R1-Items (4 BLOCKER + 14 MAJOR) verifiziert CLOSED**, davon 4 PARTIAL bei fehlender Caller-Adoption (F-B-4/5/6/11).
- **3 neue MINORs** + 1 OFFENE MAJOR (F-C2-R2-1, Design-Frage).

**Empfehlung:** F-C2-R2-1 (qty-scaling-Discard) ist konzeptuell — entweder die Pydantic-Order-Liste muss qty-Updates aus dem risk_controls-Output zurückspielen, ODER risk_controls darf qty nicht modifizieren sondern nur reject. Design-Diskussion notwendig.

---

## 2. Methodik

### 2.1 Reviewer-Setup

Drei parallele `senior-code-reviewer`-Simulationen (Opus 4.7 via `general-purpose`):
- **Batch A2** — Sensitive Zones Verifikation (execution, risk, pipeline, accounting, paper, portfolio, ops/paper_ledger).
- **Batch B2** — Decision + Data Verifikation (signals, strategies, ml, data, features, dataquality, events).
- **Batch C2** — Infrastructure + Cross-cutting Verifikation (api, ops, compliance, certify, adapters, application, ...).

Jeder Reviewer hatte:
1. Liste der Fix-Commits + Findings aus Round 1.
2. Pflicht-Verifikation: `git show <commit> -- <file>`, dann Read der Fix-Stelle.
3. Pflicht-Regression-Check: hat der Fix neue Bugs eingeführt?
4. Pflicht-Sweep-Check: Sister-Module mit gleichem Pattern.

Tool-Use-Cap: ≤30 pro Reviewer (Round 1 ging deeper; R2 ist Verifikation + Delta).

### 2.2 Bekannte Limitationen

1. **Simulation** statt registriertem Subagent (gleiche Limitation wie Round 1; `senior-code-reviewer` ist in dieser Session nicht via `subagent_type` dispatchbar).
2. **Local pytest only.** Ubuntu CI nicht re-verified für die Fix-Commits.
3. **Selektives Lesen.** Nicht alle 493 .py-Dateien erneut geprüft; Fokus auf Fix-Areas + Sister-Module.
4. **R1-MINORs nicht alle individuell verifiziert.** F-B-7/8/13-20 sind im R1-Audit nur in der Summary-Tabelle gelistet, ohne dedizierte Detail-Sektion — strukturell nicht einzeln verifizierbar.

---

## 3. Round-1 Findings — Verifikations-Status

### 3.1 BLOCKERs (alle CLOSED)

| ID | Fix-Commit | R2-Verdict | Evidence |
|---|---|---|---|
| **F-A-1** paper_ledger Shorts | `e6d1461` | **CLOSED** | Explizite 9-Branch Case-Split (BUY long-add/cover-partial/cover-exact/cover-flip; SELL long-partial/full-close/oversell-flip/open-short/add-to-short). Cash-Invariant: BUY -=, SELL +=. avg_price wird auf Flip ersetzt, nicht geblendet. 13/13 Tests grün. |
| **F-B-1** geo_risk_composite live FRED leak | `8267234` | **CLOSED** | `as_of`-Param + Path-2 short-circuits zu zero-fill. Call-Site 1290 passt `_bar_as_of`. Test `test_backtest_mode_skips_live_fred_fetch_F_B_1` bestätigt: FRED-Spy NICHT aufgerufen. |
| **F-B-2** insider_cluster_factor SEC-Form-4 leak | `8267234` | **CLOSED** | Gleicher Pattern. Call-Site 1300 passt `_bar_as_of`. Test bestätigt: `cluster_buy_score` NICHT aufgerufen. |
| **F-B-3** buyback_drift_factor 8-K leak | `8267234` | **CLOSED** | Gleicher Pattern. Call-Site 1320 passt `_bar_as_of`. Test bestätigt: `buyback_signal_score` NICHT aufgerufen. |

### 3.2 MAJORs (alle CLOSED, 4 davon PARTIAL pending caller-adoption)

| ID | Fix-Commit | R2-Verdict | Note |
|---|---|---|---|
| F-A-2 crisis_alpha-Fallback | `353583e` | CLOSED | dict-keys `engaged` + `persistent.reason` jetzt korrekt. |
| F-A-3 intel_context iloc-sort | `e4dcb7b` | CLOSED | `_pit_ts` Helper + mergesort + drop-helper-col. |
| F-A-4 _tc_features regime-sort | `e4dcb7b` | CLOSED | Sort by `date` oder `timestamp`, mergesort. |
| F-A-9 corp-actions exact-equality | `353583e` | CLOSED | `.dt.normalize()` auf beiden Seiten + match-count log. |
| F-B-4 macro_regime_quadrant | `1484264` | **PARTIAL** | Defense-in-depth korrekt, **aber keine Production-Caller** passen `as_of`. Latent. |
| F-B-5 recession_probability | `1484264` | **PARTIAL** | Gleiche Latenz. |
| F-B-6 sentiment_panel | `1484264` | **PARTIAL** | Gleiche Latenz. |
| F-B-9 corporate_actions delisting | `6b4a660` | CLOSED | Sort + skip-mit-WARN statt iloc[-1] Fallback. |
| F-B-10 filter_events_as_of default | `6b4a660` | CLOSED | Default `fallback_to_event_date: True→False`. Interne Caller explizit mit `disclosure_col`. |
| F-B-11 pead_sue today-param | `6b4a660` | **PARTIAL** | `today`-Param OK, kein Production-Caller passt ihn. |
| F-B-12 EventStore.append re-raise | `353583e` | CLOSED | `EventAppendError` + raise. ABER `append_batch` asymmetrisch → siehe B2-N1. |
| F-C-1 risk-filter cardinality | `6b4a660` | CLOSED | `_row_id`-Matching mit WARN-Fallback bei Column-Loss. |
| F-C-2 _engine Lock | `6b4a660` | CLOSED | `threading.RLock` an allen 6 _engine.* Call-Sites. |
| F-C-4 PDT date.today() | `353583e` | CLOSED | `America/New_York` via zoneinfo. **ABER:** `risk/pdt_counter.py` Sister-Modul → siehe F-C2-R2-2. |
| F-C-10 portfolio iloc-sort | `e4dcb7b` | CLOSED | `sort_values("timestamp")` vor iloc[-1]. |

---

## 4. Neue Round-2 Findings

### 4.1 F-A2-4 — BLOCKER (FIXED)

**Datei:** `src/assembled_core/execution/unified_paper_engine.py:1582-1591`
**Kategorie:** bug, Sweep-Gap zu F-A-1

**Evidence:** Same anti-pattern wie F-A-1 in der parallelen Implementation `_update_positions`:
```python
sold_qty = min(qty, current_qty)  # cannot sell more than owned
proceeds = sold_qty * fill_price
```

Bei `current_qty = 0`: `sold_qty = 0`, `proceeds = 0`, kein Short eröffnet, kein Cash. **Identischer Bug-Pattern wie F-A-1.** F-A-1 Fix-Scope erfasste `paper_ledger.py`, übersah aber `unified_paper_engine.py` — beide implementieren paper-engine-Ledger-Logic.

**Reachability:** `unified_paper_engine` ist die "unified"-Schicht, die ältere Engines ersetzt. Long-Short-Strategien (multifactor_long_short) führen Shorts durch diesen Pfad.

**Fix (d8d1034):** Volle 9-Branch Case-Split portiert. Konsistent mit F-A-1.

### 4.2 F-C2-R2-2 — MAJOR (FIXED)

**Datei:** `src/assembled_core/risk/pdt_counter.py:42, 50`
**Kategorie:** logic-error, Sweep-Gap zu F-C-4

**Evidence:** Same `date.today()`-Pattern wie F-C-4 in `compliance/pdt.py`. Beide Module implementieren PDT-Logik, aber F-C-4 Fix erfasste nur `compliance/pdt.py`.

**Fix (d8d1034):** `_us_market_today()` Helper mit `America/New_York` (UTC-Fallback) in `risk/pdt_counter.py`. Konsistent mit F-C-4.

### 4.3 F-C2-R2-1 — MAJOR (OFFEN — Design-Frage)

**Datei:** `src/assembled_core/api/routers/paper_trading.py:_apply_risk_controls_to_paper_orders` + Engine-Submit
**Kategorie:** correctness

**Evidence:** `risk_controls.py` führt qty-Scaling durch:
- Step-0 Exposure-Cap (`risk_controls.py:299-304`) skaliert qty proportional zur Drawdown-Damper-Multiplier.
- Pre-trade qty-Clip (`pre_trade_checks.py:232, 243, 385, 396, ...`) reduziert qty bei Limit-Verletzungen.

Aber: `passed_orders.append(order)` (paper_trading.py:201) reicht die **Original-Pydantic-Order** weiter, **ignoriert das aktualisierte qty** aus dem DataFrame. `_engine.submit_orders(passed_orders)` erhält das volle qty → Drawdown-Cap und pre-trade qty-Limits werden im API-Pfad **lautlos** umgangen.

**Status:** Pre-existing, vor F-C-1 da. Beim F-C-1 Review entdeckt — F-C-1 selbst ist CLOSED, aber das aufgedeckte Problem ist tieferer Natur.

**Empfohlene Fixes (Design-Frage — vor Pilot klären):**
- **Option A:** Risk-Controls dürfen qty nicht modifizieren, nur reject. Sauberer Contract.
- **Option B:** API-Pfad muss qty aus `filtered_df` zurück in die Pydantic-Order spielen vor `submit_orders`. Komplexer, behält Reduction-Pfad.

### 4.4 B2-N1 / F-A2-2 — MINOR

**Datei:** `src/assembled_core/events/store.py:106-126` `append_batch`
**Kategorie:** correctness, Asymmetrie zu F-B-12

`append()` raised `EventAppendError` nach F-B-12 Fix. `append_batch()` raised raw `sqlite3.Error`. Inconsistent. Catchers, die nur `EventAppendError` fangen, verpassen Batch-Failures.

**Fix (offen):** `try/except sqlite3.Error → raise EventAppendError(str(exc)) from exc` um `executemany` in `append_batch`.

### 4.5 F-A2-1 — MINOR (latent)

**Datei:** `src/assembled_core/ops/paper_ledger.py:396-399` `mark_to_market_equity`

HWM (High-Water-Mark) wird unbedingt mit `max(price)` aktualisiert, unabhängig vom Position-Sign. Für Shorts (jetzt nach F-A-1 erreichbar) ist das semantisch falsch — Shorts gewinnen auf Drops, der relevante HWM wäre `min(price)`. Aktuell **dormant**, weil alle HWM-Konsumenten (multifactor_v1/v2/ema_trend_v0 trailing-stop) bei `qty <= 0` early-returnen.

**Empfehlung:** Vor next-trailing-stop-on-shorts-Implementierung adressieren.

### 4.6 F-C2-R2-3 / F-C2-R2-4 — MINOR

`ops/experience_log.py:147-148` und `intel/wild_card_detector.py:36-37` haben `iloc[-1]` auf caller-supplied Series/DataFrame ohne Sort-Guard. Latent — Class-of-Bug ähnlich F-A-3/4/C-10. Fix: Sort-Pflicht im Docstring oder defensives `sort_values` in der Funktion.

### 4.7 B2-N2 — MINOR (latent)

`multifactor_v2._update_dd_damper:202`: `today = as_of or _dt.date.today()`. Defensive Default, aktueller Caller-Pfad scheint immer `as_of` zu passen — aber gleiche Risk-Klasse wie F-B-11 / die `as_of or pd.Timestamp.now()` Pattern aus R1.

### 4.8 B2-N3 — INFO

F-B-4/5/6/11 sind PARTIAL CLOSED: Implementation korrekt, aber **keine Production-Caller passen `as_of`**. Latente Korrektheit — wenn ein zukünftiger PR sie naiv verdrahtet (ohne `as_of`), ist der Leak zurück.

**Empfehlung:** Either (a) Deprecation-Warning bei `as_of=None` + Backtest-Mode-Hinweis, oder (b) Lint-Rule, die Aufrufer ohne `as_of` flaggt.

---

## 5. Residual MINORs aus Round 1 — Status

| ID | Status | Note |
|---|---|---|
| F-A-5 (vix_z `or 0.0`) | OPEN | Nicht im R1-Fix-Scope. |
| F-A-6/7/10 | NOT_LOCATED | Im R1-Audit nur in Summary-Tabelle referenziert, keine Detail-Sektion. Strukturell nicht einzeln verifizierbar. |
| F-A-8 (dead tz_localize in ledger.py:190-191, 300-301) | OPEN | 2 Sites confirmed still present. |
| F-B-7, F-B-8, F-B-13..F-B-20 | NOT_DOCUMENTED | Nicht detailliert in R1-Audit. |
| F-C-3 (Import-Prefix-Inkonsistenz, 341 Dateien) | OPEN | Not in fix scope, not regressed by R1 fixes. |
| F-C-5 (elster.py `date.today()`) | OPEN | Annual filing context, lower priority. |
| F-C-6 (drift_monitor.py:105) | OPEN | Confirmed present. |
| F-C-7 (post_trade_analyzer.py:468) | OPEN | Confirmed present. |
| F-C-8 (daily_scheduler.py:577) | OPEN | Confirmed present. |
| F-C-9, F-C-11..F-C-23 | OPEN | Nicht im Fix-Scope, nicht regressed. |

---

## 6. Sweep-Vervollständigung benötigt (Cross-Cutting)

### 6.1 `date.today()`-Sweep — 4 Sites OFFEN

Nach F-C-4 (compliance/pdt.py) und F-C2-R2-2 (risk/pdt_counter.py) bleiben offen:
- `src/assembled_core/compliance/elster.py:148` (Steuer-Doku, low impact)
- `src/assembled_core/ops/drift_monitor.py:105`
- `src/assembled_core/qa/post_trade_analyzer.py:468`
- `src/assembled_core/ops/daily_scheduler.py:577`

**Empfehlung:** Einmal-Sweep via grep + ersetzen mit `datetime.now(tz=timezone.utc).date()`. Niedriger Aufwand.

### 6.2 PIT-Live-Fetch-Pattern — 4 Latent

F-B-4/5/6/11 sind implementiert aber ohne Caller-Adoption. Wenn ein zukünftiger Wiring-Step diese naiv einbindet, kommt der Leak zurück.

**Empfehlung:** Test-Hook der bei `as_of=None` in einer detected Backtest-Umgebung einen Warning ausspielt.

### 6.3 `iloc[-1]`-without-sort — 2 weitere identifiziert (F-C2-R2-3, F-C2-R2-4)

Defensives Pattern fehlt noch in `ops/experience_log.py` und `intel/wild_card_detector.py`.

---

## 7. Was diese Audit nicht abgedeckt hat

- Keine Tests in CI gelaufen — nur lokal (Windows pytest).
- F-B-7/8/13-20 nicht einzeln verifiziert (Detail-Sektion fehlt im R1-Audit).
- F-A-6/7/10 nicht einzeln verifiziert (Detail-Sektion fehlt).
- Tiefere Sweeps (z. B. import-prefix 341 Dateien) bewusst nicht in R2-Scope.
- Performance-/Numerical-Stress-Tests nicht ausgeführt.
- Backtest-Real-Run mit Fixes nicht durchgeführt — Korrektheit nur statisch.

---

## 8. Empfehlung für nächste Schritte

**Sofort:**
- F-C2-R2-1 (qty-scaling Discard) — Design-Entscheidung zwischen Option A (no-qty-mut) und Option B (sync-back). Vor Live-/Pilot-Aktivität klären.

**Innerhalb 1 Woche:**
- B2-N1 EventStore.append_batch wrapping — 1-min Fix.
- §6.1 `date.today()`-Sweep für die 4 verbliebenen Sites.
- F-A-8 dead-tz_localize-Sweep in ledger.py.

**Mittelfristig (Backlog):**
- §6.2 Wiring der F-B-4/5/6/11 mit `as_of`-Pass-Through.
- F-C-3 Import-Prefix-Sweep (341 Dateien, größerer Aufwand).
- F-A2-1 HWM-on-Shorts vor trailing-stop-on-shorts.
- F-C2-R2-3/4 sort-Pflichten dokumentieren.

**Nicht in Scope (laut Audit-Architektur):**
- Backtest-Performance-Verifikation der Fixes (gehört in QA-Pipeline).
- Tests-Run in Ubuntu CI (gehört in CI-Workflow).

---

**Reviewer:** simulierter `senior-code-reviewer` (Opus 4.7) via `general-purpose` Subagent  
**Datum:** 2026-05-15  
**Audit-Dauer:** 3 parallele R2-Reviewer-Sessions, ~340k Tokens kumulativ + 2 inline Sweep-Fixes  
**Fix-Commits in R2:** d8d1034 (F-A2-4 + F-C2-R2-2)  
**Verifikationsstand:** statisch-analytisch, keine Backtests ausgeführt
