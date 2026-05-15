# Senior-Code-Reviewer System-Audit Round 4 — 2026-05-15

**Auftrag:** Verifikation der R3-MAJOR-Fixes + Deep-Dive in noch nicht auditierte Bereiche (state_machine, garch_vol, georisk_overlay, attribution/, intel/) + Cumulative-Integrity-Check über alle 22 Fix-Commits.

**Status:** Round 4 abgeschlossen via 3 parallele Reviewer (Opus simulation).

---

## 1. Executive Summary

**Verifikation R3-Fixes:** Alle 3 R3-MAJORs (F-A3-1, F-A3-2, F-A3-4) verifiziert **CLOSED**. Composability über alle 22 Fix-Commits (R1+R2+R3) ist **PASS** — Short-Trade-End-to-End-Flow funktioniert konsistent.

**Aber Round 4 hat 2 HIGH + 1 MAJOR Findings in bisher nicht auditierten Bereichen aufgedeckt:**

| ID | Severity | Was |
|---|---|---|
| **B4-IN-08** | **HIGH** | `intel/news_rag.py:215` ruft `Qdrant.recreate_collection()` im `__init__` → **historischer News-Korpus wird bei jedem Prozess-Restart gelöscht.** Docstring verspricht "RAG retrieval over historical events" — Realität: data-loss bei jedem Restart. |
| **B4-GR-02** | **HIGH** | `risk/georisk_overlay.py:161-180` qty-mode Downscale: scaling reduziert Shares, aber Cash-Side wird **nicht** komplementär hochskaliert. Gross-Exposure-Loss silent. |
| **F-C4-N-1** | **MAJOR** | 4 Production-Sites laden Models via rohes `joblib.load` statt `safe_load_model` (das in `ml/model_registry.py:118` Hash-Verifikation bietet). Deserialization-Attack-Surface in user-writable Pfaden. |

Plus **5 MEDIUM** (state_machine corrupted-file fallback, garch_vol numerische edges, attribution sqlite concurrency, intel news_dedupe persistence-gap, **process-gap: 10/12 Fix-Commits ohne Regression-Tests**) und **30+ LOW/INFO**.

### Pre-Pilot Gate Status

**CONDITIONAL GO.** Must-fix vor Pilot:
1. **B4-IN-08** Qdrant recreate_collection → use get_or_create pattern
2. **F-C4-N-1** Raw joblib.load → safe_load_model an 4 Sites
3. **B4-GR-02** Georisk-Overlay qty-mode: dokumentieren oder symmetrisch fixen
4. Regression-Tests für F-A2-4, F-A3-1, F-A3-2, F-A3-4 (process gap)

---

## 2. Verifikation R3-Fixes

| ID | Fix-Commit | Verdict | Evidence |
|---|---|---|---|
| **F-A3-1** unified_paper_engine fail-closed | `880cb38` | **CLOSED** | except branch produces `orders.iloc[0:0].copy()` (empty df with same columns). Downstream-kompatibel (post-validation block handles empty). ERROR-log mit exc_info=True. Normal-success path unverändert. |
| **F-A3-2** pre_trade_checks recompute | `2af9227` | **CLOSED** | `_recompute_exposures()` Closure aufgerufen nach sector loop (L426) und nach region loop (L448). FX-Check ist currency-basiert, keine exposures_df-Konsumption. Edge-case: empty filtered_orders nach Sector-Scaling wird korrekt gehandelt. |
| **F-A3-4** broker client_order_id | `f3a8b5d` | **CLOSED** | `_auto_client_order_id()` Helper nutzt existing `idempotency.build_client_order_id(signal_id=f"auto-{utc_day}", intent_hash=...)`. Wired in MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest + legacy fallback. 24-char output (well within 48-char Alpaca limit). |

**Minor concerns aus Verifikation (alle LOW/INFO):**
- ABC `BrokerAdapter` zeigt nicht den neuen `client_order_id` kwarg → Programmer-against-ABC kann nicht explizit ID übergeben.
- `submit_moc_order` / `submit_loc_order` Delegations können kein explicit coid weiterreichen.
- `order_management.py:46` nutzt andere Idempotenz-Strategie (signal_id direkt ohne day-bucket) — divergent aber non-blocking.

### Composability E2E

Reviewer hat den Short-Trade-Flow durchverfolgt: signal → portfolio → risk_controls → API → engine → broker → fills → ledger. **Alle Handoffs erhalten Information.** F-A-1 (paper_ledger) und F-A2-4 (unified_paper_engine) haben identische 9-Branch-Short-Logic. Cash-Invariant (BUY -=, SELL +=) preserved. **PASS.**

---

## 3. Neue HIGH-Findings aus R4

### 3.1 B4-IN-08 — Qdrant `recreate_collection` löscht Korpus bei Restart

**Datei:** `src/assembled_core/intel/news_rag.py:215-221`
**Kategorie:** data-loss on restart

**Evidence:**

`NewsRAG.__init__` ruft `self._qdrant_client.recreate_collection(...)`. Per Qdrant-Doku **droppt `recreate_collection` die existierende Collection.** Jeder Restart eines Prozesses, der NewsRAG mit `qdrant_host` instanziiert → **WIPED den historischen News+Outcome-Korpus.**

Docstring verspricht "RAG retrieval over historical events" — Realität: jedes neue Process-Start beginnt mit leerer Datenbank.

**Suggested Fix:** `get_collection` versuchen, `create_collection` nur wenn missing. Oder separate `.bootstrap()`-Method, die explizit opt-in ist.

**Severity:** HIGH (Daten-Verlust + Doku-Reality-Mismatch).

### 3.2 B4-GR-02 — Georisk-Overlay qty-mode: Silent Gross-Exposure-Loss

**Datei:** `src/assembled_core/risk/georisk_overlay.py:161-180`
**Kategorie:** silent weight loss / exposure semantics

**Evidence:**

`apply_exposure_multiplier_to_targets` Downscaling-Pfad: Wenn KEINE CASH-Row im target_positions vorhanden ist UND multiplier < 1.0, geht der freigesetzte Weight lautlos verloren (nur log.warning). `target_qty` wird skaliert, aber cash side NICHT — qty-basierte Caller haben **keine Cash-Absorption**.

**Konsequenz:** In qty-only Flows reduziert Downscale die Shares, aber Ledger zeigt die Gap als untracked → P&L-Inkonsistenz, Risk-Reduktion ist scheinbar wirksam aber gross-exposure-Verschwendung silent.

**Suggested Fix:**
- (a) Dokumentieren dass qty-mode REQUIRES CASH-Row, raise/error bei missing.
- (b) Synthetische CASH-Row in qty-mode skalieren mit current cash position.
- Mindestens: `log.warning` → `log.error` bei qty-mode downscale ohne cash.

**Severity:** HIGH (Risk-Overlay-Semantik bricht silent).

### 3.3 F-C4-N-1 — Raw joblib.load bypasst safe_load_model an 4 Sites

**Dateien:**
- `src/assembled_core/signals/meta_model.py:546`
- `src/assembled_core/strategies/multifactor_v2.py:1608`
- `src/assembled_core/ml/regime_hmm.py:284, 535`
- `src/assembled_core/intel/conviction_engine.py:274`

**Kategorie:** security / deserialization attack surface

**Evidence:**

`ml/model_registry.py:118` exponiert `safe_load_model(strict=False)` mit SHA256-Hash-Verifikation gegen registry. Die 4 Production-Loading-Sites umgehen es und rufen rohes `joblib.load` direkt auf. Models liegen unter `output/models/` (user-writable). **Deserialization-Attack-Surface** falls Pfad kompromittiert (pickle/joblib.load executiert arbiträren Code beim Loaden).

**Suggested Fix:** Alle 4 Sites auf `safe_load_model(model_path, strict=False)` umstellen. Verifikation gegen Registry; bei fehlender Registry warnt + lädt (strict=True würde refuse).

**Severity:** MAJOR (Rule 20 Secrets/Security; pickle ist explicit-attack-vector wenn Pfad user-writable).

---

## 4. MEDIUM-Findings aus R4 (Auswahl)

### 4.1 B4-SM-02 — state_machine: Corrupt-File-Fallback ignoriert .bak

**Datei:** `risk/state_machine.py:86-104`

`load_risk_state()` fällt auf `_default_record(_now_utc_str())` zurück wenn File missing/unreadable/non-dict. Es existiert ein `.bak`-File (geschrieben bei save), aber **load konsultiert es nicht** auf Corruption. Restart nach Crash mid-write → silent DEMOTE auf WATCH.

**Fix:** Bei unreadable primary: try `.bak`, dann default. WARN-log on bak-fallback.

### 4.2 B4-AT-01 — attribution/storage SQLite ohne WAL/timeout

**Datei:** `attribution/storage.py:28, 48, 83`

`AttributionStore` öffnet fresh `sqlite3.connect()` pro save/load, kein WAL, kein timeout. Concurrent writes (paper_runner + research process) collide mit "database is locked" → no retry.

**Fix:** `conn.execute("PRAGMA journal_mode=WAL")` + `timeout=5.0`.

### 4.3 B4-IN-03 — news_dedupe.NewsDedupeIndex thread-unsafe

**Datei:** `intel/news_dedupe.py:196-203`

`filter_new()` mutiert internal state (OrderedDicts) während Iteration. Kein Lock. Zwei concurrent Caller racen.

### 4.4 B4-X-01 — Persistence-Divergenz across subsystems

5 Subsysteme owners persisted state mit unterschiedlichen Garantien:
- `risk/state_machine`: atomic + .bak (best)
- `intel/news_dedupe`: atomic, no .bak
- `intel/news_archive`: text-append + flush + fsync, no .bak, no header-fsync
- `intel/news_trade_attribution`: tmp+replace, bare-except cleanup
- `attribution/storage`: sqlite default (no WAL, no timeout)

**Empfehlung:** Konsolidieren auf `src.assembled_core.utils.atomic_io` + ein sqlite-Helper mit WAL+timeout-Defaults.

### 4.5 F-C4-N-5 — Process-Gap: 10/12 Fix-Commits ohne Regression-Tests

| Commit | Fix | Regression-Test? |
|---|---|---|
| e6d1461 | F-A-1 paper_ledger | **Ja** (6 neue Tests) |
| 8267234 | F-B-1/2/3 mfv2 | **Ja** (6 neue Tests) |
| 1484264 | F-B-4/5/6 FRED | Nein |
| 353583e | F-A-2/9/B-12/C-4 | Nein |
| e4dcb7b | F-A-3/4/C-10 | Nein |
| 6b4a660 | F-B-9/10/11/C-1/C-2 | Nein |
| d8d1034 | **F-A2-4 (BLOCKER-grade!)** + F-C2-R2-2 | **Nein** ← Lücke |
| 20d9897 | F-C2-R2-1 | Nein |
| fc5f482 | date.today() sweep | Nein |
| 5be9ba4 | B2-N1 EventStore | Nein |
| 880cb38 | F-A3-1 fail-closed | Nein |
| 2af9227 | F-A3-2 recompute | Nein |
| f3a8b5d | F-A3-4 idempotency | Nein |

**Risiko:** Future-Refactor öffnet stille die Fixes erneut. **F-A2-4 ist BLOCKER-grade ohne Test.**

**Empfehlung:** Mindestens für F-A2-4 + F-A3-1/2/4 Regression-Tests nachziehen.

---

## 5. Cumulative-Risk-Picture nach R4

| Kategorie | R1 | R2 | R3 | R4 | Status |
|---|---|---|---|---|---|
| BLOCKER aufgedeckt | 4 | 1 (F-A2-4) | 0 | 0 | **5 total** |
| BLOCKER closed | 0 | 5 | 5 | 5 | **5/5 ✅** |
| MAJOR aufgedeckt | 14 | 2 (F-C2-R2-1/2) | 3 (F-A3-1/2/4) | 1 (F-C4-N-1) | **20 total** |
| MAJOR closed | 0 | 16 | 19 | 19 | **19/20** (F-C4-N-1 offen) |
| HIGH (R4-new) | — | — | — | 2 | **2 offen** (B4-IN-08, B4-GR-02) |
| MEDIUM offen | — | — | — | 5+ | F-C4-N-5 (Tests), B4-SM-02, B4-AT-01, etc. |
| LOW/INFO offen | — | — | — | ~30 | Backlog |

**Pre-Pilot Gate:** **CONDITIONAL GO** mit Must-Fix:
- F-C4-N-1 (joblib.load → safe_load_model an 4 Sites)
- B4-IN-08 (Qdrant recreate_collection)
- B4-GR-02 (georisk qty-mode Cash-Absorb)
- Regression-Tests für F-A2-4 + F-A3-1/2/4

---

## 6. Was R4 NICHT abgedeckt hat

- Keine Backtest-Runs.
- `execution/paper`, `pipeline`, `OMS`, `kill-switch` re-audit nicht im R4-Scope.
- 5 broad-except in `intel/news_ingest.py` und 3 in `intel/conviction_engine.py` gescannt aber nicht voll gelesen.
- `event_signal.py` und `news_validation.py` (im R3-Scope erwähnt) **existieren nicht** — Scope-Liste war stale.
- Property-based/Fuzz-Testing der Cash-Invariant nicht vorhanden.
- Alpaca-py SDK Feldname `client_order_id` nicht gegen aktuelle SDK-Source verifiziert (existing audit caveat).

---

## 7. Empfehlung für Round 5 / Pilot-Vorbereitung

**Vor Pilot (in dieser Session machbar):**
- F-C4-N-1 (4 Sites swap zu safe_load_model)
- B4-IN-08 (1-line change: recreate_collection → get_or_create-Pattern)
- B4-GR-02 (Documentation + Error-Log Eskalation)
- Regression-Test für F-A2-4

**Round 5 Scope (nach Pre-Pilot-Fixes):**
- Verifikation der oben genannten
- Test-Suite-Run (CI + Lokal)
- Backtest mit allen kumulativen Fixes
- `execution/paper` + `pipeline` re-audit (Rule 30 sensitive zones; R4 hat sie aus dem Scope gelassen)

**Out-of-Scope für jetzt:**
- F-C-3 Import-Prefix-Sweep (181 Dateien — Policy-Entscheidung)
- Hexagonal-Doku-Korrektur

---

**Reviewer:** simulierter `senior-code-reviewer` (Opus 4.7) via `general-purpose` Subagent
**Datum:** 2026-05-15
**Audit-Dauer:** 3 parallele R4-Reviewer-Sessions, ~395k Tokens kumulativ
**Verifikationsstand:** statisch-analytisch
