# 06c — Risk Controls + Order Execution + Order Lifecycle (Deep Audit, Round 2)

- **Datum:** 2026-05-30
- **Cluster:** RISK CONTROLS + ORDER EXECUTION + ORDER LIFECYCLE (safety-critical core)
- **Modus:** READ-ONLY. Einziger Schreibpfad: diese Datei. Keine `src/`-Edits.
- **Module (gelesen):**
  - `src/assembled_core/pipeline/_tc_risk.py` (374 Z., voll)
  - `src/assembled_core/pipeline/_tc_execution.py` (699 Z., voll)
  - `src/assembled_core/pipeline/trading_cycle_v2.py` (Order-Flow 532–790)
  - `src/assembled_core/pipeline/trading_cycle_shared.py` (`_apply_risk_controls_default` 1340–1558)
  - `src/assembled_core/execution/kill_switch.py` (544 Z., voll)
  - `src/assembled_core/execution/pre_trade_checks.py` (1261 Z., voll)
  - `src/assembled_core/execution/risk_controls.py` (387 Z., voll)
  - `src/assembled_core/execution/order_lifecycle.py` (284 Z., voll)
  - `src/assembled_core/execution/fat_finger_guard.py` (151 Z., voll)
  - `src/assembled_core/execution/order_gate.py`, `stale_order_guard.py`, `pdt_counter.py`, `risk/pdt_counter.py`
  - `src/assembled_core/execution/unified_paper_engine.py` (Parallel-Guard-Chain 1440–1538)
  - `src/assembled_core/ops/dead_man_switch.py` (100–328)
  - `scripts/run_live_paper.py` (`_preflight_checks` 338–382, Aufruf 566)
  - `src/assembled_core/ops/paper_runner.py` (1259)

**Round-1-Findings, die hier NICHT wiederholt werden:** M-2 (`_tc_risk.py:101-102` bare except auf Pivot), M-6 (`_tc_execution.py:99,122` verworfene meta-dicts), C2 Kill-Switch-Deactivate-Token-Gating. Diese gelten als bekannt.

---

## Produktions-Order-Pfad (autoritativ etabliert)

```
scripts/run_live_paper.py:_preflight_checks (KS-Gate)
  → run_paper_daily_one
    → paper_runner.py:1259  run_trading_cycle(ctx)
      → trading_cycle_v2.py:run_trading_cycle
          route_orders        (_tc_execution.py)
          check_risk          (_tc_risk.py:22)   ← Step 6 ruft _apply_risk_controls_default (KS-wired)
                                                    + Step 6.35–6.9 (VaR/auto-DD/CB/fat-finger/lifecycle)
          book_fills          (_tc_execution.py:226)
```

**`UnifiedPaperEngine` ist NICHT der Produktionspfad.** Es besitzt eine eigene, reichhaltigere
Guard-Chain (CB → KS → symbol-kill → fat-finger MIT history → pre-trade fail-closed, 1440–1538),
wird aber von `run_paper_daily_one`/`run_trading_cycle` nicht aufgerufen. Produktion läuft über die
schwächere `check_risk`-Kette. **Das ist eine §50-Doppelstruktur** (zwei parallele Order-Guard-Chains,
eine davon tot im Live-Pfad). Siehe NEW-C-1.

**Gate-Ordering:** Alle harten Gates (`check_risk` Step 6–6.9) laufen VOR `book_fills`. Keine
harte Schutzprüfung läuft NACH Order-Emission. Gate-Reihenfolge ist korrekt.

---

## Tabelle 1 — Protective-Checks-Status

| Check | Modul:Zeile | wired? | Bypass-Risiko | Beleg |
|---|---|---|---|---|
| Single-Flag-Total-Bypass | `_tc_risk.py:64-67` | WIRED+ACTIVE | **HOCH** — `enable_risk_controls=False` → `orders_filtered = orders.copy(); return` überspringt ALLE Steps | `if not getattr(ctx,"enable_risk_controls",True): result.orders_filtered = orders.copy(); return result` |
| QA-Gate | `_tc_risk.py:70-78` | WIRED+ACTIVE | niedrig (fail-closed: leert orders) | `if ctx.qa_block_trading: result.orders = pd.DataFrame(...); return` |
| Policy-Load | `_tc_risk.py:52-61` | WIRED-but-bypassable | **HOCH** — Load-Fehler → `policy = {}` → alle policy-gated Guards default-OFF | `except: log.warning(...); policy = {}` |
| EVT-Tail-VaR | `_tc_risk.py:104-145` | WIRED+ACTIVE | mittel (`except` → no-op, default fail-open) | try/except um evt_var |
| Risk-Controls-Default (KS + limits) | `_tc_risk.py:201-207` → `trading_cycle_shared.py:1340` | WIRED+ACTIVE | mittel — siehe Kill-Switch-Zeile; bei Raise: `status="error"`, `orders_filtered` bleibt empty-default → **fail-closed** | `result.orders_filtered = _apply_risk_controls_default(ctx, orders)` |
| Kill-Switch (prod) | `trading_cycle_shared.py:1529-1543` (`enable_kill_switch=ctx.enable_risk_controls`) | WIRED+ACTIVE | **HOCH** — an dasselbe Single-Flag gekoppelt wie pre-trade; `enable_risk_controls=False` deaktiviert KS-Check im Cycle mit | `enable_kill_switch=ctx.enable_risk_controls` |
| Kill-Switch (preflight) | `run_live_paper.py:356-358` | WIRED+ACTIVE | niedrig (fail-closed: `return False`) | `if is_kill_switch_engaged(): return False` |
| HALT-Flag-Gate | `run_live_paper.py:348-353` | WIRED+ACTIVE | niedrig (fail-closed) | `if HALT_FLAG_PATH.exists(): return False` |
| Drawdown-KS (preflight) | `run_live_paper.py:360-381` | WIRED-but-bypassable | **mittel** — `except: log.warning(...)` ohne `return False` → **fail-open**, defekter `get_account()` lässt preflight passieren | `except Exception as exc: logger.warning(...)` (kein return) |
| Pre-Trade-Checks (prod) | `trading_cycle_shared.py:1534` | WIRED+ACTIVE | **HOCH** — `enable_pre_trade_checks=ctx.enable_risk_controls` (Single-Flag) | `enable_pre_trade_checks=ctx.enable_risk_controls` |
| `_apply_risk_controls_default` outer-except | `trading_cycle_shared.py:1546-1557` | WIRED+ACTIVE | niedrig — **fail-CLOSED** (`return empty`, `log.critical`) | `return pd.DataFrame(columns=list(orders.columns))` |
| VaR-Gate | `_tc_risk.py:213-222` | WIRED-but-bypassable | **HOCH** — `except → "gate no-op"` = **fail-open** | `except: log.warning("...gate no-op..."); meta["var_gate"]={"status":"error"}` |
| Auto-DD-Kill-Switch | `_tc_risk.py:224-243` | WIRED-but-bypassable | **HOCH** — `except → "gate no-op"` = **fail-open** | `except: log.warning("...auto_dd...gate no-op...")` |
| Circuit-Breaker | `_tc_risk.py:245-254` | WIRED-but-bypassable | **HOCH** — `except → "gate no-op"` = **fail-open** | `except: log.warning("...circuit_breaker...gate no-op...")` |
| Fat-Finger notional cap | `_tc_risk.py:296-314` | WIRED-but-bypassable | **HOCH** — (a) nur aktiv wenn `policy.fat_finger_guard.enabled`; (b) `except → "hard cap not applied"` fail-open | `apply_fat_finger_guard_from_policy(result.orders_filtered, policy)` |
| Fat-Finger qty-multiple cap | `_tc_risk.py:303-305` + `fat_finger_guard.py:100` | **DEFINED-but-UNWIRED (de-facto)** | **HOCH** — Aufruf OHNE `history_qty_by_symbol` → `history={}` → qty-Multiple-Branch nie betreten → Cap TOT | `apply_fat_finger_guard_from_policy(...)` ohne history-Arg; `if max_qty_multiple is not None and history:` |
| Anti-Churn / min-notional | `_tc_risk.py:256-293` | WIRED+ACTIVE | niedrig (kein Safety-Gate; `except → debug skip`) | `except: log.debug("anti_churn filters skipped")` |
| Order-Lifecycle in-memory tracker | `_tc_risk.py:317-338` | WIRED, aber **non-persistent** | niedrig — `OrderLifecycleTracker()` pro Zyklus neu, dann verworfen; kein State über Zyklen | `_olt = OrderLifecycleTracker()` (lokal, nie gespeichert) |
| Order-Lifecycle JSONL-Log | `_tc_risk.py:339-369` | WIRED+ACTIVE | niedrig (`except → debug`); schreibt SUBMITTED-Event je Order | `append_lifecycle_event("SUBMITTED", ...)` |
| Neg-qty → abs() | `_tc_execution.py:128-138` | WIRED+ACTIVE | **mittel** — `abs()` statt Reject; side vom `side`-Feld entkoppelt; WARNING maskiert Konventions-Drift | `orders["qty"] = orders["qty"].abs()` (nur WARNING) |
| Kill-Switch-Read in route_orders | `_tc_execution.py:633-643` | WIRED, aber NUR Prometheus-Gauge | n/a — **kein Gate**, nur Telemetrie | `is_kill_switch_engaged()` → gauge.set(...) |
| `book_fills` orders_filtered None-Fallback | `_tc_execution.py:262-268` | WIRED+ACTIVE | **mittel (latent)** — bei `orders_filtered is None` → bucht `result.orders` (un-risk-checked); im v2-Pfad nicht erreichbar (default=empty DF, nie None gesetzt), aber **fail-open by design** | `if result.orders_filtered is None: result.orders_filtered = result.orders.copy()` |
| `pre_trade_gate` (Gawande) | `pre_trade_checks.py:1091-1160` | **DEFINED-but-UNWIRED** | n/a — fail-closed, aber kein Aufruf aus trading_cycle_v2/_tc_* | grep: keine Prod-Callsite |
| OrderGate (PDT + RoundTrip) | `execution/order_gate.py` | **DEFINED-but-UNWIRED** | n/a — 0 Prod-Caller (nur Tests) | grep `OrderGate`: nur order_gate.py + tests |
| `cancel_stale_orders` | `execution/stale_order_guard.py` | **DEFINED-but-UNWIRED** | n/a — nur Self-Ref + tests | grep: nur stale_order_guard.py + tests |
| PDTCounter / get_pdt_counter | `risk/pdt_counter.py`, `execution/pdt_counter.py` | **DEFINED-but-UNWIRED** + **Doppelstruktur** | n/a — 0 Prod-Caller; ZWEI Module gleichen Namens | grep: 2 Module + tests, keine Prod-Callsite |
| Dead-Man-Switch (`dms_monitor_loop`) | `ops/dead_man_switch.py:100-328` | **DEFINED-but-UNWIRED (prod)** | **HOCH** — auto-flat bei Heartbeat-Staleness; Caller nur `scripts/dms_daemon.py`; Daemon NICHT im Task Scheduler (Memory 86468b0c) → läuft nicht | grep: nur dms_daemon.py + tests |
| UnifiedPaperEngine guard-chain | `unified_paper_engine.py:1440-1538` | **WIRED-but-UNUSED (prod)** | n/a — Parallelpfad, nicht in run_trading_cycle | paper_runner.py:1259 ruft run_trading_cycle, nicht UPE |
| `run_pre_trade_checks(orders)` (UPE) | `unified_paper_engine.py:1523` | WIRED (in UPE), near-no-op | mittel — Aufruf ohne config → meiste Checks inert | `run_pre_trade_checks(orders)` ohne config |
| Backtest-KS backup/restore | `trading_cycle_v2.py:544-555, 765-788` | WIRED+ACTIVE | niedrig — restore fail-closed (`PermissionError` → KS bleibt engaged, log.critical) | `deactivate_kill_switch(..., operator_token=os.environ.get("OPERATOR_KILL_TOKEN"))` |

---

## Tabelle 2 — Silent-Except-Inventar (Risk/Execution-Pfad)

| Modul:Zeile | fail-open / closed | maskiert was |
|---|---|---|
| `_tc_risk.py:56-61` | **fail-OPEN** | Policy-Load-Fehler → `policy={}` → alle policy-gated Guards (fat-finger, anti-churn, VaR-config) default-OFF |
| `_tc_risk.py:101-102` (R1 M-2) | fail-open (degradiert) | Pivot/returns-Berechnung → `_shared_rets=None` → EVT-VaR übersprungen |
| `_tc_risk.py:115-116` | fail-open | `evt_var()` Fehler → `_evt_var_99=None` → EVT-Gate inert |
| `_tc_risk.py:194-195` | fail-open | barbell_strategy-Fehler → kein Sizing-Tilt (kein Safety-Gate) |
| `_tc_risk.py:204-207` | **fail-CLOSED** | `_apply_risk_controls_default` Raise → `status="error"`, `orders_filtered` bleibt empty-default → nichts gebucht |
| `_tc_risk.py:220-222` | **fail-OPEN** | VaR-Gate-Eval-Fehler → Gate no-op, Orders passieren |
| `_tc_risk.py:241-243` | **fail-OPEN** | Auto-DD-KS-Fehler → Gate no-op, kein KS-Trigger, Orders passieren |
| `_tc_risk.py:252-254` | **fail-OPEN** | Circuit-Breaker-Fehler → Gate no-op, Orders passieren |
| `_tc_risk.py:292-293` | fail-open (debug) | Anti-Churn-Filter übersprungen (kein Safety-Gate) |
| `_tc_risk.py:313-314` | **fail-OPEN** | Fat-Finger-Raise → "hard cap not applied", unbegrenztes Notional passiert |
| `_tc_risk.py:368-369` | fail-open (debug) | Lifecycle-Log-Hook übersprungen (nur Audit-Trail-Verlust) |
| `_tc_risk.py:370-371` | fail-open (debug) | Lifecycle-Tracking übersprungen (nur Telemetrie) |
| `_tc_execution.py:125-126` | fail-open (debug) | group_exposures-Caps übersprungen → **Exposure-Cap kann still ausfallen** |
| `_tc_execution.py:99,122` (R1 M-6) | fail-open | verworfene meta-dicts |
| `trading_cycle_shared.py:1526-1527` | fail-open (degradiert) | policy-Load → `_cycle_policy=None` (an risk_controls weitergereicht) |
| `trading_cycle_shared.py:1546-1557` | **fail-CLOSED** | Risk-Controls-Raise → `return empty`, `log.critical`, alle Orders geblockt |
| `run_live_paper.py:380-381` | **fail-OPEN** | Drawdown-Check-Fehler → WARNING, kein `return False` → preflight passiert |
| `kill_switch.py:_read_state` (R1-Nähe) | **fail-OPEN** | State-Read-Fehler → `{}` → `engaged=False` (loggt error) |

---

## NEW-Findings (Round 2, mit Severity)

| ID | Finding | file:line | Mechanismus | Severity | Blast Radius |
|---|---|---|---|---|---|
| NEW-C-1 | **Doppelstruktur Order-Guard-Chain (§50).** Produktion nutzt schwache `check_risk`-Kette; reichere `UnifiedPaperEngine`-Chain (mit fat-finger+history, symbol-kill, pre-trade fail-closed) ist im Live-Pfad tot. Divergenz: UPE blockt, Prod-Pfad nicht. | `unified_paper_engine.py:1440-1538` vs `_tc_risk.py:201-314`; Beleg `paper_runner.py:1259` | run_trading_cycle ≠ UPE; zwei Wahrheiten für „pre-trade guard" | **HOCH** | Live + OOS |
| NEW-C-2 | **3 harte Gates fail-OPEN bei Exception.** VaR-Gate, Auto-DD-Kill-Switch, Circuit-Breaker werden bei JEDER inneren Exception zu no-ops („gate no-op") und lassen Orders durch. Ein Bug/Schema-Drift in der Eval-Logik deaktiviert das Gate still. | `_tc_risk.py:220-222, 241-243, 252-254` | `except: log.warning(...no-op...)`; kein orders_filtered-Clear | **HOCH** | Live + OOS |
| NEW-C-3 | **Fat-Finger qty-Multiple-Cap de-facto tot.** `apply_fat_finger_guard_from_policy` wird OHNE `history_qty_by_symbol` aufgerufen → `history={}` → `if max_qty_multiple is not None and history:` nie wahr. Nur das absolute Notional-Cap wirkt; die mengen-relative Fat-Finger-Erkennung ist konfigurierbar, aber wirkungslos. | `_tc_risk.py:303-305`; `fat_finger_guard.py:100-114` | history-Arg default `None` → `history={}` falsy | **HOCH** | Live + OOS |
| NEW-C-4 | **Single-Flag deaktiviert KS + Pre-Trade + alle Gates gemeinsam.** `enable_risk_controls=False` ⇒ (a) `_tc_risk.py:64-67` früher Pass-Through aller Steps; (b) `enable_kill_switch=enable_pre_trade_checks=ctx.enable_risk_controls` ⇒ KS-Check UND Pre-Trade-Check im selben Zug aus. Ein einziges Flag entfernt die gesamte Schutzschicht. | `_tc_risk.py:64-67`; `trading_cycle_shared.py:1534-1535` | gemeinsame Flag-Quelle für unabhängige Schutzschichten | **HOCH** | Live + OOS |
| NEW-C-5 | **DMS nicht im Prod-Pfad.** Dead-Man-Switch (auto-flat bei Heartbeat-Staleness) existiert + getestet, aber nur via `scripts/dms_daemon.py` aufrufbar, der laut Memory (86468b0c) NICHT im Task Scheduler registriert ist. Bei hängendem Pilot-Prozess kein automatisches Flatten. | `ops/dead_man_switch.py:100-328`; Caller nur `scripts/dms_daemon.py` | Daemon nicht deployed → Guard inert | **HOCH** | Live |
| NEW-H-1 | **Drawdown-Preflight fail-OPEN.** `_preflight_checks` Drawdown-Block fängt jede Exception, loggt WARNING und kehrt NICHT mit `return False` zurück → ein defekter `adapter.get_account()` lässt den Run trotz potenziellem DD-Breach starten. | `run_live_paper.py:380-381` | `except: warning` ohne `return False` | **HOCH** | Live |
| NEW-H-2 | **Policy-Load-Fehler entwaffnet alle policy-gated Guards.** `policy={}` (Z.61) ⇒ fat_finger `enabled=False`, anti-churn aus, VaR-config leer. Eine korrupte/fehlende policy.yaml degradiert still zu „keine policy-Gates" statt zu blockieren. | `_tc_risk.py:56-61` | fail-open default-dict | **HOCH** | Live + OOS |
| NEW-H-3 | **group_exposure-Cap silent-skip.** `_apply_group_exposure_caps` in route_orders unter `except: log.debug(...skipped)` → Sektor/Region/FX-Gruppen-Caps können still ausfallen, Orders mit Überexposure passieren ohne sichtbares Signal. | `_tc_execution.py:121-126` | `except: log.debug` (debug-level, unsichtbar in prod-log) | **HOCH** | Live + OOS |
| NEW-H-4 | **Neg-qty silent abs() statt Reject.** Negative qty wird zu `abs()` korrigiert (nur WARNING), side bleibt vom `side`-Feld bestimmt. Driftet die Sizing-Konvention je zu signed-qty-shorts ohne `side='SELL'`, wird ein Short still zu BUY-Magnitude — Richtungs-Korruption ohne Abbruch. | `_tc_execution.py:128-138` | `abs()` entkoppelt qty-Vorzeichen von side; kein Reject | **HOCH** | Live + OOS |
| NEW-H-5 | **`book_fills` None-Fallback bucht un-risk-checked orders.** Bei `orders_filtered is None` bucht `book_fills` `result.orders` (Pre-Filter). Im v2-Pfad latent (default=empty DF), aber jeder alternative Caller, der None durchreicht, umgeht die gesamte Risk-Filterung am Booking-Layer. | `_tc_execution.py:262-268` | `is None` → `result.orders.copy()` fail-open-Fallback | **MITTEL (latent)** | Live + OOS |
| NEW-M-1 | **PDTCounter-Doppelstruktur.** Zwei Module `risk/pdt_counter.py` und `execution/pdt_counter.py`, beide ohne Prod-Caller. Zweite Wahrheit für PDT-Zählung, beide tot. | `risk/pdt_counter.py`, `execution/pdt_counter.py` | parallele Implementierungen, 0 Callsites | **MITTEL** | neither (unwired) |
| NEW-M-2 | **OrderGate (PDT+RoundTrip) + cancel_stale_orders unwired.** Definierte, getestete Schutz-Guards ohne Produktions-Callsite. PDT-Verstöße / stale Orders werden im Live-Pfad nicht geprüft/gecancelt. | `execution/order_gate.py`, `execution/stale_order_guard.py` | 0 Prod-Caller (nur Tests) | **MITTEL** | neither (unwired), Live-Lücke |
| NEW-M-3 | **Order-Lifecycle-Tracker non-persistent.** `OrderLifecycleTracker()` pro Zyklus neu erzeugt, Transitions VALIDATED→SUBMITTED gefahren, dann verworfen. Keine zyklusübergreifende State-Machine; `find_stuck_orders`/`find_open_orders` über Tracker ist sinnlos (nur JSONL-Log persistiert separat). | `_tc_risk.py:324-338` | lokale Instanz, nie gespeichert | **MITTEL** | neither (Telemetrie) |

---

## State-Machine-Integrität (order_lifecycle.py)

- `transition()` RAISED `ValueError` bei illegaler Transition (`_VALID_TRANSITIONS`) — **fail-closed**, korrekt.
- ABER: in `_tc_risk.py` wird der Tracker pro Zyklus neu erzeugt und verworfen (NEW-M-3). Es gibt keine
  persistente Lifecycle-State-Machine über Zyklen. `find_stuck_orders` (250-283) operiert auf einer
  In-Memory-Instanz, die nach dem Zyklus weg ist → kann keine zyklusübergreifend hängenden Orders finden.
- Die einzige persistente Lifecycle-Spur ist `order_lifecycle.jsonl` via `append_lifecycle_event`
  (`_tc_risk.py:355` SUBMITTED; FILLED-Hook laut Memory in `_tc_execution`/`unified_paper_engine`).
  Das ist ein Append-Log, KEINE State-Machine mit Transition-Validierung.

## Order-Generierung (Korrektheit)

- **qty-Vorzeichen:** Negative qty → `abs()` (NEW-H-4), kein Reject. Side aus separater `side`-Spalte.
- **Zero-qty:** Im gelesenen Pfad kein explizit dedizierter zero-qty-Reject in `check_risk`; Fat-Finger
  prüft nur obere Schranken. Zero/near-zero-qty-Orders würden bis book_fills durchlaufen (kein Beleg
  für unteren Guard im check_risk-Pfad gefunden — **UNSURE**, Sizing-Stufe nicht in diesem Cluster).
- **Idempotenz / stale-qty-reuse:** OrderLifecycleTracker non-persistent ⇒ keine zyklusübergreifende
  Dedup/Idempotenz über die Lifecycle-Schicht. order_id wird in `append_lifecycle_event` aus
  `order_id`-Spalte ODER synthetisch `f"{sym}_{side}_{run_id}"` gebildet (`_tc_risk.py:357-358`) —
  bei fehlender order_id kollidieren mehrere Orders gleichen Symbols/Side im selben run zu EINER ID.

## Verifikations-Status

- **Keine Tests ausgeführt** (READ-ONLY-Audit). Aussagen basieren auf statischer Code-Lektüre +
  grep-Callsite-Analyse, NICHT auf Laufzeit-/CI-Belegen.
- Memory-Belege (86468b0c DMS-Wiring-Status) als sekundäre Bestätigung herangezogen, nicht
  unabhängig per CI verifiziert.
