# Design — Paper-Pilot Alerting + Halt-Handling

Datum: 2026-06-22
Status: Phase 1 shipped 2026-06-22 (watchdog + Telegram-Alerts + shadow-Liquidation, Commits `26d9b930`→`b58988ce`); Phase 2 (echtes close_all_positions) weiterhin GATED. (Status aktualisiert 2026-07-23; war: APPROVED Design v2, Spec-Review ausstehend)
Autor: Fable (remote-control) + Operator
Scope-Typ: Ops/Monitoring (Phase 1, nicht-geschützt) + gated Auto-Liquidation (Phase 2, geschützt)

> **v2-Korrektur (wichtig):** Beim Code-Lesen festgestellt: es existiert **keine** "Positionen
> auf Cash verkaufen"-Primitive. `auto_flatten_on_stale` aktiviert nur den **Kill-Switch**
> (blockt neue Orders), verkauft NICHTS. Der Operator will echtes Auto-Liquidieren — **mit
> Telegram-Vorwarnung + Interventionsfenster davor**. Das berührt geschützte Pfade
> (`execution`/`paper`) und wird daher in Phase 2 isoliert, hinter eigener Freigabe + Review-Chain.

---

## 1. Problem (belegt, diese Session diagnostiziert)

Der Paper-Pilot hat am **2026-05-22** nach einem 1500s-Soft-Timeout (yfinance-429-Hang im
Prewarm) selbst eine **Halt-Flag** gesetzt (`output/ops/halt_ack_required.json`).
Danach lief er **jeden Handelstag weiter** (Exit 0), verband sich zum Broker — und
**verweigerte den Handel** (`rc=1, n_orders=0, "HALT FLAG present"` für 14+ Handelstage im
`output/pilot/pilot_manifest.json`). Die Paper-Equity driftete dabei **~100k → ~88k (−12 %)**
unmanaged. **Niemand wurde alarmiert**, weil (1) das Setzen der Halt-Flag keinen Alert auslöst,
(2) ein separater Alert-Worker nur JSON schreibt statt auszuliefern (Beob. 770), (3) der
DMS-Heartbeat-Monitor nicht im Task Scheduler läuft.

Ein System, das **stillschweigend anhält**, ist gefährlicher als eines, das gar nicht startet.

---

## 2. Bestehende Bausteine + die Lücke

| Baustein | Datei | Rolle | Status |
|---|---|---|---|
| `AlertManager.fire(rule, ctx)` | `ops/alerting.py` | echter Dispatcher (telegram/email/log_only, Cooldowns, Env-Creds, liest `configs/alerting.yaml`) | **liefert bereits** — wird aufgerufen |
| `auto_flatten_on_stale(policy, reason=…)` | `ops/dead_man_switch.py:107` | aktiviert **Kill-Switch** (`activate_kill_switch(throttle_pct=0)`) — blockt Orders, **verkauft NICHT** | aufrufbar; ≠ echtes Flatten |
| `get_positions()` / `cancel_all_orders()` | `execution/broker_adapter.py:75/226` | Positionen lesen, offene Orders canceln | vorhanden |
| **`close_all_positions()` / liquidate-to-cash** | — | **EXISTIERT NICHT** → muss in Phase 2 neu gebaut werden (geschützt) | **fehlt** |
| Halt-Flag schreiben / clearen | `scripts/run_live_paper.py`, `scripts/ack_halt.py` | setzt/entfernt `halt_ack_required.json` | Edit: +1 `fire()`-Call je |

---

## 3. Architektur — zwei Phasen

Ansatz B (Standalone-Watchdog orchestriert bestehende Primitive). Erwogen + verworfen:
(A) alles in den DMS-Daemon — schwerere Edits in risk-adjacenten Interna; (C) in den
`daily_paper_trading.bat`-Wrapper — läuft nur 1×/Tag, erkennt "Pilot läuft gar nicht mehr" nicht.

### PHASE 1 — Alerting + Watchdog + Vorwarnung (NICHT-geschützt, zuerst shippen)

Liefert den belegten Fix (3 Wochen unbemerkt) + das Interventionsfenster — ganz ohne
geschützte Edits. Die Auto-Liquidation läuft hier im **Shadow** (loggt "würde liquidieren",
sendet Telegram, verkauft NICHT).

**K1 — `configs/alerting.yaml`** (neu/erweitert; nicht-geschützt)
- Channel `telegram` für `critical`+`warning`; `log_only` immer als Fallback.
- Regeln (mit Cooldowns): `halt_flag_set`, `halt_cleared`, `liquidation_warning`,
  `liquidation_executed`, `heartbeat_stale`, `zero_orders_unexpected`, `drawdown_breach`.
- Creds aus `.env`: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (nie hardcoded/geloggt).
- Schwellen: `warn_after_trading_days` (default 1), `liquidate_after_warning_hours` (default 4),
  `heartbeat_stale_hours` (26), `zero_order_days` (2), `dd_breach_pct` (−8).

**K2 — `fire()` an der Halt-Quelle** (Edits in `scripts/`, review-chain, editierbar)
- `run_live_paper.py`: beim Schreiben der Halt-Flag → `fire("halt_flag_set", …)`.
- `ack_halt.py`: nach Clear → `fire("halt_cleared", …)`.

**K3 — `scripts/ops_watchdog.py`** (neu; nicht-geschützt) — idempotenter Einzel-Durchlauf,
per Task Scheduler alle ~15–30 min. Pro Tick:
1. **Halt-Eskalation (zweistufig):**
   - Flag neu vorhanden → `fire("halt_flag_set")`.
   - unacked **> `warn_after_trading_days`** → `fire("liquidation_warning")`
     ("Auto-Liquidation in <`liquidate_after_warning_hours`> h, sofort ack'en zum Abbrechen").
   - unacked **> warn + `liquidate_after_warning_hours`** → **Liquidation auslösen** (Phase 1:
     Shadow-Log; Phase 2: echt) + `fire("liquidation_executed")`.
   - **Ack jederzeit bricht ab** (Flag weg → keine Liquidation, `halt_cleared`).
2. **Heartbeat-Staleness** > `heartbeat_stale_hours` → `fire("heartbeat_stale")`
   (Auto-Flatten bei Heartbeat-Stale bleibt Sache des DMS-Daemons, K4).
3. **Run-Qualität:** letzter Manifest-Eintrag `rc≠0` oder ≥`zero_order_days` Werktags-Läufe mit
   `n_orders==0` → `fire("zero_orders_unexpected")`.
4. **Drawdown** unter `dd_breach_pct` → `fire("drawdown_breach")`.
- Watchdog-State (`last_seen_halt_ts`, `warning_sent_at`, `liquidation_done`, letzte Alarme) in
  `output/ops/watchdog_state.json` → "neu vs. schon-alarmiert", Eskalationsstufe, Einmaligkeit.

**K4 — DMS-Daemon in den Task Scheduler** (Ops-Schritt, kein Code) — `dms_daemon.py`
registrieren (war nie eingetragen). Arbeitsteilung: **DMS** = Heartbeat-stale → Kill-Switch;
**Watchdog** = Halt-Eskalation + Run-Qualität + Drawdown-Alarm + (Phase 2) Liquidation.

### PHASE 2 — Echte Auto-Liquidation (GESCHÜTZT, eigene Freigabe + Review-Chain)

**K5 — `close_all_positions()` Broker-Primitive** in `execution/broker_adapter.py` (GESCHÜTZT).
- Liest `get_positions()`, submitted für jede Long-Position eine schließende SELL-Order
  (market, day), cancelt vorher offene Orders via `cancel_all_orders()`. Long-only → nur SELL.
- Idempotent, logged jede Order, gibt Report zurück (geschlossen/fehlgeschlagen je Symbol).
- Eigener `flatten_mode`-Respekt: bei `shadow` nur "would close X@Y" loggen, kein Submit.
- **Geschützter Edit:** via Deny-Lift-Workflow (scoped, danach restauriert) + Review-Chain
  (`risk-execution-reviewer` → `senior-code-reviewer` → `task-completion-auditor`) + explizite
  Operator-Freigabe pro Datei. Wird ERST nach Phase-1-Verifikation gebaut.
- Watchdog K3-Schritt 1 ruft dann statt Shadow-Log `broker.close_all_positions()` auf.

---

## 4. Fehlerbehandlung / Safety

- **Fail-safe, nicht fail-open:** Watchdog-Fehler werden geloggt; eine Liquidations-Entscheidung
  wird nie still übersprungen. Erst Entscheidung berechnen, dann alarmieren.
- **Zweistufige Eskalation mit Interventionsfenster:** Vor jeder Liquidation geht eine Telegram-
  Warnung raus mit explizitem Zeitfenster + Ack-Anleitung. Nur wenn nach Ablauf STILL unacked →
  Liquidation. Das macht die irreversible Aktion menschlich abfangbar.
- **Einmaligkeit:** Nach ausgelöster Liquidation Markierung in `watchdog_state.json` → kein
  Re-Trigger pro Tick.
- **Shadow-first:** Phase 1 + initialer Phase-2-Rollout mit `flatten_mode: shadow` (loggt/alarmiert,
  verkauft nicht), bis Alert- und Liquidations-Pfad live verifiziert sind; dann bewusster Umstieg
  auf `market`.
- **Cooldowns** gegen Alarm-Spam. **Keine Secrets** in Logs/Alerts.

---

## 5. Testing

- **Watchdog-Logik (`tests/`):** synthetische State-Files je Fixture — halt-neu / unacked>warn
  (Warnung) / unacked>warn+window (Liquidation) / ack-bricht-ab / heartbeat-stale / 0-orders-streak
  / dd-breach / all-clear. Order-Invarianz + Liquidation/Warnung genau einmal.
- **AlertManager-Routing:** `fire()` mit `log_only` (kein echter Telegram-Call im Test).
- **Liquidation-Reuse:** Mock auf `broker.close_all_positions` / `auto_flatten_on_stale`, prüfen
  korrekter `reason` + `flatten_mode`; **kein echter Broker-Call** in Tests.
- **`close_all_positions` (Phase 2):** Unit gegen einen Fake-Broker (gemockte `get_positions` →
  erwartete SELL-Orders; shadow → keine Submits).
- **Smoke:** `scripts/drills/drill_halt_flag.py` → Watchdog-Ticks über die Eskalationsstufen.

---

## 6. Reversibilität / Blast-Radius

- **Phase 1:** 1 neues Skript + 1 Config + 2 Scheduler-Einträge + 2 `fire()`-Zeilen. **Null
  geschützte Edits.** Rückbau trivial.
- **Phase 2:** +1 Methode in `execution/broker_adapter.py` (geschützt, additiv) + Umstellung des
  Watchdog-Calls Shadow→real. Rückbau = Methode entfernen + Watchdog-Call auf Shadow.

---

## 7. Offene Punkte / YAGNI

- Kein eigener Channel-Code (Telegram/Email existiert in `alerting.py`).
- Toten JSON-only-Alert-Worker separat retiren (eigener Follow-up).
- 429-Ursache (Soft-Timeout) ist Thema #2 "Datenfetch entkoppeln" — separat.
- Phase 2 startet erst nach Phase-1-Live-Verifikation (Telegram-Pfad nachweislich liefert).
