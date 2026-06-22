# Design — Paper-Pilot Alerting + Halt-Handling

Datum: 2026-06-22
Status: APPROVED (Design), Spec-Review ausstehend
Autor: Fable (remote-control) + Operator
Scope-Typ: Ops/Monitoring — orchestriert bestehende Primitive, minimal-invasiv, reversibel

---

## 1. Problem (belegt, diese Session diagnostiziert)

Der Paper-Pilot hat am **2026-05-22** nach einem 1500s-Soft-Timeout (yfinance-429-Hang im
Prewarm) selbst eine **Halt-Flag** gesetzt (`output/ops/halt_ack_required.json`,
`reason: "soft-timeout fired … halted until operator acks via scripts/ack_halt.py"`).

Danach lief der Pilot **jeden Handelstag weiter** (zuletzt belegt: Exit 0), verband sich
zum Broker — und **verweigerte den Handel** (`rc=1, n_orders=0, "HALT FLAG present"` für 14+
Handelstage im `output/pilot/pilot_manifest.json`). In dieser Zeit driftete die Paper-Equity
**~100k → ~88k (−12 %)** unmanaged, weil gehaltene Positionen weiter schwankten und nicht
rebalanced/exited wurden.

**Niemand wurde alarmiert.** Ursachen:
1. Beim Setzen der Halt-Flag ruft **nichts** den `AlertManager` auf.
2. Ein separater autonomer Alert-Worker schreibt nur JSON, **liefert nie aus** (Beob. 770, 2026-06-02).
3. Der DMS-Heartbeat-Monitor (`dms_daemon.py`), der Staleness erkennen + flatten würde, ist
   **nicht im Task Scheduler** registriert → läuft nicht.

Fazit: Ein System, das **stillschweigend anhält**, ist gefährlicher als eines, das gar nicht
startet. Diese Spec schließt die Monitoring-/Alerting-Lücke.

---

## 2. Bestehende Bausteine (werden WIEDERVERWENDET, nicht neu gebaut)

| Baustein | Datei | Rolle | Status |
|---|---|---|---|
| `AlertManager.fire(rule, ctx)` | `src/assembled_core/ops/alerting.py` | echter Multi-Channel-Dispatcher (telegram/email/log_only, Cooldowns, Env-Creds, liest `configs/alerting.yaml`) | **liefert bereits** — wird nur aufgerufen |
| `auto_flatten_on_stale(cfg, reason=…)` | `src/assembled_core/ops/dead_man_switch.py:107` | generische Flatten-Primitive via Kill-Switch; `flatten_mode: market/shadow` | **aufrufbar von außen** — wird mit eigenem `reason` wiederverwendet |
| Halt-Flag schreiben | `scripts/run_live_paper.py` (`_arm_soft_timeout`) | setzt `halt_ack_required.json` | Edit: +1 `fire()`-Call |
| Halt-Flag clearen | `scripts/ack_halt.py` | entfernt die Flag | Edit: +1 `fire()`-Call (all-clear) |
| Pilot-State | `output/pilot/pilot_manifest.json`, `output/ops/scheduler_heartbeat.json`, `output/state/heartbeat.json`, `output/journal/trade_journal.jsonl` | Quellen für die Watchdog-Checks | read-only |

**Geschützte Pfade (`execution/risk/accounting/pipeline/paper/.github/workflows`): werden NICHT
editiert.** `auto_flatten_on_stale` liegt in `ops/` (nicht in der Deny-Liste) und wird nur
**aufgerufen**, nicht verändert.

---

## 3. Architektur (Ansatz B — Standalone-Watchdog orchestriert bestehende Primitive)

Erwogene Alternativen: (A) alles in den DMS-Daemon falten — verworfen, schwerere Edits in
risk-adjacenten Interna; (C) in den `daily_paper_trading.bat`-Wrapper — verworfen, läuft nur
1×/Tag und kann „Pilot läuft gar nicht mehr" nicht erkennen.

### 3.1 Komponenten

**K1 — Alert-Config `configs/alerting.yaml`** (neu oder erweitert; nicht-geschützt)
- Channel `telegram` für Severities `critical` + `warning`; `log_only` als Fallback immer aktiv.
- Regeln (mit Cooldowns): `halt_flag_set`, `halt_cleared`, `halt_unacked_grace_exceeded`,
  `heartbeat_stale`, `zero_orders_unexpected`, `drawdown_breach`.
- Creds aus `.env`: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (nie hardcoded, nie geloggt).

**K2 — `fire()` an der Halt-Quelle** (Edits in `scripts/`, review-chain-getriggert, editierbar)
- `run_live_paper.py`: beim Schreiben von `halt_ack_required.json` →
  `AlertManager().fire("halt_flag_set", {reason, ts, equity})`.
- `ack_halt.py`: nach erfolgreichem Clear → `fire("halt_cleared", {actor, ts})`.

**K3 — `scripts/ops_watchdog.py`** (neu; nicht-geschützt) — der fehlende Monitor.
Idempotenter Einzel-Durchlauf (kein Daemon-State), per Task Scheduler alle ~15–30 min.
Pro Tick:
1. **Halt-Flag-Check:** Flag vorhanden? Alter aus `ts_utc` berechnen.
   - frisch gesetzt (seit letztem Tick neu) → `fire("halt_flag_set")`.
   - vorhanden + unacked **> Grace (default 1 Handelstag)** → `auto_flatten_on_stale(cfg,
     reason="halt_unacked_grace_exceeded")` **+** `fire("halt_unacked_grace_exceeded")`.
2. **Heartbeat-Staleness:** `scheduler_heartbeat.json` / `heartbeat.json` älter als Schwelle
   (default 26 h, deckt 1 verpassten Tageslauf + Puffer) → `fire("heartbeat_stale")`.
   *(Auto-Flatten bei Heartbeat-Staleness bleibt Sache des DMS-Daemons — siehe K4.)*
3. **Run-Qualität:** letzter `pilot_manifest.json`-Eintrag `rc≠0` **oder** ≥N aufeinanderfolgende
   Werktags-Läufe mit `n_orders_detected==0` (default N=2) → `fire("zero_orders_unexpected")`.
4. **Drawdown:** aktuelle Equity vs. Pilot-Peak; Unterschreiten des Pilot-Hard-Stops
   (`max_drawdown_pct: -8%` aus dem Manifest) → `fire("drawdown_breach")`.
- Watchdog-eigener State (zuletzt gesehener Halt-ts, letzter Alarm) in
  `output/ops/watchdog_state.json`, damit „neu vs. schon-alarmiert" + Cooldown sauber sind.
- Alle Schwellen (`grace_trading_days`, `heartbeat_stale_hours`, `zero_order_days`,
  `dd_breach_pct`) in `configs/alerting.yaml` — nichts hardcoded.

**K4 — DMS-Daemon in den Task Scheduler** (Ops-Schritt, kein Code)
- `dms_daemon.py` als Scheduled Task registrieren (war nie eingetragen). Liefert die
  kontinuierliche Heartbeat-Staleness-Flatten-Absicherung, die K3 bewusst NICHT dupliziert.
- Dokumentierte Arbeitsteilung: **DMS** = Heartbeat-stale → flatten; **Watchdog** = Halt-Alarm
  + Halt-Grace-Flatten + Run-Qualität + Drawdown-Alarm.

### 3.2 Datenfluss

```
run_live_paper ──setzt──> halt_ack_required.json ──┐
ack_halt ──cleared──> (Flag weg)                   │
                                                   ▼
scheduler/heartbeat/pilot_manifest/journal ──> ops_watchdog.py (alle 15–30 min)
                                                   │  ├─ Bedingung erfüllt ─> AlertManager.fire() ─> Telegram
                                                   │  └─ Halt unacked > Grace ─> auto_flatten_on_stale() ─> Kill-Switch/Broker
dms_daemon.py (dauerhaft) ── Heartbeat stale ─> auto_flatten_on_stale()
```

---

## 4. Fehlerbehandlung / Safety

- **Fail-safe, nicht fail-open:** Watchdog-Fehler (z.B. korruptes JSON, Telegram down) werden
  geloggt und führen NIE zu einem stillen Skip einer Flatten-Entscheidung; der Flatten-Pfad
  hat Vorrang vor dem Alert-Pfad (erst flatten-entscheiden, dann alarmieren).
- **Grace-Flatten ist einmalig + markiert:** nach ausgelöstem Grace-Flatten wird der Zustand in
  `watchdog_state.json` markiert, damit nicht jeder Tick erneut flattet.
- **`flatten_mode`-Respekt:** der Watchdog ruft `auto_flatten_on_stale` mit dem in `policy.yaml`
  konfigurierten `flatten_mode`; bei `shadow` wird nur geloggt/alarmiert (kein Broker-Eingriff)
  — wichtig für einen sicheren ersten Rollout.
- **Cooldowns:** verhindern Alarm-Spam (z.B. heartbeat_stale nicht alle 15 min).
- **Keine Secrets in Logs/Alerts:** Token/Chat-ID nur aus `.env`, nie in Nachrichtentext.
- **Rollout-Sicherheit:** Watchdog startet mit `flatten_mode: shadow` (alarmiert, flattet nicht),
  bis der Alert-Pfad live verifiziert ist; dann Umstellung auf `market` als bewusster Schritt.

---

## 5. Testing

- **Unit (`tests/`):** Watchdog-Bedingungslogik mit synthetischen State-Files — je ein Fixture für
  halt-frisch / halt-grace-überschritten / heartbeat-stale / 0-orders-streak / dd-breach / all-clear.
  Order-Invarianz: gleiche Inputs → gleiche Entscheidung; Grace-Flatten genau einmal.
- **AlertManager-Routing:** `fire()` mit `log_only`-Channel testen (kein echter Telegram-Call im Test).
- **Flatten-Reuse:** Mock auf `auto_flatten_on_stale`, prüfen dass der Watchdog es mit korrektem
  `reason` + `flatten_mode` aufruft (kein echter Broker-Call).
- **Smoke:** `scripts/drills/drill_halt_flag.py` (existiert) → Watchdog-Tick → erwarteter Alarm im Log.
- **Kein** echter Telegram-/Broker-Call in der Test-Suite.

---

## 6. Reversibilität / Blast-Radius

Neu/geändert: 1 neues Skript (`ops_watchdog.py`) + 1 Config (`alerting.yaml`) + 1 Scheduler-Eintrag
(Watchdog) + 1 Scheduler-Eintrag (DMS) + 2 kleine `fire()`-Calls in `scripts/`. **Null Edits in
`execution/risk/accounting/pipeline/paper/workflows`.** Rückbau = Skript + Config + Tasks entfernen,
2 `fire()`-Zeilen revertieren.

---

## 7. Offene Punkte / bewusst ausgeklammert (YAGNI)

- Kein eigener Alert-Channel-Code (Telegram/Email existiert bereits in `alerting.py`).
- Kein Ersatz des JSON-only-Worker-Pfads in dieser Spec — der Watchdog nutzt direkt
  `AlertManager`; den toten JSON-Worker separat zu retiren ist ein eigener Follow-up.
- Keine Änderung der eigentlichen Pilot-Strategie oder des Soft-Timeout-Werts (separate Themen
  #2 „Datenfetch entkoppeln" adressiert die 429-Ursache).
