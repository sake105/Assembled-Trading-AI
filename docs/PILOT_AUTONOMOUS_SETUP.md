# Paper Pilot — Autonomous Setup

Diese Anleitung aktiviert den autonomen Daily-Mode des Paper-Pilot via Windows Task Scheduler.

---

## Architektur-Entscheidung: Task Scheduler statt Daemon

Option B (Windows Task Scheduler) wurde gegen Option A (Python-Daemon) gewählt:

| | Task Scheduler | Python Daemon |
|---|---|---|
| Reboot-Survival | ✅ läuft weiter | ❌ stoppt |
| Single Point of Failure | ✅ keine | ❌ Prozess-Crash → Pilot tot |
| Sleep/Wake | ✅ kann Rechner wecken | ❌ pausiert |
| User Logout | ✅ irrelevant | ❌ Session-gebunden |
| 30-Tage-Pilot-Survival | ✅ OS-native | ❌ ein Crash und Lauf bricht ab |

Daily-Batch-Workloads (1 Cycle/Tag) sind exakt der Anwendungsfall, für den Task Scheduler gemacht wurde.

---

## Setup (einmalig, ca. 2 Minuten)

### Schritt 1: PowerShell als Administrator öffnen

- Win+X → "Terminal (Administrator)" oder "Windows PowerShell (Admin)"
- Oder: Rechtsklick auf PowerShell-Verknüpfung → "Als Administrator ausführen"

### Schritt 2: Task registrieren

```powershell
cd F:\Python_Projekt\Aktiengerüst
.\scripts\ops\register_paper_pilot_task.ps1
```

Erwartete Ausgabe:
```
[OK] Registered task 'AssembledTradingAI-PaperPilot'.

Verify with:
  Get-ScheduledTask -TaskName 'AssembledTradingAI-PaperPilot' | Format-List
```

### Schritt 3: Verifizieren

```powershell
Get-ScheduledTask -TaskName 'AssembledTradingAI-PaperPilot' | Format-List
```

Suche nach `State: Ready` und `NextRunTime: <morgen 21:30>` (wenn heute > 21:30, sonst heute 21:30).

### Schritt 4: Test-Run jetzt (optional)

```powershell
Start-ScheduledTask -TaskName 'AssembledTradingAI-PaperPilot'
```

Status nach 1-2 Min prüfen:
```powershell
Get-ScheduledTaskInfo -TaskName 'AssembledTradingAI-PaperPilot'
```
`LastTaskResult: 0` = erfolgreich.

Log:
```powershell
Get-ChildItem logs\scheduler\daily_paper_trading_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content
```

---

## Was der Task tut

Jeden Werktag (Mo-Fr) um **21:30 Lokalzeit** (= 15:30 ET, 30 Min vor NYSE-Close):

1. `scripts/ops/prewarm_price_cache.py --years 2`
   → fetcht fehlende Watchlist-Symbole von yfinance (idempotent — wenn Cache komplett, no-op)
2. `scripts/run_paper_pilot.py --run-day`
   → führt 1 Trading-Cycle aus, updated Pilot-Manifest, position_sync, intent_store

Beide Schritte loggen nach `logs/scheduler/daily_paper_trading_YYYYMMDD.log`.

---

## Monitoring

### Pilot-Status prüfen

```powershell
.venv\Scripts\python.exe scripts\run_paper_pilot.py --status
```

Zeigt z. B. `Day 7/30 | Crashes: 0 | Verdict: pending`.

### Letzten Lauf prüfen

```powershell
Get-ScheduledTaskInfo -TaskName 'AssembledTradingAI-PaperPilot'
```

Wichtige Felder:
- `LastRunTime` — wann zuletzt gelaufen
- `LastTaskResult` — 0 = ok, sonst Fehler
- `NextRunTime` — wann der nächste Lauf geplant ist

### Logs lesen

```powershell
# Letzter Scheduler-Log
Get-ChildItem logs\scheduler\daily_paper_trading_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 50

# Letzter Trading-Cycle-Log
Get-ChildItem logs\live_paper_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 30
```

### Equity-Curve

```powershell
.venv\Scripts\python.exe -c "import json; s=json.load(open('output/runs/_paper_ledger/ledger_state.json')); print('\n'.join(f\"{e['utc'][:10]}: \${e['equity']:,.2f}\" for e in s['equity_curve']))"
```

---

## Task pausieren / wieder aktivieren

```powershell
# Pausieren (z. B. fürs Wochenende-Wartung):
Disable-ScheduledTask -TaskName 'AssembledTradingAI-PaperPilot'

# Wieder aktivieren:
Enable-ScheduledTask -TaskName 'AssembledTradingAI-PaperPilot'
```

## Task entfernen

```powershell
.\scripts\ops\register_paper_pilot_task.ps1 -Unregister
```

Oder:
```powershell
Unregister-ScheduledTask -TaskName 'AssembledTradingAI-PaperPilot' -Confirm:$false
```

---

## Bekannte Constraints / Caveats

1. **Logon-Type Interactive:** Aktueller Setup läuft nur wenn der User-Account interaktiv eingeloggt ist (oder bei Login auto-startet). Für "läuft auch ohne Login": Task editieren, "Run whether user is logged on or not" + Passwort einmal eingeben. Sicherer aber komplexer.

2. **Sommer-/Winterzeit (DST):** Die `21:30 lokal` Zeit folgt CEST/CET automatisch, aber NYSE wechselt zu DST 2 Wochen früher. Im März und November gibt es ~2 Wochen wo `21:30 CET = 16:30 ET` statt 15:30 ET. Folge: Lauf 1h näher am NYSE-Close. Akzeptabel.

3. **Holidays:** NYSE schließt an US-Feiertagen. Der Task läuft trotzdem, der Pilot-Cycle wird bei stale prices oder zero-fills im Log warnen, aber kein Crash. Optional: Holiday-Liste in den Pilot einbauen.

4. **Concurrent runs:** Pilot hat keinen expliziten Lock. Falls Task während eines laufenden Cycles erneut feuert (sollte nicht passieren, da 24h Abstand), könnte es zu Doppel-Submissions kommen. Idempotency via `client_order_id` (F-A3-4) sollte das aber abfangen.

5. **Crash-Recovery:** Wenn der Cycle crasht (rc!=0), wird der Pilot-Manifest die Run als "crashed" markieren. Bei 2+ Crashes in den 30 Tagen scheitert das GO-Kriterium. Manueller Eingriff bei wiederholten Crashes empfohlen.

6. **Speicher / Disk:** Logs in `logs/scheduler/` und `logs/live_paper_*.log` wachsen. Etwa 1-2 MB pro Run. Über 30 Tage = ~60 MB. Periodisch aufräumen oder Rotation via `logging.handlers.RotatingFileHandler` ist schon eingerichtet (live_paper-logs nutzen das).

---

## Wenn etwas schiefgeht

**Task läuft nicht zur geplanten Zeit:**
```powershell
Get-ScheduledTaskInfo -TaskName 'AssembledTradingAI-PaperPilot'
```
Check `LastRunTime` und `LastTaskResult`. Wenn `LastTaskResult = 0x41303` oder ähnlich: Sleep-State hat verhindert, dass Task lief. Setting `WakeToRun` sollte das verhindern — wenn nicht, BIOS-Power-Settings prüfen.

**Cycle crasht:**
1. Letzten Trading-Cycle-Log lesen (`logs/live_paper_*.log`)
2. Pilot manuell ausführen für Diagnose:
   ```powershell
   .venv\Scripts\python.exe scripts\run_paper_pilot.py --run-day
   ```
3. Bei wiederholten Crashes: Task disable bis Bug gefixt.

**Pilot-Manifest meldet "Verdict: failure":**
Per `pilot_manifest.json` Failure-Kriterien: CAGR ≤ 5%, Sharpe ≤ 0.5, MDD ≤ -15%, oder 2+ Crash-Days. Bei Auslösung: 14-Tage-Offline + Strategy-Review (per Manifest-Specs).

---

**Setup-Status:** PowerShell-Skript bereit unter `scripts/ops/register_paper_pilot_task.ps1`. Einmalige Admin-Registrierung erforderlich (siehe Schritt 2 oben).
