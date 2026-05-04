# Was noch zu tun ist, bevor du sicher in Paper-Trading gehst

**Datum:** 2026-05-03
**Geprüfter Stand:** `2883f64` (HEAD on main)
**Frage:** "Was wäre noch zu tun, bevor wir sicher in Paper-Trading gehen können?"

---

## TL;DR

**Du bist viel weiter als ich beim letzten Audit dachte.** Du hast bereits:

- `scripts/run_live_paper.py` (552 LOC) — vollständiger Paper-Live-Runner
- `scripts/paper_trading_scheduler.py` — Autonomer Scheduler
- `daily_paper_trading.bat` — Windows-Task-Scheduler-Wrapper
- `docs/PAPER_TRACK_PLAYBOOK.md` (723 LOC) + `PAPER_TRACK_QUICKSTART.md`
- `AlpacaAdapter` mit allen abstract-methods, `alpaca-py 0.38.0` installiert
- `broker_execution.py` (463 LOC), `intent_store.py`, `heartbeat.py`, `liveness_check.py`
- `configs/paper_track/` mit 4 Strategie-Configs + Watchlist (30 US-Large-Caps)
- `reconciliation.py` (399 LOC) — Position-Sync mit Broker

Was du **WIRKLICH** noch brauchst sind **6 konkrete Vor-Flug-Checks**, **kein** Code-Schreiben.

**Gesamt:** ~4-6h echte Arbeit + 2-3 Tage Beobachtung = du bist sicher in Paper-Trading.

---

## Die 6 Pre-Flight-Checks (in Reihenfolge)

### Check 1: Alpaca-Paper-Account einrichten + `.env` (~30 min)

**Warum kritisch:** Ohne Credentials geht gar nichts. Aber: kein Code dafür im Repo (gewollt — Secrets niemals committen).

**Schritte:**

1. **Account anlegen:** https://app.alpaca.markets/signup → Paper-Trading-Tab
   - Free, sofort verfügbar
   - $100k virtueller Account (entspricht deinem Backtest-Capital)

2. **API-Keys generieren:** Dashboard → "Generate New Key"
   - Notiere `Key ID` und `Secret`
   - Wichtig: **Paper**-Keys (nicht Live!) — URL muss `paper-api.alpaca.markets` sein

3. **`.env`-Datei lokal anlegen** (NICHT committen):

```bash
# F:\Python_Projekt\Aktiengerüst\.env
ALPACA_API_KEY=PK...your-paper-key
ALPACA_API_SECRET=...your-paper-secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Optional: andere Daten-APIs
POLYGON_API_KEY=optional
NEWSAPI_KEY=optional
```

4. **Smoke-Test:**

```powershell
# Im Repo-Root
python -c "
from src.assembled_core.execution.broker_adapter import create_adapter_from_env
from dotenv import load_dotenv
load_dotenv()

adapter = create_adapter_from_env(force_paper=True)
account = adapter.get_account()
print(f'✓ Connected to Alpaca Paper')
print(f'  Equity: \${account[\"equity\"]:,.2f}')
print(f'  Buying Power: \${account[\"buying_power\"]:,.2f}')
print(f'  Status: {account[\"status\"]}')

positions = adapter.get_positions()
print(f'  Open Positions: {len(positions)}')
"
```

**Erfolg:** Du siehst dein $100k Paper-Equity. **Misserfolg:** API-Key-Fehler, URL-Fehler, oder rate-limit.

---

### Check 2: Dry-Run einer einzelnen Cycle (~30 min)

**Warum kritisch:** Bevor irgendwas live submitted wird, gegen die echte Pipeline laufen — aber ohne Order-Submission.

**Schritte:**

1. **Eine Strategie wählen** — start mit `trend_baseline` (einfachste, keine ML-Layer):

```powershell
python scripts\run_live_paper.py --once --dry-run `
    --config configs\paper_track\trend_baseline.yaml
```

2. **Was du erwartest zu sehen:**

```
[run_live_paper] starting cycle (dry-run, mode=paper)
[run_live_paper] cache age: 1.2h (fresh)
[run_live_paper] computing features for 30 symbols...
[run_live_paper] generating signals via trend_baseline...
[run_live_paper] 8 signals positive, 2 negative
[run_live_paper] DRY RUN: would submit 5 orders:
  AAPL  buy   25 @ $187.50  ($4,687.50)
  MSFT  buy   12 @ $385.00  ($4,620.00)
  ...
[run_live_paper] DRY RUN: no orders submitted
[run_live_paper] cycle complete in 47.3s
```

3. **Was bei Problemen schief gehen kann:**
   - **Daten-Pipeline-Fehler** (yfinance rate-limit, NaN in Features) → fix vorher
   - **Adapter-Fehler** (API-Token falsch) → Check 1 nicht abgeschlossen
   - **Signal-Pipeline-Crash** (mode-handling-bug) → 1-2h Debug

4. **Wichtige Validierung:** schau dir die generierten Orders **manuell** an. Sind die Symbole sinnvoll? Größenordnung passt zu deinem Backtest?

**Erfolg:** Dry-Run produziert plausible Orders ohne Crash. **Misserfolg:** Stopp und debuggen — das ist genau der Wert von Dry-Run.

---

### Check 3: Halt-Mechanismus + Recovery testen (~30 min)

**Warum kritisch:** Wenn etwas schief geht und du den Stecker ziehen musst, **muss das funktionieren**. Ein nicht-getesteter Kill-Switch ist schlimmer als kein Kill-Switch (false sense of security).

Du hast bereits `HALT_FLAG_PATH = ROOT / "output" / "ops" / "halt_ack_required.json"` in `run_live_paper.py:46`.

**Schritte:**

1. **Halt-File anlegen** (manuell):

```powershell
mkdir -Force output\ops
echo '{"reason": "test_drill", "actor": "manual", "ts": "2026-05-03T10:00:00Z"}' > output\ops\halt_ack_required.json
```

2. **Cycle starten:**

```powershell
python scripts\run_live_paper.py --once --dry-run
```

**Erwartung:** Das Skript bricht **sofort** ab mit einer klaren Fehlermeldung wie:
```
[run_live_paper] HALT FLAG ACTIVE — cycle skipped
[run_live_paper] reason: test_drill
[run_live_paper] To resume: delete output/ops/halt_ack_required.json
```

3. **Halt-File löschen + nochmal:**

```powershell
Remove-Item output\ops\halt_ack_required.json
python scripts\run_live_paper.py --once --dry-run
```

Cycle läuft normal durch.

4. **Kill-Switch separater Test:**

```powershell
python -c "
from src.assembled_core.execution.kill_switch import activate_kill_switch, is_kill_switch_engaged
activate_kill_switch(throttle_pct=0.0, reason='test_drill', actor='manual')
print(f'Kill-Switch engaged: {is_kill_switch_engaged()}')
"

# Now run cycle — must refuse orders
python scripts\run_live_paper.py --once

# Restore
python -c "
from src.assembled_core.execution.kill_switch import deactivate_kill_switch
deactivate_kill_switch(reason='drill_done', actor='manual')
"
```

**Erfolg:** Beide Mechanismen blockieren Trading. Recovery funktioniert. **Misserfolg:** **Kein Live-Trading bis das gefixt ist.**

---

### Check 4: Reconciliation + Heartbeat-Liveness (~1h)

**Warum kritisch:** Wenn das System hängt oder abstürzt, willst du das **innerhalb von Minuten** wissen, nicht erst am nächsten Tag.

**Heartbeat-Test:**

1. **Cycle laufen lassen (live, nicht dry-run!):**

```powershell
python scripts\run_live_paper.py --once
```

2. **Heartbeat prüfen:**

```powershell
python scripts\liveness_check.py --max-age 1800 --json
```

Erwartung:
```json
{
  "alive": true,
  "last_heartbeat": "2026-05-03T15:42:18+00:00",
  "age_seconds": 47,
  "status": "ok"
}
```

3. **Stale-Heartbeat-Test:**

```powershell
# Wait 31+ minutes, then re-check
# OR: manually edit heartbeat to old timestamp
python scripts\liveness_check.py --max-age 1800
echo "Exit code: $LASTEXITCODE"  # should be 1 (NOT alive)
```

**Reconciliation-Test:**

1. **Im Paper-Account manuell etwas tun** (z.B. 10 AAPL kaufen über Alpaca-Web-UI)
2. **Reconcile-Only-Cycle starten:**

```powershell
python scripts\run_live_paper.py --reconcile-only
```

3. **Erwartung:** Logs zeigen "Found 1 broker position not in ledger: AAPL × 10"
   - Wenn deine Pipeline die Position als "fremder Trade" trackt: **gut**.
   - Wenn sie die Position als ihre eigene übernimmt: **gefährlich** — solche manuellen Trades dürfen nicht in den Strategy-State leaken

**Erfolg:** Beides funktioniert. **Misserfolg:** Heartbeat-Stale-Detection oder Reconciliation-Discrepancy → 1-3h Debug vor Paper-Live.

---

### Check 5: Daten-Frische-Check + Holiday-Calendar (~1h)

**Warum kritisch:** US-Feiertage, Halbtage (Black Friday, Christmas Eve), und yfinance-Lücken sind die häufigsten "warum hat mein Bot nichts getan" Probleme.

**Schritte:**

1. **Welcher Daten-Pfad wird genutzt?**

`run_live_paper.py` hat eine 3-Stufen-Logik:
1. Lokaler Parquet-Cache (`data/raw/equities_eod/yfinance/*.parquet`)
2. yfinance-Live (wenn Cache ≥ 3 Tage alt)
3. Failover auf alten Cache (wenn yfinance fehlschlägt)

Frage: **wie aktuell ist dein Cache?**

```powershell
python -c "
import pandas as pd
from pathlib import Path

files = list(Path('data/raw/equities_eod/yfinance').glob('*.parquet'))
if not files:
    print('❌ NO CACHE FILES — yfinance must succeed every cycle')
else:
    sample = pd.read_parquet(files[0])
    latest = pd.Timestamp(sample.index.max() if 'timestamp' not in sample.columns else sample['timestamp'].max())
    age_days = (pd.Timestamp.now(tz='UTC') - latest).days
    print(f'✓ {len(files)} cache files')
    print(f'  Latest: {latest}')
    print(f'  Age: {age_days} days')
    if age_days > 3:
        print('  ⚠️  Stale — refresh with: python scripts/update_prices.py --days 10')
"
```

2. **Holiday-Calendar-Check:**

US-Feiertage 2026 (NYSE):
- Memorial Day: Mon May 25, 2026
- Independence Day: Fri Jul 3, 2026 (observed)
- Labor Day: Mon Sep 7, 2026
- Thanksgiving: Thu Nov 26 + Half-Day Fri Nov 27, 2026
- Christmas: Half-Day Thu Dec 24 + Closed Fri Dec 25, 2026

Frage: **handelt die Pipeline am 25. Mai zwischen 13:30-20:00 UTC?** Sie sollte **nicht**, weil Markt geschlossen.

```powershell
# Test: simuliere Memorial Day
python -c "
import pandas as pd
from src.assembled_core.utils.market_calendar import is_trading_day  # check ob existiert
print(is_trading_day(pd.Timestamp('2026-05-25')))  # should be False
print(is_trading_day(pd.Timestamp('2026-05-26')))  # should be True
"
```

3. **Wenn `is_trading_day` nicht existiert:** im Scheduler entweder
   - `pandas_market_calendars` library installieren + benutzen, oder
   - manuelle Holiday-Liste in `configs/holidays.txt`

**Erfolg:** Cache ist <3 Tage alt UND Holiday-Calendar wird respektiert. **Misserfolg:** 1-2h fixing.

---

### Check 6: Risiko-Limits explizit setzen (~30 min)

**Warum kritisch:** $100k Paper-Account verleitet dazu, "voll zu fahren". Aber du willst **erst beobachten**, dann erst skalieren. Ohne harte Limits in der `.yaml` wirst du in Tag 1 schon 30 Symbole shortselen mit Vol-Targeting auf 100% — das ist nicht das Ziel.

**Schritte:**

1. **`configs/paper_track/trend_baseline_live.yaml` editieren:**

```yaml
# configs/paper_track/trend_baseline_live.yaml — ADD/MODIFY
risk_limits:
  # PILOT-PHASE LIMITS (Wochen 1-4):
  max_total_exposure_pct: 0.30      # max 30% deployed (statt voller 100%)
  max_position_pct: 0.05            # max 5% pro Symbol
  max_daily_orders: 10              # max 10 orders/Tag
  max_daily_loss_pct: -0.02         # -2% Tagesverlust → kill-switch
  
  # SAFE-DEFAULTS für Paper-Pilot:
  forbidden_symbols:                 # zu volatil für Pilot
    - GME
    - AMC
    - TSLA  # erstmal raus, später wieder rein
  
  # Order-Type-Restrictions:
  allowed_order_types: [limit]       # keine market-orders im Pilot
  default_limit_offset_bps: 5        # 5bps inside spread

# AUTO-HALT-RULES (zusätzlich zu Kill-Switch):
auto_halt:
  on_consecutive_losing_days: 3       # 3 Verlust-Tage → review
  on_drawdown_pct: -0.05              # -5% MDD → halt + review
  on_position_count_anomaly: 50       # >50 Positionen = bug
```

2. **Validierung:** Cycle laufen lassen mit `--dry-run`, schauen dass die Limits auch wirken:

```powershell
python scripts\run_live_paper.py --once --dry-run --config configs\paper_track\trend_baseline_live.yaml
```

In den Logs muss stehen sowas wie:
```
[risk] applied position cap 5%: AAPL trimmed from 8.2% to 5.0%
[risk] total exposure 28.4% (limit 30.0%)
[risk] 6 orders generated (daily cap: 10)
```

**Erfolg:** Limits greifen, Logs zeigen das klar. **Misserfolg:** Wenn der Cycle 23 Orders generiert oder 60% Exposure: Limits werden ignoriert → 1h Debug.

---

## Die 4 "Run-und-Beobachte"-Schritte (3 Tage)

Nach Pre-Flight-Checks 1-6 (4-6h Arbeit), kommt **Beobachtung**. Das ist Echtzeit-Arbeit, kein Code.

### Tag 1: Manueller Single-Cycle Live (heute oder morgen)

**Ziel:** Erster echter Trade im Paper-Account.

**Wann:** US-Markt offen, am besten 14:00-19:00 UTC (ein Werktag).

**Was tun:**

```powershell
# 1. Daten frisch
python scripts\update_prices.py --days 10

# 2. Cycle live (KEIN dry-run mehr!)
python scripts\run_live_paper.py --once --config configs\paper_track\trend_baseline_live.yaml
```

**Was du beobachtest:**

- Logs vom Cycle (sollte 30-90 Sekunden dauern)
- Alpaca Web-UI: erscheinen die Orders dort?
- Alpaca Web-UI: werden sie gefüllt?
- `output/paper_live/<date>/` — Cycle-Summary-File

**Wenn alles gut läuft:** 3-5 Trades im Paper-Account, alle gefüllt, Positionen sichtbar. Schließen vor 21:00 UTC oder Über-Nacht-Halten.

**Wenn Probleme auftauchen:** Heute fixen, Tag 1 morgen wiederholen.

---

### Tag 2-3: 2 Tage Daily-Cycle mit Scheduler

```powershell
# Im Hintergrund laufen lassen
python scripts\paper_trading_scheduler.py --hour 19 --minute 30  # 19:30 UTC = 14:30 ET (1.5h vor close)
```

Oder als Windows-Task-Scheduler-Job mit `daily_paper_trading.bat`.

**Was du beobachtest täglich:**

1. **Hat der Cycle gelaufen?** (Heartbeat-Check vor Schlafengehen)
2. **Wieviele Orders?** (Sollte 0-10 pro Tag sein, nicht 50)
3. **Wieviele Fills?** (Sollte ≥95% Fill-Rate sein, ohne große Slippage)
4. **P&L?** (Klein, vielleicht ±$200 — egal welche Richtung; Volatilität ist das eigentliche Signal)
5. **Reconciliation?** (Broker-Position = Ledger-Position? Wenn nein: Bug)
6. **Logs auf Warnings?** (Oft "ratemibit" oder "stale data" Warnungen — fix oder akzeptieren)

**Was nicht:** Kein Stress-Trading wenn ein Tag rot ist. Pilot ist Beobachtung, nicht Performance.

---

### Tag 4-7: Erste Wochen-Review

Nach 4-7 Tagen Daily-Cycle hast du **5-25 Trades**. Genug für erste Beobachtung:

**Run:**

```powershell
python scripts\generate_daily_qa_report.py --since 7d --out reports\paper_pilot_week1.html
```

**Was prüfen:**

- **Sharpe (annualisiert) vs. Backtest** — sollte bei 50-70% des Backtest-Wertes sein. Wenn deutlich niedriger: Sim-to-Real-Gap zu groß
- **Avg-Slippage:** In bps. Bei US-Large-Caps und Limit-Orders: typisch 2-5bps. Wenn ≥15bps: Order-Pricing-Bug
- **Fill-Rate:** ≥95% wenn limit-orders nahe an Quote sind
- **Order-Reasoning:** schau dir 5 Beispiel-Trades an mit `python scripts\explain_trade.py --order-id <id>` (wenn dieses Skript existiert; sonst manuell aus Logs)

---

## Was ist NICHT nötig vor Paper-Trading

Wichtig zu wissen, was du **NICHT** brauchst:

- ❌ **Multi-Strategy-Ensemble** (Phase 1.1 aus dem 11/10-Plan) — pure `trend_baseline` ist OK für Paper
- ❌ **News-Features im Meta-Model** — alle ML-Layer disabled, ist OK
- ❌ **Conformal-Sizing aktiviert** — kann später dazukommen
- ❌ **Stress-Tests gegen historische Krisen** — wichtig für Live, nicht für Paper
- ❌ **Pager-Duty-Alerting** — Email-only ist OK für Paper
- ❌ **Trade-Reasoning-Logs** — schön zu haben, nicht kritisch
- ❌ **30-Tage-Pilot mit go/no-go-Skript** — das kommt nach den ersten 7 Tagen

Paper-Trading ist **kein** "Mini-Live-Trading". Es ist **Validierung**:
- Funktioniert die Pipeline 24/7?
- Stimmen die Annahmen über Slippage?
- Triggern die Risk-Gates korrekt?
- Halt-Mechanismen funktionieren?

Die Antwort auf alle 4 ist nicht aus mehr Code zu bekommen — sondern aus **Beobachtung**.

---

## Konkrete Aufwands-Schätzung

| Schritt | Aufwand | Charakter |
|---|---|---|
| Check 1: Alpaca + .env | 30 min | One-time setup |
| Check 2: Dry-Run | 30 min | Verification |
| Check 3: Halt + Recovery | 30 min | Drill |
| Check 4: Heartbeat + Reconciliation | 1h | Verification |
| Check 5: Daten + Holiday-Calendar | 1h | Verification + maybe fix |
| Check 6: Risk-Limits in YAML | 30 min | Configuration |
| **Pre-Flight Total** | **4-6h** | **vor erstem Live-Cycle** |
| Tag 1: Erster Single-Cycle | 1h | Beobachtung |
| Tag 2-3: Scheduler läuft | 30 min/Tag | Beobachtung |
| Tag 4-7: Wochen-Review | 2h | Analyse |
| **Beobachtung Total** | **~5h verteilt über 7 Tage** | |

---

## Mein konkreter Vorschlag

**Mache morgen Vormittag** Check 1 + Check 2 (1h). Nimm dir Zeit, in Ruhe.

**Mache morgen Nachmittag** während US-Markt offen ist Check 3 + Check 4 (1.5h). Da kannst du auch direkt Tag 1 (erster echter Cycle) anhängen.

**Übermorgen** Check 5 + Check 6 (1.5h), dann Scheduler scharf schalten.

**3 Tage später** erstes Wochen-Review machen, dann **entscheiden**:
- ✅ Wenn alles glatt: 30-Tage-Pilot starten mit `paper_trading_scheduler.py`
- ⚠ Wenn Probleme: fixen, dann nochmal 7 Tage beobachten

Wenn du jetzt anfängst, kannst du **innerhalb einer Woche** im Paper-Pilot sein — und in **5 Wochen** bereit für Live-Mikro-Capital.

Soll ich konkret ein Pre-Flight-Check-Skript schreiben, das die ersten 4 Checks automatisch durchläuft und dir am Ende eine Go/No-Go-Auswertung gibt? Wäre ~150 LOC und macht den Validation-Prozess reproduzierbar.
