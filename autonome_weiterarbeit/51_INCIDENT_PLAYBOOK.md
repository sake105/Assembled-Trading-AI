# 51 — Incident-Playbook und Disaster-Recovery

**Zweck:** Runbooks für die wahrscheinlichsten Ausfälle. Wenn du nachts um 3 Uhr aufwachst, weil Sentry eine Kritikal-Mail geschickt hat, sollst du **nicht nachdenken müssen** — sondern nur der Checkliste folgen.

**Regel:** Jeder Runbook-Eintrag muss ohne Zugriff auf dein Gedächtnis ausführbar sein. Auch wenn du krank bist, müde, oder 4 Monate nicht am System warst.

---

## Struktur jedes Runbooks

```
Titel
├─ Symptom (wie sieht es aus?)
├─ Schweregrad (SEV1 bis SEV4)
├─ Detection (wie wird es erkannt?)
├─ Sofort-Maßnahme (erste 5 Minuten)
├─ Diagnose (was ist die Ursache?)
├─ Behebung
├─ Post-Incident (Dokumentation, Prävention)
```

**Schweregrade:**

| Level | Bedeutung | Reaktionszeit |
|---|---|---|
| **SEV1** | Geld-gefährdend, System tradet falsch | sofort, alle Hände |
| **SEV2** | Kill-Switch hat getripped, System gestoppt | 1h |
| **SEV3** | Signal-Qualität degradiert, Paper-System OK | 24h |
| **SEV4** | Kosmetisch, Monitoring-Noise | nächster Werktag |

---

## Die zehn wahrscheinlichsten Incidents

1. Alpaca-API returns 500er-Fehler / Connection-Loss
2. Broker-Positionen weichen vom System-State ab (Reconcile-Drift)
3. News-Pipeline liefert keine Daten mehr (feed stale)
4. PostgreSQL-Container unresponsive / Out-of-Memory
5. Hetzner-Server komplett down (Outage oder gelöscht)
6. Kill-Switch hat hart getripped, du weißt nicht warum
7. Anthropic-API-Budget überschritten, News-LLM fehlt
8. Feature-Drift-Alert: PSI>0.35 auf mehreren Features
9. Consecutive Losses > 5 — System fährt schlecht
10. Redis-Data-Loss (Cache weg, Signale "schreien")

Dazu drei erweiterte:

11. Datenquelle ändert Format / API Breaking-Change
12. Python-Package-Security-Vulnerability
13. Du kommst 4 Wochen nicht ans System und kehrst zurück

---

## Incident #1 — Alpaca-API-Ausfall

**Symptom:** Ihre Requests bekommen HTTP 500/502/503 oder Timeouts. Order-Submits scheitern. Positions-Calls hängen.

**Schweregrad:** SEV1 (während Trading-Hours), SEV3 (außerhalb).

**Detection:**
- Sentry: `APIError` oder `httpx.TimeoutException` in Spike
- Prometheus: `alpaca.request.success_rate < 0.5` für 2 Minuten
- Uptime Kuma: Alpaca-Health-Endpoint-Probe fails

### Sofort-Maßnahme (5 Minuten)

1. **Alpaca-Status-Page prüfen:** https://status.alpaca.markets
2. Wenn Alpaca-Outage bestätigt:
   ```bash
   # SSH zum Server
   tailscale ssh trading@ata-prod
   # Kill-Switch hart triggern
   docker compose exec app python -m src.ops.kill_switch trip \
     --level hard --reason "alpaca_api_outage" --context "status.alpaca.markets"
   ```
3. Bestehende Positionen sind bei Alpaca — sie bleiben, aber keine neuen Orders.

### Diagnose (wenn nicht Alpaca-Outage)

```bash
# Eigene Netzwerk-Probleme?
curl -v https://paper-api.alpaca.markets/v2/clock

# DNS-Problem?
dig paper-api.alpaca.markets

# Eigene Rate-Limit überschritten?
docker compose exec app python -c "
import httpx
r = httpx.get('https://paper-api.alpaca.markets/v2/account',
              headers={'APCA-API-KEY-ID': '$KEY', 'APCA-API-SECRET-KEY': '$SECRET'})
print(r.status_code, r.headers.get('X-Ratelimit-Remaining'))
"
```

**Wenn Rate-Limit:** Alpaca gibt 200 req/min free. Dein Polling-Pattern verbraucht zu viel. Siehe Behebung.

### Behebung

| Ursache | Aktion |
|---|---|
| Alpaca-Outage | Warten, Status-Page refreshen. Kill-Switch bleibt bis Resolution. |
| Eigene Netzwerk-Probleme | `systemctl restart networking` auf Hetzner |
| Rate-Limit-Überschreitung | Polling-Frequenz in `src/polling.py` reduzieren, Batch-Endpoints nutzen |
| Expired API-Key | Neuen Key auf dashboard.alpaca.markets generieren, SOPS-File updaten |

### Nach Resolution

```bash
# Reconcile-Run bevor Kill-Switch zurückgesetzt wird
docker compose exec app python -m src.ops.reconcile --full

# Wenn kein Drift → Kill-Switch zurücksetzen
docker compose exec app python -m src.ops.kill_switch reset --token "$RESET_TOKEN"
```

### Post-Incident

- [ ] Sentry-Alert-Schwellen prüfen (waren sie rechtzeitig?)
- [ ] Uptime-Kuma-History als Beweis archivieren
- [ ] Wenn häufig: Secondary-Broker evaluieren (IBKR Paper als Backup?)

---

## Incident #2 — Broker-Position weicht ab (Reconcile-Drift)

**Symptom:** Alert "🔄 Position-Drift: N Symbols" aus Slack.

**Schweregrad:** SEV1 wenn >1 Symbol oder Cash-Drift > $100, sonst SEV2.

**Detection:** Automatischer Reconcile-Worker (alle 5 Min) vergleicht Postgres vs. Alpaca und loggt Diffs.

### Sofort-Maßnahme

1. **Alle neuen Orders pausieren:**
   ```bash
   docker compose exec app python -m src.ops.kill_switch trip \
     --level hard --reason "reconcile_drift"
   ```
2. **Aktuellen Broker-State als Wahrheit speichern:**
   ```bash
   docker compose exec app python -m src.ops.reconcile --snapshot broker > /tmp/broker_truth.json
   ```

### Diagnose

Die häufigsten Ursachen für Drift:

| Ursache | Wahrscheinlichkeit | Indikator |
|---|---|---|
| WebSocket-Event verloren | hoch | Polling zeigt andere Qty als letztes DB-Update |
| Partial-Fill nicht korrekt gebucht | hoch | `order_events` zeigt `partial_fill` aber Position-Update fehlt |
| Manueller Trade in Alpaca-UI | mittel | Order existiert in Alpaca, aber nicht in `orders`-Tabelle |
| Corporate Action (Split, Spin-off) | niedrig | Alpaca Activity zeigt `SC` (split) oder `SPIN` |
| Dividend-Reinvestment | niedrig | Activity zeigt `DIV` |

```bash
# Check: manuelle Trades in Alpaca (Order in Alpaca aber nicht in DB)
docker compose exec app python -m src.ops.reconcile --find-orphan-broker-orders

# Check: DB zeigt Position, Alpaca nicht
docker compose exec app python -m src.ops.reconcile --find-orphan-db-positions

# Check: Corporate Actions letzte 7 Tage
docker compose exec app python -m src.ops.activity --days 7 --types SC,SPIN,DIV
```

### Behebung

**Regel:** Der Broker ist immer die Wahrheit. Dein System passt sich an, nie umgekehrt.

```bash
# Sicherung vorher
docker compose exec postgres pg_dump -U trading trading > /tmp/before_reconcile_$(date +%F_%H%M).sql

# Force-Correct
docker compose exec app python -m src.ops.reconcile --force-correct --confirm
```

**Bei Corporate Action:** Position-Verlauf manuell nachziehen. `tax_lots` muss korrekt sein.

```sql
-- Beispiel: Split 2:1 auf ticker X
UPDATE positions SET qty = qty * 2, avg_entry_price = avg_entry_price / 2 
WHERE symbol = 'X';
UPDATE tax_lots SET qty = qty * 2, price_usd = price_usd / 2, price_eur = price_eur / 2 
WHERE symbol = 'X' AND status = 'open';
```

### Post-Incident

- [ ] 24h nach Kill-Switch-Reset erneut reconcilen, 0 Drift erwartet
- [ ] Wenn WebSocket-Event-Loss: Polling-Intervall prüfen
- [ ] Root-Cause in `docs/incidents/` dokumentieren

---

## Incident #3 — News-Pipeline liefert keine Daten

**Symptom:** `news.feed.staleness_seconds > 600` für 10+ Minuten. Keine neuen Artikel in Redis.

**Schweregrad:** SEV2 wenn News-Gewichtung im aktuellen Regime > 15%, sonst SEV3.

**Detection:** Prometheus-Metrik `news.last_ingest_timestamp`, Alert wenn >10 Min alt.

### Sofort-Maßnahme

```bash
# Kill-Switch soft triggern (News-Signale unzuverlässig)
docker compose exec app python -m src.ops.kill_switch trip \
  --level soft --reason "news_feed_stale"

# Worker-Status prüfen
docker compose logs --tail 100 worker | grep -i "news\|gdelt\|finnhub\|sec"
```

### Diagnose

Welche Pipeline ist tot? News hat mehrere Quellen.

```bash
docker compose exec app python -c "
from src.news.health import check_all
import asyncio
print(asyncio.run(check_all()))
"
```

Erwartete Ausgabe:
```
gdelt:       last_fetch=2min_ago,  articles_last_hour=120 ✓
finnhub:     last_fetch=5min_ago,  articles_last_hour=45  ✓
sec_edgar:   last_fetch=3min_ago,  articles_last_hour=8   ✓
rss_pr:      last_fetch=40min_ago, articles_last_hour=0   ✗
rss_bw:      last_fetch=40min_ago, articles_last_hour=0   ✗
rss_gnw:     last_fetch=40min_ago, articles_last_hour=0   ✗
```

Im Beispiel: alle RSS-Feeds tot → wahrscheinlich ein feedparser-Problem oder Outbound-Connectivity.

| Problem | Behebung |
|---|---|
| Alle Quellen tot | Outbound-Connectivity-Problem, `curl -v google.com` |
| Eine Quelle tot | Source-Health-Check, ggf. URL-Change |
| FinBERT-Container crashed | `docker compose restart worker` |
| Redis out-of-memory | `docker compose exec redis redis-cli INFO memory` |
| Embedding-Modell nicht geladen | Logs checken, neu deployen |

### Behebung

```bash
# Einzelnen Service neustarten
docker compose restart worker

# Wenn immer noch tot: komplett neu
docker compose down worker
docker compose up -d worker

# Force-Pull der letzten Stunde News
docker compose exec app python -m src.news.backfill --hours 1
```

### Post-Incident

- [ ] Staleness-Schwelle in Prometheus prüfen (war 10min zu spät?)
- [ ] Fallback-Pipeline: wenn eine Quelle tot ist, andere höher gewichten

---

## Incident #4 — PostgreSQL Unresponsive / OOM

**Symptom:** Queries hängen, Container-Restart-Loop, Logs zeigen `Out of memory: Killed process`.

**Schweregrad:** SEV1.

**Detection:** Grafana-Dashboard `pg_up = 0`, Sentry-Error auf DB-Connections.

### Sofort-Maßnahme

```bash
# Kill-Switch hart
docker compose exec app python -m src.ops.kill_switch trip \
  --level hard --reason "postgres_oom"

# Postgres-Status prüfen
docker compose ps postgres
docker compose logs --tail 200 postgres | grep -i "memory\|kill\|error"
```

### Diagnose

```bash
# RAM-Gesamtnutzung auf Server
free -m

# Container-RAM-Nutzung
docker stats --no-stream

# Postgres-Running-Queries
docker compose exec postgres psql -U trading -c "
  SELECT pid, now() - query_start AS duration, state, query 
  FROM pg_stat_activity 
  WHERE state != 'idle' 
  ORDER BY duration DESC LIMIT 20;
"
```

Häufige Ursachen:
- Langlaufende Query auf großer Tabelle (typisch: `order_events` ohne Index)
- Zu viele Connections (Default 100, Worker-Pool zu groß konfiguriert)
- `shared_buffers` zu hoch auf CX22

### Behebung

```bash
# Notfall-Restart (Daten-safe wegen WAL + Volume)
docker compose restart postgres
sleep 30
docker compose exec postgres pg_isready -U trading

# Lange Queries killen (nur wenn spezifisch schuldig)
docker compose exec postgres psql -U trading -c "SELECT pg_cancel_backend(PID);"

# Falls persistentes RAM-Problem: Config tunen
# docker-compose.yml postgres service:
#   command: 
#     - "postgres"
#     - "-c" 
#     - "shared_buffers=512MB"          # nicht mehr als 25% RAM
#     - "-c"
#     - "effective_cache_size=2GB"
#     - "-c"
#     - "max_connections=50"            # statt 100
```

**Bei korrupter DB:** Aus Backup wiederherstellen.

```bash
# Letztes Backup finden
ls -la /home/trading/ata/backups/*.sql.gz | tail -5

# Restore
docker compose stop app worker scheduler
docker compose exec -T postgres psql -U trading -c "DROP DATABASE trading;"
docker compose exec -T postgres psql -U trading -c "CREATE DATABASE trading;"
zcat /home/trading/ata/backups/pg-YYYY-MM-DD.sql.gz | \
  docker compose exec -T postgres psql -U trading trading
docker compose start app worker scheduler
```

**Danach zwingend:** Reconcile gegen Alpaca, Tax-Lots prüfen.

### Post-Incident

- [ ] Backup-Frequenz prüfen (täglich reicht? stündliche WAL?)
- [ ] Hetzner-Monitoring: RAM-Alert bei >85%
- [ ] ggf. CX32-Upgrade (16 GB RAM)

---

## Incident #5 — Hetzner-Server komplett weg

**Symptom:** SSH timeout, Tailscale zeigt Device offline, Uptime Kuma zeigt alles rot.

**Schweregrad:** SEV1 (während Trading-Hours).

**Detection:** Uptime Kuma Alert, Grafana Cloud no-data für >5min.

### Sofort-Maßnahme (kritisch!)

**Du musst das System stoppen können, auch wenn Hetzner down ist.**

1. **Alpaca manuell erreichen:**
   - Browser: https://app.alpaca.markets/paper/dashboard
   - Login: in Password-Manager
   - **Tab "Positions" → "Close All" ist der NOT-AUS**
   - **Tab "Orders" → "Cancel All"**
2. Alpaca-UI bleibt funktional, auch wenn dein Server tot ist.

Nach manuellem Flatten:
- System tradet nicht (es gibt ja keinen Server)
- Du hast Zeit zur Diagnose

### Diagnose

```bash
# Von einem anderen Rechner (Handy-Tethering, Freund)
# Hetzner-Cloud-Console öffnen: console.hetzner.cloud

# Server-Status prüfen:
# - Läuft er? (grünes Power-Symbol)
# - Letzte Events: wurde er neugestartet, gestoppt?
# - Monitoring: wann kam der letzte Metric-Wert?
```

Häufige Ursachen:
- Hetzner-Outage (check: status.hetzner.com)
- Rate-Limit / Automated-Suspension wegen verdächtigem Traffic
- Zahlung fehlgeschlagen → Suspendierung
- SSD-Voll (Server läuft, aber keine Writes mehr)
- Kernel-Panic
- Netzwerk-Interface down

### Behebung

```bash
# Via Hetzner-Console: Power-Reset
# console.hetzner.cloud → Server → Power → Reset

# Wenn Boot-Loop: Rescue-Mode
# console.hetzner.cloud → Server → Rescue → Enable

# SSD-Voll? In Rescue-Mode:
mount /dev/sda1 /mnt
df -h /mnt
du -sh /mnt/var/lib/docker/* | sort -h | tail
# Docker-Images bereinigen
chroot /mnt docker system prune -a --volumes
```

**Bei komplett verlorener Maschine (unwahrscheinlich):**

1. Neuen Server provisionieren (Hetzner Cloud oder Oracle Always-Free als Notfall)
2. Latestes Backup aus Backblaze B2:
   ```bash
   rclone copy b2:ata-worm-backup/pg-latest.sql.gz ./
   rclone copy b2:ata-worm-backup/parquet-latest.tar.gz ./
   ```
3. Docker-Compose wiederherstellen (Git-Clone des Repos)
4. Daten einspielen, Secrets aus SOPS + age
5. Tailscale-Tag neu assignen
6. **Reconcile gegen Alpaca** (Broker ist Wahrheit)

Erwartete Restore-Zeit: **2-4 Stunden**.

### Post-Incident

- [ ] Monatliches Restore-Drill: aus Backup auf Test-Server wiederherstellen
- [ ] Payment-Methode prüfen (Hetzner nimmt Lastschrift oder Kreditkarte — beide haben Stolperfallen)
- [ ] Backup-Verifikation: SHA256 der letzten Backups prüfen

---

## Incident #6 — Kill-Switch getripped, Grund unklar

**Symptom:** Slack-Alert "🛑 Kill-Switch HARD: [reason]" — aber du verstehst nicht, warum.

**Schweregrad:** SEV2.

### Sofort-Maßnahme

**Nicht sofort zurücksetzen.** Erst verstehen, was passiert ist.

```bash
docker compose exec app python -m src.ops.kill_switch status

# Sollte zeigen:
# State: hard
# Tripped at: 2026-05-01 14:23:00 UTC
# Reason: consecutive_losses
# Context: {"consec": 10, "losses": [...]}
```

### Diagnose

```sql
-- Letzte 20 Kill-Switch-Events
SELECT * FROM kill_switch_events ORDER BY tripped_at DESC LIMIT 20;

-- Welche Trades haben zum Trigger geführt?
SELECT * FROM positions_closed 
WHERE closed_at > (SELECT tripped_at FROM kill_switch_events ORDER BY tripped_at DESC LIMIT 1) - INTERVAL '6 hours'
ORDER BY closed_at DESC;
```

**Interpretation:**

| Trigger | Bedeutung | Aktion |
|---|---|---|
| `daily_drawdown_exceeded` | Tagesverlust > 3% | normales Markt-Event, Reset wenn Regime-korrekt |
| `consecutive_losses` (10x) | Strategie läuft im aktuellen Regime schlecht | **nicht einfach resetten**, Regime-Check |
| `reconcile_drift` | siehe Incident #2 | dort behandeln |
| `high_reject_rate` | Order-Rejects > 30%/h | siehe Incident #1 oder PDT-Issue |
| `data_feed_stale` | News oder Market-Data-Feed tot | siehe Incident #3 |

### Behebung

**Wichtig:** Bei `consecutive_losses` ist die richtige Antwort meist nicht "Reset und weiter". Es ist "Regime-Shift verstehen".

```bash
# Aktuelles Regime prüfen
docker compose exec app python -c "
from src.signals.regime import RegimeClassifier
r = RegimeClassifier().classify_now()
print(f'Regime: {r.label}, Confidence: {r.confidence}')
"

# Performance pro Signal in letzten 30 Tagen
docker compose exec app python -m src.ops.signal_performance --days 30
```

Wenn ein Signal eindeutig im aktuellen Regime versagt:
```bash
# Signal einzeln deaktivieren (nicht Gesamtsystem)
docker compose exec app python -m src.ops.signal disable --name pead_sue

# Kill-Switch zurücksetzen
docker compose exec app python -m src.ops.kill_switch reset --token "$RESET_TOKEN"
```

### Post-Incident

- [ ] Incident in `docs/incidents/` dokumentieren
- [ ] Prüfen ob Trigger-Schwelle richtig ist (zu sensibel? zu lax?)
- [ ] Bei Regime-bedingtem Trigger: Regime-Weighting-Matrix überprüfen

---

## Incident #7 — Anthropic-Budget überschritten

**Symptom:** Slack-Alert "💰 Anthropic-Budget 90% spent". News-LLM-Zweitrunde läuft nicht mehr.

**Schweregrad:** SEV3.

**Detection:** `LLMBudgetGuard` in Code (siehe `21_PAID_MODELLE.md`).

### Sofort-Maßnahme

Kein Kill-Switch nötig — News-LLM ist Enhancement, nicht Kern. FinBERT-Tone läuft weiter.

```bash
# Aktuelle Ausgaben prüfen
curl -H "x-api-key: $ANTHROPIC_API_KEY" \
  https://api.anthropic.com/v1/organizations/usage_report/messages

# Fallback aktivieren: lokales Ollama
docker compose exec app python -m src.ops.llm switch --provider ollama
```

### Diagnose

Warum ist Budget weg?

```sql
-- Anthropic-Calls der letzten 30 Tage
SELECT DATE(created_at), COUNT(*), SUM(estimated_cost_usd) 
FROM llm_calls 
WHERE provider = 'anthropic' 
  AND created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY DATE(created_at);
```

Häufige Ursachen:
- News-Volumen gestiegen (Earnings-Season, Crisis-Regime)
- Retry-Schleife ohne Exponential-Backoff (Bug)
- LLM-Veto wird für zu viele Trades aufgerufen

### Behebung

**Option 1 — Provider-Switch (sofort):**
```bash
# Ollama lokal, keine Kosten
docker compose exec app python -m src.ops.llm switch --provider ollama_llama31
```

**Option 2 — Budget erhöhen (für Monatsende OK):**
- Anthropic-Console → Billing → Add 10 USD
- Budget-Guard in Code updaten

**Option 3 — Qualität reduzieren:**
```bash
# Nur noch Top-10 News pro Tag (statt Top-20)
docker compose exec app python -m src.ops.config set \
  llm.max_news_per_day=10
```

### Post-Incident

- [ ] Budget-Pattern analysieren: wann Peaks? (Earnings-Weeks?)
- [ ] Budget-Kurve in Grafana für Trend-Tracking

---

## Incident #8 — Feature-Drift-Alert (PSI>0.35)

**Symptom:** Slack-Alert "⚠ PSI-Drift: 3 Features über 0.35". Betroffene Features z.B. `news_velocity`, `vix_term_z`, `sector_rs`.

**Schweregrad:** SEV3.

**Detection:** Daily-Cron `evidently` vergleicht letzte 7 Tage vs. Training-Baseline.

### Sofort-Maßnahme

```bash
# Drift-Report laden
docker compose exec app python -m src.ops.drift_report --days 7

# Auto-Pause für drift-verdächtige Signale
docker compose exec app python -m src.ops.signal throttle --psi-threshold 0.35
```

### Diagnose

Drift hat drei Ursachen:

| Ursache | Indikator | Beispiel |
|---|---|---|
| **Markt-Regime-Wechsel** | mehrere korrelierte Features driften zusammen | Calm→Crisis Übergang |
| **Daten-Quelle-Änderung** | einzelnes Feature driftet abrupt | Finnhub-API-Schema geändert |
| **Echtes Signal-Decay** | Drift stabil über Wochen, IC fällt | Strategie ausgestorben |

```bash
# Alle drei prüfen
docker compose exec app python -c "
from src.monitoring.drift import diagnose_drift
diagnose_drift('news_velocity', days=30)
"

# Output:
# Feature: news_velocity
# PSI 7d: 0.42, PSI 30d: 0.18, PSI 90d: 0.08
# Correlation-with-regime-shift: 0.71  ← Regime-Wechsel wahrscheinlich
# Correlation-with-other-features: [...]
```

### Behebung

**Bei Regime-Wechsel:** Kein Eingreifen. Die Regime-Weighting-Matrix passt sich bereits an. PSI wird sich nach 14-21 Tagen normalisieren, wenn das neue Regime Baseline wird.

**Bei Daten-Quelle-Change:**
- API-Docs prüfen
- Eingabe-Schema validieren
- Rollback zu altem Parser oder Schema-Adapter schreiben

**Bei echtem Signal-Decay:**
- Signal-Retraining-Pipeline triggern
- Wenn Retraining keinen Lift bringt: Signal hart abschalten
- Live-Capital-Allokation auf andere Signale verschieben

### Post-Incident

- [ ] Drift-Report als Markdown in `docs/drift/YYYY-MM.md`
- [ ] Bei Regime-Wechsel: neue Training-Baseline erwägen (alter Markt ist Vergangenheit)

---

## Incident #9 — Consecutive Losses > 5

**Symptom:** Soft-Kill-Switch triggered nach 5 Verlust-Trades in Folge.

**Schweregrad:** SEV3.

### Sofort-Maßnahme

Soft-Kill → keine neuen Entries, existierende laufen. Nicht panisch alles schließen.

```bash
# Aktuelle Lage
docker compose exec app python -m src.ops.portfolio_status
```

### Diagnose

**Die große Frage:** Systematisch oder normal-statistisch?

Bei Sharpe 0.8 und 100 Trades ist Wahrscheinlichkeit für 5 Losses in Folge ~4-6%. Kein Grund zur Panik.

Bei Sharpe 0.8 und 10 Losses in Folge → Signal-Problem wahrscheinlich.

```bash
# Binomial-Test
docker compose exec app python -c "
from scipy.stats import binom
# Annahme: Win-Rate historisch 55%
# Probability: 5 consecutive losses
p_loss = 0.45
prob = p_loss ** 5
print(f'P(5 losses in a row | win_rate=55%): {prob:.3%}')
print(f'Over 100 trades: {1 - (1 - prob)**96:.3%} chance of 5+ streak')
"

# Output:
# P(5 losses in a row | win_rate=55%): 1.845%
# Over 100 trades: 83.7% chance of 5+ streak  ← normal!
```

**Realistisch: Verlustserie ist normal.** Nur systematisches Muster ist gefährlich.

```bash
# Welche Signale haben verloren?
docker compose exec app python -m src.ops.recent_losses --count 10

# Alle das gleiche Signal? Alle im gleichen Regime? Alle gleiche Sektor?
```

### Behebung

**Normal-statistische Serie:** Nicht eingreifen. Soft-Kill 24h laufen lassen, dann automatisch zurücksetzen.

**Systematisches Muster (z.B. alle Losses aus PEAD-Signal):**
- Signal isoliert analysieren
- Wenn Regime-bedingt: Regime-Gewichtung reduzieren
- Wenn echter Decay: Signal deaktivieren

### Post-Incident

- [ ] 30-Tages-Performance pro Signal-Gruppe prüfen
- [ ] Keine emotionalen Overrides — Statistik vertrauen

---

## Incident #10 — Redis-Data-Loss

**Symptom:** Cache leer, alle Features müssen neu berechnet werden. Feature-Store-Reads langsam.

**Schweregrad:** SEV3 (Performance-Degradation, nicht Geld-Verlust).

**Detection:** Grafana: Redis-Keys-Count = 0 abrupt.

### Sofort-Maßnahme

Kein Kill-Switch. Redis ist Cache + Event-Bus — Daten sind persistent in Postgres + Parquet.

```bash
# Redis-Status
docker compose exec redis redis-cli INFO

# Wurde Container neugestartet?
docker compose ps redis
docker inspect --format='{{.RestartCount}}' $(docker compose ps -q redis)
```

### Diagnose

Redis hat **kein eingebautes Persistenz-Default** für AOF (Append-Only-File). Wenn Container restart ohne AOF → alle Keys weg.

```bash
# Prüfen ob AOF aktiv
docker compose exec redis redis-cli CONFIG GET appendonly
# Antwort sollte "yes" sein

# Falls "no": aktivieren (siehe unten)
docker compose exec redis redis-cli CONFIG SET appendonly yes
```

### Behebung

**Kurzzeitig:** Cache wird in 1-2 Zyklen neu aufgebaut.

**Langfristig:** AOF aktivieren.

```yaml
# docker-compose.yml, redis service:
redis:
  image: redis:7.4-alpine
  command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru --appendonly yes --save 60 1000
  volumes:
    - ./redisdata:/data   # <- wichtig, damit AOF persistiert!
```

### Post-Incident

- [ ] AOF in Backup-Rotation aufnehmen
- [ ] Feature-Store-Reads ohne Cache messen (Fallback-Latenz)

---

## Incident #11 — API-Breaking-Change

**Symptom:** Eine Daten-Quelle liefert plötzlich HTTP 410 oder neues Schema. Beispiel: Finnhub ändert `/stock/earnings` von `array` zu `{data: array}`.

**Schweregrad:** SEV2.

### Sofort-Maßnahme

```bash
# Soft-Kill wenn das betroffene Signal wichtig ist
docker compose exec app python -m src.ops.signal disable --name pead_sue
```

### Diagnose

```bash
# Raw-Response checken
curl -H "X-Finnhub-Token: $KEY" \
  "https://finnhub.io/api/v1/stock/earnings?symbol=AAPL" | jq .

# Mit erwartetem Schema vergleichen
docker compose exec app python -m src.parsers.validate --source finnhub_earnings
```

### Behebung

1. Schema-Adapter schreiben (in `src/adapters/`)
2. Pydantic-Modelle aktualisieren
3. Test schreiben gegen neues Schema (VCR-Cassette)
4. Signal wieder aktivieren

### Post-Incident

- [ ] **API-Change-Detection-Layer:** alle eingehenden Responses gegen Pydantic-Schema validieren, nicht naiv parsen
- [ ] Beobachtungs-Liste: welche APIs haben Änderungs-Historie?

---

## Incident #12 — Security-Vulnerability in Dependency

**Symptom:** Sentry-Alert oder GitHub-Dependabot-Alert: `CVE-2026-XXXX in transformers < 4.46.8`.

**Schweregrad:** SEV3 (falls nicht exposed als Netzwerkdienst) / SEV1 (falls exponiert).

### Sofort-Maßnahme

Bewertung der CVE:

```bash
# Welche Version ist installiert
docker compose exec app pip show transformers

# Was betrifft es? CVE auf nvd.nist.gov oder github.com/advisories
# Für uns relevant: attack vector (local? network? user-assisted?)
```

### Behebung

```bash
# Dependency-Bump im lokalen Dev
# pyproject.toml: "transformers==4.46.8"
uv sync

# Testen
pytest tests/

# Deploy
docker compose build app
docker compose up -d app worker scheduler
```

### Post-Incident

- [ ] Monatlicher `uv pip list --outdated`-Review in Kalender
- [ ] Dependabot auf GitHub aktivieren
- [ ] `pip-audit` als Pre-commit-Hook

---

## Incident #13 — Rückkehr nach 4 Wochen Abwesenheit

**Symptom:** Urlaub, Krankheit, "ich war lange weg und weiß nicht, was gelaufen ist".

**Schweregrad:** SEV3 (Risiko, dass Problem unentdeckt blieb).

### Systematisches Vorgehen (in dieser Reihenfolge)

1. **Kill-Switch-Status prüfen**
   ```bash
   docker compose exec app python -m src.ops.kill_switch status
   ```
   Wenn "tripped": siehe Incident #6.

2. **Uptime-History anschauen**
   - Grafana Cloud → Uptime-Dashboard
   - Letzte 30 Tage: wie viele Downtimes?
   - Wenn >99%: weiter

3. **Alpaca-Portfolio real checken**
   - Browser-Login bei Alpaca
   - Equity-Kurve anschauen
   - Suspicious-Trades? Force-Close ungewöhnlicher Positionen

4. **Reconcile-Report**
   ```bash
   docker compose exec app python -m src.ops.reconcile --full --report
   ```

5. **P&L-Historie**
   ```bash
   docker compose exec app python -m src.ops.pnl_report --days 30
   ```

6. **Incident-Log der letzten 30 Tage**
   ```sql
   SELECT * FROM kill_switch_events WHERE tripped_at > NOW() - INTERVAL '30 days';
   SELECT * FROM system_alerts WHERE created_at > NOW() - INTERVAL '30 days';
   ```

7. **Dependency-Updates**
   ```bash
   uv pip list --outdated
   pip-audit
   ```

8. **Backup-Health**
   ```bash
   # Letztes erfolgreiches Backup?
   ls -la /home/trading/ata/backups/ | tail -5
   rclone ls b2:ata-worm-backup | tail -5
   ```

9. **Budget-Check**
   - Hetzner-Rechnung OK?
   - EODHD-Account aktiv?
   - Anthropic-Credit-Saldo?

10. **Wenn alles OK:** Wechsel von Shadow zu normalen Operations nach 24h Beobachtung.

---

## Chaos-Test-Schedule

**Alle 3 Monate:** Planmäßiger Failure-Test.

### Q1: Postgres Restart

```bash
# Während Market-Hours (Paper!)
docker compose restart postgres
# Beobachten: wie lange bis System normal läuft?
# Erwartung: <60 Sekunden, keine Order verloren
```

### Q2: Redis Kill

```bash
docker compose stop redis
sleep 30
docker compose start redis
# Erwartung: Features werden neu berechnet, keine Kill-Switch-Trigger
```

### Q3: Restore aus Backup

```bash
# Auf Test-Server aus Backup wiederherstellen
# Erwartung: identischer State wie Production (außer letzter Stunde)
```

### Q4: Fake-Alpaca-Outage

```bash
# iptables-Regel: Alpaca-API blockieren
iptables -A OUTPUT -d paper-api.alpaca.markets -j DROP
sleep 180  # 3 Minuten
iptables -D OUTPUT -d paper-api.alpaca.markets -j DROP
# Erwartung: Kill-Switch trippt nach 2min, Retry danach erfolgreich
```

---

## Post-Mortem-Template

Nach jedem SEV1/SEV2-Incident:

```markdown
# Post-Mortem: [Titel]

**Date:** YYYY-MM-DD
**Duration:** HH:MM - HH:MM
**Impact:** [Was war betroffen, wie lange]
**Severity:** SEV1/SEV2
**Root Cause:** [1-2 Sätze]

## Timeline

- HH:MM Symptom zuerst beobachtet
- HH:MM Detection (wie?)
- HH:MM Sofort-Maßnahme ausgeführt
- HH:MM Root-Cause identifiziert
- HH:MM Behebung abgeschlossen
- HH:MM Normal-Betrieb

## Was lief gut

- [Monitoring hat rechtzeitig gewarnt]
- [Runbook war hilfreich]

## Was lief schlecht

- [Detection zu spät]
- [Runbook fehlte]

## Action Items

- [ ] Code-Fix für Root Cause
- [ ] Monitoring-Alert für ähnliche Fälle
- [ ] Runbook-Update
- [ ] Chaos-Test-Szenario hinzufügen
```

Archiviert in `docs/incidents/YYYY-MM-DD_[kurztitel].md`.

---

## Umsetzungs-Checkliste

**Phase 1 (erste 3 Monate):**
- [ ] 10 Basis-Runbooks als Markdown in `docs/runbooks/`
- [ ] Incident-DB: `CREATE TABLE incidents (id, title, severity, started, resolved, root_cause)`
- [ ] Post-Mortem-Template als Git-Template
- [ ] Grafana-Dashboard "Incident-Overview"
- [ ] Slack-Alerts auf SEV-Level-Mapping

**Phase 2 (Monat 4-6):**
- [ ] Runbooks #11-#13 hinzugefügt
- [ ] Erster Chaos-Test: Postgres-Restart während Test-Markt
- [ ] Backup-Restore-Drill erfolgreich auf Test-Server
- [ ] On-Call-Rotation (nur du, aber dokumentierte Verfügbarkeit)

**Phase 3 (Monat 7-9):**
- [ ] Quartalsweise Chaos-Tests etabliert
- [ ] Alle Runbooks mindestens einmal "durchgespielt" (auch wenn nicht real)
- [ ] Post-Mortem-Historie: 3+ dokumentierte Incidents

---

## Ehrliche Einschätzung

**Die meisten Solo-Quant-Systeme haben keine Runbooks.** Das ist der Grund, warum ein einziger Incident sie zerstört — bei Nacht, im Stress, ohne Plan reagiert man falsch.

**Der Unterschied zwischen "Hobby-Projekt" und "ernsthaftem System":**

- Hobby: nachts 3 Uhr, Panik, manuelles Eingreifen, häufig Fehler
- Ernst: nachts 3 Uhr, Runbook, Schritt-für-Schritt, selten Fehler

**Die wichtigsten drei Regeln:**

1. **Der Broker ist immer die Wahrheit** — bei jedem Drift.
2. **Kill-Switch vor Debuggen** — erst stoppen, dann denken.
3. **Nach jedem Incident: Post-Mortem** — sonst lernst du nichts.

**Was Chaos-Tests bringen:** Du findest Probleme in kontrollierten Situationen (Paper-Account, unkritische Zeiten) statt im echten Notfall. Ein Postgres-Restart zu einem ruhigen Sonntag ist langweilig. Ein Postgres-Crash am Earnings-Friday mit $5k offenen Orders ist teuer.

**Die einzige Wahrheit über Incidents:** Sie werden passieren. Nicht ob, sondern wann. Ein gutes System ist nicht eines, das nie ausfällt — es ist eines, das schnell und vorhersagbar wiederkommt.
