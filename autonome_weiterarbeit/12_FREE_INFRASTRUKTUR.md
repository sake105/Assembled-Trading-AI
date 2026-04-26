# 12 — Free Infrastruktur (0 EUR/Monat)

**Zweck:** Das System komplett kostenlos hosten und betreiben — von lokalem Windows-Start über Oracle Always-Free bis zum FastAPI-Deployment ohne Subscription-Kosten.

---

## Module in diesem Dokument

| # | Modul | Kategorie |
|---|---|---|
| 12.1 | Lokales Windows-Setup | Development |
| 12.2 | Oracle Cloud Always-Free | Cloud (ohne Kreditkarten-Belastung) |
| 12.3 | FastAPI + Uvicorn | Backend |
| 12.4 | Redis Streams / Valkey | Event-Bus + Cache |
| 12.5 | Postgres (lokal) + SQLite | State-Persistence |
| 12.6 | DuckDB + Parquet | Analytics + Feature-Store |
| 12.7 | APScheduler | Scheduling |
| 12.8 | MLflow (SQLite-Backend) | Model Registry + Tracking |
| 12.9 | Docker + Docker-Compose | Containerization |
| 12.10 | Caddy | Reverse-Proxy + Auto-HTTPS |
| 12.11 | Uptime Kuma + HealthChecks.io | Monitoring |
| 12.12 | Grafana Cloud Free-Tier | Observability (Cloud-Dashboards free) |
| 12.13 | Sentry Free-Tier | Error-Tracking |
| 12.14 | SOPS + age | Secrets-Management |
| 12.15 | Git + Pre-commit + gitleaks | Code-Hygiene |

---

## 12.1 Lokales Windows-Setup

**Der realistischste Start.** Dein System läuft aktuell Windows-lokal. Das ist für Phase 1 (Monate 0-3) absolut richtig. Erst wenn FastAPI 24/7 laufen muss oder du an der Alpaca-Paper-Connection für Live-Intraday testest, brauchst du Cloud.

**Python:** 3.12 via `pyenv-win` oder direkter Installer.
**Shell:** PowerShell 7+ als Standard, WSL2 nur wenn du hftbacktest oder Linux-spezifische Tools brauchst.
**IDE:** VSCode (kostenlos) mit Python-Extension + Pylance + `uv`-Support.
**Package-Manager:** **`uv` von Astral** (in Rust, ~10× schneller als pip). Install: `pip install uv`.

**Struktur:**
```
F:\Python_Projekt\Aktiengerüst\
  assembled_trading_ai/
    .env                   # im .gitignore
    .venv/
    src/
    tests/
    pyproject.toml
```

**Achtung:** Dein Windows-Pfad hat einen Umlaut + Leerzeichen (`Aktiengerüst`, siehe Teil 1 des Audits). Das kann PowerShell-Scripts brechen. **Umbenennen zu `Aktiengeruest`** oder `AssembledTradingAI` vorteilhaft.

---

## 12.2 Oracle Cloud Always-Free

**Der Gamechanger für 0-EUR-Deployment.** Oracle bietet **dauerhaft** free:
- 4 OCPUs (ARM Ampere A1) + 24 GB RAM
- 200 GB Block-Storage
- 10 TB Egress/Monat
- 2 × AMD64 VMs mit 1 OCPU + 1 GB RAM (klein, aber nutzbar)

**Region-Caveat:** Frankfurt/London oft "Out of Capacity". Lösungen:
1. Andere Region nehmen (Amsterdam, Marseille, Mailand)
2. Pay-As-You-Go-Upgrade (100 USD Karten-Hold, refundable) erhöht Success-Rate

**Setup-Schritte:**
1. Account auf `cloud.oracle.com` erstellen.
2. Compartment erstellen: `assembled-trading`.
3. VM-Instance: Shape `VM.Standard.A1.Flex` mit 4 OCPUs + 24 GB.
4. Ubuntu 22.04 LTS als OS.
5. SSH-Key generieren, im Browser hochladen.
6. Firewall-Regel: Port 80, 443, 22 (SSH).

**Was du damit baust:**
- FastAPI-App + Postgres + Redis + Grafana-Agent — alles in Docker-Compose.
- Einziger laufender Compute für das gesamte System.
- Backup via Object-Storage (10 GB free).

**Einzige Sorge:** ARM64-Architektur — du musst Docker-Images für `linux/arm64` bauen oder Multi-Arch. Die meisten Libraries (FastAPI, PyTorch, Numpy) haben ARM-Wheels. **LightGBM, TA-Lib, hftbacktest können problematisch sein** — vorher testen.

---

## 12.3 FastAPI + Uvicorn

**Install:** `pip install fastapi==0.115.0 uvicorn[standard]==0.30.0`

**Production-Start:**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 2 --loop uvloop
```

**Pattern `@asynccontextmanager lifespan`** (nicht deprecated `@app.on_event`):
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    app.state.redis = await aioredis.from_url("redis://localhost")
    app.state.db = create_engine(...)
    yield
    # shutdown
    await app.state.redis.close()

app = FastAPI(lifespan=lifespan)
```

**Config:** `pydantic-settings 2.x` für `.env`-basiertes Setup.

---

## 12.4 Redis Streams / Valkey

**Zwei Optionen:**

| Option | Lizenz | Empfehlung |
|---|---|---|
| **Redis 7.4** | RSAL/SSPL seit 2024 | Für private Nutzung OK |
| **Valkey 8.x** | BSD-3 (Linux-Foundation-Fork) | Für SaaS-Pfad |

**Install lokal (Docker):**
```yaml
redis:
  image: redis:7.4-alpine
  # oder: valkey/valkey:8.0-alpine
  ports: ["6379:6379"]
```

**Patterns:**
- **Event-Bus:** `XADD`/`XREAD` für Signal-Emissions
- **Cache:** Set-With-TTL für Feature-Cache
- **Rate-Limiter:** `INCR` + `EXPIRE` für API-Quotas
- **Pub/Sub:** WebSocket-Fanout im Dashboard

**Benchmark:** 480k msg/s, p99 ~0.8ms — schnellster Event-Bus für Solo.

---

## 12.5 Postgres + SQLite

**Dual-Database-Pattern:**

| DB | Use-Case | Warum |
|---|---|---|
| **Postgres 16** (lokal Docker oder Oracle) | Shared-State, Event-Source, Audit-Trail | Multi-Process-Safe, JSONB, Partitioning |
| **SQLite** mit `PRAGMA journal_mode=WAL` | Tracking-Logs, Single-Process-Writes | Zero-Ops, atomare Writes |

**Postgres-Install (Docker):**
```yaml
postgres:
  image: postgres:16-alpine
  environment:
    POSTGRES_USER: trading
    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    POSTGRES_DB: trading
  volumes:
    - ./pgdata:/var/lib/postgresql/data
```

**TimescaleDB Extension** (OSS Apache-2.0) — nur wenn du Intraday-Tick-Daten in Postgres schiebst. Für EOD ist reines Postgres 16 ausreichend.

**Achtung:** TimescaleDB Hypercore (columnar) hat seit 2024 eine kommerzielle Zusatz-Lizenz. Die **klassische Hypertable-Funktion bleibt OSS**.

---

## 12.6 DuckDB + Parquet als Feature-Store

**Das ist dein Feature-Store.** Feast ist Overkill, eigene SQL-Tabellen sind langsam.

**DuckDB 1.1.3** + Parquet + Hive-Partitioning:
```
features/
  view=rsi/
    year=2025/
      month=01/
        ticker=AAPL.parquet
```

**ASOF-JOIN-Pattern** (3-10× schneller als pandas `merge_asof`):
```python
import duckdb
con = duckdb.connect()
con.execute("""
  CREATE OR REPLACE VIEW fv_rsi AS
  SELECT * FROM read_parquet('features/view=rsi/**/*.parquet',
                             hive_partitioning=1)
""")
df = con.execute("""
  SELECT e.*, fv_rsi.rsi_14
  FROM entities e
  ASOF LEFT JOIN fv_rsi
    ON e.ticker = fv_rsi.ticker
    AND e.inference_ts - INTERVAL '1 minute' >= fv_rsi.available_at
""").df()
```

**Regel:** Jedes Feature speichert `available_at` (nicht `event_time`!), beim Join immer `available_at <= inference_time − embargo`. Das verhindert Look-Ahead-Bias strukturell.

**Install:** `pip install duckdb==1.1.3 pyarrow==17.0.0`

---

## 12.7 APScheduler

**Install:** `pip install apscheduler==3.10.4`

**Pattern für Scheduler-Daemon:**
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

# EOD-Pipeline jeden Werktag 16:15 ET
scheduler.add_job(run_eod_pipeline, 'cron',
                  day_of_week='mon-fri', hour=16, minute=15,
                  timezone='America/New_York')

# News-Poll alle 5 Minuten während Trading-Hours
scheduler.add_job(poll_news, 'cron',
                  minute='*/5', hour='9-16',
                  day_of_week='mon-fri',
                  timezone='America/New_York')

scheduler.start()
```

**Warum nicht Airflow/Prefect/Dagster:**
- Airflow: >2 GB RAM, Overkill für 10 Jobs.
- Prefect Hobby: free-Tier OK, aber Cloud-Abhängigkeit.
- Dagster+: Credits-Modell ab 01.05.2026 — Self-Host oder APScheduler wählen.

APScheduler ist für Solo **die richtige Antwort** bis du >50 Jobs hast.

---

## 12.8 MLflow (SQLite-Backend)

**Install:** `pip install mlflow==3.11.0`

**Setup ohne Cloud:**
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000
```

**Was du bekommst:**
- Experiment-Tracking (Runs, Metrics, Params, Artifacts)
- Model-Registry mit Aliases (seit 2.9: `@production`, `@canary` statt deprecated Stages)
- Artifact-Storage lokal oder S3-kompatibel

**SQLite-Limit:** ~10k Runs, dann auf Postgres wechseln. Reicht für Jahre Solo-Betrieb.

**Was NICHT nutzen:**
- **W&B Cloud:** Lock-in + Alpha-Signal-Confidentiality-Risiko.
- **Neptune/Aim:** Aim akzeptabel, aber MLflow ist Standard.

---

## 12.9 Docker + Docker-Compose

**Drei-Service-Layout für Solo:**

```yaml
version: '3.9'
services:
  app:
    build: .
    environment:
      - DATABASE_URL=postgresql://trading:${POSTGRES_PASSWORD}@postgres/trading
      - REDIS_URL=redis://redis:6379
    depends_on: [postgres, redis]
    ports: ["8000:8000"]

  worker:
    build: .
    command: python -m src.workers.main
    depends_on: [postgres, redis]

  scheduler:
    build: .
    command: python -m src.scheduler.main
    depends_on: [postgres, redis]

  postgres:
    image: postgres:16-alpine
    # ...

  redis:
    image: redis:7.4-alpine
    # ...

  caddy:
    image: caddy:2-alpine
    ports: ["80:80", "443:443"]
    volumes: ["./Caddyfile:/etc/caddy/Caddyfile"]
```

**Keine Kubernetes.** Keine Service-Mesh. Keine gRPC. Docker-Compose ist für Solo-Quant der richtige Level.

---

## 12.10 Caddy — Reverse-Proxy + Auto-HTTPS

**Install:** via Docker (oben).

**Caddyfile (10 Zeilen reichen):**
```
trading.deine-domain.de {
    reverse_proxy app:8000
    encode gzip
}

mlflow.deine-domain.de {
    reverse_proxy mlflow:5000
    basicauth {
        admin JDJhJDE0JG...  # bcrypt
    }
}
```

**Was du bekommst:** automatische Let's-Encrypt-Zertifikate, HTTP/2, gzip. **Keine manuelle SSL-Konfiguration mehr.**

**Alternative:** Traefik mit Docker-Label-Config — gleichermaßen gut, etwas mehr Komplexität.

---

## 12.11 Uptime Kuma + HealthChecks.io

**Uptime Kuma (self-hosted):**
```yaml
uptime-kuma:
  image: louislam/uptime-kuma:1
  volumes: ["./uptime-kuma:/app/data"]
  ports: ["3001:3001"]
```
- Liveness-Checks auf deine FastAPI-Endpoints.
- Status-Seite öffentlich oder privat.
- Notifications via Telegram, Discord, E-Mail.

**HealthChecks.io (Cloud-Free):**
- 20 Checks free.
- Für Cron-Job-Liveness (z.B. "EOD-Pipeline hat heute gelaufen").
- Python-Client: `pip install healthchecks-io-client`.

---

## 12.12 Grafana Cloud Free-Tier

**Das beste Free-Monitoring-Angebot 2026.**

**Limits:**
- 10k Active Series (Prometheus-Metriken)
- 50 GB Logs
- 50 GB Traces
- 14 Tage Retention
- **Keine Kreditkarte** nötig

**Setup:**
1. Account auf `grafana.com` — free.
2. Grafana Cloud Stack erstellen.
3. **Grafana Alloy Agent** (Nachfolger von Grafana Agent) auf Oracle-VM installieren.
4. OpenTelemetry-Export von FastAPI aktivieren.

**Was du bekommst:**
- Dashboards für Portfolio-P&L, Signal-IC, Order-Fills
- Alerts bei PSI-Drift, Kill-Switch-Trigger, Reconcile-Fail
- Log-Aggregation via Loki

**Alternative self-hosted:** Prometheus + Grafana auf Oracle-VM — braucht 2 GB RAM extra, nicht free beim Betrieb.

---

## 12.13 Sentry Free-Tier

**Limits:** 5k Errors/Monat.

**Install:** `pip install sentry-sdk[fastapi]`

**Setup:**
```python
import sentry_sdk
sentry_sdk.init(
    dsn=settings.sentry_dsn,
    traces_sample_rate=0.1,
    profiles_sample_rate=0.1,
)
```

**Für Solo-Trading-System absolut ausreichend.** Error-Tracking ist kritisch — eine einzige Python-Exception in der Order-Pipeline kann 4-stellige Verluste verursachen.

---

## 12.14 SOPS + age — Secrets-Management

**Die beste Solo-Secrets-Lösung 2026.**

**Install:**
```bash
# SOPS
brew install sops   # oder Windows: scoop install sops
# age
brew install age    # oder Windows: scoop install age
```

**Pattern:**
1. age-Key generieren: `age-keygen -o key.txt` → Public-Key in `.sops.yaml`.
2. `.env.sops.yaml` committen (verschlüsselt), `.env` plain nie committen.
3. In Python: `sops -d .env.sops.yaml | python-dotenv`.

**Vorteil gegenüber HashiCorp Vault:** Zero-Infra, Git-native, funktioniert lokal und in CI.

**HashiCorp Vault:** Overkill. HCP Vault Secrets wurde Juni 2025 sunsetted — nicht darauf bauen.
**Infisical Cloud:** Free-Tier nur 3 Members — für Solo unnötig.

---

## 12.15 Git + Pre-commit + gitleaks

**Pre-commit-Hooks** (Pflicht):

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.0
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.12.0
    hooks:
      - id: mypy
```

**gitleaks** verhindert, dass API-Keys versehentlich committed werden. **Absolute Pflicht** nach dem .env-Incident (siehe Teil 1 des Audits).

**detect-secrets-baseline** zusätzlich als Zweit-Gürtel.

---

## Beispiel: kompletter Stack auf Oracle Always-Free

```
Oracle A1.Flex VM (4 OCPU, 24 GB, ARM, Ubuntu 22.04)
├── Caddy (Port 80/443, Auto-HTTPS)
│   ├── trading.domain.de → FastAPI:8000
│   ├── mlflow.domain.de → MLflow:5000
│   └── uptime.domain.de → Uptime Kuma:3001
├── Docker-Compose:
│   ├── app (FastAPI + Uvicorn)
│   ├── worker (APScheduler + News-Pipeline)
│   ├── postgres (16-alpine)
│   ├── redis (7.4-alpine)
│   ├── mlflow (SQLite backend)
│   ├── uptime-kuma
│   └── grafana-alloy (exports to Grafana Cloud)
├── /data/parquet/ (Feature-Store)
├── /data/mlruns/ (MLflow Artifacts)
└── Backup: /data → Oracle Object-Storage (10 GB free)

Kosten: 0 EUR/Monat dauerhaft
Monitoring: Grafana Cloud Free
Error-Tracking: Sentry Free
Domain: Freenom oder 10-EUR/Jahr .de-Domain
```

---

## Umsetzungs-Checkliste

**Phase 0 — lokal:**
- [ ] Python 3.12 + `uv` installiert
- [ ] VSCode + Python-Extension konfiguriert
- [ ] Git mit Pre-commit-Hooks + gitleaks
- [ ] SOPS + age für Secrets
- [ ] Repo-Pfad ohne Umlaut/Leerzeichen

**Phase 1 — Core-Services lokal:**
- [ ] Docker-Compose mit Postgres + Redis + MLflow
- [ ] FastAPI läuft lokal mit Uvicorn
- [ ] APScheduler-Daemon mit ≥3 Jobs
- [ ] DuckDB + Parquet-Feature-Store-Skeleton

**Phase 2 — Cloud-Migration:**
- [ ] Oracle Always-Free Account + A1-Flex-VM
- [ ] Domain + Caddy + Auto-HTTPS
- [ ] Grafana Cloud Free-Account + Alloy-Agent
- [ ] Sentry-Account + SDK integriert
- [ ] Uptime Kuma für 5+ Endpoints
- [ ] HealthChecks.io für Cron-Jobs

**Phase 3 — Skalierung:**
- [ ] Backup-Strategie (Oracle Object-Storage)
- [ ] Disaster-Recovery-Playbook dokumentiert
- [ ] Multi-Service-Deployment via systemd oder Docker-Compose

---

## Ehrliche Einschätzung

Der Free-Infrastruktur-Stack ist **zu 100 %** für ein Solo-Quant-System geeignet. Oracle Always-Free ist eine ernsthafte Compute-Plattform (4 OCPUs + 24 GB sind mehr als manche Startups haben). Die 0-EUR-Grenze ist real einhaltbar, solange du keine Paid-Data-Feeds hinzunimmst.

**Wenn du upgradest (→ `22_PAID_INFRASTRUKTUR.md`), dann aus genau einem Grund:** Du willst Hetzner statt Oracle, weil du bessere Latenz zum US-Broker oder mehr Storage brauchst. Das sind dann 4-9 EUR/Monat — nicht dramatisch. Alles andere (Cloud-Monitoring, Secrets-Management, Deployment) bleibt free.
