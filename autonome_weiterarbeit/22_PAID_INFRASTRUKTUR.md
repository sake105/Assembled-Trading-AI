# 22 — Paid Infrastruktur (mit Kosten, <10 EUR/Monat für Hosting)

**Zweck:** Hosting-Optionen jenseits von Oracle Always-Free. Die wichtigste Entscheidung: **Hetzner vs. Oracle** — beide unter 10 EUR.

---

## Module

| # | Service | Kosten | Empfehlung |
|---|---|---|---|
| 22.1 | **Hetzner CX22** | **4.25 EUR/Monat** | **Primär-Empfehlung** |
| 22.2 | Hetzner CX32 | 7.67 EUR/Monat | Upgrade wenn Tier-2 voll läuft |
| 22.3 | Hetzner CPX21 | 7.05 EUR/Monat | Alternative mit mehr Storage |
| 22.4 | Hetzner Object Storage | 4.75 EUR/TB | Backups, MLflow-Artifacts |
| 22.5 | Domain + DNS | 0.8-1 EUR/Monat | Optional, aber empfohlen |
| 22.6 | Offsite-Backup (Backblaze B2) | ~1 EUR/Monat | Disaster-Recovery |

---

## 22.1 Hetzner CX22 — Primär-Empfehlung

**Specs:**
- 2 vCPU (AMD EPYC)
- 4 GB RAM
- 40 GB NVMe SSD
- 20 TB Traffic
- Standort: Nürnberg oder Falkenstein

**Preis:** 4.25 EUR/Monat (inkl. USt), 0.008 EUR/h.

**Warum CX22 statt Oracle Always-Free:**

| Kriterium | Hetzner CX22 | Oracle Always-Free A1 |
|---|---|---|
| Kosten | 4.25 EUR | 0 EUR |
| Architektur | **x86_64** | ARM64 |
| Latenz zu US-Broker | ~90ms | ~110ms |
| Storage | 40 GB NVMe | 200 GB (langsamer) |
| Availability-Guarantee | **99.9%** | keine |
| Library-Kompatibilität | **alle Python-Wheels** | nur ARM-Wheels |

**Entscheidungskriterium:**
- **Oracle** wenn: Budget absolut 0 EUR, ARM-Kompatibilität aller deiner Libs geprüft, Out-of-Capacity akzeptabel.
- **Hetzner** wenn: 4 EUR/Monat OK, Zuverlässigkeit wichtig, x86_64 einfacher.

**Für ein ernstzunehmendes Trading-System: Hetzner.**

### Setup-Schritte

```bash
# 1. Hetzner Cloud Console: VM erstellen
#    - Image: Ubuntu 24.04 LTS
#    - Type: CX22
#    - Network: IPv4 + IPv6
#    - SSH-Key generiert und hochgeladen

# 2. Initial-Hardening
ssh root@<IP>
adduser trading
usermod -aG sudo trading
# SSH-Key auch für 'trading' User einrichten
ufw allow 22
ufw allow 80
ufw allow 443
ufw enable

# 3. Disable root SSH + password auth
sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

# 4. Docker Install
curl -fsSL https://get.docker.com | sh
usermod -aG docker trading

# 5. fail2ban + automatic-updates
apt install -y fail2ban unattended-upgrades

# 6. swap für 4 GB RAM VM
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

---

## 22.2 Hetzner CX32 — für Phase-2-Expansion

**Specs:**
- 4 vCPU
- 8 GB RAM
- 80 GB NVMe SSD
- 20 TB Traffic

**Preis:** 7.67 EUR/Monat.

**Wann upgrade:**
- Tier-2-Universum voll aktiv (1800 Ticker)
- LightGBM-Training für 11 Sektor-Modelle parallel
- Postgres + Redis + FastAPI + 2 Worker + MLflow alles auf einer VM
- RAM-Usage regelmäßig >80 %

---

## 22.3 Hetzner CPX21 — Alternative

**Specs:**
- 3 vCPU (AMD, Performance)
- 4 GB RAM
- 80 GB NVMe SSD (doppelt so groß wie CX22)

**Preis:** 7.05 EUR/Monat.

**Wann:** Wenn 40 GB CX22-SSD für Parquet-Feature-Store zu knapp. Bei Tier-1-only (1.9 GB Feature-Storage) reicht CX22.

---

## 22.4 Hetzner Object Storage

**Preis:** 4.75 EUR/TB/Monat, S3-kompatibel.

**Verwendung:**
- MLflow-Artifacts (sonst lokaler Disk-Space)
- Parquet-Daten-Backup (Disaster-Recovery)
- Deep-Archive für historische Bars

**MLflow-Integration:**
```bash
# MLflow mit S3-compatiblem Backend
mlflow server \
  --backend-store-uri postgresql://... \
  --default-artifact-root s3://hetzner-storage-bucket/mlartifacts \
  --host 0.0.0.0
```

```python
# .env
import os
os.environ["AWS_ACCESS_KEY_ID"] = "hetzner-access-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "hetzner-secret"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://fsn1.your-objectstorage.com"
```

---

## 22.5 Domain + DNS

**Optionen:**

| Anbieter | Preis/Jahr | Bemerkung |
|---|---|---|
| **INWX** | 5-10 EUR | .de-Domain, deutsches Unternehmen |
| Cloudflare Registrar | At-Cost | International, z.B. .com ~9 USD/Jahr |
| Freenom | 0 EUR | .tk/.ml — **unzuverlässig, nicht empfehlen** |

**Empfehlung:** INWX .de für 8 EUR/Jahr = 0.67 EUR/Monat.

**DNS:** Cloudflare Free-Tier (unbegrenzt Records, Proxying, SSL-Edge).

**Caddy auf Hetzner** liefert Let's-Encrypt-Zertifikate gratis, Cloudflare-Proxy optional für DDoS-Protection.

---

## 22.6 Offsite-Backup (Backblaze B2)

**Preis:** 6 USD/TB/Monat (~5.50 EUR). Free-Tier bis 10 GB.

**Pattern:**
```bash
# restic mit B2-Backend für verschlüsselte Incremental-Backups
restic -r b2:assembled-trading-backups init
restic -r b2:assembled-trading-backups backup /data/parquet /data/mlruns

# Retention: 7 daily, 4 weekly, 6 monthly, 1 yearly
restic -r b2:assembled-trading-backups forget \
  --keep-daily 7 --keep-weekly 4 --keep-monthly 6 --keep-yearly 1 \
  --prune
```

**Cron:**
```
0 3 * * * /usr/local/bin/restic-backup.sh
```

**Kosten realistisch:** Bei Tier-1-only Setup (1.9 GB + MLflow-Runs) passt du in 10 GB Free-Tier.

---

## 22.7 Der realistische Deployment-Stack

```
┌─────────────────────────────────────────────────┐
│ Hetzner CX22 / CX32 (Ubuntu 24.04, Docker)      │
│   4-8 GB RAM, 40-80 GB SSD, 4.25-7.67 EUR/Mo    │
├─────────────────────────────────────────────────┤
│ Caddy (80/443, Auto-HTTPS)                       │
│   → app (FastAPI), mlflow, grafana-agent         │
├─────────────────────────────────────────────────┤
│ Docker-Compose:                                  │
│   • app (FastAPI + Uvicorn)                     │
│   • worker (APScheduler + News-Pipeline)        │
│   • postgres 16-alpine                          │
│   • redis 7.4-alpine                            │
│   • mlflow (Postgres-Backend)                   │
├─────────────────────────────────────────────────┤
│ Monitoring:                                      │
│   • Grafana Cloud Free (Metrics, Logs)          │
│   • Sentry Free (Errors)                        │
│   • Uptime Kuma self-hosted (optional)          │
├─────────────────────────────────────────────────┤
│ Backups:                                         │
│   • Hetzner Object Storage (Hot, 4.75 EUR/TB)   │
│   • Backblaze B2 (Offsite, 10 GB free)          │
└─────────────────────────────────────────────────┘
```

**Monatliche Kosten Infrastruktur:**

| Posten | Kosten |
|---|---|
| Hetzner CX22 | 4.25 EUR |
| Domain INWX | 0.67 EUR |
| Object Storage 10 GB | 0.05 EUR |
| B2-Offsite 10 GB | 0 EUR (Free-Tier) |
| **Summe Infrastruktur** | **~5 EUR/Monat** |

---

## 22.8 Optional Paid Monitoring (Skip in Phase 1-2)

**Datadog / New Relic:** 15-30 USD/Host/Monat. **Nicht nötig** bei Grafana Cloud Free.

**Papertrail / Loggly / Better Stack:** Log-Aggregation. Grafana Cloud 50 GB Logs reichen.

**PagerDuty:** 19 USD/User/Monat für Alerting. Grafana Cloud OnCall Free bis 5 User.

**Verdict:** Skip alle Paid-Monitoring. Grafana Cloud Free + Sentry Free + Uptime Kuma self-hosted ist ausreichend.

---

## 22.9 Was NICHT in diesem Katalog ist

**Explizit NICHT empfohlen:**

- **AWS/GCP/Azure:** 4-10× teurer als Hetzner bei gleichen Specs.
- **DigitalOcean:** ok, aber 6 USD/Monat für schwächere VMs als Hetzner 4.25 EUR.
- **Linode/Akamai:** ähnlich DigitalOcean.
- **Kubernetes (EKS/GKE):** Control-Plane-Kosten allein >70 USD/Monat. **Overkill** für Solo.
- **Heroku:** Eco-Dyno 7 USD/Monat, Postgres 9 USD, Redis 15 USD = 31 USD für weniger als Hetzner CX22.
- **Railway / Render:** Free-Tier zu limitiert, Paid-Tier teurer als Hetzner.
- **Fly.io:** Interessant, aber Pricing-Updates 2024 haben Attraktivität gemindert.

---

## Umsetzungs-Checkliste

**Phase 1 (lokal Windows):**
- [ ] System lokal mit Docker-Compose lauffähig
- [ ] SOPS-encrypted `.env` im Repo

**Phase 2 (Cloud-Migration):**
- [ ] Hetzner-Account + Cloud-API-Token
- [ ] CX22-VM aufgesetzt, Ubuntu 24.04
- [ ] UFW + fail2ban + disable root SSH
- [ ] Docker + Docker-Compose installiert
- [ ] Git-Deploy-Pfad (deploy via SSH+rsync oder Git-Pull-Hook)
- [ ] Caddyfile mit Auto-HTTPS für deine Domain
- [ ] systemd-Service oder `docker-compose up -d` als Persistenz
- [ ] Grafana-Alloy-Agent exportiert zu Grafana Cloud
- [ ] Sentry-DSN in FastAPI-App integriert
- [ ] restic-Backups zu Object Storage täglich 03:00

**Phase 3 (Upgrade bei Bedarf):**
- [ ] Hetzner CX32 wenn RAM knapp
- [ ] Backblaze B2 für Offsite wenn Daten >10 GB
- [ ] DNS über Cloudflare mit Proxying

---

## Ehrliche Einschätzung

**Für Solo-Quant:** Hetzner CX22 für 4.25 EUR/Monat ist unschlagbar. x86_64, 99.9% SLA, deutsche Datenzentren (DSGVO-konform), alle Python-Libraries laufen.

**Die einzige Alternative mit 0 EUR ist Oracle Always-Free A1-Flex mit 4 OCPU + 24 GB RAM.** Deutlich mehr Power als CX22, **aber** ARM-Kompatibilität und Out-of-Capacity in EU-Regionen sind reale Probleme.

**Praktischer Vorschlag:**
- Phase 1-2: Windows lokal (0 EUR Infrastruktur)
- Phase 2: Entscheide ob Oracle Free funktioniert (alle Libs, ARM-Wheels geprüft?) → wenn ja, bleib bei 0 EUR; wenn nein → Hetzner CX22 für 4.25 EUR
- Phase 3: Hetzner CX32 (7.67 EUR) wenn mehr RAM nötig

**Gesamt-Infrastruktur bleibt unter 8 EUR/Monat** selbst bei voller Auslastung.
