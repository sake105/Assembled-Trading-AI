# 36 — Multi-Environment-Setup (Dev / Staging / Prod)

**Zweck:** Trennung der Laufzeit-Umgebungen, damit ein Bug im Entwicklungspfad nicht die Live-Positionen anfasst. Aktuell hat das Repo **ein** `.env`, **eine** Alpaca-API-Key-Pair, **eine** Postgres-Datenbank — alles mischt sich. Das ist der typische Single-Environment-Zustand und eines der gefährlichsten Anti-Patterns für ein System, das irgendwann echtes Geld bewegen soll.

**Scope:** Rang 4 aus der Gap-Analyse. Kann parallel zu Migration-Playbook (Rang 1) umgesetzt werden. Sollte **vor** jedem Live-Betrieb fertig sein.

**Kern-Idee:** Drei strikt getrennte Environments — Dev, Staging, Prod — mit getrennten Credentials, getrennten Datenbanken, getrennten Logs. Promotion von Dev zu Staging zu Prod erfolgt nur über definierte Gates.

---

## 0. Warum das wichtig ist — auch für einen Einzel-Quant

### Die naive Sichtweise: "Ich bin allein, ich brauche das nicht"

Typisches Argument: "Ich arbeite alleine, mein System läuft Paper, ich brauche keine Multi-Environment-Trennung."

**Das stimmt genau so lange, bis einer dieser fünf Fälle eintritt:**

1. **Du experimentierst im Live-Paper-System.** Du änderst schnell einen Parameter, testest, änderst wieder. Eine dieser Änderungen hat einen Bug. Das Paper-System verliert über Nacht 15 %.
2. **Du gehst zu echtem Geld über.** Jetzt ist derselbe Code, der gestern Bugs hatte, auf echten Dollar. Kein Puffer.
3. **Dein DB-Schema ändert sich.** Du migratest die einzige Postgres — Live-Positionen sind eine Migration später kaputt oder weg.
4. **Du hast einen Infinite-Loop-Bug.** Ein Script im Dev-Modus hämmert die Alpaca-API, dein API-Key wird rate-limited. Dein echtes Trading-System hat 15 Minuten keinen Marktzugang.
5. **Du bist im Urlaub und willst ein Update einspielen.** Du hast nur die Prod-Version auf dem Server. Jede Änderung ist Live-Risiko.

**Ohne Multi-Environment-Trennung ist jedes dieser Szenarien ein Katastrophen-Szenario.** Mit Trennung ist es ein "tritt im Dev auf, behoben vor Prod".

### Die Alpaca-Paper-Falle

Alpaca bietet Paper- und Live-Accounts. Man könnte meinen: "Paper = Dev, Live = Prod, fertig." Das ist **nicht genug**:

- **Paper-Account** ist nicht Dev. Paper ist deine **Staging**-Umgebung — realistische Simulation, aber du willst trotzdem nicht während einer Session rumspielen.
- **Dev** muss **komplett offline** sein. Keine Alpaca-API-Calls, keine Marktdaten-API-Calls. Mock-Responses aus Fixtures. Warum? Weil Dev-Experimente Endless-Loops produzieren können. Rate-Limit-Bans kosten dich keinen einzigen USD, aber sie knocken dich für Stunden aus dem Live-Betrieb.
- **Live** ist Prod. Kein Feature geht dahin, das nicht in Staging mindestens 7 Tage stabil lief.

### Die typische Regel: "Mehr als du denkst"

Erfahrene Trading-System-Betreiber haben typischerweise **5 Environments**, nicht 3:

| Env | Zweck | Alpaca | DB |
|---|---|---|---|
| `local_dev` | Entwickler-Laptop, offline Tests | Mock | SQLite in-memory |
| `ci_test` | CI-Pipeline, automatisierte Tests | Mock | SQLite / Postgres-Test-Container |
| `shadow` | Parallel zu Prod laufend, aber kein P&L | Paper | Postgres-Staging |
| `staging` | Finaler Acceptance-Test | Paper | Postgres-Staging |
| `prod` | Echtes Geld | Live | Postgres-Prod |

Für dich als Einzel-Quant können `local_dev` und `ci_test` zusammenfallen, ebenso `shadow` und `staging`. **Das Minimum ist 3**: dev, staging, prod.

---

## 1. Das Tool-Set

```bash
# Config-Management
uv pip install pydantic-settings==2.14.0   # Typsichere Config-Klassen
uv pip install python-dotenv==1.0.1         # .env-Dateien laden

# Secrets-Management (für Prod)
uv pip install keyring==25.6.0              # OS-Keychain-Zugriff (macOS Keychain, Windows Credential Manager)
uv pip install python-gnupg==0.5.3          # Optional: .env.prod.gpg verschlüsselt

# Container (empfohlen für Dev/Prod-Parität)
# Docker oder Podman. Auf deinem Linux-Hetzner: Docker.
```

**Versions-Stand:** April 2026. `pydantic-settings 2.14` ist der aktuelle Stand, released 2026-04-20.

---

## 2. Grundprinzip: Config-Klasse statt 30 `os.environ.get`

### 2.1 Das Anti-Pattern (das du wahrscheinlich hast)

Irgendwo im aktuellen Repo:

```python
# config.py (vermutlich so ähnlich)
import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
POSTGRES_URL = os.environ.get("POSTGRES_URL")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
RUN_MODE = os.environ.get("RUN_MODE", "paper")
MAX_POSITION_USD = float(os.environ.get("MAX_POSITION_USD", "1000"))
# ... 20+ weitere
```

**Probleme:**
- Kein Typ-Check. `MAX_POSITION_USD="foo"` crasht erst, wenn du versuchst zu handeln.
- Keine Pflicht-Check. `ALPACA_KEY=None` läuft durch, crasht später beim ersten API-Call.
- Keine Environment-Trennung. Ein .env für alles.
- Keine Validierung. `ALPACA_BASE_URL="not-a-url"` akzeptiert das System klaglos.

### 2.2 Das Ziel-Pattern mit `pydantic-settings`

```python
# src/assembled_core/config/settings.py
from enum import Enum
from pathlib import Path
from pydantic import Field, HttpUrl, PostgresDsn, RedisDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Strikt getrennte Laufzeit-Umgebungen."""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class TradingMode(str, Enum):
    """Was darf der Execution-Layer tun?"""
    MOCK = "mock"                # nur Fixture-basierte Antworten, keine API-Calls
    PAPER = "paper"              # Alpaca-Paper
    LIVE = "live"                # Alpaca-Live (Echtgeld)


class AlpacaSettings(BaseSettings):
    """Alpaca-spezifische Konfiguration. Nur für PAPER und LIVE benötigt."""
    api_key: SecretStr
    secret_key: SecretStr
    base_url: HttpUrl = Field(default="https://paper-api.alpaca.markets/")
    data_url: HttpUrl = Field(default="https://data.alpaca.markets/")

    model_config = SettingsConfigDict(env_prefix="ATA_ALPACA_")


class DatabaseSettings(BaseSettings):
    """Postgres-Verbindung, pro Environment getrennt."""
    url: PostgresDsn
    pool_size: int = Field(default=5, ge=1, le=50)
    max_overflow: int = Field(default=10, ge=0)

    model_config = SettingsConfigDict(env_prefix="ATA_DB_")


class RiskLimits(BaseSettings):
    """Harte Grenzen, die nie überschritten werden dürfen.
    
    Diese sind environment-spezifisch: in Dev willst du vielleicht 50$ max,
    in Prod 500$. Nie umgekehrt.
    """
    max_position_usd: float = Field(gt=0)
    max_daily_loss_usd: float = Field(gt=0)
    max_open_positions: int = Field(ge=1, le=100)
    kill_switch_loss_usd: float = Field(gt=0)

    @field_validator("kill_switch_loss_usd")
    @classmethod
    def kill_switch_gt_daily_loss(cls, v, info):
        daily = info.data.get("max_daily_loss_usd")
        if daily is not None and v <= daily:
            raise ValueError(
                f"kill_switch_loss ({v}) muss > max_daily_loss ({daily}) sein"
            )
        return v

    model_config = SettingsConfigDict(env_prefix="ATA_RISK_")


class Settings(BaseSettings):
    """Die zentrale Config-Klasse. Top-Level.
    
    Lädt aus der .env-Datei des entsprechenden Environments.
    """
    environment: Environment
    trading_mode: TradingMode
    
    # Nested settings
    alpaca: AlpacaSettings | None = None
    database: DatabaseSettings
    risk: RiskLimits
    
    # Observability
    log_level: str = Field(default="INFO")
    sentry_dsn: HttpUrl | None = None
    
    # Features
    enable_news_features: bool = Field(default=True)
    enable_shadow_mode: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=".env",                    # wird pro-env-Datei überschrieben
        env_file_encoding="utf-8",
        env_nested_delimiter="__",          # für nested: ATA_ALPACA__API_KEY
        case_sensitive=False,
        extra="forbid",                     # unbekannte Env-Vars → ValidationError
    )
    
    @field_validator("trading_mode")
    @classmethod
    def validate_mode_matches_environment(cls, v, info):
        """Dev darf NIE live traden. Prod darf NIE mocken."""
        env = info.data.get("environment")
        if env == Environment.DEV and v == TradingMode.LIVE:
            raise ValueError("Dev-Environment darf niemals trading_mode=live nutzen")
        if env == Environment.PROD and v == TradingMode.MOCK:
            raise ValueError("Prod-Environment mit trading_mode=mock ist sinnlos")
        return v
```

**Schlüssel-Details:**

- `SecretStr` statt `str` für API-Keys → werden bei `print(settings)` als `**********` angezeigt, Leak in Logs fast unmöglich.
- `extra="forbid"` → unbekannte Env-Vars (Tippfehler wie `ATA_ALPACE_API_KEY`) crashen beim Start, nicht beim ersten Alpaca-Call.
- `field_validator` für Invariants: Kill-Switch > Daily-Loss, Dev ≠ Live.
- `env_prefix` pro Section verhindert Namenskollisionen.

### 2.3 Der Setup-Loader

```python
# src/assembled_core/config/__init__.py
import os
from functools import lru_cache
from pathlib import Path
from .settings import Settings, Environment

REPO_ROOT = Path(__file__).parent.parent.parent.parent
ENV_DIR = REPO_ROOT / "config" / "env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Lädt die Config-Datei für das aktive Environment.
    
    ATA_ENVIRONMENT muss gesetzt sein (dev/staging/prod).
    Default: dev.
    """
    env_name = os.environ.get("ATA_ENVIRONMENT", "dev").lower()
    
    if env_name not in {"dev", "staging", "prod"}:
        raise RuntimeError(
            f"Unbekanntes ATA_ENVIRONMENT={env_name}. "
            f"Erlaubt: dev, staging, prod."
        )
    
    env_file = ENV_DIR / f".env.{env_name}"
    if not env_file.exists():
        raise RuntimeError(
            f"Config-Datei {env_file} existiert nicht. "
            f"Kopiere aus config/env/.env.{env_name}.template."
        )
    
    return Settings(_env_file=env_file)


# Für Dependency-Injection in Tests
def override_settings(**kwargs) -> Settings:
    """Nur für Tests. Erzeugt neue Settings mit Overrides."""
    base = get_settings().model_dump()
    base.update(kwargs)
    get_settings.cache_clear()
    return Settings(**base)
```

---

## 3. Die Verzeichnisstruktur

```
assembled-trading-ai/
├── config/
│   ├── env/
│   │   ├── .env.dev.template          # committed, ohne Secrets
│   │   ├── .env.staging.template      # committed
│   │   ├── .env.prod.template         # committed
│   │   ├── .env.dev                   # gitignored, persönlich
│   │   ├── .env.staging               # gitignored, nur Hans
│   │   └── .env.prod                  # gitignored, in Secret-Manager
│   ├── schemas/
│   │   └── db_schema.sql              # Pro-Env anwendbar
│   └── promotion_gates.yaml           # Staging→Prod Checklist
├── src/
│   └── assembled_core/
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       └── ...
├── scripts/
│   └── env/
│       ├── bootstrap_dev.sh           # Anfangs-Setup Dev
│       ├── bootstrap_staging.sh
│       ├── promote_to_staging.sh      # Dev → Staging Promotion
│       └── promote_to_prod.sh         # Staging → Prod Promotion
└── .gitignore
```

### 3.1 Die `.env.dev.template`-Datei (committed)

```bash
# config/env/.env.dev.template
# ================================================
# DEV Environment — keine echten Credentials
# Kopiere zu .env.dev und fülle Werte aus.
# ================================================

ATA_ENVIRONMENT=dev
ATA_TRADING_MODE=mock              # NIE live im Dev!

# Alpaca — bleibt leer im Dev-Mock-Modus
# ATA_ALPACA__API_KEY=
# ATA_ALPACA__SECRET_KEY=
# ATA_ALPACA__BASE_URL=

# Database — lokal, SQLite ist auch ok für Dev
ATA_DB__URL=postgresql://ata_dev:ata_dev_pw@localhost:5432/ata_dev
ATA_DB__POOL_SIZE=2

# Risk Limits — kleine Zahlen im Dev, damit Bugs nicht explodieren
ATA_RISK__MAX_POSITION_USD=100
ATA_RISK__MAX_DAILY_LOSS_USD=50
ATA_RISK__MAX_OPEN_POSITIONS=5
ATA_RISK__KILL_SWITCH_LOSS_USD=100

# Observability
ATA_LOG_LEVEL=DEBUG
# ATA_SENTRY_DSN=                   # kein Sentry im Dev

# Features
ATA_ENABLE_NEWS_FEATURES=true
ATA_ENABLE_SHADOW_MODE=false
```

### 3.2 Die `.env.staging.template`

```bash
# config/env/.env.staging.template
# ================================================
# STAGING — Alpaca Paper, echte Marktdaten, kein echtes Geld
# ================================================

ATA_ENVIRONMENT=staging
ATA_TRADING_MODE=paper

# Alpaca — EIGENER Paper-Account (nicht derselbe wie Dev falls Dev auch mal Paper)
ATA_ALPACA__API_KEY=<FILL_ME_IN>
ATA_ALPACA__SECRET_KEY=<FILL_ME_IN>
ATA_ALPACA__BASE_URL=https://paper-api.alpaca.markets/
ATA_ALPACA__DATA_URL=https://data.alpaca.markets/

# Database — eigene Staging-DB
ATA_DB__URL=postgresql://ata_staging:<PW>@staging-db.example.com:5432/ata_staging
ATA_DB__POOL_SIZE=5

# Risk Limits — moderat
ATA_RISK__MAX_POSITION_USD=2000
ATA_RISK__MAX_DAILY_LOSS_USD=500
ATA_RISK__MAX_OPEN_POSITIONS=10
ATA_RISK__KILL_SWITCH_LOSS_USD=1000

# Observability
ATA_LOG_LEVEL=INFO
ATA_SENTRY_DSN=<FILL_ME_IN>

# Features
ATA_ENABLE_NEWS_FEATURES=true
ATA_ENABLE_SHADOW_MODE=true         # Shadow-Mode an, um Prod-Vorbereitung zu testen
```

### 3.3 Die `.env.prod.template`

```bash
# config/env/.env.prod.template
# ================================================
# PROD — ECHTES GELD. Vorsicht.
# ================================================

ATA_ENVIRONMENT=prod
ATA_TRADING_MODE=live

# Alpaca — LIVE Account
ATA_ALPACA__API_KEY=<FROM_SECRETS_MANAGER>
ATA_ALPACA__SECRET_KEY=<FROM_SECRETS_MANAGER>
ATA_ALPACA__BASE_URL=https://api.alpaca.markets/
ATA_ALPACA__DATA_URL=https://data.alpaca.markets/

# Database — Prod-DB, separate Maschine
ATA_DB__URL=postgresql://ata_prod:<PW>@prod-db.example.com:5432/ata_prod
ATA_DB__POOL_SIZE=10

# Risk Limits — real, aber konservativ für Start
ATA_RISK__MAX_POSITION_USD=5000
ATA_RISK__MAX_DAILY_LOSS_USD=1000
ATA_RISK__MAX_OPEN_POSITIONS=15
ATA_RISK__KILL_SWITCH_LOSS_USD=3000

# Observability
ATA_LOG_LEVEL=INFO
ATA_SENTRY_DSN=<FROM_SECRETS_MANAGER>

# Features
ATA_ENABLE_NEWS_FEATURES=true
ATA_ENABLE_SHADOW_MODE=false       # Prod ist kein Shadow mehr
```

### 3.4 `.gitignore` — kritisch

```
# .gitignore
# NIE .env-Dateien committen, nur Templates
config/env/.env.dev
config/env/.env.staging
config/env/.env.prod
config/env/.env.*.local

# Ausnahmen: Templates sind ok
!config/env/.env.*.template
```

**Pre-commit-Hook verifiziert das:**

```yaml
# .pre-commit-config.yaml (relevanter Teil)
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

---

## 4. Datenbank-Separation

### 4.1 Drei Databases, keine Abkürzungen

```
Dev Postgres:      localhost:5432/ata_dev
Staging Postgres:  staging-db.hetzner:5432/ata_staging
Prod Postgres:     prod-db.hetzner:5432/ata_prod
```

**Nein**, eine einzige Postgres-Instanz mit drei Datenbanken ist **nicht** ausreichend. Ein falscher Befehl (`DROP DATABASE ata_prod`) oder eine Migration-Race-Condition und alles ist weg.

**Ja**, auf einer einzigen Hetzner-Maschine kannst du zwei Postgres-Instanzen auf verschiedenen Ports laufen lassen (5432, 5433, 5434), als Docker-Container. Das ist billig und separat genug.

### 4.2 Migrations mit Alembic pro Environment

```bash
# Dev: Migration ausprobieren
ATA_ENVIRONMENT=dev alembic upgrade head

# Staging: nach 7 Tagen Dev-Stabilität
ATA_ENVIRONMENT=staging alembic upgrade head

# Prod: nur über promote_to_prod.sh
```

**Goldene Regel:** Eine Migration, die in Dev nicht 24h stabil gelaufen ist, geht nicht in Staging. Eine Migration, die in Staging nicht 7 Tage gelaufen ist, geht nicht in Prod.

### 4.3 Dump-Strategie

```bash
# scripts/env/backup_prod.sh
#!/usr/bin/env bash
# Nächtlich per Cron, Dump Prod-DB
set -euo pipefail

BACKUP_DIR="/var/backups/ata"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Prod-Dump
pg_dump -U ata_prod -h prod-db -d ata_prod \
    --format=custom \
    --file="${BACKUP_DIR}/prod_${TIMESTAMP}.dump"

# Ältere Dumps rotieren (30 Tage behalten)
find "${BACKUP_DIR}" -name "prod_*.dump" -mtime +30 -delete
```

**Wichtig:** Der Dump geht auf eine separate Maschine, nicht auf dieselbe wie die Prod-DB.

---

## 5. Secrets-Management

### 5.1 Die drei Levels

| Level | Lokation | Zugriff |
|---|---|---|
| Dev | `.env.dev` auf Laptop | nur Hans lokal |
| Staging | `.env.staging` in OS-Keychain | Hans via `keyring` |
| Prod | Nur auf Prod-Server | via Environment-Variablen beim systemd-Start |

### 5.2 OS-Keychain-Integration

```python
# scripts/env/load_secrets.py
"""
Lädt Staging/Prod-Secrets aus OS-Keychain statt aus Datei.
Build im Prompt-Interaktion, einmalig beim Setup ausgeführt.
"""
import keyring
import getpass

SERVICE = "assembled-trading-ai"

def store_secret(env: str, key: str):
    """Einmalig: speichere ein Secret in Keychain."""
    value = getpass.getpass(f"{env.upper()} {key}: ")
    keyring.set_password(SERVICE, f"{env}:{key}", value)
    print(f"Stored {env}:{key} in keychain")

def get_secret(env: str, key: str) -> str:
    """Lese Secret aus Keychain."""
    value = keyring.get_password(SERVICE, f"{env}:{key}")
    if value is None:
        raise RuntimeError(f"Secret {env}:{key} nicht in Keychain gefunden")
    return value

# Nutzung beim Start:
# ATA_ALPACA__API_KEY=$(python scripts/env/load_secrets.py staging alpaca_api_key)
```

**Vorteil:** Selbst wenn dein Laptop kompromittiert wird, braucht der Angreifer zusätzlich das Keychain-Master-Passwort.

### 5.3 Systemd-Service für Prod

```ini
# /etc/systemd/system/ata-trader.service
[Unit]
Description=Assembled Trading AI (Prod)
After=network.target postgresql.service

[Service]
Type=simple
User=ata
WorkingDirectory=/opt/ata
Environment="ATA_ENVIRONMENT=prod"
EnvironmentFile=/etc/ata/prod.env      # chmod 600, nur root lesbar
ExecStart=/opt/ata/venv/bin/python -m assembled_core.run
Restart=on-failure
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

**Die Prod-Secrets liegen in `/etc/ata/prod.env` mit `chmod 600`**, nur vom root lesbar. Der `ata`-User bekommt sie über `EnvironmentFile` zum Startzeitpunkt.

**Warum keine Secret-Manager wie Vault?** Für Einzel-Quant-Setup zu schwer. Vault rentiert sich ab 5+ Services. Für dich reicht File + chmod 600.

---

## 6. Der Promotion-Workflow

### 6.1 Dev → Staging

```bash
# scripts/env/promote_to_staging.sh
#!/usr/bin/env bash
# Gates: alle müssen grün sein
set -euo pipefail

echo "=== Promote Dev → Staging ==="

# 1. Alle Tests grün
echo "[1/6] Running test suite..."
pytest tests/ -v --tb=short
if [ $? -ne 0 ]; then
    echo "FAIL: Tests not green"
    exit 1
fi

# 2. Characterization Tests grün
echo "[2/6] Running characterization tests..."
pytest tests/characterization/ -v
if [ $? -ne 0 ]; then
    echo "FAIL: Characterization tests not green"
    exit 1
fi

# 3. Linter sauber
echo "[3/6] Running linter..."
ruff check src/ tests/
if [ $? -ne 0 ]; then
    echo "FAIL: Linting issues"
    exit 1
fi

# 4. Type check
echo "[4/6] Running mypy..."
mypy src/ --ignore-missing-imports
if [ $? -ne 0 ]; then
    echo "FAIL: Type errors"
    exit 1
fi

# 5. Architektur-Check
echo "[5/6] Running tach..."
tach check
if [ $? -ne 0 ]; then
    echo "FAIL: Architecture violations"
    exit 1
fi

# 6. Commit-Check
echo "[6/6] Verifying clean working tree..."
if [ -n "$(git status --porcelain)" ]; then
    echo "FAIL: Uncommitted changes"
    exit 1
fi

# Alles grün: Tag erzeugen
SHA=$(git rev-parse HEAD)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TAG="staging-${TIMESTAMP}-${SHA:0:7}"

git tag -a "$TAG" -m "Promoted to staging: $TIMESTAMP"
git push origin "$TAG"

echo ""
echo "=== PROMOTION APPROVED ==="
echo "Tag: $TAG"
echo ""
echo "Deploy to Staging:"
echo "  ssh staging@staging-host"
echo "  cd /opt/ata && git fetch --tags && git checkout $TAG"
echo "  ATA_ENVIRONMENT=staging ./scripts/deploy.sh"
```

**Die 6 Gates:**
1. **Alle Unit/Integration-Tests**
2. **Alle Characterization-Tests** (aus `35_GOLDEN_EQUITY_SCENARIO_TESTS.md`)
3. **Linter** sauber (ruff)
4. **Type-Checker** sauber (mypy)
5. **Architektur-Check** (tach, siehe `60_MIGRATION_PLAYBOOK.md`)
6. **Git-Tree** sauber (kein "quick fix" ungetracked)

### 6.2 Staging → Prod

Dies ist der gefährlichste Schritt. Hier gehen **echtes Geld** in die Verantwortung.

```bash
# scripts/env/promote_to_prod.sh
#!/usr/bin/env bash
set -euo pipefail

echo "=== Promote Staging → PROD ==="
echo "ACHTUNG: Dies betrifft echtes Geld."
echo ""

# 1. Muss als Staging-Tag existieren
read -p "Staging-Tag (z.B. staging-20260424_143022-a1b2c3d): " STAGING_TAG
if ! git rev-parse "$STAGING_TAG" > /dev/null 2>&1; then
    echo "FAIL: Tag $STAGING_TAG existiert nicht"
    exit 1
fi

# 2. Mindestlaufzeit in Staging: 7 Tage
TAG_DATE=$(git log -1 --format=%at "$STAGING_TAG")
NOW=$(date +%s)
AGE_DAYS=$(( (NOW - TAG_DATE) / 86400 ))
if [ $AGE_DAYS -lt 7 ]; then
    echo "FAIL: Staging-Tag nur $AGE_DAYS Tage alt. Minimum: 7 Tage."
    exit 1
fi

# 3. Staging-Monitoring prüfen
echo "Staging-Laufzeit: $AGE_DAYS Tage"
echo ""
read -p "Staging läuft ohne Alerts/Incidents? [y/N]: " STAGING_OK
if [ "$STAGING_OK" != "y" ]; then
    echo "Abort."
    exit 1
fi

# 4. Scenario-Tests gegen Staging-Daten
echo "Running scenario tests against staging snapshot..."
ATA_ENVIRONMENT=staging pytest tests/characterization/test_scenarios.py
if [ $? -ne 0 ]; then
    echo "FAIL: Scenario tests against staging failed"
    exit 1
fi

# 5. Manuelle Checkliste
echo ""
echo "=== MANUAL CHECKLIST ==="
echo "Beantworte mit 'ja' für jedes Item:"

CHECKS=(
    "Alle Risk-Limits für Prod sind in .env.prod gesetzt"
    "Kill-Switch ist aktiv und getestet"
    "Backup der aktuellen Prod-DB existiert (<24h alt)"
    "Rollback-Plan ist vorbereitet (alter Git-Tag notiert)"
    "Sentry-DSN für Prod ist konfiguriert"
    "Pager-Duty/Telegram-Bot für Alerts ist konfiguriert"
    "Geldbetrag für erste 7 Tage ist begrenzt (max_daily_loss)"
    "Test-Order (1 Share) wurde vorbereitet für Go-Live-Verification"
)

for check in "${CHECKS[@]}"; do
    read -p "  [ ] $check — ja? " RESP
    if [ "$RESP" != "ja" ]; then
        echo "Abort: Checklist-Item nicht erfüllt."
        exit 1
    fi
done

# 6. Erstelle Prod-Tag
SHA=$(git rev-parse "$STAGING_TAG")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROD_TAG="prod-${TIMESTAMP}-${SHA:0:7}"
git tag -a "$PROD_TAG" -m "Promoted to PROD from $STAGING_TAG"
git push origin "$PROD_TAG"

echo ""
echo "=== PROD PROMOTION TAGGED ==="
echo "Tag: $PROD_TAG"
echo ""
echo "Nächste Schritte:"
echo "  1. ssh prod@prod-host"
echo "  2. cd /opt/ata"
echo "  3. git fetch --tags && git checkout $PROD_TAG"
echo "  4. Verify: ATA_ENVIRONMENT=prod python -c 'from assembled_core.config import get_settings; print(get_settings().trading_mode)'"
echo "  5. systemctl restart ata-trader"
echo "  6. Monitor logs: journalctl -u ata-trader -f"
echo "  7. Test-Order nach 5 Minuten Live-Zeit"
```

**Die 8-Item-Checkliste ist nicht übertrieben.** Jedes Item verhindert eine spezifische Art von Desaster:
- Risk-Limits: verhindert Runaway-Loss
- Kill-Switch-Test: verhindert "Switch funktioniert nicht, wenn man ihn braucht"
- Backup: verhindert Datenverlust bei Migration
- Rollback-Plan: verhindert Minuten-Panik bei Problemen
- Sentry: verhindert blinde Flug-Ops
- Pager: verhindert 24h-Unentdeckte-Probleme
- Max-Daily-Loss: verhindert Katastrophen-Verlust
- Test-Order: verhindert "läuft, aber keine Orders gehen raus"

---

## 7. Entwicklungs-Workflow

### 7.1 Eine normale Feature-Session

```bash
# Montag: Feature entwickeln
export ATA_ENVIRONMENT=dev
# Arbeite in src/ — alle Tests laufen gegen Dev-DB, Mock-Alpaca

pytest tests/ -v
# ... bis grün

git commit -m "feat: add new signal XYZ"
git push origin main
```

### 7.2 Freitag: Staging-Promotion

```bash
./scripts/env/promote_to_staging.sh
# Durchläuft 6 Gates, taggt bei Erfolg

# SSH auf Staging-Server
ssh staging
cd /opt/ata
git fetch --tags && git checkout staging-20260426_170000-f4c2a1b
systemctl restart ata-staging
journalctl -u ata-staging -f
# Lass Staging übers Wochenende laufen mit Alpaca-Paper
```

### 7.3 Nächsten Freitag: Prod-Promotion

```bash
# Nach 7 Tagen Staging-Beobachtung
./scripts/env/promote_to_prod.sh
# Gates + 8-Item-Checkliste
```

**Das entspricht einer natürlichen Wochen-Kadenz:** Feature entwickeln Montag-Donnerstag, Freitag ins Staging, nächsten Freitag in Prod. Falls Probleme auftauchen, genug Zeit zum Reagieren.

---

## 8. Feature-Flags für schrittweises Rollout

### 8.1 Warum Flags

Manchmal willst du ein Feature in Prod **haben**, aber **nicht sofort aktivieren**. Beispiel: neuer Signal-Type, der in Staging super funktioniert, aber in Prod erstmal Shadow-Mode laufen soll.

### 8.2 Simples Flag-System

```python
# src/assembled_core/features/flags.py
from dataclasses import dataclass
from ..config import get_settings


@dataclass
class FeatureFlags:
    """Schritte zum Rollout eines Features.
    
    off      → nicht ausgeführt
    shadow   → parallel ausgeführt, ignoriert
    canary   → für 10% der Signale aktiv
    on       → voll aktiv
    """
    news_sentiment_v2: str = "off"
    regime_ml_model: str = "shadow"
    news_topic_clustering: str = "canary"
    trend_baseline: str = "on"


def load_flags() -> FeatureFlags:
    """Flags sind environment-spezifisch, kommen aus Settings."""
    settings = get_settings()
    
    if settings.environment == "dev":
        # Dev: alles an, für Entwicklung
        return FeatureFlags(
            news_sentiment_v2="on",
            regime_ml_model="on",
            news_topic_clustering="on",
            trend_baseline="on",
        )
    elif settings.environment == "staging":
        # Staging: wie Prod geplant
        return FeatureFlags(
            news_sentiment_v2="shadow",
            regime_ml_model="shadow",
            news_topic_clustering="canary",
            trend_baseline="on",
        )
    else:  # prod
        return FeatureFlags(
            news_sentiment_v2="off",
            regime_ml_model="off",
            news_topic_clustering="shadow",
            trend_baseline="on",
        )
```

### 8.3 Nutzung

```python
flags = load_flags()

if flags.news_sentiment_v2 == "on":
    signal = run_news_sentiment_v2(data)
elif flags.news_sentiment_v2 == "shadow":
    shadow_signal = run_news_sentiment_v2(data)
    log_shadow_diff(shadow_signal, actual_signal)
    signal = actual_signal  # ignoriert v2
elif flags.news_sentiment_v2 == "canary":
    if hash(ticker) % 10 == 0:
        signal = run_news_sentiment_v2(data)
    else:
        signal = actual_signal
else:  # off
    signal = actual_signal
```

**Goldene Regel:** Neue Features in Prod starten immer als `shadow`, dann `canary`, dann `on`. Nie direkt `on`.

---

## 9. Das Environment-Banner im Log

Damit du nie versehentlich denkst du bist in Dev, wenn du in Prod bist:

```python
# src/assembled_core/__init__.py
import logging
from .config import get_settings

def emit_startup_banner():
    settings = get_settings()
    
    banner_char = {"dev": "_", "staging": "-", "prod": "!"}[settings.environment]
    banner_width = 60
    
    lines = [
        banner_char * banner_width,
        f"  Assembled-Trading-AI",
        f"  Environment: {settings.environment.upper()}",
        f"  Trading Mode: {settings.trading_mode.upper()}",
        f"  Max Position: ${settings.risk.max_position_usd:,.0f}",
        f"  Kill Switch: ${settings.risk.kill_switch_loss_usd:,.0f}",
        banner_char * banner_width,
    ]
    
    if settings.environment == "prod":
        logging.warning("\n" + "\n".join(lines))  # WARNING-Level für Prod
    else:
        logging.info("\n" + "\n".join(lines))
```

**Beim Start siehst du sofort:**

```
____________________________________________________________
  Assembled-Trading-AI
  Environment: DEV
  Trading Mode: MOCK
  Max Position: $100
  Kill Switch: $100
____________________________________________________________
```

Oder in Prod:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  Assembled-Trading-AI
  Environment: PROD
  Trading Mode: LIVE
  Max Position: $5,000
  Kill Switch: $3,000
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

Das `!!!`-Banner im Prod ist absichtlich aufdringlich. Du sollst es niemals übersehen.

---

## 10. Docker-Compose für lokale Parität

### 10.1 Warum Docker lokal

Die größte Gefahr ist "läuft auf meinem Laptop, läuft nicht auf dem Server". Mit Docker-Compose in Dev und denselben Images in Prod vermeidest du das.

### 10.2 `docker-compose.dev.yml`

```yaml
# docker-compose.dev.yml
version: "3.9"

services:
  postgres-dev:
    image: postgres:16.2
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: ata_dev
      POSTGRES_PASSWORD: ata_dev_pw
      POSTGRES_DB: ata_dev
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      - ./config/schemas/db_schema.sql:/docker-entrypoint-initdb.d/schema.sql

  redis-dev:
    image: redis:7.4
    ports:
      - "6379:6379"
    volumes:
      - redis_dev_data:/data

  trader-dev:
    build: .
    depends_on:
      - postgres-dev
      - redis-dev
    environment:
      ATA_ENVIRONMENT: dev
    env_file:
      - config/env/.env.dev
    volumes:
      - ./src:/app/src           # Dev: source mount für Live-Reload

volumes:
  postgres_dev_data:
  redis_dev_data:
```

### 10.3 `docker-compose.staging.yml`

```yaml
# docker-compose.staging.yml — auf Staging-Host
version: "3.9"

services:
  postgres-staging:
    image: postgres:16.2        # EXAKT dieselbe Version wie Dev + Prod
    ports:
      - "5433:5432"              # anderer Host-Port, damit Dev und Staging parallel laufen
    environment:
      POSTGRES_USER: ata_staging
      POSTGRES_PASSWORD_FILE: /run/secrets/pg_staging_pw
      POSTGRES_DB: ata_staging
    secrets:
      - pg_staging_pw
    volumes:
      - postgres_staging_data:/var/lib/postgresql/data

  trader-staging:
    image: ata-trader:${STAGING_TAG}    # KEIN source mount, fixes Image
    environment:
      ATA_ENVIRONMENT: staging
    env_file:
      - /etc/ata/staging.env
    restart: on-failure

volumes:
  postgres_staging_data:

secrets:
  pg_staging_pw:
    file: /etc/ata/pg_staging_pw.txt
```

**Schlüsselunterschied:** Staging nutzt ein festes Image-Tag (`ata-trader:${STAGING_TAG}`), kein Source-Mount. Dev nutzt Mount für schnelle Iteration.

---

## 11. Umsetzungs-Checkliste

**Phase 1 — Config-Refactor (Woche 1):**
- [ ] `pydantic-settings` installieren
- [ ] `Settings`-Klasse mit nested Alpaca/Database/Risk erstellen
- [ ] `field_validator` für Dev≠Live und Kill-Switch-Invariants
- [ ] Bestehende `os.environ.get`-Stellen auf `get_settings()` umstellen
- [ ] Unit-Tests, die Settings-Validation prüfen

**Phase 2 — Dateistruktur (Woche 1):**
- [ ] `config/env/` mit allen 3 Templates
- [ ] `.gitignore` Update, detect-secrets pre-commit hook
- [ ] `scripts/env/load_secrets.py` für keyring

**Phase 3 — DB-Separation (Woche 2):**
- [ ] 2 Postgres-Instanzen auf Hetzner (Dev + Staging) oder lokal Dev + Staging remote
- [ ] Alembic-Setup pro Env
- [ ] Migration-Dry-Run in Dev, dann Staging

**Phase 4 — Promotion-Scripts (Woche 2):**
- [ ] `promote_to_staging.sh` mit 6 Gates
- [ ] `promote_to_prod.sh` mit 8-Item-Checkliste
- [ ] Erste Promotion von Dev → Staging durchspielen

**Phase 5 — Deployment (Woche 3):**
- [ ] Docker-Images bauen
- [ ] systemd-Service für Prod vorbereiten (ohne zu starten)
- [ ] Staging-Dauerbetrieb 7 Tage laufen lassen

**Phase 6 — Prod-Go-Live (Woche 4):**
- [ ] Alle Checklisten-Items grün
- [ ] Erste Prod-Go-Live-Session mit sehr kleinen Risk-Limits
- [ ] 24h Beobachtung, dann Normalbetrieb

**Gesamt-Aufwand:** 3-4 Wochen bei 10-15 h/Woche.

---

## 12. Quellen

**12-Factor-App:**
- [The Twelve-Factor App](https://12factor.net/) — Adam Wiggins, Heroku (Original)
- [12factor.net/config](https://12factor.net/config)
- [12factor.net/dev-prod-parity](https://12factor.net/dev-prod-parity)
- Pradeep Loganathan (2025): [12 Factor App — Modern Tools](https://pradeepl.com/blog/12-factor-cloud-native-apps/)
- pyyne (2025): [Twelve-Factor App in 2025](https://www.pyyne.com/post/the-twelve-factor-app-methodology)
- DEV (2025): [12 Factor App Methodology](https://dev.to/adeleke123/12-factor-app-methodology-1e57)

**Pydantic Settings:**
- [pydantic-settings 2.14.0](https://pypi.org/project/pydantic-settings/)
- [Pydantic Docs: Settings Management](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- CodeCut (2025): [Pydantic-settings: Type-Safe Config Management](https://codecut.ai/pydantic-settings-type-safe-config-management/)
- FastAPI (2025): [Settings and Environment Variables](https://fastapi.tiangolo.com/advanced/settings/)

**Alpaca Paper/Live:**
- [Alpaca Paper Trading Docs](https://docs.alpaca.markets/docs/paper-trading)
- [Alpaca Learn: Start Paper Trading](https://alpaca.markets/learn/start-paper-trading)
- [alpaca-trade-api-python](https://github.com/alpacahq/alpaca-trade-api-python)

**Secrets-Management:**
- [python-keyring](https://github.com/jaraco/keyring)
- [detect-secrets (Yelp)](https://github.com/Yelp/detect-secrets)
- Docker Docs: [Secrets](https://docs.docker.com/engine/swarm/secrets/)

**Docker-Compose Dev/Prod Parity:**
- EKS Developers Workshop (2025): [Refactoring Python Apps Using Twelve-Factor Principles](https://developers.eksworkshop.com/docs/python/introduction/refactoring/)
- GeekCoding101 (2025): [12 Factor Crash Course in Python](https://geekcoding101.com/tech/system-design/12-factor-crash-course/)

---

## 13. Ehrliche Einschätzung

**Was dieses Playbook dir gibt:**
- Saubere Trennung zwischen Experimentieren und echtem Betrieb
- Typsichere Config mit automatischen Validators (Dev ≠ Live, Kill-Switch-Konsistenz)
- Promotion-Gates, die menschliche Fehler abfangen
- Paritäts-Garantie durch Docker zwischen lokalem Dev und Prod-Server

**Was es dir nicht gibt:**
- Hochverfügbarkeit — dein Prod läuft auf einer Maschine, die kann ausfallen. Für mehr bräuchtest du Kubernetes und das ist für Einzel-Quant Overkill
- Automatisches Rollback — du hast Git-Tags, aber Rollback ist manuell (`git checkout <prev-tag>`). Automatisch wäre möglich, aber komplex
- Zero-Downtime-Deploys — `systemctl restart` ist 5-30 Sekunden Downtime. Für Strategie, die stündlich handelt, ok; für HFT zu langsam

**Die drei Sachen, die du nicht auslassen darfst:**
1. **Die Dev≠Live-Validation.** `@field_validator` verhindert den häufigsten Fehler ("ich habe versehentlich Live-Keys im Dev geladen").
2. **Der Kill-Switch-Invariant.** Wenn `kill_switch` <= `daily_loss` ist, kann der Switch nie triggern. Garantierter Verlust, wenn eine Strategie ausufert.
3. **Der 7-Tage-Staging-Mindestlauf.** Viele Bugs brauchen Tage, bis sie auftauchen (z.B. Weekend-Handling, Month-End-Handling, Dividendenzahlungen). Ohne 7-Tage-Lauf fliegen dir die in Prod um die Ohren.

**Der wichtigste Punkt des ganzen Playbooks:** Multi-Environment ist keine technische Optimierung, sondern eine **psychologische**. Mit der Trennung **traust du dich, im Dev zu experimentieren**, weil du weißt: selbst wenn ich alles kaputt mache, Live läuft weiter. Ohne Trennung wirst du zu vorsichtig in Dev (verlangsamst Entwicklung) oder zu nachlässig in Live (katastrophales Risiko). Mit Trennung kriegst du beides: schnell in Dev, sicher in Prod.
