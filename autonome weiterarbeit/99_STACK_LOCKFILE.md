# 99 — Stack-Lockfile (pinned Python-Dependencies)

**Zweck:** Vollständige Liste aller Python-Libraries mit Versionen. Kopiere dies als `pyproject.toml` oder `requirements.txt`.

**Python-Version:** 3.12 (alle Libraries getestet).
**Package-Manager:** `uv` (Empfehlung) oder `pip`.

---

## pyproject.toml (Empfohlen)

```toml
[project]
name = "assembled-trading-ai"
version = "0.2.0"
requires-python = ">=3.12,<3.13"

dependencies = [
    # Core API & Config
    "fastapi==0.115.0",
    "uvicorn[standard]==0.30.0",
    "pydantic==2.9.0",
    "pydantic-settings==2.6.0",
    "httpx==0.27.2",
    
    # Daten & Broker
    "alpaca-py==0.34.1",
    "yfinance==0.2.50",
    "pandas-datareader==0.10.0",
    "eodhd==1.0.0",
    "finnhub-python==2.4.21",
    "openbb==4.7.0",
    "edgartools==5.22.0",
    "fredapi==0.5.2",
    
    # News & NLP
    "transformers==4.46.3",
    "sentence-transformers==3.3.1",
    "torch==2.5.1",
    "spacy==3.7.5",
    "gliner==0.2.13",
    "hdbscan==0.8.38",
    "hnswlib==0.8.0",
    "trafilatura==1.12.0",
    "fasttext-langdetect==1.0.5",
    "feedparser==6.0.11",
    "gdeltdoc==1.9.0",
    "praw==7.8.1",
    "atproto==0.0.46",  # Bluesky
    "mwviews==0.3.0",
    "anthropic==0.39.0",  # Claude Haiku (Paid)
    
    # ML Core
    "scikit-learn==1.5.2",
    "lightgbm==4.6.0",
    "xgboost==2.1.2",
    "skfolio==0.9.0",
    "mlfinpy==0.1.2",
    "shap==0.48.0",
    
    # Statistics & Time-Series
    "arch==7.2.0",  # GARCH
    "hmmlearn==0.3.3",
    "pykalman==0.9.7",
    "linearmodels==6.0",
    "statsmodels==0.14.4",
    "pingouin==0.5.5",
    "ruptures==1.1.10",
    
    # TA
    "ta-lib==0.6.8",
    "pandas-ta-classic==0.4.47",
    "talipp==2.5.0",
    
    # Pattern Recognition
    "stumpy==1.14.0",
    "tslearn==0.6.3",
    "dtaidistance==2.3.13",
    
    # Options
    "py_vollib==1.0.1",
    "py_vollib_vectorized==0.1.1",
    
    # Optimization & Calibration
    "optuna==4.7.0",
    "mapie==0.9.2",
    "cvxpy==1.8.2",
    "riskfolio-lib==7.2.1",
    
    # Data Engineering
    "numpy==2.1.3",
    "pandas==2.2.3",
    "polars==1.12.0",
    "pyarrow==17.0.0",
    "duckdb==1.1.3",
    
    # Ops & Infra
    "redis==5.2.0",
    "faststream==0.5.0",
    "apscheduler==3.10.4",
    "sqlalchemy==2.0.36",
    "alembic==1.14.0",
    "psycopg[binary]==3.2.3",
    
    # Experiment Tracking
    "mlflow==3.11.0",
    
    # Monitoring
    "sentry-sdk[fastapi]==2.17.0",
    "evidently==0.7.0",
    "prometheus-client==0.21.0",
    
    # Utilities
    "python-dotenv==1.0.1",
    "rich==13.9.4",
    "typer==0.13.0",
    "orjson==3.10.11",
    "loguru==0.7.2",
]

[project.optional-dependencies]
dev = [
    "pytest==8.3.3",
    "pytest-asyncio==0.24.0",
    "pytest-recording==0.13.4",
    "pytest-cov==5.0.0",
    "hypothesis==6.115.0",
    "ruff==0.7.0",
    "black==24.10.0",
    "mypy==1.12.1",
    "pre-commit==4.0.1",
    "ipython==8.28.0",
    "jupyterlab==4.2.5",
]

phase3 = [
    "pymc==5.16.2",  # optional für Bayesian-Research
    "neuralforecast==1.7.5",  # Phase 3 Neural-TS
    "fastdtw==0.3.4",  # Alternative zu dtaidistance
    "voyageai==0.2.3",  # Optional Embeddings-Upgrade
]

gnn = [
    "torch-geometric==2.5.3",  # Phase 3 Experiment
]

build-system = [
    "requires = ['hatchling']",
    "build-backend = 'hatchling.build'",
]
```

---

## requirements.txt (Alternative)

```
# Core API & Config
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.9.0
pydantic-settings==2.6.0
httpx==0.27.2

# Daten & Broker
alpaca-py==0.34.1
yfinance==0.2.50
pandas-datareader==0.10.0
eodhd==1.0.0
finnhub-python==2.4.21
openbb==4.7.0
edgartools==5.22.0
fredapi==0.5.2

# News & NLP
transformers==4.46.3
sentence-transformers==3.3.1
torch==2.5.1
spacy==3.7.5
gliner==0.2.13
hdbscan==0.8.38
hnswlib==0.8.0
trafilatura==1.12.0
fasttext-langdetect==1.0.5
feedparser==6.0.11
gdeltdoc==1.9.0
praw==7.8.1
atproto==0.0.46
mwviews==0.3.0
anthropic==0.39.0

# ML Core
scikit-learn==1.5.2
lightgbm==4.6.0
xgboost==2.1.2
skfolio==0.9.0
mlfinpy==0.1.2
shap==0.48.0

# Statistics & Time-Series
arch==7.2.0
hmmlearn==0.3.3
pykalman==0.9.7
linearmodels==6.0
statsmodels==0.14.4
pingouin==0.5.5
ruptures==1.1.10

# TA
ta-lib==0.6.8
pandas-ta-classic==0.4.47
talipp==2.5.0

# Pattern Recognition
stumpy==1.14.0
tslearn==0.6.3
dtaidistance==2.3.13

# Options
py_vollib==1.0.1
py_vollib_vectorized==0.1.1

# Optimization & Calibration
optuna==4.7.0
mapie==0.9.2
cvxpy==1.8.2
riskfolio-lib==7.2.1

# Data Engineering
numpy==2.1.3
pandas==2.2.3
polars==1.12.0
pyarrow==17.0.0
duckdb==1.1.3

# Ops & Infra
redis==5.2.0
faststream==0.5.0
apscheduler==3.10.4
sqlalchemy==2.0.36
alembic==1.14.0
psycopg[binary]==3.2.3

# Experiment Tracking
mlflow==3.11.0

# Monitoring
sentry-sdk[fastapi]==2.17.0
evidently==0.7.0
prometheus-client==0.21.0

# Utilities
python-dotenv==1.0.1
rich==13.9.4
typer==0.13.0
orjson==3.10.11
loguru==0.7.2
```

---

## Install-Commands

### Mit uv (empfohlen, ~10× schneller als pip)

```bash
# 1. uv installieren
pip install uv

# 2. Venv erstellen + Deps installieren
uv venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

uv pip install -e .
uv pip install -e ".[dev]"  # mit dev-Tools
```

### Mit pip (klassisch)

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## spaCy-Modell

Nach Install separat:

```bash
python -m spacy download en_core_web_lg
# 560 MB Download
```

**Nicht** `en_core_web_trf` — 20× langsamer, 10 GB RAM.

---

## Ollama (optional für lokale LLMs)

```bash
# Install von https://ollama.com (Windows-tauglich)
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull mistral:7b-instruct-q4_K_M
```

---

## Windows-spezifische Hinweise

**TA-Lib 0.6.8:** Pre-built Wheels für Python 3.10-3.14. `pip install TA-Lib` funktioniert jetzt ohne Visual-Studio-Build-Tools.

**LightGBM 4.6:** Pre-built Windows-Wheels verfügbar.

**cvxpy 1.8.2:** Auch auf Windows per pip-wheel. CLARABEL-Solver default.

**torch 2.5.1:**
- CPU-Version: `pip install torch==2.5.1`
- CUDA 12.1: `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`

---

## Linux/ARM-Hinweise (Oracle Always-Free)

**ARM64-Kompatibilität prüfen:**

```bash
# Test nach Install
python -c "
import lightgbm
import xgboost
import torch
import talib
import duckdb
print('✅ All critical libraries loaded')
"
```

**Bekannte ARM-Probleme:**
- TA-Lib: neuere Wheels (0.6+) haben ARM-Support, ältere müssen kompiliert werden
- hftbacktest: **kein** ARM-Wheel — bei ARM kein hftbacktest
- Manche Niche-Libraries (ruptures ARM-Wheel existiert 2026)

---

## Upgrade-Pfad

```bash
# Check für veraltete Packages
uv pip list --outdated

# Selective Updates
uv pip install --upgrade anthropic fastapi

# Lockfile generieren
uv pip freeze > requirements.lock.txt
```

**Empfehlung:** Monatlich `uv pip list --outdated` ausführen, Updates im Dev-Branch testen, dann mergen.

---

## Monats-Budget für Libraries

**0 EUR** — alle Libraries in diesem Lockfile sind Open-Source, lokal installierbar, keine Subscription.

Die einzige Ausnahme ist `anthropic` — die Library ist free, aber die API-Nutzung kostet (siehe `21_PAID_MODELLE.md` §21.1).

---

## Häufige Installations-Probleme

| Problem | Lösung |
|---|---|
| `Building wheel for TA-Lib ...failed` | Python 3.12 und `ta-lib>=0.6.8` erzwingen |
| `psycopg` Error beim Install | `psycopg[binary]` statt `psycopg` |
| `torch` Download sehr langsam | PyTorch-Mirror via `--index-url` |
| `lightgbm` Import-Error Mac | `brew install libomp` vor pip install |
| `cvxpy` ECOS-Error | `pip install clarabel` und ECOS ignorieren |
| `hmmlearn` ABI-Mismatch | `numpy==2.1.3` und `hmmlearn==0.3.3` zusammen neu installieren |

---

## Dependency-Konflikte vorbeugen

**Bekannte Konflikte in 2026:**

1. `mlfinlab` (Hudson & Thames) vs `mlfinpy` — **nicht beide gleichzeitig installieren**. `mlfinpy` is der Open-Source-Ersatz.
2. `pandas-ta` (original twopirllc) — **nicht installieren**. Verwende `pandas-ta-classic`.
3. `backtrader` — **deprecated seit 2018**. Nicht in Neuinstallationen.
4. `stellargraph` — seit >12 Monaten kein Release. **Nicht installieren.**

---

## Nächste Schritte nach Install

1. `python -m spacy download en_core_web_lg`
2. `pre-commit install` (falls dev-deps installiert)
3. `pytest` → alle Tests sollten passen (nach ersten Tests: Cassettes aufzeichnen via `pytest --record-mode=all`)
4. `mlflow ui` in separatem Terminal starten

---

## Wartungsplan

**Monatlich:**
- `uv pip list --outdated` prüfen
- Minor-Updates anwenden (z.B. 4.46.3 → 4.46.5)

**Quartalsweise:**
- Major-Updates prüfen (z.B. 4.46 → 4.47)
- Im Dev-Branch testen, Regression-Tests fahren

**Jährlich:**
- Python-Minor-Upgrade (3.12 → 3.13 nach Release)
- Major-Library-Upgrades (z.B. LightGBM 4.x → 5.x)
