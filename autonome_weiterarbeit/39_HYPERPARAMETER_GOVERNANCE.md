# 39 — Hyperparameter-Governance

**Zweck:** Jedes ML-Modell, jede Strategie, jeder Composite-Score hat Hyperparameter. EMA-Perioden (20, 50), FinBERT-Schwellwerte (0.6), Position-Sizing-Parameter, Composite-Gewichte. Ohne Governance sind diese Werte im Code verteilt, dokumentationslos, unversioniert. Du weißt nicht, **welche Kombination** zu der Version geführt hat, die letzte Woche so gut lief.

**Scope:** Rang 7 aus der Gap-Analyse. Voraussetzung für reproduzierbare Experimente und jede Form von Modell-Iteration.

**Kern-Idee:** Hyperparameter werden als versionierte Artefakte behandelt — genauso wie Code oder Daten. Jeder Trainings-Lauf wird getrackt. Jede Live-Deploy-Konfiguration wird registriert und auditierbar.

---

## 0. Warum das wichtig ist — "Welche Version hat das gehandelt?"

### Das Szenario, das dich frustriert

Montag: du deployst Strategy v4. EMA-Short=20, FinBERT-Threshold=0.6, Composite-Gewicht-News=0.15.

Mittwoch: Strategy läuft gut, +2.3 %.

Freitag: du experimentierst im Dev, probierst EMA-Short=15 und FinBERT-Threshold=0.7 aus. Du pushst commit `a3f2b1c` mit diesen neuen Default-Werten in einem `config.yaml`.

Nächsten Montag: du willst zurück zum "guten Stand". Du weißt nicht mehr, welche Werte in Strategy v4 waren. Der alte Git-Commit hat die Werte, aber du hast auch die Modell-Files (`finbert_ft_v3.pkl`) lokal ersetzt. Du bist nicht mehr reproduzierbar.

**Diese Situation ist der Regelfall bei Hobby-Quants ohne Governance.** Mit Governance ist es ein 10-Sekunden-Job: "Restore config for deployment `strategy_v4_20260315_prod`".

### Die drei Ebenen der Parameter

Nicht alle Parameter sind gleichwertig:

1. **Code-Hyperparameters:** EMA-Perioden, Fenstergrößen, Standard-Quantile. Gehören als `Default` in Code.
2. **Run-Hyperparameters:** Gewichte des Composite-Scores, Thresholds, Klassifikator-Settings. Gehören in Config-Files, versioniert.
3. **Training-Hyperparameters:** ML-spezifisch, z.B. Learning-Rate, Batch-Size, Number of Trees. Gehören in Experiment-Tracker (MLflow).

**Fehler: alle in einen Topf werfen.** Entweder im Code gehardcoded (schlecht, weil unversioniert) oder alle in einer einzigen `settings.yaml` (unübersichtlich bei 50+ Parametern).

---

## 1. Die Tool-Entscheidung

### 1.1 Die Kandidaten (Stand 2026)

| Tool | Version | Primary-Use | Für dich geeignet? |
|---|---|---|---|
| **MLflow** | 2.17+ | Experiment-Tracking, Model-Registry | **Ja** — self-hostable, kostenlos, deckt alle 3 Ebenen |
| **Weights & Biases** | cloud | Experiment-Tracking, Visualisierung | Eventuell — kostenloses Tier für Einzel-Nutzer |
| **Neptune.ai** | cloud/SaaS | Metadata-Tracking | Eher nicht — SaaS-Focus, Pricing |
| **DVC** | 3.x | Data + Model Versioning | Komplementär, nicht alternativ |
| **Git + YAML** | n/a | Plain | Nur für Run-Hyperparameters |

### 1.2 Empfehlung für deinen Stack

**Primary: MLflow (self-hosted auf Hetzner).** Warum?

1. **Open-Source, kostenlos.** Läuft auf deiner Hetzner-Maschine als separater Service.
2. **Self-hosted = keine Vendor-Lock-in.** Wenn du später wechselst, kannst du deine Datenbank behalten.
3. **Deckt alle 3 Ebenen:** Tracking (Runs), Registry (Versioned Models), Projects (reproduzierbare Runs).
4. **Python-API ist minimal-invasiv.** `mlflow.log_param("ema_short", 20)` — mehr brauchst du oft nicht.

**Ergänzung: DVC (optional).** Für grosse Modell-Dateien (>100 MB) ist Git-LFS alternative zu DVC. MLflow Registry kann aber auch einfach Artefakte speichern. **Für dich reicht MLflow allein.**

**Ergänzung: Git-versioniertes YAML** für Run-Hyperparameters. Strategy-Configs als `configs/strategy_v4.yaml` in Git. Einfach, auditierbar, diff-bar.

**Nicht W&B.** Für Hobby-Quant ist die Cloud-Abhängigkeit unnötig, und bei sensitiven Trading-Daten willst du eh Self-hosted.

### 1.3 Installation

```bash
# MLflow
uv pip install mlflow==2.17.0
```

**Version 2.17 ist stable Stand April 2026.** Die 3.x-Serie ist noch im Beta.

---

## 2. Die Parameter-Klassifikation

### 2.1 Was gehört wohin

```
┌───────────────────────────────────────────────────┐
│  Code-Hyperparameters                              │
│  → Python-Code, als Default-Argumente              │
│  → Beispiele: min_observations=100, buffer_size=   │
│    1024, cache_ttl_seconds=300                     │
│  → Änderung = Code-Commit, nicht Config-Change     │
└───────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────┐
│  Run-Hyperparameters (Strategy-Config)             │
│  → YAML in configs/                                │
│  → Beispiele: composite_weights, FinBERT threshold,│
│    risk_per_trade_pct, regime_multipliers          │
│  → Änderung = YAML-Commit, versioniert             │
└───────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────┐
│  Training-Hyperparameters                          │
│  → MLflow-Runs + Model-Registry                    │
│  → Beispiele: n_estimators, max_depth, learning-   │
│    rate, batch_size                                │
│  → Änderung = neuer MLflow-Run mit neuen Params    │
└───────────────────────────────────────────────────┘
```

### 2.2 Beispiele

**Code-Hyperparameters (im Python-Code):**

```python
# src/assembled_core/features/ema.py
DEFAULT_EMA_SHORT = 20
DEFAULT_EMA_LONG = 50
DEFAULT_MIN_OBSERVATIONS = 100  # unter dieser Zahl Bars: kein Signal

def compute_ema_cross(
    prices: pd.Series,
    short: int = DEFAULT_EMA_SHORT,
    long: int = DEFAULT_EMA_LONG,
    min_obs: int = DEFAULT_MIN_OBSERVATIONS,
) -> pd.Series:
    ...
```

**Run-Hyperparameters (in YAML):**

```yaml
# configs/strategies/trend_news_v4.yaml
strategy_id: "trend_news_v4"
description: "Trend + News Composite, moderate Gewichtung News"
created: "2026-04-20"
author: "Hans"

composite_weights:
  trend: 0.25
  momentum: 0.15
  news: 0.15
  volatility: 0.10
  regime: 0.10
  sentiment: 0.08
  volume: 0.07
  microstructure: 0.05
  fundamentals: 0.05

thresholds:
  buy: 0.6
  sell: -0.6
  
regime_multipliers:
  bull: 1.0
  bear: 0.7
  neutral: 0.9

news:
  finbert_threshold_positive: 0.6
  finbert_threshold_negative: 0.6
  source_weights:
    reuters: 1.0
    bloomberg: 1.0
    seeking_alpha: 0.5
    
risk:
  max_position_pct_of_equity: 0.05
  max_daily_loss_pct: 0.02
  kill_switch_loss_pct: 0.06

model_versions:
  news_classifier: "news_clf_v3"    # MLflow Registry-Alias
  regime_classifier: "regime_rf_v2"
```

**Training-Hyperparameters (in MLflow gespeichert):**

```python
import mlflow

with mlflow.start_run(run_name="news_classifier_training_v3"):
    params = {
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 50,
        "random_state": 42,
        "training_data_sha": "a3f2b1c",
        "training_period_start": "2023-01-01",
        "training_period_end": "2025-12-31",
    }
    mlflow.log_params(params)
    
    model = train_news_classifier(**params)
    
    metrics = evaluate_model(model, test_data)
    mlflow.log_metrics(metrics)
    
    mlflow.sklearn.log_model(model, "model", registered_model_name="news_classifier")
```

---

## 3. MLflow-Setup auf Hetzner

### 3.1 Server-Installation

```bash
# Auf Hetzner-Box als separater systemd-Service
sudo useradd -m mlflow
sudo su - mlflow

# Python-Venv
python -m venv /home/mlflow/venv
source /home/mlflow/venv/bin/activate
pip install mlflow==2.17.0 psycopg2-binary boto3

# Datenordner
mkdir -p /home/mlflow/mlruns
mkdir -p /home/mlflow/artifacts
```

### 3.2 Backend-Store

MLflow speichert Metadata (Runs, Params, Metrics) in einer Datenbank. SQLite reicht für Einzel-Nutzer, aber Postgres ist robuster.

Wenn du schon Postgres für die Trading-DB hast (siehe `36_MULTI_ENVIRONMENT_SETUP.md`), nutze sie auch für MLflow. Separate Database.

```bash
sudo -u postgres psql
CREATE USER mlflow WITH PASSWORD 'your_secure_pw';
CREATE DATABASE mlflow_tracking;
GRANT ALL PRIVILEGES ON DATABASE mlflow_tracking TO mlflow;
\q
```

### 3.3 Systemd-Service

```ini
# /etc/systemd/system/mlflow.service
[Unit]
Description=MLflow Tracking Server
After=network.target postgresql.service

[Service]
Type=simple
User=mlflow
WorkingDirectory=/home/mlflow
Environment="PATH=/home/mlflow/venv/bin"
ExecStart=/home/mlflow/venv/bin/mlflow server \
    --backend-store-uri postgresql://mlflow:PASSWORD@localhost:5432/mlflow_tracking \
    --default-artifact-root file:/home/mlflow/artifacts \
    --host 127.0.0.1 \
    --port 5000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable mlflow
sudo systemctl start mlflow
```

**Access via SSH-Tunnel:**

```bash
ssh -L 5000:localhost:5000 hans@hetzner
# Browser: http://localhost:5000
```

**Nicht öffentlich auf Port 5000 exposen.** MLflow hat kein Auth-System eingebaut — das wäre ein Security-Hole.

### 3.4 Environment-Variable im Code

```python
# src/assembled_core/mlflow_setup.py
import mlflow
import os

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",  # via SSH-Tunnel erreichbar
)

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Experiment = Gruppierung von Runs
    mlflow.set_experiment("assembled_trading_ai")
```

---

## 4. Der Training-Workflow mit MLflow

### 4.1 Basis-Struktur eines Training-Scripts

```python
# scripts/train/train_news_classifier.py
"""
Trainiert den News-Classifier und registriert im MLflow Model-Registry.

Usage:
    python -m scripts.train.train_news_classifier \
        --data data/news_labeled_2024_2025.parquet \
        --tag experiment_2026_q2
"""
import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

from assembled_core.mlflow_setup import setup_mlflow


def file_sha256(path: Path) -> str:
    """SHA-256 Hash eines Files für Data-Version-Tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--tag", default="dev")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()
    
    setup_mlflow()
    
    with mlflow.start_run(run_name=f"news_clf_{args.tag}_{datetime.utcnow():%Y%m%d_%H%M}"):
        # ---- Log all inputs ----
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "random_state": 42,
            "data_path": str(args.data),
            "data_sha": file_sha256(args.data),
            "git_sha": get_git_sha(),
            "tag": args.tag,
        })
        
        # ---- Load data ----
        df = pd.read_parquet(args.data)
        X = df.drop(columns=["label"])
        y = df["label"]
        
        # TimeSeriesSplit statt random
        cutoff = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
        y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
        
        mlflow.log_params({
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "n_features": X.shape[1],
            "train_start": df.iloc[0]["timestamp"].isoformat(),
            "train_end": df.iloc[cutoff-1]["timestamp"].isoformat(),
            "test_start": df.iloc[cutoff]["timestamp"].isoformat(),
            "test_end": df.iloc[-1]["timestamp"].isoformat(),
        })
        
        # ---- Train ----
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        
        # ---- Evaluate ----
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, output_dict=True)
        
        mlflow.log_metrics({
            "macro_f1": f1,
            "accuracy": report["accuracy"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        })
        
        # ---- Feature Importance als artifact ----
        importances = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importances.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        
        # ---- Classification Report als artifact ----
        pd.DataFrame(report).to_json("classification_report.json")
        mlflow.log_artifact("classification_report.json")
        
        # ---- Model registrieren ----
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            registered_model_name="news_classifier",
        )
        
        print(f"Run complete. Macro-F1: {f1:.4f}")
        print(f"View: http://localhost:5000")


def get_git_sha() -> str:
    import subprocess
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:12]
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
```

**Der wichtige Teil:** `mlflow.log_params` und `mlflow.log_metrics` für **alles**, was die Ergebnisse beeinflussen könnte. Data-Hash, Git-Sha, Trainingszeitraum.

### 4.2 Model-Registry-Workflow

MLflow hat eine Model-Registry mit Staging-Funktionalität. Bedenke: du brauchst **keine volle Staging/Production-Pipeline** wie ein 10-Personen-Team. Aber **Aliases** sind wertvoll:

```python
from mlflow import MlflowClient

client = MlflowClient()

# Nach dem Training: Modell version taggen
latest_version = client.get_latest_versions("news_classifier", stages=["None"])[-1].version

# Als "candidate" taggen
client.set_registered_model_alias(
    name="news_classifier",
    alias="candidate",
    version=latest_version,
)

# Nach manueller Review, falls ok: als "production" taggen
client.set_registered_model_alias(
    name="news_classifier",
    alias="production",
    version=latest_version,
)
```

**In deinem Pipeline-Code:**

```python
import mlflow.sklearn

model = mlflow.sklearn.load_model("models:/news_classifier@production")
```

**So geht das Live-Deploy:** du änderst nur den Alias. Code ändert sich nicht. Rollback ist ein Kommando.

```python
# Rollback zu vorherigem Modell
client.set_registered_model_alias(
    name="news_classifier",
    alias="production",
    version=previous_version,  # z.B. 12 statt 13
)
```

---

## 5. Die Strategy-Config-Versionierung

MLflow ist für ML-Modelle. Für **Strategy-Configs** (Gewichte, Thresholds, Regime-Multipliers) ist Git mit YAML ausreichend und sogar besser.

### 5.1 Directory-Layout

```
configs/
├── strategies/
│   ├── trend_news_v1.yaml
│   ├── trend_news_v2.yaml
│   ├── trend_news_v3.yaml
│   └── trend_news_v4.yaml      # aktuelle Version
├── schemas/
│   └── strategy_schema.yaml    # Pydantic-Schema für Validierung
└── deployments/
    ├── dev_active.yaml          # symlink zu configs/strategies/trend_news_v4.yaml
    ├── staging_active.yaml      # zu trend_news_v3.yaml (ältere Version in Staging)
    └── prod_active.yaml         # zu trend_news_v2.yaml (noch ältere Version)
```

### 5.2 Schema-Validierung

```python
# src/assembled_core/strategy/config.py
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator
from typing import Dict


class CompositeWeights(BaseModel):
    trend: float = Field(ge=0, le=1)
    momentum: float = Field(ge=0, le=1)
    news: float = Field(ge=0, le=1)
    volatility: float = Field(ge=0, le=1)
    regime: float = Field(ge=0, le=1)
    sentiment: float = Field(ge=0, le=1)
    volume: float = Field(ge=0, le=1)
    microstructure: float = Field(ge=0, le=1)
    fundamentals: float = Field(ge=0, le=1)
    
    @field_validator("*", mode="after")
    @classmethod
    def sum_close_to_1(cls, v, info):
        # Wird erst nach vollständiger Validierung geprüft — siehe unten
        return v


class Thresholds(BaseModel):
    buy: float = Field(gt=0, lt=1)
    sell: float = Field(gt=-1, lt=0)


class StrategyConfig(BaseModel):
    strategy_id: str
    description: str
    created: str
    author: str
    
    composite_weights: CompositeWeights
    thresholds: Thresholds
    regime_multipliers: Dict[str, float]
    
    news: Dict = Field(default_factory=dict)
    risk: Dict
    model_versions: Dict[str, str]
    
    @field_validator("composite_weights", mode="after")
    @classmethod
    def weights_sum_to_one(cls, v):
        total = sum(v.model_dump().values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Composite weights must sum to ~1.0, got {total:.3f}")
        return v


def load_strategy_config(path: Path) -> StrategyConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return StrategyConfig(**data)


def load_active_config(environment: str) -> StrategyConfig:
    """Lädt die aktive Strategy-Config für das Environment."""
    path = Path(f"configs/deployments/{environment}_active.yaml")
    if path.is_symlink():
        path = path.resolve()
    return load_strategy_config(path)
```

### 5.3 Der Deployment-Workflow

```bash
# Szenario: trend_news_v5 ist ready für Staging
# 1. Erstelle v5 config (kopiere v4, ändere, validiere)
cp configs/strategies/trend_news_v4.yaml configs/strategies/trend_news_v5.yaml
# Edit v5.yaml, ändere z.B. composite_weights

# 2. Validate
python -c "from assembled_core.strategy.config import load_strategy_config; \
           load_strategy_config('configs/strategies/trend_news_v5.yaml')"

# 3. Commit
git add configs/strategies/trend_news_v5.yaml
git commit -m "strategy: new trend_news_v5 - reduce news weight 0.15 -> 0.12"

# 4. Symlink für Staging ändern (lokal, für einen Test-Lauf)
cd configs/deployments
rm staging_active.yaml
ln -s ../strategies/trend_news_v5.yaml staging_active.yaml
git add staging_active.yaml
git commit -m "deploy: promote trend_news_v5 to staging"

# 5. Nach 7 Tagen Staging → Prod (siehe 36_MULTI_ENVIRONMENT_SETUP.md)
rm prod_active.yaml
ln -s ../strategies/trend_news_v5.yaml prod_active.yaml
git commit -m "deploy: promote trend_news_v5 to prod after 7d staging"
```

**Der Vorteil:** `git log configs/deployments/prod_active.yaml` zeigt dir die komplette Deployment-History.

---

## 6. Die Koppelung: Strategy ↔ ML-Modelle

Strategy-Config referenziert ML-Modell-Versionen:

```yaml
model_versions:
  news_classifier: "news_classifier@v3"     # MLflow registry-alias
  regime_classifier: "regime_rf@v2"
```

Dein Code löst das auf:

```python
import mlflow.sklearn

def load_models_from_config(config: StrategyConfig):
    models = {}
    for model_key, mlflow_ref in config.model_versions.items():
        # "news_classifier@v3" → "models:/news_classifier/v3"
        name, version = mlflow_ref.split("@")
        if version.startswith("v"):
            # Explicit version
            uri = f"models:/{name}/{version[1:]}"
        else:
            # Alias
            uri = f"models:/{name}@{version}"
        
        models[model_key] = mlflow.sklearn.load_model(uri)
    return models
```

**Diese Koppelung gibt dir volle Reproduzierbarkeit:**
- Gegeben: Strategy-Config-Hash (`trend_news_v5.yaml`, Git-Sha `a3f2b1c`)
- Referenziert: MLflow-Model-Versionen
- → du kannst den Stand von vor 6 Monaten **exakt** rekonstruieren

---

## 7. Hyperparameter-Tuning (wenn nötig)

### 7.1 Die Frage: brauchst du Hyperparameter-Tuning überhaupt?

**Ehrlicher Rat für Hobby-Quant:** Wahrscheinlich **nicht** in der ersten Jahr.

- Die Leistung deines Systems ist eher durch **Feature-Qualität** und **Strategie-Logik** limitiert als durch Tuning.
- Hyperparameter-Tuning auf Trading-Daten ist ein **Multiple-Testing-Problem** (siehe `32_VALIDIERUNG.md`). Wenn du 200 Kombinationen probierst, finest du zufällig eine, die im Backtest glänzt — aber Live versagt.

**Bessere Reihenfolge:**
1. Erst: gute Features entwickeln (Ebene 1)
2. Dann: simple Modelle (LogReg, einfache Random-Forest) statt Tuning-fancy-Modelle
3. Erst wenn du sicher bist, dass Feature-Signal stark ist: vorsichtig tunen

### 7.2 Wenn du tunen willst: Optuna mit MLflow

```python
# scripts/train/tune_news_classifier.py
import optuna
import mlflow
from optuna.integration.mlflow import MLflowCallback

from assembled_core.mlflow_setup import setup_mlflow


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
    }
    
    # Training-Code analog zu train_news_classifier.py,
    # mit params dict
    model = train_model(params)
    score = evaluate(model)
    return score


if __name__ == "__main__":
    setup_mlflow()
    
    mlflow_cb = MLflowCallback(
        tracking_uri="http://localhost:5000",
        metric_name="macro_f1",
    )
    
    study = optuna.create_study(
        direction="maximize",
        study_name="news_classifier_tuning_2026_q2",
    )
    study.optimize(objective, n_trials=50, callbacks=[mlflow_cb])
    
    print(f"Best params: {study.best_params}")
    print(f"Best score: {study.best_value:.4f}")
```

**Wichtig:** `n_trials=50` ist genug. Darüber hinaus Overfitting-Risiko. Wenn du 500 Trials brauchst, ist dein Search-Space zu groß.

### 7.3 Walk-Forward-Tuning (empfohlen für Trading)

Statt eines einzigen Train/Test-Splits nutze **rolling windows**:

```python
def walk_forward_objective(trial, data):
    params = {...}  # wie oben
    
    scores = []
    for fold_start in range(0, len(data) - 12, 3):  # 3-Monat-Stride
        train = data.iloc[fold_start:fold_start+9]    # 9 Monate Train
        test = data.iloc[fold_start+9:fold_start+12]  # 3 Monate Test
        
        model = train_model(train, params)
        score = evaluate(model, test)
        scores.append(score)
    
    return np.mean(scores)  # average über alle Folds
```

**Dies verhindert:** Hyperparameters, die zufällig auf dem einen Test-Window gut sind aber in anderen Zeiträumen versagen.

---

## 8. Monitoring: Parameter-Drift

### 8.1 Was ist Parameter-Drift?

Wenn deine Strategy-Configs **ungeplant** divergieren zwischen den Environments:

- Dev hat `news_weight=0.15`
- Staging hat `news_weight=0.20` (vergessen zu updaten?)
- Prod hat `news_weight=0.12` (manuell gepatcht?)

Das ist Drift. Es passiert, wenn Leute (auch du selbst nach 3 Monaten) vergessen, welche Änderungen wohin propagiert wurden.

### 8.2 Drift-Detection

```python
# scripts/monitoring/check_config_drift.py
"""
Prüft ob Dev/Staging/Prod Configs sich aus Versehen unterscheiden.
Läuft täglich via cron.
"""
from pathlib import Path
import yaml
import sys


def check_drift():
    configs = {}
    for env in ["dev", "staging", "prod"]:
        path = Path(f"configs/deployments/{env}_active.yaml").resolve()
        configs[env] = yaml.safe_load(path.read_text())
    
    # Compare
    dev_keys = set(flatten_keys(configs["dev"]))
    staging_keys = set(flatten_keys(configs["staging"]))
    prod_keys = set(flatten_keys(configs["prod"]))
    
    drift_report = []
    
    # Unterschiedliche Keys
    for key in (dev_keys | staging_keys | prod_keys):
        dev_val = get_nested(configs["dev"], key)
        staging_val = get_nested(configs["staging"], key)
        prod_val = get_nested(configs["prod"], key)
        
        if len({str(dev_val), str(staging_val), str(prod_val)}) > 1:
            drift_report.append({
                "key": key,
                "dev": dev_val,
                "staging": staging_val,
                "prod": prod_val,
            })
    
    # Bericht
    if drift_report:
        print("⚠️  Config-Drift erkannt:")
        for item in drift_report:
            print(f"  {item['key']}:")
            print(f"    dev:      {item['dev']}")
            print(f"    staging:  {item['staging']}")
            print(f"    prod:     {item['prod']}")
    else:
        print("✓ Kein Config-Drift.")
```

**Regel:** Prod sollte **niemals** neuer sein als Staging. Staging sollte **niemals** neuer sein als Dev. Wenn Drift-Detection das Gegenteil zeigt, wurde direkt in Prod/Staging gepatcht → Incident.

---

## 9. Deployment-Inventory

### 9.1 Was läuft gerade, wo?

```python
# scripts/monitoring/deployment_inventory.py
"""
Erstellt Snapshot: welche Strategy-Version + ML-Model-Versions laufen 
aktuell in jedem Environment?
"""
import json
from datetime import datetime
from pathlib import Path
import yaml
from mlflow import MlflowClient


def inventory():
    client = MlflowClient()
    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "environments": {},
    }
    
    for env in ["dev", "staging", "prod"]:
        path = Path(f"configs/deployments/{env}_active.yaml").resolve()
        config = yaml.safe_load(path.read_text())
        
        models_info = {}
        for model_key, mlflow_ref in config.get("model_versions", {}).items():
            name, version = mlflow_ref.split("@")
            try:
                if version.startswith("v"):
                    mv = client.get_model_version(name, version[1:])
                else:
                    mv = client.get_model_version_by_alias(name, version)
                models_info[model_key] = {
                    "name": name,
                    "version": mv.version,
                    "run_id": mv.run_id,
                    "created_at": mv.creation_timestamp,
                }
            except Exception as e:
                models_info[model_key] = {"error": str(e)}
        
        snapshot["environments"][env] = {
            "strategy_file": str(path),
            "strategy_id": config.get("strategy_id"),
            "strategy_author": config.get("author"),
            "strategy_created": config.get("created"),
            "models": models_info,
        }
    
    # Write
    out_path = Path(f"data/deployment_snapshots/snapshot_{datetime.utcnow():%Y%m%d_%H%M}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"Snapshot saved: {out_path}")
    return snapshot
```

**Täglich per cron.** Gibt dir einen historischen Audit-Trail: "am 15.03. lief in Prod strategy_v4 mit news_classifier_v3".

---

## 10. Umsetzungs-Checkliste

**Phase 1 — MLflow-Server (Tag 1-2):**
- [ ] MLflow auf Hetzner installieren
- [ ] Postgres-Backend
- [ ] systemd-Service
- [ ] SSH-Tunnel-Access validieren

**Phase 2 — Training-Integration (Tag 3-5):**
- [ ] Basis-Training-Script mit `mlflow.log_params/metrics`
- [ ] Model-Registry mit Aliases (candidate/production)
- [ ] Rollback-Plan für Prod-Models

**Phase 3 — Strategy-Configs (Tag 6-7):**
- [ ] Directory-Struktur `configs/strategies/`
- [ ] Pydantic-Schemas mit Validierung
- [ ] `configs/deployments/{env}_active.yaml` Symlinks

**Phase 4 — Integration (Tag 8-10):**
- [ ] Pipeline liest Config + Models pro Environment
- [ ] Models werden aus MLflow via `models:/...@alias` geladen
- [ ] Deployment-Workflow dokumentiert

**Phase 5 — Monitoring (Tag 11-12):**
- [ ] Config-Drift-Detection täglich
- [ ] Deployment-Inventory täglich
- [ ] Audit-Log für Model-Version-Wechsel

**Phase 6 — Optional: Tuning (Tag 13-15):**
- [ ] Walk-Forward-Tuning-Script
- [ ] Optuna-Integration mit MLflow
- [ ] Nur nach Feature-Arbeit, nicht vorher

**Gesamt-Aufwand:** 2-3 Wochen bei 10-15 h/Woche.

---

## 11. Quellen

**MLflow:**
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- MLflow 2.17 Release Notes
- ML Journey (2026): [MLflow vs Weights & Biases vs Neptune](https://mljourney.com/mlflow-vs-weights-and-biases-vs-neptune-choosing-an-experiment-tracker/)
- Uplatz Blog (2025): [The 2025 MLOps Landscape: MLflow, W&B, Neptune](https://uplatz.com/blog/the-2025-mlops-landscape-a-comparative-analysis-of-mlflow-weights-biases-and-neptune/)
- ZenML: [MLflow vs W&B vs ZenML](https://www.zenml.io/blog/mlflow-vs-weights-and-biases)

**Model-Versioning:**
- ML Journey (2025): [Model Versioning Strategies: DVC vs MLflow vs W&B](https://mljourney.com/model-versioning-strategies-dvc-vs-mlflow-vs-weights-biases/)
- [DVC Documentation](https://dvc.org/doc)
- Reintech (2026): [Experiment Tracking Tools Compared 2026](https://reintech.io/blog/mlflow-vs-weights-and-biases-vs-neptune-experiment-tracking-comparison)

**Hyperparameter Tuning:**
- [Optuna Documentation](https://optuna.readthedocs.io/)
- Markaicode (2025): [W&B for LLM Experiment Tracking](https://markaicode.com/wandb-llm-experiment-tracking-guide/)
- [Optuna-MLflow Integration](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.MLflowCallback.html)

**Config Management:**
- [Pydantic Documentation](https://docs.pydantic.dev/) (siehe `36_MULTI_ENVIRONMENT_SETUP.md` für Details)

---

## 12. Ehrliche Einschätzung

**Was dieses Playbook dir gibt:**
- Volle Reproduzierbarkeit: du kannst jede Strategy-Version + ML-Modell-Stand 6 Monate zurück rekonstruieren
- Audit-Trail: "wer hat wann welche Konfiguration deployed?"
- Rollback in Sekunden statt Stunden
- Ein klares Bild: "welche Version läuft gerade wo?"

**Was es dir nicht gibt:**
- **Bessere Modelle.** MLflow trackt; es optimiert nicht.
- **Automatische Deployment-Gates.** Du musst die Regeln aus `36_MULTI_ENVIRONMENT_SETUP.md` einhalten.
- **Schutz vor Overfitting beim Tuning.** Optuna probiert Parameter aus; die statistische Validität musst du via Walk-Forward sicherstellen.

**Die drei Sachen, die du nicht auslassen darfst:**
1. **`data_sha` und `git_sha` bei jedem Training-Run.** Ohne diese weißt du nicht, auf welchen Daten + Code-Stand dein Modell trainiert wurde.
2. **Strategy-Config als YAML in Git.** Nicht in Python-Konstanten, nicht in Environment-Variablen, nicht in Database. YAML in Git.
3. **Config-Drift-Detection.** Der häufigste Fehler ist nicht falsche Params, sondern **unbemerkte** Divergenz zwischen Environments. Detection kostet 50 Zeilen Code, spart 50 Stunden Debug-Zeit.

**Der wichtigste psychologische Punkt:** Ohne Governance traust du dich nicht, zu experimentieren. Jede Änderung fühlt sich irreversibel an. Mit Governance kannst du frei experimentieren, weil du weißt: "falls das Scheiße ist, `mlflow alias set news_classifier production v12` und ich bin in 10 Sekunden zurück." Diese Sicherheit ist das, was dich wirklich produktiv macht.
