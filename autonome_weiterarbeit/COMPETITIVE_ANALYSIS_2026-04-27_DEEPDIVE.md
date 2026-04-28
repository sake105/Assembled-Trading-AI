# Wettbewerbsanalyse: Tiefen-Recherche — Teil 2

**Datum:** 2026-04-27
**Vorausgegangen:** `COMPETITIVE_ANALYSIS_2026-04-27.md` mit 7 Themen-Bereichen und 15+ Projekten.
**Diese Datei:** **8 weitere Themen-Bereiche**, die in Teil 1 nicht oder zu wenig drin waren. Tiefer und breiter, mit anderen Schwerpunkten — Production-Engineering, Security, Execution, Forecasting, Observability, MLOps.

**Inhaltsverzeichnis:**
1. [Execution-Algorithmen](#1-execution-algorithmen-vwap--twap--almgren-chriss)
2. [Secrets-Management & API-Security](#2-secrets-management--api-security)
3. [Reinforcement Learning für Trading](#3-reinforcement-learning-für-trading)
4. [Time-Series-Forecasting](#4-time-series-forecasting--probabilistic--anomaly)
5. [Observability-Stack](#5-observability-stack-grafana--prometheus--loki--tempo)
6. [MLOps: Experiment Tracking & Model Registry](#6-mlops-experiment-tracking--model-registry)
7. [Order-Management & Production-Patterns](#7-order-management--production-patterns)
8. [Feature-Engineering & López de Prado Methoden](#8-feature-engineering--lópez-de-prado-methoden)
9. [Erweiterter Adoption-Plan](#erweiterter-adoption-plan-mit-aufwandschätzung)

---

## 1. Execution-Algorithmen: VWAP / TWAP / Almgren-Chriss

**Status bei dir:** Slippage-Modellierung in `risk/slippage.py` (gut, A8 erledigt). Aber **keine echten Execution-Algorithmen** — d.h. wenn du eine Order über $50.000 schickst, geht sie als 1 Market-Order raus, mit dem entsprechenden Impact.

**Warum das relevant wird:** Wenn dein NAV wächst und du in liquideren oder weniger liquiden Märkten handelst, beginnt der **Implementation Shortfall** (Differenz zwischen Entscheidungspreis und tatsächlichem Ausführungspreis) signifikant zu werden. Bei Mid-Caps oder Small-Caps können das 30-50bps sein.

### 1.1 Almgren-Chriss-Model

**Was es ist:** Akademisches Standard-Modell für optimale Order-Execution. Definiert eine **optimal trajectory** — wie schnell soll eine große Order liquidiert werden, um Permanent + Temporary Market Impact gegen Volatilitäts-Risiko abzuwägen.

Die Mathematik:
- **Permanent Impact**: dauerhafter Preisversatz proportional zur Liquidationsrate
- **Temporary Impact**: kurzfristiger Spread proportional zur Order-Geschwindigkeit
- Optimum: minimiere `E[Cost] + λ × Var[Cost]` bei Risk-Aversion λ

**Praktischer Code-Block** (aus `meridianalgo`-Library, Apache 2.0):

```python
class AlmgrenChrissExecution:
    def __init__(self, total_quantity, total_time, volatility, 
                 risk_aversion=1e-6, eta=2.5e-6, gamma=2.5e-7):
        self.X = total_quantity
        self.T = total_time
        self.sigma = volatility
        self.lambda_ = risk_aversion
        self.eta = eta      # temporary impact
        self.gamma = gamma  # permanent impact
    
    def calculate_optimal_trajectory(self, n_steps=100):
        """Returns array of position holdings over time."""
        kappa = np.sqrt(self.lambda_ * self.sigma**2 / self.eta)
        tau = self.T / n_steps
        
        trajectory = np.zeros(n_steps + 1)
        trajectory[0] = self.X
        
        for k in range(1, n_steps + 1):
            t = k * tau
            trajectory[k] = self.X * (
                np.sinh(kappa * (self.T - t)) / 
                np.sinh(kappa * self.T)
            )
        return trajectory
    
    def calculate_expected_cost(self):
        """Implementation shortfall in absolute units."""
        kappa = np.sqrt(self.lambda_ * self.sigma**2 / self.eta)
        permanent_cost = 0.5 * self.gamma * self.X**2
        temporary_cost = (
            self.eta * self.X**2 * kappa / 
            (2 * np.tanh(kappa * self.T / 2))
        )
        return permanent_cost + temporary_cost
```

**Was du damit machst:** Wenn deine Order > X% des Average-Daily-Volumes ist, statt direkt Markt-Order zu schicken: Almgren-Chriss-Trajectory berechnen, in N Slices teilen, über Zeit verteilt ausführen.

### 1.2 VWAP- und TWAP-Schedules

**TWAP (Time-Weighted Average Price):** trivial einfach — teile Order in N gleiche Slices über T Minuten:

```python
def twap_schedule(total_quantity, total_minutes, slice_count):
    slice_size = total_quantity / slice_count
    interval_minutes = total_minutes / slice_count
    return [(i * interval_minutes, slice_size) for i in range(slice_count)]
```

**VWAP (Volume-Weighted Average Price):** schwerer, weil Volume-Profile geschätzt werden muss. Standard: U-Shape (mehr Volume Open + Close, weniger Mittag).

```python
def vwap_schedule(total_quantity, intraday_volume_profile):
    """
    intraday_volume_profile: dict[time_bucket -> volume_fraction]
    z.B. {'09:30': 0.08, '10:00': 0.05, ..., '15:30': 0.07, '16:00': 0.10}
    """
    schedule = {}
    for time_bucket, vol_frac in intraday_volume_profile.items():
        schedule[time_bucket] = total_quantity * vol_frac
    return schedule
```

Volume-Profile baust du aus 30-Tage-rolling Daten von `fxec_daily_*`.

### 1.3 Empfehlung für dich

**Phase 1 (kurz, ~6h):** Triggerschwellen einbauen.

In `pipeline/trading_cycle_v2.py` bei Order-Generierung: wenn `order_size_usd > threshold_for_execution_algo`, dann statt direkt zu `submit_order()` → Routing zu einem `ExecutionAlgo`-Modul.

```python
# src/assembled_core/execution/execution_router.py

def route_order(
    order: Order,
    daily_volume: float,
    config: ExecutionConfig,
) -> list[ChildOrder]:
    """Splits parent order into child orders if needed."""
    notional_usd = order.quantity * order.price
    pct_of_adv = notional_usd / daily_volume
    
    if pct_of_adv < config.direct_threshold:
        # Klein genug: direkt
        return [ChildOrder.from_parent(order)]
    elif pct_of_adv < config.twap_threshold:
        # TWAP
        return twap_split(order, n_slices=config.twap_slices)
    else:
        # Almgren-Chriss
        return ac_split(order, config)
```

**Phase 2 (~12h):** Tatsächliche Almgren-Chriss-Implementation, mit Volatility/η/γ aus eigenen Daten kalibriert.

**Aufwand gesamt:** 18h für robustes Execution-Routing.

**Lizenz Quelle:** `meridianalgo` Apache 2.0 — Code-Adaption erlaubt.

---

## 2. Secrets-Management & API-Security

**Status bei dir:** Vermutlich `.env`-Datei mit Alpaca-API-Keys, oder Konstanten im Code (was du schon nicht machst). Aber: keine Rotation, keine zentrale Verwaltung, kein Secret-Scanning.

**Warum das wichtig ist:** Bei `git log --all -p | grep -i "api_key"` einen kompromittierten Key zu finden, ist trivial. **GitGuardian fand 21.000 OpenClaw-Instanzen mit exposed Tokens in 2 Wochen** (lt. Recherche). Bei einem Trading-Bot mit Live-Account ist das ein direkter Hebel für Schaden.

### 2.1 Threat-Model für Trading-Bots

Drei Levels von Bedrohung:
1. **Public Repo Leak**: Du committest versehentlich `.env` → öffentliche Schlüssel → Alpaca-Account übernommen
2. **Private Repo Compromise**: Jemand kommt in dein Repo (geleakter GitHub-Token, Lieferanten-Kompromittierung) → gleicher Effekt
3. **Runtime Compromise**: Server geht hoch, Schlüssel im Memory leak via Crash-Logs

**Zentrale Defense-Strategie:** API-Keys haben **Permission-Scopes** und **Time-To-Live**. Statt einem "All-Permissions, never expire"-Key viele kleine Keys mit klaren Scopes.

Bei Alpaca z.B.:
- Read-only Key fürs Datenholen
- Trading-only Key (kein Withdraw) für die Trading-Pipeline
- Beide IP-whitelisted auf deinen Server

### 2.2 Tooling-Optionen (sortiert nach Aufwand)

#### Option A — SOPS + age (30-Min-Setup)

**Was es ist:** Mozilla-Tool. Verschlüsselt nur die **Werte** in YAML/JSON-Files, Keys bleiben lesbar (gut für Git-Diff). MIT-Lizenz.

**Setup:**
```bash
# Tools installieren
brew install sops age
# oder: scoop install sops age

# Keypair generieren
age-keygen -o ~/.age/key.txt

# Secrets-File anlegen und verschlüsseln
cat > secrets.yaml << EOF
alpaca_paper:
  api_key: PK...
  secret_key: ...
alpaca_live:
  api_key: AK...
  secret_key: ...
EOF

PUBKEY=$(grep "public key" ~/.age/key.txt | cut -d: -f2 | tr -d ' ')
sops --encrypt --age "$PUBKEY" secrets.yaml > secrets.enc.yaml

# secrets.enc.yaml committen, secrets.yaml in .gitignore
```

**Decryption im Code:**
```python
import subprocess
import os
from pathlib import Path
import yaml

def load_secrets() -> dict:
    encrypted_path = Path(__file__).parent / "secrets.enc.yaml"
    os.environ["SOPS_AGE_KEY_FILE"] = str(Path.home() / ".age/key.txt")
    result = subprocess.run(
        ["sops", "--decrypt", str(encrypted_path)],
        capture_output=True, text=True, check=True,
    )
    return yaml.safe_load(result.stdout)

secrets = load_secrets()
alpaca_key = secrets["alpaca_paper"]["api_key"]
```

**Vorteil:** verschlüsselte Datei kann in Git committed werden. Du brauchst nur die Age-Privatkey-Datei lokal.

#### Option B — Infisical (mehr Aufwand, mehr Features)

**Was es ist:** Self-hosted Secret-Manager mit Web-UI. MIT-Lizenz. Vault-Alternative (Vault wechselte 2023 zu BSL).

**Setup:**
```yaml
# docker-compose.yml
version: '3.8'
services:
  infisical:
    image: infisical/infisical:latest
    ports:
      - "8080:8080"
    environment:
      - ENCRYPTION_KEY=...
      - JWT_SECRET=...
    volumes:
      - infisical_data:/data
volumes:
  infisical_data:
```

**Code-Integration:**
```python
from infisical_client import InfisicalClient

client = InfisicalClient({
    "site_url": "http://localhost:8080",
    "client_id": "...",
    "client_secret": "...",
})

alpaca_key = client.get_secret("ALPACA_API_KEY", environment="prod").value
```

**Vorteil:** zentrale Verwaltung, Audit-Log, automatische Rotation, RBAC.

**Nachteil:** Infrastruktur-Aufwand. Lohnt sich erst wenn du > 20 Secrets hast oder Team.

### 2.3 GitGuardian / pre-commit Secret-Scanning

Egal welche Option du wählst, **ein Pre-Commit-Hook** sollte verhindern, dass jemals plain Secrets committed werden:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.2
    hooks:
      - id: gitleaks
```

`gitleaks` (MIT) scannt deinen Diff nach Pattern wie API-Keys, AWS-Tokens, Stripe-Keys. Wenn etwas matcht → commit blockiert.

### 2.4 Empfehlung für dich

**Phase 1 (~30 Min):** SOPS + age einrichten, alle Secrets dorthin migrieren, GitHub-Pre-Commit-Hook mit gitleaks installieren.

**Phase 2 (optional, bei Skalierung):** Wenn du mehrere Bots, mehrere Umgebungen, mehrere Schlüssel-Typen hast — Infisical aufstellen.

**Aufwand:** 1-2h für Phase 1.

**Lizenzen:** SOPS Mozilla, age MIT, gitleaks MIT, Infisical MIT — alles erlaubt.

---

## 3. Reinforcement Learning für Trading

**Status bei dir:** Klassisches Supervised Learning (vermutlich). RL ist für viele Trading-Aufgaben überlegen, weil:
- Action-Selektion ist sequenziell (Position halten oder schließen heute beeinflusst morgen)
- Reward kann direkt Sharpe sein, statt Proxy-Loss
- Market-Frictions (Transaktionskosten, Slippage) sind direkt im Environment einbaubar

**Aber:** RL für Trading ist bekannt schwer. Sample-Effizienz ist mies, Reward Hacking häufig, Live ≠ Backtest noch dramatischer als bei Supervised.

### 3.1 FinRL (`AI4Finance-Foundation/FinRL`)

**Was es ist:** Standard-Library. PyTorch-basiert, integriert mit Stable-Baselines3, OpenAI-Gym-konform. Implementiert DQN, DDPG, PPO, SAC, A2C, TD3 für Trading. Aktiv weiterentwickelt; in 2026 PAKDD-Paper "FinRL-X" für AI-native modular infrastructure.

**Architektur in 3 Schichten:**
1. **Environment**: Gymnasium-Env mit OHLCV + Portfolio-State + Market-Frictions
2. **Agent**: Stable-Baselines3 (PPO/SAC/etc.)
3. **Application**: Multi-Stock-Trading, Portfolio-Allocation, Single-Stock-Trading

**Code-Pattern:**
```python
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import PPO

env = StockTradingEnv(
    df=ohlcv_df,
    stock_dim=len(symbols),
    hmax=100,                  # max shares per trade
    initial_amount=1_000_000,
    transaction_cost_pct=0.001,
    state_space=(...),
    action_space=(...),
    tech_indicator_list=["macd", "rsi", "cci", "dx"],
    reward_scaling=1e-4,
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

### 3.2 Wann RL Sinn macht — und wann nicht

**Macht Sinn für:**
- Position-Sizing-Probleme (wie viel?)
- Optimal-Execution (wann genau ausführen?)
- Multi-Asset-Allocation mit dynamischen Constraints

**Macht KEINEN Sinn für:**
- Simple Direction-Forecasting (Long/Short) — Supervised ist hier besser
- Niederfrequente Signal (täglich/wöchentlich) — zu wenig Sample-Daten für RL
- Strategien wo du keine simulator-fähige Environment hast

### 3.3 Empfehlung für dich

**Konkret:** Bevor du auf RL setzt, frag dich: Hast du schon einen funktionierenden Supervised-Stack mit out-of-sample Sharpe > 1? Wenn nein → erst mal das. Wenn ja → RL kann **incremental** Mehrwert bringen, aber nicht als Ersatz.

**Mein Vorschlag:** RL als **separate Experimentier-Spur** in `experiments/rl/`, nicht in deine produktive `pipeline/`. Dort kannst du FinRL spielerisch ausprobieren, ohne Risiko für deine bestehende Logik.

```
src/
├── pipeline/           # Produktiv: trading_cycle_v2 etc.
├── models/             # Supervised: aktuelles Setup
└── experiments/
    └── rl/             # FinRL-Spielwiese, nicht in CI gemerged
        ├── envs/
        ├── train.py
        └── README.md
```

**Aufwand für initiales Setup:** 8-12h, hauptsächlich Environment-Tweaking.

**Lizenz:** FinRL ist MIT — frei nutzbar.

---

## 4. Time-Series-Forecasting — Probabilistic + Anomaly

**Status bei dir:** Vermutlich klassische Indikatoren + Tree-basiertes ML. Kein dediziertes Time-Series-Forecasting-Framework.

**Lücke:** Probabilistische Forecasts (Konfidenzintervalle statt Punktprognosen) und Anomaly-Detection auf Markt-Daten (Regime-Wechsel, Vol-Spikes).

### 4.1 darts (`unit8co/darts`)

**Was es ist:** "Sklearn for time series". Apache 2.0. Modelle von ARIMA bis Transformer in einer einheitlichen API (`fit()` / `predict()`).

**Was es kann (was wir bei dir nicht haben):**

1. **`historical_forecasts()`** — eingebautes Walk-Forward für TS-Modelle:
```python
from darts.models import ExponentialSmoothing, Prophet

models = [ExponentialSmoothing(), Prophet()]
backtests = [
    model.historical_forecasts(
        series, start=0.5, forecast_horizon=5,
        retrain=True, verbose=True,
    )
    for model in models
]
```
Das ist genau dein Walk-Forward (A9), aber als off-the-shelf-Funktion.

2. **Probabilistic Forecasting** — Konfidenz-Intervalle out-of-the-box:
```python
from darts.models import TFTModel  # Temporal Fusion Transformer

model = TFTModel(input_chunk_length=20, output_chunk_length=5,
                 likelihood=QuantileRegression(quantiles=[0.05, 0.5, 0.95]))
model.fit(series)
prediction = model.predict(n=10, num_samples=200)
# prediction ist eine probabilistische TimeSeries mit 200 Samples pro Zeitpunkt
```

3. **Conformal Prediction** — kalibrierte Quantil-Intervalle für **jedes** vortrainierte Modell:
```python
from darts.models import ConformalPredictionModel

cpm = ConformalPredictionModel(model, alpha=0.1)
cpm.fit(series)
forecast = cpm.predict(n=10)  # mit garantierter 90% Coverage
```

4. **Anomaly Detection** in `darts.ad`:
```python
from darts.ad import KMeansScorer, ThresholdDetector

scorer = KMeansScorer(window=20, k=4)
scorer.fit(reference_series)
anomaly_scores = scorer.score(test_series)

detector = ThresholdDetector(high_threshold=0.95)
binary_anomalies = detector.detect(anomaly_scores)
```

5. **Multiple-Series Training**: ein Modell auf alle deine Symbole gleichzeitig trainieren (Transfer-Learning-artig).

### 4.2 PyOD (`yzhao062/pyod`) — Anomaly Detection deep dive

**Was es ist:** 60+ Anomaly-Detection-Algorithmen in einer Library. 38M+ Downloads. BSD-Lizenz. 

**Standard-Algorithmen für Markt-Daten:**
- **Isolation Forest**: schneller Outlier-Detector, gut für niederfrequente Märkte
- **AutoEncoder**: für komplexe Multivariate-Anomalies
- **LOF (Local Outlier Factor)**: dichte-basierte Detection
- **ECOD**: parameterfrei, sehr schnell

```python
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD

# Ensemble von mehreren Detektoren
detectors = [IForest(contamination=0.1), ECOD(contamination=0.1)]
for d in detectors:
    d.fit(features_train)

# Score auf neuen Daten
scores = np.mean([d.decision_function(features_test) for d in detectors], axis=0)
```

### 4.3 Empfehlung für dich

**Konkret 1 (klein, ~4h):** PyOD als zweite Defense-Line in deinem `qa/`-Bereich:

```python
# src/assembled_core/qa/anomaly_detection.py

from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD

class MarketAnomalyDetector:
    def __init__(self, contamination=0.05):
        self.detectors = {
            'iforest': IForest(contamination=contamination, random_state=42),
            'ecod': ECOD(contamination=contamination),
        }
        self.is_fit = False
    
    def fit(self, features: pd.DataFrame):
        for name, det in self.detectors.items():
            det.fit(features.values)
        self.is_fit = True
    
    def detect(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fit:
            raise RuntimeError("Detector must be fit first")
        return {
            name: det.predict(features.values)  # 0/1 binary
            for name, det in self.detectors.items()
        }
    
    def consensus_anomaly(self, features: pd.DataFrame) -> np.ndarray:
        results = self.detect(features)
        return np.mean(list(results.values()), axis=0) > 0.5
```

Im Trading-Cycle dann als **kill-switch**: wenn `consensus_anomaly` für aktuelle Markt-Daten True → keine neuen Trades öffnen, bestehende halten.

**Konkret 2 (mittel, ~10h):** Darts für Probabilistic Forecasting integrieren.

Statt "Predicted Return = +0.5%" lieferst du "Predicted Return = +0.5% [-0.3%, +1.4%] mit 90% Konfidenz". Der untere Wert wird dann zum Stop-Loss-Anchor, der obere zum Take-Profit-Anchor.

**Lizenzen:** beide Apache 2.0 / BSD — frei nutzbar.

---

## 5. Observability-Stack: Grafana + Prometheus + Loki + Tempo

**Status bei dir:** Vermutlich Logging in Files + manchmal Print-Statements. Bei einem Live-Bot ist das fragil — wenn was schiefgeht, gräbst du in Log-Files statt zu sehen, dass Sharpe gerade kippt.

### 5.1 Was eine richtige Observability-Stack tut

Drei Säulen:
1. **Metrics** (Prometheus): Zahlen über Zeit. "Anzahl Orders pro Stunde", "Aktuelle Position-Größe", "Sharpe der letzten 30 Tage".
2. **Logs** (Loki): strukturierte Log-Events. "Order #1234 abgelehnt von Alpaca: insufficient_buying_power".
3. **Traces** (Tempo): Request-Traces durch deinen Code. "trading_cycle_v2 dauert 8s, davon 6s in `size_positions`, davon 4s in `_sp_apply_correlation_cap`".

Drüber liegt **Grafana** als Visualisierungs- und Alert-Layer.

### 5.2 OpenTelemetry als Instrumentations-Standard

**Was es ist:** Vendor-neutrale API. Du instrumentierst dein Python einmal mit OTel, kannst dann **jeden** Backend nutzen (Grafana Cloud, Datadog, Jaeger, eigene Prometheus).

**Code-Pattern:**
```python
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup einmalig
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

# Im Code:
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Custom Metric definieren
order_counter = meter.create_counter(
    "trading.orders_submitted",
    description="Number of orders submitted",
)

# Im trading_cycle_v2:
def run_trading_cycle():
    with tracer.start_as_current_span("trading_cycle"):
        with tracer.start_as_current_span("generate_signals"):
            signals = generate_signals(...)
        
        with tracer.start_as_current_span("size_positions"):
            sized = size_positions(signals)
        
        with tracer.start_as_current_span("submit_orders"):
            for order in sized:
                broker.submit(order)
                order_counter.add(1, {"symbol": order.symbol, "side": order.side})
```

**Was du in Grafana siehst:**
- **Trace-View**: Wasserfall-Diagramm, welcher Schritt im Cycle wie lange dauert
- **Metrics-Dashboard**: Live-Counter "Orders heute", "Aktuelle Positions", "PnL"
- **Logs**: korreliert mit Traces — du klickst auf einen Span, siehst die Logs während der Ausführung

### 5.3 Trading-spezifische Dashboards

Was du in Grafana sehen willst:
- **Equity Curve** (live, mit Annotations für Trades)
- **Drawdown** vs Threshold (mit Alert wenn > X%)
- **Sharpe rolling 30d** (mit Alert wenn fällt unter Y)
- **Order Latency Distribution** (p50, p95, p99)
- **Broker-API Errors** über Zeit (mit Alert)
- **News-Pipeline-Lag** (Zeit zwischen Publish und Verarbeitung)

### 5.4 Empfehlung für dich

**Phase 1 (~6h):** Lokales Grafana + Prometheus + Loki setup mit Docker-Compose:

```yaml
# docker-compose.observability.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  loki:
    image: grafana/loki:latest
    ports: ["3100:3100"]
  
  tempo:
    image: grafana/tempo:latest
    ports: ["3200:3200"]
  
  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  grafana_data:
```

Dann OpenTelemetry-Instrumentierung in deinem Code. Bei jedem `trading_cycle_v2`-Run werden Metriken, Traces und Logs an Grafana geschickt.

**Phase 2 (~4h):** Alert-Setup:
- "Drawdown > 8% → Telegram-Notification an dich"
- "Order-Latency p95 > 5s → Email"
- "Broker-API Error-Rate > 5% in 5min → Pause Trading via Webhook"

**Aufwand gesamt:** 10h für komplettes Setup.

**Kosten:** Grafana Cloud Free Tier reicht (10k Metrics, 50GB Logs/Monat). Self-hosted kostet nur Server-Ressourcen.

**Lizenzen:** Prometheus Apache 2.0, Loki AGPL3 (Self-Host OK, Code nicht übernehmen!), Grafana AGPL3, Tempo Apache 2.0, OpenTelemetry Apache 2.0.

---

## 6. MLOps: Experiment Tracking & Model Registry

**Status bei dir:** Wenn du Modelle trainierst, vermutlich als `pickle`-Files irgendwo. Welche Hyperparams, welche Features, welche Daten? Hoffentlich im Git-Commit dokumentiert.

**Lücke:** Reproduzierbarkeit. In 6 Monaten willst du wissen "warum war der LightGBM-Forecast vom 2026-04-15 schlechter als der vom 2026-04-10?" — und du brauchst die exakten Daten + Hyperparams.

### 6.1 MLflow als Standard

**Was es ist:** Apache 2.0. 60k Downloads/Tag. Vier Komponenten:
1. **Tracking**: log Hyperparams, Metrics, Artifacts pro Run
2. **Projects**: package Code für Reproduzierbarkeit
3. **Models**: Standard-Format für Modelle
4. **Model Registry**: Stage-Transitions (Staging → Production → Archived)

**Code-Pattern:**
```python
import mlflow
import mlflow.lightgbm

mlflow.set_experiment("price_forecast_v3")

with mlflow.start_run():
    # Hyperparams
    params = {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "max_depth": 5,
        "feature_fraction": 0.8,
    }
    mlflow.log_params(params)
    
    # Daten-Version (vom DVC)
    mlflow.log_param("data_version", get_dvc_version("data/features.parquet"))
    
    # Training
    model = lgb.train(params, train_data, num_boost_round=100)
    
    # Metrics
    val_metrics = compute_metrics(model, val_data)
    mlflow.log_metrics(val_metrics)
    
    # Artifact (Modell + Plots)
    mlflow.lightgbm.log_model(model, "model")
    mlflow.log_artifact("feature_importance.png")
    
    # Tag für Filtering
    mlflow.set_tag("strategy", "momentum_v3")
    mlflow.set_tag("symbol_universe", "sp500_top100")
```

**Was du gewinnst:**
- Web-UI mit Filtering: "zeige alle Runs mit `strategy=momentum_v3` und `val_sharpe > 1.5`"
- Compare-Ansicht: 5 Runs Side-by-Side
- Model-Registry: "promote Run #1234 zu Production-Modell"
- Model-Loading: `mlflow.lightgbm.load_model("models:/forecast/Production")`

### 6.2 DVC für Daten-Versionierung

**Was es ist:** Apache 2.0. Git-like Tool für große Daten-Files. Du committest **Pointer** in Git, **Daten** liegen separat (S3, lokales Verzeichnis).

```bash
# Daten-File trackken
dvc add data/features.parquet
git add data/features.parquet.dvc .gitignore
git commit -m "Add features v1.0"

# Daten-File in remote Storage pushen
dvc remote add -d myremote s3://mybucket/dvc-store
dvc push

# Bei Reproduzierung
git checkout v1.0
dvc checkout  # holt zugehörige Daten-Version
```

**Kombination MLflow + DVC:** Bei jedem MLflow-Run wird der DVC-Hash der Trainings-Daten als Param gelogged. → 100% Reproduzierbar.

### 6.3 Empfehlung für dich

**Phase 1 (~3h):** MLflow lokal aufsetzen, alle aktuellen Modell-Trainings dort tracken.

```bash
pip install mlflow
mlflow ui  # http://localhost:5000
```

In `models/`-Trainings-Skripten den Tracking-Code einbauen. Dauer pro Skript ~10 Min.

**Phase 2 (~4h):** DVC für `data/features/` einbauen. Sobald du eine neue Feature-Generation machst, wird sie versioniert.

**Phase 3 (~2h):** Model-Registry nutzen. Statt `models/forecast_lgb.pkl` → `mlflow.lightgbm.load_model("models:/forecast/Production")`.

**Aufwand gesamt:** ~9h für komplettes MLOps-Setup.

**Lizenzen:** MLflow Apache 2.0, DVC Apache 2.0.

---

## 7. Order-Management & Production-Patterns

**Status bei dir:** A2 (Idempotency) erledigt. A8/A14 (Slippage) erledigt. Aber noch nicht systematisch durchdacht: Bracket-Orders, Order-State-Machines, Crash-Recovery.

### 7.1 OMSPY (`uberdeveloper/omspy`) — CompoundOrder-Pattern

**Was es ist:** Broker-agnostisches OMS. Hat das `CompoundOrder`-Konzept: eine "logische" Order besteht aus mehreren Atomic-Orders.

**Bracket-Order-Pattern:**
```python
from omspy.order import CompoundOrder

bracket = CompoundOrder(broker=broker)

# Entry-Order
bracket.add_order(
    symbol="TSLA", side="buy", quantity=10,
    order_type="LIMIT", price=250.00,
    key="entry",
)

# Profit-Take
bracket.add_order(
    symbol="TSLA", side="sell", quantity=10,
    order_type="LIMIT", price=275.00,
    key="profit_target",
)

# Stop-Loss
bracket.add_order(
    symbol="TSLA", side="sell", quantity=10,
    order_type="STOP", trigger_price=240.00,
    key="stop_loss",
)

# Atomar ausführen oder gar nicht
bracket.execute_all()
```

**Warum das gut ist:** Wenn dein Bot crashed nachdem die Entry-Order gefilled wurde, aber bevor die Stop-Loss platziert wurde → ungeschützte Position. CompoundOrder-Pattern macht "atomic" — entweder beide Orders, oder keine.

### 7.2 Per-Symbol Circuit-Breaker

Aus dem DEV-Community-Crypto-Bot-Post: **jedes Symbol hat eigenen Circuit-Breaker-State.**

```python
class SymbolCircuitBreaker:
    """Per-symbol failure tracking with auto-reset."""
    
    def __init__(self, threshold=3, reset_minutes=30):
        self.failures: dict[str, int] = {}
        self.last_failure: dict[str, datetime] = {}
        self.threshold = threshold
        self.reset_minutes = reset_minutes
    
    def record_failure(self, symbol: str):
        # Reset Counter wenn letzter Fehler älter als reset_minutes
        if symbol in self.last_failure:
            if (datetime.now() - self.last_failure[symbol]).seconds > self.reset_minutes * 60:
                self.failures[symbol] = 0
        
        self.failures[symbol] = self.failures.get(symbol, 0) + 1
        self.last_failure[symbol] = datetime.now()
    
    def is_open(self, symbol: str) -> bool:
        return self.failures.get(symbol, 0) >= self.threshold
    
    def record_success(self, symbol: str):
        self.failures[symbol] = 0
```

In deinem `pipeline/trading_cycle_v2.py`:
```python
if circuit_breaker.is_open(symbol):
    logger.warning(f"Circuit breaker open for {symbol}, skipping")
    continue
```

### 7.3 Pydantic für Config-Validation

**Was es ist:** MIT-Lizenz. Schema-Validation für Python-Configs.

Bei dir hast du vermutlich `configs/*.yaml` mit Strategie-Parametern. Ohne Validation kann `max_position_size: "0.5"` (String!) zu Runtime-Fehlern führen.

```python
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

class StrategyConfig(BaseSettings):
    name: str
    max_position_size: float = Field(gt=0, lt=1)  # 0 < x < 1
    max_correlation: float = Field(gt=0, le=1)
    purge_days: int = Field(ge=0, le=30)
    embargo_days: int = Field(ge=0, le=10)
    universe: list[str] = Field(min_length=1)
    
    @field_validator("universe")
    @classmethod
    def validate_universe(cls, v):
        if not all(s.isupper() for s in v):
            raise ValueError("Symbols must be uppercase")
        return v

# Beim Laden:
try:
    config = StrategyConfig(**yaml.safe_load(open("configs/momentum.yaml")))
except ValidationError as e:
    print(f"Config invalid: {e}")
    sys.exit(1)
```

**Bei Fehler:** Fail-Fast beim Startup statt Crash mitten im Live-Trading.

### 7.4 Graceful Shutdown

**Pattern:** SIGTERM/SIGINT abfangen, alle offenen Aufgaben beenden, dann clean exit.

```python
import signal
import asyncio

class GracefulExit:
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)
    
    def _handler(self, signum, frame):
        logger.info(f"Received {signum}, shutting down gracefully")
        self.shutdown = True

# In Main-Loop:
exit_handler = GracefulExit()
while not exit_handler.shutdown:
    run_trading_cycle()
    time.sleep(60)

# Cleanup
logger.info("Closing positions, persisting state, disconnecting...")
```

### 7.5 Empfehlung für dich

**Konkret 1 (Pydantic, ~4h):** Alle deine YAML-Configs durch Pydantic validieren. Verhindert Runtime-Fehler die heute bei dir auftreten könnten.

**Konkret 2 (Per-Symbol Circuit-Breaker, ~3h):** Erweitere deinen bestehenden `symbol_kill_switch` (der ist global oder per-symbol?) um ein **automatisches Reset nach X Minuten ohne Fehler**. Aktuell vermutlich manueller Reset.

**Konkret 3 (CompoundOrder-Pattern, ~6h):** Wenn du Bracket-Orders willst (Entry + TP + SL), implementiere als CompoundOrder. Atomare Ausführung oder Rollback.

**Konkret 4 (Graceful Shutdown, ~1h):** Quick-Win. Ohne das verlierst du bei Ctrl+C möglicherweise State.

**Aufwand gesamt:** ~14h für robusteres Production-Setup.

---

## 8. Feature-Engineering & López de Prado Methoden

**Status bei dir:** Triple-Barrier-Labeling (A1 erledigt). Aber wahrscheinlich noch nicht alle López de Prado-Konzepte.

### 8.1 Meta-Labeling

**Was es ist:** Two-Stage-Modell. Stage 1 sagt **Direction** (long/short/neutral). Stage 2 sagt **Confidence** (sollten wir wirklich traden?). Stage 2 nutzt Stage 1's Output als Feature.

**Warum besser als Single-Model:**
- Stage 1 kann recall-orientiert sein (viele Signale generieren)
- Stage 2 filtert auf Precision (nur die guten ausführen)
- Position-Sizing fällt aus Stage 2 raus (höhere Confidence = größere Position)

**Code-Pattern:**
```python
# Stage 1: Direction
primary_model = LGBMClassifier(...)
primary_model.fit(X_train, y_train_direction)
y_pred_direction = primary_model.predict(X_val)

# Stage 2: Was-richtig (Meta-Label)
# y_meta = 1 wenn primary_model auf Sample i richtig lag, sonst 0
y_meta_train = (primary_model.predict(X_train) == y_train_direction).astype(int)

# Features für Stage 2: original Features + Direction-Prediction
X_meta = np.hstack([X_train, primary_model.predict_proba(X_train)])

meta_model = LGBMClassifier(...)
meta_model.fit(X_meta, y_meta_train)

# Inference
def predict_with_size(X):
    direction = primary_model.predict(X)
    direction_proba = primary_model.predict_proba(X)
    X_meta = np.hstack([X, direction_proba])
    confidence = meta_model.predict_proba(X_meta)[:, 1]
    return direction, confidence
```

**Position Sizing aus Confidence:**
```python
direction, confidence = predict_with_size(X_now)
# confidence ist die Wahrscheinlichkeit, dass primary richtig liegt
# Mappe: confidence ∈ [0.5, 1.0] auf size ∈ [0.0, max_size]
size = max(0, (confidence - 0.5) * 2 * max_size)
```

### 8.2 Sample-Weighting für überlappende Labels

**Problem:** Triple-Barrier-Labels überlappen oft (Trade 1 hält 5 Tage, Trade 2 startet nach 2 Tagen). Wenn du beide als unabhängig trainierst → Daten-Leakage durch Überlappung.

**López de Prado-Lösung:** Sample-Weights basierend auf Uniqueness.

```python
def compute_sample_weights(events: pd.DataFrame, prices: pd.Series) -> pd.Series:
    """Weight = 1 / (number of overlapping events at this time)."""
    counts = pd.Series(0, index=prices.index)
    for t_in, t_out in zip(events['t_in'], events['t_out']):
        counts.loc[t_in:t_out] += 1
    
    weights = pd.Series(0.0, index=events.index)
    for i, (t_in, t_out) in enumerate(zip(events['t_in'], events['t_out'])):
        weights.iloc[i] = (1.0 / counts.loc[t_in:t_out]).mean()
    return weights

# Beim Training:
weights = compute_sample_weights(events, prices)
model.fit(X, y, sample_weight=weights.values)
```

### 8.3 Fractional Differentiation

**Problem:** Returns (`pct_change()`) verlieren Memory. Prices haben Memory aber sind nicht-stationär.

**Lösung:** Fractional Differentiation — `d=0.5` statt `d=1` (was Returns sind). Behält Memory **und** macht stationär.

```python
def fractional_diff(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """López de Prado fractional differentiation with weight cutoff."""
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] / k * (d - k + 1)
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    weights = np.array(weights[::-1])
    
    output = []
    for i in range(len(weights), len(series) + 1):
        window = series.iloc[i - len(weights):i].values
        output.append(np.dot(weights, window))
    
    return pd.Series(output, index=series.index[len(weights) - 1:])
```

### 8.4 tsfresh für automatisches Feature-Engineering

**Was es ist:** MIT-Lizenz. Berechnet ~800 Features auf Time-Series automatisch (Mean, Std, Skew, Autocorrelation, Frequency-Components, etc.) und macht Feature-Selection nach statistischer Signifikanz.

```python
from tsfresh import extract_features, select_features

# Auto-Extract: ~800 Features
features = extract_features(df, column_id='symbol', column_sort='date')

# Auto-Select: nur signifikante
selected = select_features(features, target)
```

**Plus:** Featuretools-tsfresh-Primitives kombinieren tsfresh mit Featuretools' Deep Feature Synthesis — multi-table Joins + automatische Features.

### 8.5 Empfehlung für dich

**Konkret 1 (Meta-Labeling, ~10h):** Für deinen aktuellen Forecast-Stack einen Meta-Model-Layer dazu. Sollte Sharpe um 0.2-0.5 verbessern wenn dein Primary-Model schon was kann.

**Konkret 2 (Sample-Weights, ~3h):** Bei deinen Triple-Barrier-Labels (A1) Sample-Weights basierend auf Overlap berechnen und ans Training übergeben.

**Konkret 3 (Fractional Diff, ~2h):** Als Feature einbauen — `frac_diff(close, d=0.4)` statt nur Returns. Geht oft als zusätzlicher prädiktiver Input.

**Konkret 4 (tsfresh, optional ~6h):** Als Exploration. Kann sehr viele Features in kurzer Zeit liefern, aber mit großem Overfitting-Risiko. **Nur mit strenger Out-of-Sample-Validation einbauen.**

**Aufwand gesamt:** 15h für solide López-de-Prado-Pipeline-Erweiterung.

---

## Erweiterter Adoption-Plan mit Aufwandsschätzung

Kombiniert mit Teil-1-Empfehlungen, hier die **vollständige priorisierte Roadmap**:

### Phase A — Quick Wins (~15h, sofortige Verbesserung)

| # | Item | Quelle | Aufwand |
|---|---|---|---|
| 1 | Ledoit-Wolf Cov-Shrinkage | PyPortfolioOpt | 1-2h |
| 2 | empyrical-reloaded Metriken | empyrical | 3-4h |
| 3 | quantstats HTML-Reports | quantstats | 2-3h |
| 4 | SOPS+age Secret-Mgmt | Mozilla | 1-2h |
| 5 | Pydantic Config-Validation | pydantic | 4h |
| 6 | Graceful Shutdown | DEV-Pattern | 1h |
| 7 | Healthcheck-Pattern (Heartbeat-Replacement) | Carver | 30min |

### Phase B — Strategische Investments (~50h, 4-6 Wochen)

| # | Item | Quelle | Aufwand |
|---|---|---|---|
| 8 | Anti-Leakage-Tool (Lookahead + Recursive) | Freqtrade | 8-12h |
| 9 | Bootstrap-CIs für Performance | PyBroker | 3-4h |
| 10 | FinBERT Sentiment-Backend | ProsusAI | 4-6h |
| 11 | PyOD Anomaly-Detector | PyOD | 4h |
| 12 | MLflow + DVC | Databricks/Iterative | 9h |
| 13 | Per-Symbol Circuit-Breaker | DEV-Pattern | 3h |
| 14 | Meta-Labeling Layer | López de Prado | 10h |
| 15 | Sample-Weights aus Overlap | López de Prado | 3h |

### Phase C — Architektur-Refactor (~50h, nach B3/B5)

| # | Item | Quelle | Aufwand |
|---|---|---|---|
| 16 | Broker-Adapter-Pattern | NautilusTrader | 8-16h |
| 17 | DataBlob-Pattern | pysystemtrade | 8-12h |
| 18 | RiskModel-Chain | QSTrader | 6-10h |
| 19 | Execution-Routing (TWAP/VWAP/AC) | meridianalgo | 18h |
| 20 | CompoundOrder-Pattern | OMSPY | 6h |

### Phase D — Observability & MLOps (~25h)

| # | Item | Quelle | Aufwand |
|---|---|---|---|
| 21 | Grafana+Prometheus+Loki+OTel | OpenTelemetry | 10h |
| 22 | Trading-Dashboards + Alerts | custom | 4h |
| 23 | CPCV (Combinatorial Purged CV) | López de Prado | 6-8h |
| 24 | Darts Probabilistic Forecasting | unit8co | 10h |

### Phase E — Optional / Spekulativ

| # | Item | Quelle | Aufwand |
|---|---|---|---|
| 25 | FinRL Experiment-Setup | AI4Finance | 8-12h |
| 26 | tsfresh Feature-Engineering | tsfresh | 6h |
| 27 | Polars Hot-Path-Migration | polars | 4-8h pro Path |
| 28 | BERTopic News-Clustering | BERTopic | 8-12h |
| 29 | Riskfolio Drop-In-Migration | Riskfolio | 12-20h |

---

## Zusammenfassung der NEUEN Themen-Bereiche

Verglichen mit Teil 1 (`COMPETITIVE_ANALYSIS_2026-04-27.md`) bringen diese 8 neuen Bereiche **20+ konkrete neue Adoption-Items**:

**Was wirklich neu ist:**
1. ✨ **Execution-Algorithmen** (TWAP/VWAP/Almgren-Chriss) — kompletter Bereich, nicht in Teil 1
2. ✨ **Secrets-Management** (SOPS, Infisical, gitleaks) — kompletter Bereich
3. ✨ **RL für Trading** (FinRL/SB3) — kompletter Bereich
4. ✨ **Probabilistic Forecasting** (Darts, Conformal Prediction) — Teil 1 nur erwähnt
5. ✨ **Observability-Stack** (Grafana+Prometheus+OTel) — kompletter Bereich
6. ✨ **MLOps Tools** (MLflow + DVC) — kompletter Bereich
7. ✨ **Production-Patterns** (Pydantic, Graceful Shutdown, CompoundOrder) — neu detailliert
8. ✨ **López de Prado deep dive** (Meta-Labeling, Sample-Weighting, FracDiff) — Teil 1 nur Triple-Barrier

**Vorschlag für dich:** 

Schau dir die **Phase A** an (15h, sofortige Quick-Wins). Davon ist der **Pydantic-Config-Validator** und **SOPS-Secrets** das wertvollste, weil sie deine bestehende Codebasis sicherer machen, ohne dass du was Neues lernst.

Danach **Phase B** — und davon ist das **Anti-Leakage-Tool** (#8) immer noch der höchste-Hebel-Punkt: es schützt dich vor der gesamten Klasse von Bugs, die A1 und A9 nur teilweise abdecken.

Wenn du willst, schreibe ich dir den konkreten Implementierungs-Code für eines der Items aus Phase A oder B. Sag mir nur, welches.
