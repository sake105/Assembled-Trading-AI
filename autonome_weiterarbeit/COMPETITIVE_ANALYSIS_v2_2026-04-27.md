# Wettbewerbsanalyse v2: Die erweiterte Tiefenrecherche

**Datum:** 2026-04-27
**Vorgänger:** `COMPETITIVE_ANALYSIS_2026-04-27.md` (v1, 909 Zeilen, 7 Themen, 15+ Projekte)
**Was hier neu ist:** v2 erweitert die Recherche um **9 weitere Themenbereiche** und **30+ neue Projekte**, die in v1 nicht abgedeckt waren — mit Fokus auf Bereiche, die deinem Repo aktuell **fehlen** und die du bisher nicht im Blickfeld hattest.

**Wichtiger Lesehinweis:** Die Empfehlungen aus v1 sind weiterhin gültig und priorisiert. v2 fügt **strategische Erweiterungen** hinzu — keine Ersetzung, sondern Vervollständigung. Lies v1 zuerst für die Quick Wins.

---

## Inhaltsverzeichnis (v2)

8. [Microsoft Qlib — Das große, unterschätzte Vorbild](#8-microsoft-qlib--das-große-unterschätzte-vorbild)
9. [mlfinlab + López de Prado — Meta-Labeling als Game-Changer](#9-mlfinlab--lópez-de-prado--meta-labeling-als-game-changer)
10. [Execution-Algorithmen (VWAP/TWAP/Almgren-Chriss)](#10-execution-algorithmen)
11. [Secrets Management — der unterschätzte Sicherheitsbereich](#11-secrets-management)
12. [Reinforcement Learning für Trading (FinRL)](#12-reinforcement-learning-für-trading)
13. [Time-Series Forecasting mit darts & Co.](#13-time-series-forecasting)
14. [Observability Stack (Prometheus + Grafana + OpenTelemetry)](#14-observability-stack)
15. [Drift Detection & Model Monitoring](#15-drift-detection--model-monitoring)
16. [Hyperparameter-Optimierung (Optuna, Ray Tune)](#16-hyperparameter-optimierung)
17. [Market Microstructure & Order Flow](#17-market-microstructure--order-flow)
18. [Feature Stores (Feast, Hopsworks)](#18-feature-stores)
19. [Production-Pattern: pre-commit, ruff, mypy, GitGuardian](#19-production-pattern)
20. [Erweiterter Adoption-Plan (v2)](#erweiterter-adoption-plan-v2)
21. [Lizenz-Übersicht (vollständig, ~30 Projekte)](#lizenz-übersicht-vollständig)

---

## 8. Microsoft Qlib — Das große, unterschätzte Vorbild

### 8.1 Was Qlib ist und warum es relevant ist

**Repo:** `microsoft/qlib`
**Lizenz:** MIT (sehr freundlich für Adoption)
**Sponsor:** Microsoft Research, aktiv weiterentwickelt

Qlib ist mit Abstand das **architektonisch reifste Open-Source-Projekt für AI-getriebenes Quant Trading**. Es ist explizit für die Pipeline gebaut, die du bei dir aktuell hast: Daten → Features → Modell → Backtest → Live. Der entscheidende Punkt: Qlib trennt diese Stufen in **lose gekoppelte Module**, die einzeln nutzbar sind.

### 8.2 Architektur-Schichten, die wir lernen können

Qlib teilt seine Komponenten in vier explizite Layer:

1. **Infrastructure Layer**
   - `DataServer`: Hochperformante Datenmanagement-Schicht
   - `Trainer`: Steuert den Trainings-Lebenszyklus
2. **Workflow Layer**
   - `InformationExtractor` (Feature-Engineering)
   - `ForecastModel` (produziert alpha-/risk-Signale)
   - `PortfolioGenerator` (wandelt Signale in Gewichte um)
   - `OrderExecutor` (Order-Routing/Execution)
3. **Strategy Layer**
   - Modulare Strategien als austauschbare Klassen
4. **Interface Layer**
   - `Analyser` (generiert Reports)

**Was uns das lehrt:**

Bei dir verteilen sich diese Verantwortlichkeiten über `pipeline/`, `signals/`, `features/`, `risk/`, `portfolio/`, `intel/`. Qlib zeigt: **das ist eine Funktion, kein Bug**, ABER nur wenn die Schnittstellen zwischen den Layern explizit dokumentiert und stabil sind. Bei uns sind viele Schnittstellen implizit (man muss in den Code schauen, um sie zu finden).

### 8.3 Konkrete Adoption-Idee: Qlib's `qrun`-Pattern

Qlib hat ein YAML-getriebenes Workflow-Tool namens `qrun`, das die ganze Pipeline aus einer Konfig startet:

```yaml
# config_alpha158.yaml
qlib_init:
    provider_uri: "~/.qlib/qlib_data/cn_data"
    region: cn

market: &market csi300
benchmark: &benchmark SH000300

data_handler_config: &data_handler_config
    start_time: 2008-01-01
    end_time: 2020-08-01
    fit_start_time: 2008-01-01
    fit_end_time: 2014-12-31
    instruments: *market

task:
    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
        kwargs:
            loss: mse
            learning_rate: 0.0421
    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [2008-01-01, 2014-12-31]
                valid: [2015-01-01, 2016-12-31]
                test: [2017-01-01, 2020-08-01]
```

Dann: `qrun config_alpha158.yaml` — fertig.

**Warum das genial wäre für dich:**

Du hast aktuell wahrscheinlich pro Strategie ein eigenes Skript. Mit `qrun`-Pattern hast du **eine deklarative YAML pro Strategie**, der gleiche Runner ausführt sie alle. Vorteile:
- Komplette Reproduzierbarkeit (YAML committet → Strategie reproduzierbar)
- Kein duplizierter Bootstrap-Code mehr in 91 Skripten
- A/B-Tests werden trivial: zwei YAMLs vergleichen
- Backtest, Paper, Live alle mit dem gleichen Runner

**Aufwand für eine Adaption:** Hoch (16-30h). Aber das Ergebnis wäre eine **massive Reduktion deiner 91 Skripte** auf wenige zentrale Runner + viele YAMLs. Macht in Verbindung mit C4 (CLI-Konsolidierung — schon erledigt) und einem zukünftigen "Strategy-Registry"-Pattern viel Sinn.

### 8.4 Qlib's Alpha158/Alpha360 — Vorgefertigte Feature-Sets

Qlib hat zwei kanonische Feature-Sets, die sehr aufschlussreich sind:

- **Alpha158**: 158 technische Features aus OHLCV-Daten (Returns, Rolling Stats, RSI/MACD-Style, etc.)
- **Alpha360**: 360 Features mit längeren Lookback-Perioden

Den Code dafür findest du in `qlib/contrib/data/handler.py`. **Empfehlung:** Bevor du eigene Features baust, schau dir Alpha158 an. Sehr wahrscheinlich überschneiden sich 70-80% mit deinen, und einige clevere Features hast du noch nicht. Der Code ist MIT-Lizenz und kann direkt zitiert/adaptiert werden.

**Aufwand:** 4-6h für Code-Review von Alpha158 + Lücken-Identifikation in deinen Features.

### 8.5 Qlib's RD-Agent (NEU 2026)

Microsoft hat seit 2025 einen **automatischen R&D-Agenten** (`microsoft/RD-Agent`) gebaut, der Qlib-Konfigurationen automatisch generiert und Hypothesen testet. Ist im Frühstadium, aber wert anzuschauen:

```
RD-Agent automates:
- Hypothesis generation (e.g. "factor X might predict Y")
- Backtesting
- Result analysis
- Iteration to better hypotheses
```

**Empfehlung:** Sehr explorativ, aber falls du irgendwann LLM-getrieben Strategien automatisch testen willst, ist das die Richtung.

---

## 9. mlfinlab + López de Prado — Meta-Labeling als Game-Changer

### 9.1 Hudson and Thames `mlfinlab`

**Repo:** `hudson-and-thames/mlfinlab`
**Lizenz:** Hat sich geändert — früher MIT, jetzt **kommerziell mit Free-Tier** (nicht alle Module sind frei). Ältere Versionen (≤1.5) sind MIT.

mlfinlab ist die Referenz-Implementierung von López de Prado's "Advances in Financial Machine Learning" (AFML). Macht eine ganze Reihe akademisch fundierter Konzepte zugänglich, die wir bei uns ansatzweise haben:

- **Triple-Barrier Method** (du nutzt das in `triple_barrier.py`)
- **Meta-Labeling** (das ist der Game-Changer — siehe unten)
- **CUSUM Filter** für Event-Sampling
- **Fractional Differentiation** (Nicht-stationäre Reihen stationär machen ohne Memory-Verlust)
- **Sequentially Bootstrapped Ensembles** (für überlappende Labels)
- **Hierarchical Risk Parity** (HRP) mit korrekten Algorithmen
- **Correlation Filtering** für Strategien
- **Backtest Statistics** mit korrekten Standard-Definitionen

### 9.2 Meta-Labeling — der Game-Changer

Das wichtigste Konzept aus mlfinlab, das du **wahrscheinlich noch nicht nutzt**:

**Klassisches Modell:**
```
Daten → Modell → "Long" oder "Short" oder "Flat"
```

**Meta-Labeling:**
```
Daten → Primary Model → "Long" oder "Short" (= Side, Richtung)
       → Secondary Model → "Trade" oder "Don't trade" (= Confidence, Bet-Size)
```

**Warum das viel besser ist:**

Statt das Modell den Two-in-One-Job machen zu lassen (Richtung UND Vertrauen), trennst du die zwei Aufgaben. Das **Primary Model** ist oft eine simple regelbasierte Strategie (Mean-Reversion, Trend-Following, Breakout). Das **Meta Model** ist ein ML-Klassifikator, der lernt, **wann das Primary Model gut funktioniert** und wann nicht.

**Effekte (laut López de Prado):**
- Höhere Precision (weniger False Positives)
- Klarere Position Sizing (Meta-Output = Probability → Kelly-fraction)
- Easier Interpretierbarkeit
- Recall sinkt (manche Trades werden ausgelassen) — das ist OK

### 9.3 Konkrete Implementierung für uns

**Schritt 1:** Identifiziere ein bestehendes signal-modul bei dir (z.B. `signals/momentum.py`).

**Schritt 2:** Daraus wird das **Primary Model**. Es generiert Side-Predictions (Long/Short).

**Schritt 3:** Trainier ein Meta-Model:

```python
# Pseudocode
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score

# Primary signals from your existing strategy:
primary_signals = momentum_strategy(prices)  # -1, 0, +1

# Triple-barrier labels: did the primary signal work?
true_outcomes = triple_barrier_labels(
    prices,
    events=primary_signals.index[primary_signals != 0],
    pt_sl=[1.0, 1.0],  # 1 vol up, 1 vol down
    target=daily_volatility,
    num_days=5,
)
# true_outcomes: 1 if profit barrier hit, 0 if loss/timeout

# Meta-features: anything that might predict whether primary will succeed
meta_features = pd.DataFrame({
    "vol_regime": rolling_vol_regime(prices),
    "trend_strength": adx_indicator(prices),
    "spread": bid_ask_spread,
    "primary_signal": primary_signals,  # the prediction itself is a feature!
    "primary_confidence": primary_confidence,
    "session": session_label,  # AM/PM/EOD
    ...
})

# Train Meta Model on whether to take primary signal:
meta_model = GradientBoostingClassifier()
meta_model.fit(meta_features.loc[true_outcomes.index], true_outcomes)

# At inference:
primary_pred = momentum_strategy(new_prices)  # -1, 0, +1
if primary_pred != 0:
    meta_pred = meta_model.predict_proba(new_meta_features)[0, 1]  # P(success)
    if meta_pred > 0.6:  # threshold
        # take the trade with size proportional to meta_pred
        bet_size = (2 * meta_pred - 1) * max_bet  # Kelly-style
    else:
        bet_size = 0  # skip the trade
```

**Was du gewinnst:**
- Du kannst deine **bestehenden Strategien** als Primary-Models behalten
- ML kommt obendrauf, nicht statt der Strategie
- Position Sizing wird deutlich rationaler
- Bei schlechten Marktregimes filtert das Meta-Model Trades raus

**Aufwand:** 12-20h für Meta-Labeling-Wrapper um eine bestehende Strategie. Wenn das funktioniert, kannst du es auf alle Strategien anwenden.

**Lizenz-Hinweis:** Nicht den mlfinlab-Code übernehmen (Lizenz unklar geworden). Aber das Konzept ist **akademisch frei** (López de Prado's Buch). Eigene Implementierung schreiben.

### 9.4 Fractional Differentiation

Ein weiteres mlfinlab-Konzept: Statt klassischer Differentiation (Returns oder Log-Returns), die Memory komplett zerstört, nutzt **Fractional Differentiation** einen Float-Parameter `d` zwischen 0 und 1. Bei `d=1` hast du normale Returns. Bei `d=0.5` hast du etwas, das **gerade so stationär ist, aber Memory behält**.

```python
def fractional_diff(series, d, threshold=0.01):
    """Fractional differentiation with FFD (fixed-width window)."""
    weights = get_weights_ffd(d, threshold)
    width = len(weights)
    output = pd.Series(index=series.index, dtype=float)
    for i in range(width, len(series)):
        output.iloc[i] = np.dot(weights[::-1], series.iloc[i-width:i])
    return output
```

**Warum relevant:** Wenn deine ML-Modelle `log_return`-Features als Input bekommen, verlierst du fast die ganze Memory. Mit `d=0.4` hättest du Memory + Stationarität. Stefan Jansen's Buch hat ein ganzes Kapitel dazu.

**Aufwand:** 4-6h Implementation + Tests + Migration einzelner Features.

---

## 10. Execution-Algorithmen

### 10.1 Was wir aktuell vermutlich nicht haben

Du hast ein `pipeline/`, das Orders generiert, und einen `paper_trading_engine`, der sie versendet. Aber: **wenn die Orders zu groß sind, kollidieren sie mit dem Markt**. Das ist relevant ab ~5% des durchschnittlichen Tagesvolumens eines Symbols.

**Problem:** Eine 10000-Stück-Order auf einmal ans Limit-Orderbuch zu schicken bewegt den Preis. Der Slippage frisst dann mehr als deine erwartete Rendite. **Execution-Algorithmen** schneiden den großen Auftrag in viele kleine.

### 10.2 Drei Standard-Algorithmen

| Algorithm | Wie es funktioniert | Wann nutzen |
|---|---|---|
| **TWAP** (Time-Weighted Average Price) | Gleichmäßig über N Zeit-Slots verteilen | Einfachster Algo, wenig Volume-Profile-Annahmen |
| **VWAP** (Volume-Weighted Average Price) | Anteile proportional zum erwarteten Volume-Profile | Bei Symbolen mit klarem Tagesprofil (Eröffnung/Schluss-Volume) |
| **POV** (Percentage of Volume) | "Sei immer X% des aktuellen Marktvolumens" | Adaptiv, gut bei unsicherem Profil |
| **Implementation Shortfall** (Almgren-Chriss) | Optimiert Trade-Off zwischen Market-Impact und Timing-Risk | Theoretisch optimal, mathematisch komplex |

### 10.3 Implementierungs-Beispiele

**TWAP** (das Einfachste):
```python
def twap_schedule(
    parent_qty: int,
    start_time: datetime,
    end_time: datetime,
    n_slices: int,
) -> list[ChildOrder]:
    """Split parent order evenly over time."""
    duration = (end_time - start_time).total_seconds()
    slice_qty = parent_qty // n_slices
    remainder = parent_qty - (slice_qty * n_slices)
    
    children = []
    for i in range(n_slices):
        slice_time = start_time + timedelta(seconds=duration * i / n_slices)
        qty = slice_qty + (1 if i < remainder else 0)
        children.append(ChildOrder(time=slice_time, qty=qty))
    return children
```

**VWAP** (mit historischem Volume-Profile):
```python
def vwap_schedule(
    parent_qty: int,
    start_time: datetime,
    end_time: datetime,
    historical_volume_profile: pd.Series,  # 1 row per minute, normalized
    n_slices: int,
) -> list[ChildOrder]:
    """Slice in proportion to expected volume."""
    profile_window = historical_volume_profile.loc[start_time.time():end_time.time()]
    profile_normalized = profile_window / profile_window.sum()
    
    # Sample n_slices times based on profile distribution
    slice_minutes = profile_normalized.cumsum()
    
    children = []
    for i in range(n_slices):
        # Find minute where cumulative volume reaches (i+1)/n_slices
        target = (i + 1) / n_slices
        slice_minute = slice_minutes[slice_minutes >= target].index[0]
        qty = parent_qty // n_slices
        children.append(ChildOrder(time=slice_minute, qty=qty))
    return children
```

**Almgren-Chriss Implementation Shortfall** (komplex, aber optimal):

```python
def almgren_chriss_optimal_trajectory(
    parent_qty: int,
    total_time: float,           # in days
    sigma: float,                # daily volatility
    eta: float,                  # temporary impact (bps per share traded)
    gamma: float,                # permanent impact (bps per share permanently)
    risk_aversion: float = 1e-6, # higher = trade faster (less timing risk)
    n_slices: int = 50,
) -> pd.Series:
    """
    Optimal liquidation trajectory minimizing E[cost] + lambda * Var[cost].
    From Almgren & Chriss (2000).
    """
    kappa = np.sqrt(risk_aversion * sigma**2 / eta)
    tau = total_time / n_slices
    
    # Time grid
    t = np.linspace(0, total_time, n_slices + 1)
    
    # Optimal holdings trajectory (sinh decay)
    holdings = parent_qty * np.sinh(kappa * (total_time - t)) / np.sinh(kappa * total_time)
    
    # Trades (differences in holdings)
    trades = -np.diff(holdings)
    return pd.Series(trades, index=pd.to_datetime(t[:-1], unit='D', origin='start_of_day'))
```

### 10.4 Empfehlung für uns

**Nicht alle drei auf einmal bauen.** Mein Vorschlag:

1. **TWAP zuerst** (4-6h) — der einfachste, deckt 80% der Use-Cases ab.
2. **VWAP danach** (6-8h) — wenn Symbole mit klarem Volume-Profile (Equities, Open/Close).
3. **Almgren-Chriss später** (10-15h) — nur wenn du wirklich große Positionen hast, wo Market-Impact relevant wird.

**Wo das eingehängt werden müsste:**

Bei dir aktuell vermutlich `paper_trading_engine.submit_order(symbol, qty, side)`. Die Erweiterung wäre:

```python
def submit_order(
    self,
    symbol: str,
    qty: int,
    side: str,
    execution_algo: ExecutionAlgo | None = None,
):
    if execution_algo is None or qty < self.threshold_for_slicing:
        return self._submit_market_or_limit(symbol, qty, side)
    
    # Use slicing
    parent_id = self.create_parent_order(symbol, qty, side)
    children = execution_algo.schedule(symbol, qty, ...)
    for child in children:
        self.schedule_child_order(parent_id, child)
    return parent_id
```

**Lizenz:** Kein direktes Code-Copy — alle drei Algorithmen sind akademisch publiziert (Almgren-Chriss 2000, etc.) und frei zu implementieren.

---

## 11. Secrets Management

### 11.1 Warum das relevanter ist, als du denkst

Bei dir hast du Alpaca-API-Keys, vielleicht Polygon, Anthropic, IBKR, Datenbroker. Jede dieser Keys ist:
- Eine Tür in dein Geld
- Eine Tür in deinen Code
- Eine Tür in deine Daten

**Häufige Fehler bei Solo-Devs:**
1. `.env` committet (passiert öfter als man denkt)
2. Hardcoded API-Key in einem Test-Script "vergessen"
3. Logs, die volle URLs mit Keys ausgeben
4. PowerShell-Profil mit `$env:ALPACA_KEY = "..."` für Convenience
5. Keys ohne Expiration/Rotation für Jahre genutzt

### 11.2 Drei Tools, die wir kombinieren sollten

**1. GitGuardian (oder gitleaks) — Pre-Commit Secret Scanning**

Verhindert, dass Keys überhaupt in Git committet werden:

```yaml
# .pre-commit-config.yaml
- repo: https://github.com/gitleaks/gitleaks
  rev: v8.22.1
  hooks:
    - id: gitleaks
      name: "🔒 Detect hardcoded secrets"
```

Wenn du `OPENAI_API_KEY=sk-proj-...` versuchst zu committen, blockiert pre-commit dich.

**2. SOPS + age — Encrypted Secrets in Git**

`secrets.yaml` direkt in Git, aber **verschlüsselt**:

```yaml
# secrets.enc.yaml (encrypted with age)
alpaca:
    key: ENC[AES256_GCM,data:H8sNk...,iv:...,tag:...,type:str]
    secret: ENC[AES256_GCM,data:f3nKj...,iv:...,tag:...,type:str]
```

Dein Repo kann öffentlich/halböffentlich sein — die Secrets sind verschlüsselt. Nur wer den `age`-Private-Key hat, kann sie entschlüsseln.

```bash
# Install
brew install sops age   # oder choco/scoop unter Windows

# Generate key
age-keygen -o ~/.age/key.txt

# Encrypt
sops --encrypt --age age1xx... secrets.yaml > secrets.enc.yaml

# Decrypt at runtime
SOPS_AGE_KEY_FILE=~/.age/key.txt sops --decrypt secrets.enc.yaml
```

Setup-Aufwand: ~30 Minuten. Lizenz: MIT (SOPS), MIT (age).

**3. Infisical (oder OpenBao) — Vault für lokale Multi-Service-Secrets**

Wenn du mehr als 5-10 Secrets hast und mehrere Komponenten unterschiedliche Subsets brauchen, lohnt sich ein lokaler Vault. Infisical ist die moderne MIT-lizenzierte Alternative zu HashiCorp Vault (die jetzt BSL ist).

```yaml
# infisical-permissions.yaml
permissions:
  - service: backtest-engine
    access: [POLYGON_API_KEY, YFINANCE_USER_AGENT]
  - service: paper-trading-engine
    access: [ALPACA_KEY, ALPACA_SECRET]
  - service: claude-research-tool
    access: [ANTHROPIC_API_KEY]
```

Damit hat dein Backtest-Skript **keinen Zugriff** auf den Trading-API-Key, selbst wenn es kompromittiert wird. Das ist Defense-in-Depth.

### 11.3 Konkrete Empfehlung

**Sofort (1h):**
- gitleaks oder GitGuardian's CLI als pre-commit-Hook installieren
- `git history` einmal scannen mit `gitleaks detect --source . --log-opts="--all"` — checkt deine ganze History auf historische Leaks. Falls Treffer: Keys rotieren (egal wie alt).

**Kurzfristig (2-4h):**
- SOPS + age einführen
- Ein `secrets.enc.yaml` ins Repo, decrypted-version in `.gitignore`
- Bootstrap-Code in `pyproject.toml`/`Makefile`, der bei jedem Run sops decrypt aufruft

**Mittelfristig (4-8h):**
- Wenn du mehrere unabhängige Services hast: Infisical oder OpenBao docker-compose
- Pro Service IAM-Style Access-Policies

**API-Key-Hygiene (immer beachten):**
- **Trennung nach Funktion**: Trading-Key hat NIE `withdraw`-Permission
- **IP-Whitelisting** auf den Exchange-Seiten (Alpaca, IBKR können das)
- **Rotation** alle 60-90 Tage (per Kalender erinnert)
- **Read-Only-Keys** für Backtest-/Research-Scripts

---

## 12. Reinforcement Learning für Trading

### 12.1 FinRL — die Standard-Library

**Repo:** `AI4Finance-Foundation/FinRL`
**Lizenz:** MIT
**Status:** Mainstream-Reference, akademisch publiziert (NeurIPS 2020, ICAIF 2020, KDD 2022)

FinRL bietet RL-Trading-Environments out of the box. Drei Lagen:
1. **Market Environments** (`finrl.meta.env_*`): Stock-Trading, Crypto-Trading, Portfolio-Allocation als OpenAI-Gym-Environments
2. **DRL Agents** (`finrl.agents.*`): A2C, DDPG, PPO, SAC, TD3 (via Stable-Baselines3)
3. **Applications** (`finrl.applications.*`): Komplette End-to-End-Beispiele

### 12.2 Wann RL bei uns Sinn machen würde

**RL ist NICHT für jede Strategie sinnvoll.** Es ist sinnvoll für:

- **Order-Execution-Optimierung** (Almgren-Chriss als Belohnungsfunktion → RL findet adaptive Schedules)
- **Portfolio-Rebalancing** (statt fester Frequenz, Agent lernt wann)
- **Position-Sizing** unter Constraints (Risk-Limits, Inventory-Limits)
- **Regime-Erkennung + Strategie-Auswahl** (welche Strategie wann)

**RL ist NICHT sinnvoll für:**
- Direkte Preis-Vorhersage (das machen Supervised-Models besser)
- Hochfrequente Mikrostrukturen (zu rauschig, RL braucht zu viele Episoden)
- Erstmalige Strategie-Erkundung (zu opak, schwer zu validieren)

### 12.3 Praktischer Use-Case bei dir: Ensemble-Strategy

Eines der besten FinRL-Papers (ICAIF 2020): **"Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy"**.

Das Konzept:
1. Trainiere drei separate RL-Agenten: PPO, A2C, DDPG
2. Bei jedem Test-Fenster: nimm den Agent mit dem **besten Sharpe im letzten Validation-Window**
3. Verwende dessen Aktion für das nächste Fenster

Effekt: Adaptive Strategy-Switching basierend auf Marktregime.

```python
# Pseudocode for ensemble switching
class EnsembleAgent:
    def __init__(self):
        self.agents = {
            "ppo": PPOAgent(),
            "a2c": A2CAgent(),
            "ddpg": DDPGAgent(),
        }
    
    def select_agent(self, validation_window):
        # Each agent ran in validation, compute Sharpe
        sharpes = {
            name: compute_sharpe(agent.predictions(validation_window))
            for name, agent in self.agents.items()
        }
        return max(sharpes, key=sharpes.get)
    
    def act(self, state, last_validation_window):
        chosen = self.select_agent(last_validation_window)
        return self.agents[chosen].predict(state)
```

**Aufwand:** Hoch (40-80h für sauberes Setup). RL ist nicht trivial. Aber FinRL nimmt 80% der Plumbing-Arbeit ab.

**Empfehlung:** **Niedrige Priorität.** Erst wenn die anderen Bereiche (Anti-Leakage, Drift, Meta-Labeling) gemacht sind. RL ist kein Quick-Win.

---

## 13. Time-Series Forecasting

### 13.1 darts (`unit8co/darts`) — der "scikit-learn für Time Series"

**Lizenz:** Apache 2.0
**Was es bietet:** Eine einheitliche `fit()`/`predict()`-API für **30+ Forecasting-Modelle**, von ARIMA bis N-BEATS bis LSTM. Plus Backtesting, Probabilistische Forecasts, Anomaly Detection.

### 13.2 Warum darts für uns interessant ist

Wenn du in deinem Repo irgendwo Vorhersagen für **kontinuierliche Werte** machst (z.B. nächster Return, nächste Volatilität, IV-Term-Structure-Forecast), nutzt du wahrscheinlich custom-Code mit Sklearn. Die Probleme dabei:

1. Für Time-Series ist Sklearn's `cross_val_score` falsch (Zeitliches Leakage!)
2. Probabilistische Forecasts (Konfidenzintervalle) musst du selber bauen
3. Multivariate Modelle sind frickelig
4. Static Covariates (z.B. Symbol-Sektor) als Conditioning brauchen Custom-Code

**darts löst all das in einer einzigen API:**

```python
from darts import TimeSeries
from darts.models import ExponentialSmoothing, Prophet, NBEATSModel
from darts.utils.statistics import check_seasonality, plot_acf

# TimeSeries-Klasse hat eingebaute Time-Aware-Methoden
series = TimeSeries.from_dataframe(prices, "timestamp", "close")

# Einheitliche API über alle Modelle:
models = [
    ExponentialSmoothing(),
    Prophet(),
    NBEATSModel(input_chunk_length=30, output_chunk_length=7),
]

# Backtest gleicht für alle Modelle
backtests = []
for model in models:
    bt = model.historical_forecasts(
        series,
        start=0.5,                # ab 50% der Series
        forecast_horizon=7,       # 7-Tage-Forecasts
        stride=1,                 # täglich neu
        retrain=False,            # Nicht jeden Tag retrain (zu langsam)
        last_points_only=True,
    )
    backtests.append(bt)

# Mean Absolute Percentage Error
from darts.metrics import mape
for name, bt in zip(["ETS", "Prophet", "NBEATS"], backtests):
    print(f"{name}: MAPE = {mape(series, bt):.2%}")
```

### 13.3 Conformal Prediction in darts

Eines der wertvollsten neuen darts-Features: **Conformal Prediction**. Erlaubt **kalibrierte Quantile-Intervalle** für jedes Vorhersage-Modell.

```python
from darts.models.forecasting.conformal_models import ConformalNaiveModel

# Wrap any pre-trained forecasting model
conformal_model = ConformalNaiveModel(
    model=trained_nbeats,
    alpha=[0.05, 0.95],  # 90%-Konfidenz-Intervall
    cal_length=100,      # Kalibrierungs-Window
)

# Prediction with calibrated bounds
pred = conformal_model.predict(n=7)
# pred ist eine ProbabilisticTimeSeries mit Median + Quantilen
```

**Warum das relevant ist:** Statt nur "ich erwarte +0.5% Return morgen", bekommst du "[-1.2%, +2.1%] mit 90% Konfidenz". Das ist **direkt nutzbar für Position-Sizing** (Kelly-Fraction skaliert mit Konfidenz).

### 13.4 Empfehlung

Wenn du irgendwo zukünftig Continuous-Predictions brauchst (z.B. für Volatility-Forecasting, oder Carry/Spread-Forecasting), **mach es nicht selbst, nutze darts**.

**Aufwand:** Erste Integration 4-6h. Pro Use-Case dann jeweils 2-4h. **Lizenz: Apache 2.0**, sehr freundlich.

---

## 14. Observability Stack

### 14.1 Was du aktuell hast vs. was Standard ist

Du hast wahrscheinlich:
- Logs in Files (JSON-Lines mit Loguru oder Standard-Logging)
- Manuell aufgerufene QA-Reports
- Vielleicht ein paar Plots in `qa/`-Outputs

Was Standard ist (laut allen produktiven Trading-Setups, die ich gefunden habe):
- **Metrics**: Prometheus (Pull-Modell, Time-Series-DB)
- **Logs**: Loki (von Grafana, Label-basiert, nicht volltextindiziert)
- **Traces**: Tempo oder Jaeger (Distributed Tracing)
- **Visualization**: Grafana
- **Profiling**: Pyroscope (Continuous Profiling)
- **Instrumentation Standard**: OpenTelemetry (vendor-neutral)

### 14.2 Was wir bauen sollten — "LGTM"-Stack

Grafana nennt es selbst **LGTM-Stack**: Loki, Grafana, Tempo, Mimir (Metrics).

**Setup für Solo-Dev:**

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports: ["9090:9090"]
  
  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
  
  loki:
    image: grafana/loki:latest
    ports: ["3100:3100"]
  
  tempo:
    image: grafana/tempo:latest
    ports: ["3200:3200"]
```

3 Stunden Setup, dann hast du eine Production-Grade-Observability, die Hedgefunds nutzen.

### 14.3 Konkrete Metrik-Idee für dich

Aus dem Trading-Domain-Wissen einige relevante Metriken:

```python
# Bei jedem Trade-Event:
from prometheus_client import Counter, Histogram, Gauge

trades_total = Counter(
    "trading_orders_submitted_total",
    "Total orders submitted",
    ["strategy", "symbol", "side"],
)

slippage_bps = Histogram(
    "trading_slippage_bps",
    "Slippage in basis points (signed)",
    ["strategy", "symbol"],
    buckets=[-50, -10, -5, -1, 0, 1, 5, 10, 50],
)

position_pnl_unrealized = Gauge(
    "trading_position_pnl_unrealized_usd",
    "Unrealized PnL per position",
    ["strategy", "symbol"],
)

drawdown_current = Gauge(
    "trading_drawdown_current",
    "Current drawdown percentage",
    ["strategy"],
)

api_latency = Histogram(
    "trading_api_latency_seconds",
    "API request latency",
    ["endpoint", "broker"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Bei jedem Order-Submit:
trades_total.labels(strategy=name, symbol=sym, side=side).inc()
slippage_bps.labels(strategy=name, symbol=sym).observe(slip)

# Im Trading-Loop:
position_pnl_unrealized.labels(strategy=name, symbol=sym).set(pnl)
drawdown_current.labels(strategy=name).set(dd_pct)
```

### 14.4 Alerting — die Killer-Funktion

Mit Grafana-Alerts kannst du Folgendes einrichten:

```yaml
# Beispiel-Alert
- alert: DrawdownExceeded
  expr: trading_drawdown_current > 0.05
  for: 5m
  annotations:
    summary: "Strategy {{ $labels.strategy }} drawdown > 5%"
  labels:
    severity: critical
```

→ Wenn das passiert, Telegram/Slack/Email-Notification + automatisches Kill-Switch-Trigger.

### 14.5 Empfehlung

**Sofort (3-5h):**
- docker-compose mit Prometheus + Grafana + Loki + Tempo aufsetzen
- 5-10 Standard-Trading-Metriken instrumentieren
- 3-4 Standard-Dashboards (Drawdown, PnL, Slippage, API-Latency)

**Mittelfristig (zusätzlich 5-10h):**
- OpenTelemetry SDK für Distributed Tracing
- Alerts mit Webhook-Integration zu Telegram

**Lizenz:** Alles Open-Source und frei.

---

## 15. Drift Detection & Model Monitoring

### 15.1 Das Problem: ML-Modelle altern

Du trainierst ein Modell auf 2022-2024 Daten. Es ging 2025 noch gut. 2026 fängt es an, schlechter zu werden — aber wann genau? Und warum? Drift hat zwei Hauptformen:

- **Data Drift**: Die Verteilung der Inputs ändert sich (z.B. Volatility-Regime ändert sich)
- **Concept Drift**: Die Beziehung zwischen Inputs und Outputs ändert sich (z.B. eine Indikator-Strategie funktioniert nicht mehr)

### 15.2 Drei Tools, die wir nutzen sollten

**1. Evidently AI (`evidentlyai/evidently`)** — die Standard-Library
- Lizenz: Apache 2.0
- Features: 100+ Drift-Metriken (Jensen-Shannon, KS-Test, PSI, Wasserstein), Pandas-friendly, HTML-Reports, Dashboards
- Use-Case: Daily check, ob Production-Daten von Training-Daten driften

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=training_features,    # was das Modell gesehen hat
    current_data=last_30d_features,       # was es jetzt sieht
)
report.save_html("drift_report.html")
```

**2. NannyML (`NannyML/nannyml`)** — performance estimation ohne Labels
- Lizenz: Apache 2.0
- Features: Estimating Performance Without Targets (CBPE, DLE), PCA-based multivariate drift detection
- Use-Case: Wenn du noch nicht weißt, ob deine Trades 2 Wochen später profitabel waren — NannyML schätzt die Performance vorab

```python
import nannyml as nml

estimator = nml.CBPE(
    y_pred_proba="prediction_probability",
    y_pred="prediction",
    y_true="actual",  # für Reference
    metrics=["roc_auc", "f1"],
    chunk_size=5000,
)
estimator.fit(reference_data)
estimated_results = estimator.estimate(production_data)
estimated_results.plot().show()
```

**3. Frouros (`IFCA-Advanced-Computing/frouros`)** — Klassische Drift-Detection-Algorithmen
- Lizenz: BSD-3
- Features: 25+ Drift-Detektoren (DDM, EDDM, ADWIN, CUSUM-Drift, Page-Hinkley)
- Use-Case: Online-Detektoren für Streaming-Settings

### 15.3 Konkrete Implementierung für uns

**In deinem `trading_cycle_v2`** sollte vor jedem Modell-Inference ein Drift-Check laufen:

```python
# qa/drift_monitor.py

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

class DriftMonitor:
    def __init__(self, reference_features: pd.DataFrame, drift_threshold: float = 0.5):
        self.reference = reference_features
        self.threshold = drift_threshold
        self.recent_window = []
    
    def check(self, current_features: pd.DataFrame) -> dict:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference, current_data=current_features)
        
        # Parse drift result
        drift_dict = report.as_dict()
        n_drifted = drift_dict["metrics"][0]["result"]["number_of_drifted_columns"]
        n_total = drift_dict["metrics"][0]["result"]["number_of_columns"]
        share = n_drifted / n_total
        
        return {
            "share_drifted": share,
            "n_drifted": n_drifted,
            "should_pause": share > self.threshold,
            "report_path": "drift_report.html",
        }

# In trading_cycle_v2:
def run_trading_cycle(ctx):
    features = compute_features(ctx)
    drift_check = ctx.drift_monitor.check(features)
    if drift_check["should_pause"]:
        ctx.logger.warning(f"DRIFT: {drift_check['share_drifted']:.0%} of features drifted. Pausing.")
        return  # skip trading this cycle
    ...
```

### 15.4 Empfehlung

**Phase 1 (3-5h):**
- Evidently für täglichen Drift-Check einbauen
- HTML-Reports automatisch in `qa/drift_reports/` ablegen
- In Pipeline einhängen: bei Drift > Threshold → Strategie pausieren

**Phase 2 (5-8h):**
- NannyML für Performance-Estimation **vor** dem Truth-Window
- Alerts in Grafana

**Lizenz:** Alles Apache 2.0 / BSD — voll frei nutzbar.

---

## 16. Hyperparameter-Optimierung

### 16.1 Was wir aktuell vermutlich nutzen

Vermutlich Grid-Search oder Random-Search in deinen Sweeps (`scripts/run_sweep_*.py`). Das ist OK für 2-3 Parameter, aber bei 10+ Parametern ist Bayesian-Optimization deutlich effizienter.

### 16.2 Optuna (`optuna/optuna`) — Standard für Bayesian HP-Opt

**Lizenz:** MIT
**Stand:** v4.8.0 (März 2026), aktiv weiterentwickelt
**Features:**
- TPE-Sampler (Tree-Parzen-Estimator)
- CMA-ES für continuous spaces
- GP-based Bayesian Optimization (seit v4.4)
- Multi-Objective-Optimization
- Pruner für frühen Abbruch schlechter Trials
- SQLite/PostgreSQL/MySQL als Storage (parallele Workers!)
- Web Dashboard für Visualisierung

### 16.3 Was es konkret für uns bringt

**Beispiel: HP-Sweep für eine Trading-Strategie**

Bisher:
```python
# Grid-search über 5 Params, je 10 Werte = 100,000 Backtests
for fast_window in range(5, 50, 5):
    for slow_window in range(20, 200, 20):
        for vol_lookback in range(20, 100, 10):
            for stop_loss in [0.01, 0.02, 0.03, 0.05]:
                for tp in [0.02, 0.04, 0.08]:
                    sharpe = run_backtest(...)
                    results.append(...)
```

Mit Optuna:
```python
import optuna

def objective(trial):
    fast_window = trial.suggest_int("fast_window", 5, 50)
    slow_window = trial.suggest_int("slow_window", 20, 200)
    vol_lookback = trial.suggest_int("vol_lookback", 20, 100)
    stop_loss = trial.suggest_float("stop_loss", 0.005, 0.10, log=True)
    tp = trial.suggest_float("tp", 0.01, 0.20, log=True)
    
    # Constraint: slow > fast
    if slow_window <= fast_window:
        raise optuna.TrialPruned()  # Skip this trial
    
    sharpe = run_backtest(fast_window, slow_window, vol_lookback, stop_loss, tp)
    return sharpe

study = optuna.create_study(
    direction="maximize",
    storage="sqlite:///optuna_studies.db",  # persistent!
    study_name="momentum_v1",
    load_if_exists=True,
)
study.optimize(objective, n_trials=200)
print(f"Best Sharpe: {study.best_value:.2f}")
print(f"Best params: {study.best_params}")
```

**Was du gewinnst:**
- 200 Trials statt 100.000 (500× schneller)
- Bessere Ergebnisse, weil TPE schlechte Bereiche überspringt
- SQLite-Storage = **mehrere Worker parallel auf demselben Study** möglich
- Optuna-Dashboard zeigt Hyperparameter-Importance live

### 16.4 Multi-Objective-Optimization

Eines der mächtigsten Optuna-Features für Trading: **mehrere Ziele gleichzeitig optimieren**.

Beispiel: Du willst Sharpe maximieren, Drawdown minimieren, Trades-pro-Tag minimieren.

```python
def objective(trial):
    params = {...}
    sharpe, max_dd, trades_per_day = run_backtest(**params)
    return sharpe, -max_dd, -trades_per_day  # all to maximize

study = optuna.create_study(
    directions=["maximize", "maximize", "maximize"],
)
study.optimize(objective, n_trials=500)

# Get the Pareto-front (non-dominated solutions)
for trial in study.best_trials:
    print(trial.params, trial.values)
```

Statt einen einzigen "besten" Punkt bekommst du eine **Pareto-Front** — eine Familie von Lösungen, wo jede einen anderen Trade-off macht.

### 16.5 Empfehlung

**Sofort einbauen (4-6h):**
- Optuna als Dependency
- Ein bestehender `run_sweep_*`-Script auf Optuna umstellen
- Dashboard starten, Visualisierung anschauen

**Mittelfristig (zusätzlich 4-8h):**
- Multi-Objective für Sharpe/DD/Turnover
- Pruner für lange Backtests (Trial-Abbruch wenn frühe Periode schon schlecht)

---

## 17. Market Microstructure & Order Flow

### 17.1 Bereich, den wir noch gar nicht abdecken

Du arbeitest aktuell vermutlich auf OHLCV-Bar-Level. **Microstructure** ist die Schicht darunter: Bid/Ask-Spread, Order-Book-Imbalance, Order-Flow, Trade-Sign-Klassifikation.

### 17.2 Wann das relevant wird

- **Slippage-Schätzung** — präzise nur mit Microstructure-Modellen
- **Execution-Timing** — Order-Book-Imbalance prädiziert Mikrobewegungen
- **Trade-Sign** — ohne Tick-Daten weißt du nicht, ob ein Trade Buyer- oder Seller-initiated war (relevant für deine Sentiment-Aggregation)

### 17.3 `mansoor-mamnoon/limit-order-book` als Lehrmaterial

C++ Order-Book-Engine + Python-SDK, mit eingebauten Microstructure-Analytics:

- **Realized Volatility** (Parkinson, Garman-Klass) — bessere Volatility-Schätzer als Daily-Close-to-Close
- **Impact Curves** — Trade-Size vs. Mid-Price-Move
- **Order-Flow-Autokorrelation** — wie persistent sind Buy-/Sell-Sequenzen
- **Imbalance Drift** — wie L1-Imbalance kurze Bewegungen prädiziert

Wenn du dich entscheidest, mit Tick-Daten zu arbeiten, ist das Repo **eine Goldgrube für die Algorithmen**.

### 17.4 Konkrete Algorithmen, die wir nutzen sollten

**1. Realized Volatility mit besseren Schätzern**

```python
def parkinson_volatility(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """Parkinson 1980 estimator using only High/Low."""
    log_hl = np.log(high / low) ** 2
    return np.sqrt(log_hl.rolling(period).sum() / (4 * period * np.log(2)))

def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Garman-Klass 1980 estimator."""
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    daily_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(daily_var.rolling(period).sum() / period)
```

Beide sind **deutlich präziser** als Standard `np.log(close).diff().rolling(20).std()` — weil sie Intra-Day-Range mitberücksichtigen.

**2. Tick-Rule für Trade-Sign-Klassifikation (ohne Bid/Ask)**

```python
def tick_rule_signs(prices: pd.Series) -> pd.Series:
    """Lee-Ready 1991 tick rule. +1 if price up, -1 if down, ffill if same."""
    diffs = prices.diff()
    signs = np.sign(diffs)
    signs = signs.replace(0, np.nan).ffill().fillna(1)
    return signs.astype(int)
```

Damit kannst du dein Volume in **Buyer-initiated** und **Seller-initiated** trennen, ohne Bid/Ask-Daten.

**3. Hawkes-Prozesse für Order-Flow-Clustering**

Order-Flow ist nicht Poisson-verteilt — Trades clustern. Ein **selbsterregender Hawkes-Prozess** modelliert das:

λ(t) = λ_0 + α · Σ exp(-β(t - tᵢ))

Findest du in `topics/order-flow` als implementierte Library.

### 17.5 Empfehlung

**Niedrige Priorität, aber strategisch wichtig:**

Wenn du **kurzfristig auf Bar-Level bleibst**, sind diese Sachen nicht akut. Wenn du irgendwann auf Tick/Sub-Minute umsteigst, brauchst du:

1. Realized-Volatility-Schätzer (Parkinson/Garman-Klass) **können sofort eingebaut werden**, nutzen nur OHLC und sind deutlich besser. Aufwand: 2-3h.
2. Tick-Rule für Trade-Sign — wenn du SIP-Trade-Daten hast (Polygon liefert das). Aufwand: 4-6h.
3. Hawkes-Prozesse für hochfrequente Strategien. Aufwand: 15-25h.

---

## 18. Feature Stores

### 18.1 Was Feature-Stores sind (und warum sie hilfreich sind)

Bei dir aktuell: Features werden in jedem Lauf neu berechnet, oder einmal in Parquet gecacht. Das skaliert OK für ein paar Strategien.

Feature-Store löst:
- **Feature-Versioning** (welche Features wurden zu welchem Modell-Training genutzt?)
- **Time-Travel** (Features as-of einem bestimmten Zeitpunkt — wichtig für PIT-Universe!)
- **Online/Offline-Konsistenz** (gleiche Features in Training und Live)
- **Feature-Reuse** über Strategien hinweg

### 18.2 Drei Optionen

**1. Feast (`feast-dev/feast`)** — Open-Source-Standard
- Lizenz: Apache 2.0
- Pro: Pip-installable, keine Cloud-Abhängigkeit
- Con: Eigentlich für Online-Inference gebaut, weniger für Backtest-Time-Travel

**2. Hopsworks (`logicalclocks/hopsworks`)** — Vollständiger ML-Lakehouse
- Lizenz: AGPL-3 (heikel) oder Commercial
- Pro: Sehr mächtig, Time-Travel built-in, Drift-Detection eingebaut
- Con: Schwergewichtig

**3. DIY mit Parquet + DuckDB**
- Lizenz: deine eigene
- Pro: Kein Dependency-Sumpf
- Con: Du baust Feature-Store-Logik selber

### 18.3 Empfehlung: DIY mit DuckDB

Für deine Repo-Größe wäre eine **eigene minimale Feature-Store-Schicht** mit DuckDB unter dem Hood die beste Lösung:

```python
# data/feature_store.py

import duckdb
import pandas as pd
from pathlib import Path

class FeatureStore:
    """Time-travel-aware feature store for backtest correctness."""
    
    def __init__(self, root: Path):
        self.root = root
        self.con = duckdb.connect(str(root / "feature_store.duckdb"))
    
    def register_feature_set(self, name: str, df: pd.DataFrame, ttl_days: int | None = None):
        """Register a new feature snapshot, indexed by (symbol, timestamp)."""
        df["snapshot_ts"] = pd.Timestamp.now()
        if ttl_days:
            df["expires_ts"] = df["snapshot_ts"] + pd.Timedelta(days=ttl_days)
        self.con.execute(f"CREATE TABLE IF NOT EXISTS {name} AS SELECT * FROM df")
        self.con.execute(f"INSERT INTO {name} SELECT * FROM df")
    
    def get_features_as_of(
        self,
        feature_set: str,
        as_of_ts: pd.Timestamp,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Get features that were available at as_of_ts (PIT-correct!)."""
        query = f"""
            SELECT * FROM {feature_set}
            WHERE timestamp <= ?
              AND snapshot_ts <= ?
              AND (expires_ts IS NULL OR expires_ts > ?)
        """
        params = [as_of_ts, as_of_ts, as_of_ts]
        if symbols:
            query += f" AND symbol IN ({','.join(['?'] * len(symbols))})"
            params.extend(symbols)
        query += " ORDER BY symbol, timestamp"
        return self.con.execute(query, params).df()
```

**Was du gewinnst:**
- Backtest-Code spricht: `features = store.get_features_as_of("alpha158", t)` — **kein PIT-Bug mehr möglich**
- Versioning durch `snapshot_ts`
- Trivialer Live-Mode: gleiche API mit `as_of_ts=now()`

**Aufwand:** 8-12h für eine saubere Implementation.

---

## 19. Production-Pattern

### 19.1 Pre-commit-Hooks im 2026er Standard

Bei dir wahrscheinlich aktuell:
- pytest läuft in CI
- Ruff/mypy vielleicht?
- Black?

**State-of-the-Art 2026 (was ich in den großen Trading-Repos gesehen habe):**

```yaml
# .pre-commit-config.yaml
repos:
  # 1. Standard hygiene
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-json
      - id: check-merge-conflict
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: detect-private-key
      - id: mixed-line-ending
  
  # 2. Python lint + format (Ruff has replaced black + isort + flake8 + pyupgrade)
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.12
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
  
  # 3. Type checking
  - repo: https://github.com/RobertCraigie/pyright-python
    rev: v1.1.391
    hooks:
      - id: pyright
  
  # 4. Security scanning
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.22.1
    hooks:
      - id: gitleaks
  
  # 5. Bandit for known security anti-patterns
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.10
    hooks:
      - id: bandit
        args: [-c, pyproject.toml]
        additional_dependencies: ["bandit[toml]"]
  
  # 6. pyproject.toml validation
  - repo: https://github.com/abravalheri/validate-pyproject
    rev: v0.23
    hooks:
      - id: validate-pyproject
        additional_dependencies: ["validate-pyproject-schema-store[all]"]
```

**Effekt:** Nichts kommt mehr ungefiltert in dein Repo. Code-Qualität, Type-Sicherheit, Sicherheits-Lecks — alles automatisch.

### 19.2 Pyright vs. mypy

**Pyright** (Microsoft, in Pylance integriert) ist mittlerweile das schnellere und feature-reichere Type-Checker. Für ein 120k-LOC-Repo ist Pyright spürbar schneller. mypy hat einen Vorsprung bei Plugin-Support.

**Empfehlung:** Wenn du aktuell mypy nutzt, kannst du das stehen lassen. Wenn du noch nicht typisierst, fang mit Pyright in `basic`-mode an.

### 19.3 GitHub Actions — moderne Patterns

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      
      # uv für 10x schnelleren Install (vs pip)
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      
      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: uv sync --frozen --all-extras
      
      - name: Run pre-commit
        run: uv run pre-commit run --all-files
      
      - name: Run tests with coverage
        run: uv run pytest -n auto -p randomly --cov --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v5
        with:
          file: ./coverage.xml
  
  # Pre-commit-CI als zusätzlicher Layer
  pre-commit-autoupdate:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - uses: pre-commit-ci/lite-action@v1
```

Highlights:
- **uv** statt pip (Astral.sh, 10x schneller)
- **pytest-xdist** für Parallel-Tests
- **pytest-randomly** für versteckte Test-Reihenfolge-Abhängigkeiten
- Matrix für mehrere Python-Versionen
- Codecov-Integration

### 19.4 uv als pip-Ersatz

`uv` (Astral.sh, Rust-implementiert) ist **massiv schneller** als pip. Bei deinen vielen Dependencies kann der Install-Step von 3 Minuten auf 10 Sekunden fallen.

```bash
# Install
pip install uv  # einmalig

# uv lockfile generieren (pyproject.toml + uv.lock)
uv sync

# Run script in env
uv run python scripts/run_backtest.py
```

Lock-File-Format ist kompatibel mit `pyproject.toml` — kein Migrations-Pain.

**Aufwand:** 2-3h für die Migration deines aktuellen Setups.

---

## Erweiterter Adoption-Plan v2

Konsolidiert aus v1 + v2, sortiert nach **(Nutzen / Aufwand / Risiko)**:

### Tier 1: Sofort einbauen (~10-15h, sehr niedriges Risiko)

| # | Empfehlung | Aufwand | Quelle |
|---|---|---|---|
| 1 | gitleaks/GitGuardian als pre-commit-Hook | 1h | v2 §11 |
| 2 | Ledoit-Wolf Shrinkage statt `df.cov()` | 1-2h | v1 §3.2 |
| 3 | empyrical-reloaded für kanonische Metriken | 3-4h | v1 §4.2 |
| 4 | Healthcheck-Pattern statt Heartbeat-Commits | 30min | v1 §7.3 |
| 5 | Parkinson/Garman-Klass Vol-Schätzer | 2-3h | v2 §17.4 |
| 6 | Pre-commit-Stack vollständig (Ruff+Pyright+Bandit) | 2-3h | v2 §19.1 |
| 7 | uv als pip-Ersatz | 2-3h | v2 §19.4 |

### Tier 2: Sprint einplanen (~30-50h, niedrig-mittleres Risiko)

| # | Empfehlung | Aufwand | Quelle |
|---|---|---|---|
| 8 | Anti-Leakage-Tool (Lookahead + Recursive) | 8-12h | v1 §2.1 |
| 9 | Bootstrap-CIs für Performance-Metriken | 3-4h | v1 §1.3 |
| 10 | Evidently AI Drift-Monitor | 3-5h | v2 §15.3 |
| 11 | Optuna für Hyperparameter-Sweeps | 4-6h | v2 §16.5 |
| 12 | Prometheus + Grafana + Loki Observability-Stack | 5-8h | v2 §14.5 |
| 13 | quantstats HTML-Tearsheets | 2-3h | v1 §4.1 |
| 14 | TWAP Order-Slicing | 4-6h | v2 §10.4 |
| 15 | SOPS+age für encrypted Secrets | 2-4h | v2 §11.3 |

### Tier 3: Strategische Investments (~50-100h, mittleres Risiko)

| # | Empfehlung | Aufwand | Quelle |
|---|---|---|---|
| 16 | Meta-Labeling für eine bestehende Strategie | 12-20h | v2 §9.3 |
| 17 | DIY-Feature-Store mit DuckDB | 8-12h | v2 §18.3 |
| 18 | Combinatorial Purged CV (López de Prado) | 6-8h | v1 §2.2 |
| 19 | Broker-Adapter-Pattern (Nautilus-style) | 8-16h | v1 §1.1 |
| 20 | DataBlob-Pattern (pysystemtrade-style) | 8-12h | v1 §1.4 |
| 21 | RiskModel-Chain (QSTrader-style) | 6-10h | v1 §1.2 |
| 22 | qrun-Pattern (Qlib-style YAML-Workflow) | 16-30h | v2 §8.3 |
| 23 | Fractional Differentiation für Features | 4-6h | v2 §9.4 |
| 24 | VWAP Order-Slicing + Almgren-Chriss | 10-15h | v2 §10.4 |

### Tier 4: Spekulativ / Zukunft (~weitere 50-100h)

| # | Empfehlung | Aufwand | Quelle |
|---|---|---|---|
| 25 | FinBERT für Sentiment-Pipeline | 4-6h | v1 §5.1 |
| 26 | darts für Time-Series-Forecasting | 4-6h | v2 §13.4 |
| 27 | NannyML für Performance-Estimation | 5-8h | v2 §15.3 |
| 28 | FinRL für Order-Execution-RL | 40-80h | v2 §12.3 |
| 29 | Hawkes-Prozesse für Order-Flow | 15-25h | v2 §17.4 |
| 30 | BERTopic News-Clustering | 8-12h | v1 §5.2 |
| 31 | Infisical Secrets-Vault | 4-8h | v2 §11.3 |
| 32 | Riskfolio-Lib volle Migration | 12-20h | v1 §3.1 |

---

## Lizenz-Übersicht (vollständig)

Erweiterte Tabelle aller in v1 + v2 erwähnten Projekte:

### MIT / BSD / Apache 2.0 (frei nutzbar, sogar Code-Übernahme)

| Projekt | Lizenz | Was wir damit dürfen |
|---|---|---|
| QSTrader | MIT | Alles ✓ — Copy-Paste mit Vermerk |
| PyBroker | Apache 2.0 | Alles ✓ |
| PyPortfolioOpt | MIT | Alles ✓ |
| Riskfolio-Lib | BSD-3 | Alles ✓ |
| quantstats | Apache 2.0 | Alles ✓ |
| empyrical-reloaded | Apache 2.0 | Alles ✓ |
| FinBERT | Apache 2.0 | Alles ✓ |
| BERTopic | MIT | Alles ✓ |
| polars | MIT | Alles ✓ |
| **Microsoft Qlib** | **MIT** | **Alles ✓ — sehr wichtig** |
| FinRL | MIT | Alles ✓ |
| darts | Apache 2.0 | Alles ✓ |
| NannyML | Apache 2.0 | Alles ✓ |
| Evidently AI | Apache 2.0 | Alles ✓ |
| Frouros | BSD-3 | Alles ✓ |
| Optuna | MIT | Alles ✓ |
| Prometheus | Apache 2.0 | Alles ✓ |
| Grafana | AGPL-3 (Komponente, nicht Code-Linkage) | Als externes Tool ✓ |
| Loki | AGPL-3 (gleich wie Grafana) | Als externes Tool ✓ |
| Tempo | AGPL-3 | Als externes Tool ✓ |
| OpenTelemetry SDK | Apache 2.0 | Alles ✓ |
| SOPS | MPL-2.0 | Als externes Tool ✓ |
| age | BSD-3 | Alles ✓ |
| Infisical | MIT | Alles ✓ |
| OpenBao | MPL-2.0 | Als externes Tool ✓ |
| gitleaks | MIT | Alles ✓ |
| Feast | Apache 2.0 | Alles ✓ |
| DuckDB | MIT | Alles ✓ |
| stefan-jansen/m4t (Buch-Repo) | MIT | Alles ✓ |
| pre-commit | MIT | Alles ✓ |
| Ruff | MIT | Alles ✓ |
| Pyright | MIT | Alles ✓ |
| uv | Apache 2.0 / MIT | Alles ✓ |

### LGPL (Konzept ja, Code nur in separates Modul)

| Projekt | Lizenz | Was wir damit dürfen |
|---|---|---|
| NautilusTrader | LGPL-3 | Konzept ✓, Code-Linkage nur als externe Library |

### GPL/AGPL (Konzept ja, Code-Übernahme NEIN)

| Projekt | Lizenz | Was wir damit dürfen |
|---|---|---|
| pysystemtrade | GPL-3 | Konzept ✓, Code-Übernahme NEIN |
| Freqtrade | GPL-3 | Konzept ✓, Code-Übernahme NEIN |
| Backtrader | GPL-3 | Konzept ✓, Code-Übernahme NEIN |
| Hopsworks | AGPL-3 | Als externes Tool, Code-Linkage gefährlich |
| Riskfolio-Lib (alte Versionen) | BSD | OK |

### Kommerziell / Verändert

| Projekt | Lizenz | Was wir damit dürfen |
|---|---|---|
| HashiCorp Vault | BSL (seit 2023) | Vorsicht — OpenBao stattdessen |
| Tecton | Kommerziell | Nur als SaaS |
| Hopsworks Pro | Kommerziell | Kein Code-Reuse |
| **mlfinlab** | **Hat sich zu kommerziell geändert** | **Konzepte ja, Code-Übernahme nicht mehr** |
| vectorbt-pro | Kommerziell | Konzept ja, Code-Übernahme NEIN |

---

## Schluss-Bemerkung

Wenn du das alles ernst nimmst: Du hast jetzt einen **Plan für die nächsten 6-12 Monate**, mit klar priorisierten Schritten. Der Plan ist:

**Quartal 1 (~30h):** Tier 1 + Tier 2 → schnelle Qualitätssprünge.
**Quartal 2 (~50h):** Top-3 aus Tier 3 → Meta-Labeling + Feature-Store + ein Architektur-Refactor (z.B. Broker-Adapter).
**Quartal 3 (~30h):** restliches Tier 3, je nach Use-Cases.
**Quartal 4 (~50h):** Tier 4 selektiv, je nach was sich aus Q1-Q3 als nächste Priorität herausstellt.

Ich würde dir konkret raten:

1. **Diese Woche:** Tier-1-Items 1-4 (gitleaks, Ledoit-Wolf, empyrical, Healthcheck). ~6-8h, alle haben sehr niedriges Risiko und du siehst sofort Ergebnisse.

2. **Nächste 2 Wochen:** Anti-Leakage-Tool (Tier 2 #8). Das ist die einzelne wichtigste Verbesserung. Es deckt eine Klasse von Bugs ab, die A1/A9 nicht abdecken.

3. **Danach:** Du entscheidest. Meta-Labeling (Tier 3 #16) bringt potentielle Performance-Sprünge. qrun-Pattern (Tier 3 #22) bringt massive Repo-Aufräum-Wirkung. Beide sind enorme Hebel, aber unterschiedliche Richtungen.

Lass mich wissen, womit du anfangen willst — bei jedem dieser Items kann ich dir den konkreten Implementierungs-Code, die genaue Repo-Stelle wo es hingehört, und Tests dazu schreiben.
