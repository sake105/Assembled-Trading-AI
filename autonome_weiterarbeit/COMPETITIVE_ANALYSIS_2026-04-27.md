# Wettbewerbsanalyse: Was wir von Open-Source-Projekten lernen können

**Datum:** 2026-04-27
**Scope:** Recherche von 15+ relevanten Open-Source-Projekten in den Bereichen Backtesting, Risk-Management, Portfolio-Optimierung, ML-Trading, News-Sentiment, Anti-Leakage. Pro Projekt: was sie gut machen, was wir adaptieren sollten, mit konkretem Aufwand und Implementierungs-Strategie.

**Wichtige Vorbemerkung zur Strategie:**

Bei "Adaption" gibt es drei Stufen, die wir je nach Lizenz und Aufwand wählen:

1. **Library als Dependency** (z.B. PyPortfolioOpt MIT-Lizenz, Riskfolio-Lib BSD): einfacher Import via `pip`. Geringster Aufwand, aber externe Abhängigkeit und meist deren API.
2. **Konzept-Adaption** (Pattern, Architektur, Idee): wir bauen unsere eigene Version inspiriert vom Original. Vorteilhaft, wenn wir die Lizenz nicht wollen (GPL!) oder mehr Kontrolle brauchen.
3. **Code-Übernahme mit Refactoring**: Quellcode lesen, zentrale Stelle in unsere Codebasis bringen, Lizenz-Header beibehalten oder Lizenz prüfen. Funktioniert nur bei MIT/BSD/Apache, niemals bei GPL/AGPL.

**Warnung:** Mehrere prominente Projekte (Freqtrade, pysystemtrade, vectorbt) sind **GPL** oder kommerziell. Davon dürfen wir Konzepte adaptieren, aber **keinen Code übernehmen**, wenn wir unser Repo nicht selbst GPL-isieren wollen. Für unsere Vorhaben heißt das: **Inspiration ja, Copy-Paste nein.**

---

## Inhaltsverzeichnis

1. [Themenbereich Backtesting & Pipeline-Architektur](#1-backtesting--pipeline-architektur)
2. [Themenbereich Walk-Forward & Anti-Leakage](#2-walk-forward--anti-leakage)
3. [Themenbereich Portfolio-Optimierung](#3-portfolio-optimierung)
4. [Themenbereich Risk-Management](#4-risk-management)
5. [Themenbereich News-Sentiment & NLP](#5-news-sentiment--nlp)
6. [Themenbereich Datenpipeline & Storage](#6-datenpipeline--storage)
7. [Themenbereich Code-Hygiene & Tooling](#7-code-hygiene--tooling)
8. [Priorisierter Adoption-Plan](#priorisierter-adoption-plan)

---

## 1. Backtesting & Pipeline-Architektur

### 1.1 NautilusTrader (`nautechsystems/nautilus_trader`)

**Was sie machen:** Production-grade Trading-Engine in Rust mit Python-Bindings. Deterministisch, event-driven. Bi-weekly Releases. Identische Strategie-Implementierung zwischen Research und Live. Im Backtest mit Nanosekunden-Auflösung.

**Was uns interessiert:**

Nautilus erzwingt **strukturell**, dass dein Strategie-Code in Backtest und Live identisch ist. Bei uns hast du `trading_cycle_v2` mit `mode="backtest"` und `mode="live"` — und die Tatsache, dass du A8 (Slippage in Backtest fehlt) gefunden hast, zeigt: bei Mode-Branching ist Drift einprogrammiert.

**Konzept zur Adaption:** **"Same code, different adapters"-Pattern.**

Statt `mode`-Branches in der Pipeline-Logik:
```python
# Aktuell bei dir (vereinfacht):
def run_trading_cycle(ctx, mode):
    ...
    if mode == "backtest":
        orders = simulate_orders(...)
    elif mode == "live":
        orders = submit_orders(...)
```

Statt dessen:
```python
# Nautilus-Style:
class BrokerAdapter(Protocol):
    def submit_order(self, order: Order) -> Fill: ...

class BacktestBroker(BrokerAdapter):
    def submit_order(self, order: Order) -> Fill:
        # Apply slippage, commission, spread
        return simulated_fill

class AlpacaBroker(BrokerAdapter):
    def submit_order(self, order: Order) -> Fill:
        return real_fill_from_alpaca

def run_trading_cycle(ctx, broker: BrokerAdapter):
    # No mode branching! Same code, different broker.
    orders = generate_orders(...)
    fills = [broker.submit_order(o) for o in orders]
```

**Vorteil:** Slippage kann nicht mehr "vergessen" werden — der `BacktestBroker` enthält sie als zentralen Schritt. Plus: Die Trennung zwischen Strategielogik und Order-Routing ist sauberer.

**Aufwand für uns:** Mittel-Hoch (8-16h). Macht in Verbindung mit B5 Sinn (Pipeline-Konsolidierung). Empfehlung: **erst nach B3/B5 angehen**, weil wir sonst doppelt refactorn.

**Lizenz:** LGPL v3 — Konzept ist frei, aber direkter Code-Übernahme wegen LGPL nur bei striktem Modul-Trennen.

---

### 1.2 QSTrader (`mhallsmoore/qstrader`)

**Was sie machen:** "Loosely-coupled collection of modules" für end-to-end Backtests mit modularer Architektur. MIT-Lizenz. Kleines aber sehr sauber strukturiertes Repo. Gemacht von Mike Halls-Moore (QuantStart), erfahrener Quant-Trainer.

**Was uns interessiert:**

QSTrader hat eine sehr klare Trennung zwischen:
- Signal-Generierung
- Portfolio-Konstruktion
- Risk-Management
- Execution
- Simulated Brokerage Accounting

Jedes als **separates austauschbares Modul**. Genau die Struktur, die du bei deinem `trading_cycle_v2` mit den 14 `_sp_*`-Helpern angefangen hast (C1 erledigt), aber QSTrader führt es konsequenter durch.

**Was wir konkret übernehmen können:**

QSTrader's `RiskModel`-Interface. Bei dir verteilt sich Risk-Logik über `risk/`, `pipeline/trading_cycle_v2.py` (Step 6-13 in `size_positions`), und mehrere überlay-Module. QSTrader macht es als ein einziges Interface:

```python
class RiskModel(Protocol):
    def __call__(
        self,
        dt: datetime,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply risk rules and return modified weights."""
```

Beispielimplementierungen wären bei uns:
- `VolTargetRiskModel`
- `CorrelationCapRiskModel`
- `CrowdingHHIRiskModel`

Statt **alle Risk-Schritte in `size_positions` zu machen**, bekommt jeder seinen eigenen aufgepluggten Risk-Modell, die in Reihe oder parallel angewendet werden:

```python
risk_chain = [
    VolTargetRiskModel(target=0.15),
    CrowdingHHIRiskModel(max_hhi=0.4),
    CorrelationCapRiskModel(max_corr=0.7),
]
weights = signal_weights
for risk_model in risk_chain:
    weights = risk_model(dt, weights)
```

**Vorteil bei uns:** A/B-Tests zwischen Risk-Konfigurationen werden trivial (`risk_chain_v1` vs `risk_chain_v2`). Plus: jedes Risk-Model ist isoliert testbar.

**Aufwand:** Mittel (6-10h). Kann inkrementell aus deinen schon extrahierten `_sp_*`-Helpern entstehen.

**Lizenz:** MIT — kannst du frei adaptieren, sogar copy-pasten mit Lizenz-Vermerk.

---

### 1.3 PyBroker (`edtechre/pybroker`)

**Was sie machen:** ML-fokussiertes Backtest-Framework, NumPy + Numba. **Das prominente Feature**: Walk-Forward-Analysis, die explizit "expanding window" macht — alle vergangenen Train-Daten werden für jedes neue Test-Fenster genutzt, nicht nur das letzte rolling window.

**Was uns interessiert:**

Ihre Walk-Forward-API ist konzeptionell klar:
```python
result = strategy.walkforward(
    timeframe='1m',
    windows=5,
    train_size=0.5,
)
```

Plus: ihre **bootstrapped trading metrics** — statt einfacher Sharpe/Sortino-Berechnung machen sie Bootstrap-Resampling über die Trades und liefern Konfidenzintervalle. Ein Sharpe von 1.5 ± 0.4 ist ehrlicher als ein punktgenauer 1.5.

**Was wir konkret adaptieren sollten:**

**Bootstrap-Konfidenzintervalle für Performance-Metriken.** Bei dir hat `qa/metrics.py` nur Punktwerte. Erweiterung:

```python
def compute_sharpe_with_ci(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    bootstrap_sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(returns.values, size=len(returns), replace=True)
        sr = sample.mean() / sample.std() * np.sqrt(252)
        bootstrap_sharpes.append(sr)
    return {
        "sharpe": returns.mean() / returns.std() * np.sqrt(252),
        "sharpe_ci_lower": np.percentile(bootstrap_sharpes, 2.5),
        "sharpe_ci_upper": np.percentile(bootstrap_sharpes, 97.5),
        "sharpe_p_value": (np.array(bootstrap_sharpes) <= 0).mean(),
    }
```

Damit bekommst du in deinen Backtest-Reports nicht nur "Sharpe = 1.5", sondern "Sharpe = 1.5 [0.9, 2.1], p-value = 0.012". Dramatisch ehrlicher und aufschlussreicher.

**Plus: Expanding Window in Walk-Forward**

Dein A9-Fix hat `purge_days` und `embargo_days` — sehr gut. Nächster Schritt wäre: Default-Konfiguration auf **expanding window** umstellen. Statt fixed `train_days=252`, lass ihn auf `train_start = corpus_start, train_end = test_start - purge_days` skalieren. Bei jedem neuen Test-Fenster wächst Train mit.

**Aufwand:** Klein (3-4h für Bootstrap-CIs, 2-3h für expanding option).

**Lizenz:** Apache 2.0 — frei nutzbar, sogar Code-Übernahme mit Vermerk.

---

### 1.4 pysystemtrade (`robcarver17/pysystemtrade`)

**Was sie machen:** Rob Carver, Autor von "Systematic Trading", betreibt damit live sein eigenes Geld 20h/Tag, 5 Tage/Woche. Sehr konservativ, futures-fokussiert, IB-spezifisch. Nicht hübsch, aber kampferprobt.

**Was uns interessiert:** Das `dataBlob`-Pattern.

```python
data = dataBlob([
    arcticFuturesContractPriceData,  # Time-series storage
    mongoFuturesContractData,        # Static config
    ibFuturesContractPriceData,      # Live broker
])
# Autoabstraktion:
data.broker_contract_price       = ibFuturesContractPriceData(ib_conn)
data.db_futures_contract_price   = arcticFuturesContractPriceData(mongo_db=mongo_db)
data.db_futures_contract         = mongoFuturesContractData(mongo_db=mongo_db)
```

**Warum genial:** Egal ob Live, Simulation oder Test — der Strategie-Code spricht immer mit `data.broker_contract_price.get_prices(symbol, date)`. Das tatsächliche Backend (CSV im Test, Mongo in Sim, IB Live) wird beim Bootstrap gewählt.

**Was wir konkret adaptieren sollten:**

Bei dir gibt es das ansatzweise mit `TradingContext`, aber unausgereift. Wir bauen einen `DataBlob`:

```python
# src/assembled_core/data/data_blob.py

class DataBlob:
    """Container for data sources with consistent interface."""
    
    def __init__(self, sources: list[type[DataSource]]):
        for src_cls in sources:
            instance = src_cls()
            attr_name = self._derive_attr_name(src_cls.__name__)
            setattr(self, attr_name, instance)
    
    @staticmethod
    def _derive_attr_name(class_name: str) -> str:
        # ParquetPriceData -> db_price_data
        # AlpacaBrokerData  -> broker_data
        # CSVUniverseData   -> static_universe_data
        ...

# Usage:
data = DataBlob([
    ParquetPriceData,
    AlpacaBrokerData,
    StaticUniverseData,
])
prices = data.db_price_data.get_prices("AAPL", "2024-01-01", "2024-12-31")
```

**Vorteil:** Tests können mit `MockBrokerData` arbeiten, Live mit `AlpacaBrokerData` — gleiche Schnittstelle. **Hilft direkt bei B3/B7** (Schema-Drift): wenn alle News-Quellen das gleiche `NewsData`-Interface implementieren, kann es nur ein Schema geben.

**Aufwand:** Mittel (8-12h für sauberen Bootstrap und Migration aller Aufrufer).

**Lizenz:** GPL v3 — Konzept-Adaption ist OK, Code-Übernahme NEIN.

---

## 2. Walk-Forward & Anti-Leakage

### 2.1 Freqtrade (`freqtrade/freqtrade`)

**Was sie machen:** Crypto-Trading-Bot, 47k Stars, sehr ausgereift. Hat in 2024 zwei genialen Tools eingeführt:

- **`lookahead-analysis`**: Führt Backtest aus, ändert dann selektiv einzelne Indikatoren/Signale, vergleicht Outputs. Wenn ein Indikator sich ändert, weil eine Future-Bar entfernt wurde → Lookahead-Bias detektiert.
- **`recursive-analysis`**: Berechnet Indikatoren mit unterschiedlich langen Startup-Fenstern (199, 499, 999, 1999 candles). Wenn die Werte am Test-Datum unterschiedlich sind → recursive Formula mit unbegrenztem Lookback (Problem: deinen Live-Daten reicht der Lookback nicht aus → dein Indikator gibt im Live andere Werte als im Backtest).

**Warum das relevant ist:**

Deine A9-Behebung (Walk-Forward mit Embargo/Purge) verhindert Label-Leakage. Aber **Indikator-Leakage und Recursive-Bias sind zwei andere Klassen** — die du nicht abdeckst.

Beispiel Indikator-Leakage:
```python
# Bei dir in features/ vermutlich:
df["rolling_mean"] = df["price"].rolling(20).mean()
# OK, kein Leak
df["centered_mean"] = df["price"].rolling(20, center=True).mean()
# LEAK! center=True nutzt 10 Bars Vergangenheit + 10 Bars Zukunft
```

Beispiel Recursive-Bias:
```python
# Bei dir vermutlich:
df["ema_50"] = df["price"].ewm(span=50).mean()
# Im Backtest mit 5 Jahren History: ema bei Bar 1000 hat 1000 Bars Lookback
# Im Live mit 200 Bars History: ema bei Bar 1000 hat nur 200 Bars Lookback
# → unterschiedliche Werte zwischen Backtest und Live!
```

**Was wir konkret bauen sollten:**

Eigenes Anti-Leakage-Tool, inspiriert von Freqtrade:

```python
# src/assembled_core/qa/leakage_analyzer.py

def detect_lookahead_bias(
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    prices: pd.DataFrame,
    test_dates: list[pd.Timestamp],
) -> dict:
    """
    For each test date t, compute features twice:
    - With full data (including bars after t)
    - With truncated data (only bars <= t)
    Compare features at date t. Difference = lookahead bias.
    """
    full_features = feature_fn(prices)
    biased_features = []
    for t in test_dates:
        truncated = prices[prices.index <= t]
        truncated_features = feature_fn(truncated)
        # Compare last row
        diff = full_features.loc[t] - truncated_features.iloc[-1]
        if (diff.abs() > 1e-9).any():
            biased_features.append({
                "date": t,
                "feature_diffs": diff.to_dict(),
            })
    return {
        "has_bias": len(biased_features) > 0,
        "biased_dates": biased_features,
    }


def detect_recursive_bias(
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    prices: pd.DataFrame,
    test_date: pd.Timestamp,
    startup_candles: list[int] = [199, 499, 999, 1999],
) -> dict:
    """
    Compute features at test_date using different startup window sizes.
    If results differ → recursive formula needs unbounded history.
    """
    features_by_startup = {}
    for n in startup_candles:
        truncated = prices.iloc[-n - 1:]
        if test_date not in truncated.index:
            continue
        feats = feature_fn(truncated)
        features_by_startup[n] = feats.loc[test_date]
    
    # Variance check
    variances = {}
    for col in features_by_startup[startup_candles[-1]].index:
        values = [features_by_startup[n][col] for n in startup_candles if n in features_by_startup]
        variances[col] = float(np.std(values) / (np.mean(np.abs(values)) + 1e-12))
    
    return {
        "has_recursive_bias": any(v > 0.01 for v in variances.values()),
        "feature_variances": variances,
    }
```

Das wäre dann ein neuer **CI-Check**: vor jedem Merge wird `detect_lookahead_bias` über alle `features/*.py`-Module ausgeführt. Ein Bug, der eine `shift(-1)` einschleicht, wird sofort gefangen.

**Aufwand:** 6-10h für Implementierung + 2h für CI-Integration.

**Lizenz:** Freqtrade ist GPL → Konzept-Adaption nur, **kein Code-Copy**. Aber das Konzept ist gut dokumentiert in deren Docs, das reicht für eine eigene Implementierung.

**Priorität:** **Hoch.** Das ist die Anti-Leakage-Versicherung, die A9 ergänzt.

---

### 2.2 López de Prado-Pattern (Combinatorial Purged Cross-Validation)

**Was es ist:** Akademisches Konzept aus "Advances in Financial Machine Learning" (López de Prado 2018). Standard im Hedgefund-Bereich.

**Warum besser als simples Walk-Forward:**

Standard Walk-Forward macht ein Train/Test-Pair pro Window — wenig statistische Power. **CPCV** macht alle möglichen Train/Test-Kombinationen aus N Folds, mit Purge zwischen Train- und Test-Bereichen. Bei N=6 Folds und K=2 Test-Folds kriegst du 15 Train/Test-Kombinationen.

**Was wir konkret machen sollten:**

Erweitere `qa/walk_forward.py` um eine `combinatorial_purged_cv`-Variante:

```python
def combinatorial_purged_splits(
    n_folds: int = 6,
    test_folds: int = 2,
    purge_days: int = 10,
    embargo_days: int = 5,
) -> list[CPCVWindow]:
    """
    Returns all C(N,K) combinations of train/test splits
    with purge gap and embargo between train and test folds.
    """
    from itertools import combinations
    fold_indices = list(range(n_folds))
    splits = []
    for test_combo in combinations(fold_indices, test_folds):
        train_combo = [f for f in fold_indices if f not in test_combo]
        splits.append(CPCVWindow(
            train_folds=train_combo,
            test_folds=list(test_combo),
            purge_days=purge_days,
            embargo_days=embargo_days,
        ))
    return splits
```

**Vorteil bei uns:** Statt 5 Walk-Forward-Windows, bekommst du 15 statistisch unabhängige Train/Test-Setups. Sharpe-Konfidenzintervalle werden viel enger.

**Aufwand:** Mittel (6-8h Implementation + 2h Test).

**Lizenz:** Konzept ist frei, López de Prado hat selber Sample-Code in seinem Buch (akademisch zitierbar).

---

## 3. Portfolio-Optimierung

### 3.1 Riskfolio-Lib (`dcajasn/Riskfolio-Lib`)

**Was es ist:** Portfolio-Optimization-Library mit **24 konvexen Risk-Measures** plus **35 Risk-Measures für HRP/HERC**. Black-Litterman, Bayesian BL, Augmented BL, Risk Factors, NCO. Auf CVXPY aufbauend. BSD-Lizenz, Python 3.7+.

**Was uns interessiert:**

Bei uns hast du eigene Implementierungen von:
- Kelly Criterion
- HRP (Hierarchical Risk Parity)
- Black-Litterman
- BL-Blend
- Mean-Variance
- ERC (Equal Risk Contribution)
- Cost-Aware Wrapper

**Riskfolio-Lib hat das alles fertig**, geprüft, und mit deutlich mehr Optionen (z.B. CVaR, EVaR, Drawdown-at-Risk als Risk-Measure statt nur Variance).

**Empfehlung:**

**Drop-in-Ersatz prüfen.** Schreibe einen Test, der für deine bestehenden HRP/BL-Implementierungen Riskfolio-Lib's Output gegenüberstellt und schaut, ob sie äquivalent sind. Wenn ja → entferne deine eigenen Implementierungen, importiere Riskfolio. Wenn nein → schaue nach, woher der Unterschied kommt (vermutlich subtile Unterschiede in Schrumpfung der Cov-Matrix etc.).

```python
# tests/portfolio/test_riskfolio_equivalence.py
import riskfolio as rp

def test_our_hrp_matches_riskfolio():
    returns = ...  # synthetisch
    
    # Our implementation:
    our_weights = our_hrp(returns)
    
    # Riskfolio's implementation:
    port = rp.HCPortfolio(returns=returns)
    rf_weights = port.optimization(model='HRP', codependence='pearson')
    
    np.testing.assert_array_almost_equal(
        our_weights.values,
        rf_weights.values.flatten(),
        decimal=4,
    )
```

**Was wir gewinnen würden:**

- 5-7 hauseigene Optimization-Module weniger zu warten (~2000 LOC)
- 24 Risk-Measures statt 1-2
- Drawdown-at-Risk als Risk-Measure (interessant für deinen `risk/`-Bereich)
- Bewährter, geprüfter Code statt eigene Implementierungen

**Was es kostet:**

- Externe Dependency (CVXPY+MOSEK können tricky beim Install sein)
- API umstellen (deine Funktionen geben aktuell `dict[str, float]` zurück, Riskfolio gibt DataFrame)

**Aufwand:** 4-6h für Equivalence-Tests, 12-20h für vollständige Migration falls äquivalent.

**Empfehlung-Konkret:** Erst Equivalence-Tests, dann entscheiden. Wenn nur 80% äquivalent: hybrid lassen — eigene Implementierungen behalten, Riskfolio als zusätzliche Option für CVaR-/Drawdown-Optimization aufbohren.

**Lizenz:** BSD — frei nutzbar als Dependency oder Code-Adaption.

---

### 3.2 PyPortfolioOpt (`PyPortfolio/PyPortfolioOpt`)

**Was es ist:** Sklearn-Style Portfolio-Optimization. Klassisches Markowitz, BL, HRP, mit Shrinkage und L2-Regularisation. MIT-Lizenz. Sehr saubere API.

**Was uns interessiert:**

Ihre **Covariance-Estimator**-Sammlung. Bei uns hast du `risk_metrics.py` mit Eigenvalue-Clipping (gut!), aber keine modernen Shrinkage-Estimators wie Ledoit-Wolf oder Oracle-Approximating.

**Was wir konkret übernehmen sollten:**

PyPortfolioOpt's `risk_models.py` — das ist eine fast eigenständige Sammlung von:
- Sample Covariance
- Semi-Covariance (downside only)
- **Ledoit-Wolf Shrinkage**
- Oracle-Approximating Shrinkage
- Constant Correlation Shrinkage
- Exponentially-Weighted Covariance

Bei uns reicht aktuell `df.cov()`. Mit Shrinkage werden out-of-sample Optimierungen deutlich stabiler. Der Mainstream-Konsens ist: **Ledoit-Wolf ist überlegen** für Sample-Sizes < 5×N (bei dir wahrscheinlich der Normalfall, weil du 252 Tage × 30 Symbole = 8 Stichproben pro Symbol-Paar hast).

**Konkrete Nutzung:**

```python
from pypfopt import risk_models

# Statt:
cov = returns.cov()

# Lieber:
cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
```

Das kannst du **direkt als Drop-in** in `risk/risk_metrics.py` einbauen — eine Zeile geändert, signifikant bessere Risk-Estimates.

**Aufwand:** Sehr klein (1-2h).

**Lizenz:** MIT.

---

## 4. Risk-Management

### 4.1 quantstats (`ranaroussi/quantstats`)

**Was es ist:** Performance-Reporting. Liefert ein "tearsheet" wie eines von QuantConnect oder Bloomberg. Gibt Sharpe, Sortino, Calmar, Tail-Ratio, VaR, CVaR, Max-Drawdown, Recovery-Time, Volatility, Skewness, Kurtosis, Best/Worst-Day/Month/Year, Stability, Common Sense Ratio etc.

**Was uns interessiert:**

Bei dir hast du `qa/metrics.py` mit den Basis-Metriken. quantstats hat 50+ Metriken plus HTML-/PDF-Reports out-of-the-box.

**Empfehlung:**

In `qa/daily_qa_report.py` als Backend integrieren:

```python
import quantstats as qs

def generate_daily_qa(equity_curve: pd.Series, output_path: Path):
    qs.reports.html(
        equity_curve,
        benchmark="SPY",
        output=str(output_path / "tearsheet.html"),
        title="Daily QA Report",
    )
```

Bekommst du eine professionelle HTML-Report mit allen Standard-Metriken, Equity-Curve, Drawdown-Plot, Monthly-Returns-Heatmap, Distribution etc.

**Aufwand:** 2-3h.

**Lizenz:** Apache 2.0.

---

### 4.2 empyrical (`quantopian/empyrical`, jetzt `stefan-jansen/empyrical-reloaded`)

**Was es ist:** Reine Performance-Metriken-Library. Sehr fokussiert, keine Reports. Statistically validated. War ursprünglich Quantopian-Standard.

**Was uns interessiert:**

Bei deinem A4-Befund hast du gesehen, dass die Implementierung von VaR/CVaR sloppy war. **empyrical** hat die Standard-Implementierungen geprüft und mit akademischen Definitionen gemappt.

**Empfehlung:**

Statt eigene VaR/CVaR/Sortino/Calmar-Funktionen in `qa/metrics.py` zu pflegen, importiere die kanonischen Versionen:

```python
import empyrical as ep

sharpe = ep.sharpe_ratio(returns, risk_free=0.0)
sortino = ep.sortino_ratio(returns)
max_dd = ep.max_drawdown(returns)
calmar = ep.calmar_ratio(returns)
var_95 = ep.value_at_risk(returns, cutoff=0.05)
cvar_95 = ep.conditional_value_at_risk(returns, cutoff=0.05)
omega = ep.omega_ratio(returns)
tail = ep.tail_ratio(returns)
```

**Vorteil:** Wenn dein Investor fragt "wie berechnest du Sharpe genau?", verweist du auf empyrical. Wenn dein eigener Code abweicht, ist das ein Bug.

**Aufwand:** 3-4h für komplette Migration der Metrik-Funktionen.

**Lizenz:** Apache 2.0.

---

## 5. News-Sentiment & NLP

### 5.1 FinBERT (ProsusAI + HuggingFace, mehrere Forks)

**Was es ist:** BERT-Modell, finegetuned auf Reuters TRC2 + Financial PhraseBank. Kategorisiert Texte in positive/negative/neutral. State-of-the-art für Financial Sentiment.

**Was uns interessiert:**

Bei dir hast du in `intel/news_*` und `events/news/*` eigene Sentiment-Logik. Schauen wir uns mal an, ob FinBERT besser ist:

**Empfehlung:**

Drop-in als Sentiment-Backend. Bei dir vermutlich aktuell:
```python
sentiment_score = naive_keyword_sentiment(text)  # vermutlich
```

Stattdessen:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def finbert_sentiment(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    # Output: [positive, negative, neutral]
    return {
        "positive": float(probs[0]),
        "negative": float(probs[1]),
        "neutral": float(probs[2]),
        "score": float(probs[0] - probs[1]),  # signed sentiment
    }
```

**Was es kostet:**
- FinBERT-Modell ist ~440MB → muss erstmal gedownloadet werden
- GPU-Inference ist 100× schneller als CPU
- Erste Inference braucht 1-2s, danach <50ms pro Text auf CPU

**Was wir gewinnen:**
- Domain-spezifische Sentiment-Klassifikation statt VADER/TextBlob (general purpose)
- 92% F1-Score auf Financial PhraseBank
- Sehr gut in Tonalität: "Earnings beat expectations" → positive, "Earnings missed estimates" → negative

**Aufwand:** 4-6h für Integration + Tests, plus Modell-Download.

**Lizenz:** Apache 2.0.

---

### 5.2 BERTopic für News-Clustering

**Was es ist:** Topic-Modeling mit Transformers. Embeddet Dokumente, clustert mit HDBSCAN, extrahiert Topics. Im Search-Result fand sich ein Projekt das "100+ meaningful topics" auf Financial Headlines clustert.

**Was uns interessiert:**

Bei deiner News-Pipeline hast du `dedupe`-Logik, aber wahrscheinlich keine Topic-Klassifikation.

Vorteil von Topic-Klassifikation:
- "Apple reports record iPhone sales" → Topic: Tech-Earnings
- "Fed signals rate cut" → Topic: Macro-Monetary
- Topic-Aggregation → "Topic-X-Sentiment trend over time"

**Empfehlung:**

**Niedrige Priorität.** Coole Erweiterung, aber nicht akut. Wenn dein News-Pipeline nach B3 (NewsEvent Schema-Drift behoben) stabilisiert ist, kannst du in Phase 2 darüber nachdenken.

**Aufwand:** 8-12h für saubere Integration.

**Lizenz:** MIT.

---

## 6. Datenpipeline & Storage

### 6.1 polars statt pandas für Hot-Paths

**Was es ist:** DataFrame-Library, Rust-implementiert, ~10× schneller als pandas bei großen DataFrames. API ähnlich zu pandas, aber mit lazy evaluation.

**Was uns interessiert:**

Dein C2-Befund: 122 `.iterrows()`-Aufrufe, davon 19 in trading_cycle_v2 (jetzt 5 nach Phase). Selbst nach C2-Vektorisierung bleiben Hot-Paths in `qa/backtest_engine.py`, die mit Polars 10× schneller wären.

**Empfehlung:**

**Selektiv migrieren.** Nicht alles, sondern nur identifizierte Hot-Paths nach Profiling:

```bash
python -m cProfile -o profile.out scripts/run_backtest_strategy.py ...
python -c "import pstats; pstats.Stats('profile.out').sort_stats('cumulative').print_stats(20)"
```

Top-3-Hotspots auf Polars umstellen, Rest mit pandas. Für deine Daten-Pipeline:

```python
import polars as pl

# Statt:
df = pd.read_parquet(path).set_index("timestamp").sort_index()
# Nutzen:
df = pl.scan_parquet(path).sort("timestamp").collect()
```

Polars's lazy evaluation kann oft mehrere Operationen in einem Pass kombinieren.

**Aufwand:** Sehr selektiv, 4-8h pro Hot-Path-Migration.

**Lizenz:** MIT.

---

### 6.2 PyArrow + Apache Arrow als Standard-Format

**Was es ist:** Columnar in-memory Data-Format. PyArrow gibt dir extrem schnelles I/O zu Parquet, plus zero-copy zwischen Polars/pandas/numpy.

**Was uns betrifft:**

Du hast bereits PyArrow als Dependency. Aber: pandas 3.0 (Januar 2026 released) ändert den **String-Dtype-Default** auf PyArrow-backed. Das hat zwei Implikationen:

1. **Performance**: String-Operations werden 3-10× schneller
2. **Memory**: Kann dein Speicher um 30-50% reduzieren bei String-heavy DataFrames (wie News-Pipelines)

**Empfehlung:**

Beim Pandas-3.0-Migration (D20 in deinem Audit, NPY201 Ruff-Rule schon aktiv) explizit dtype-Backend auf PyArrow umstellen:

```python
import pandas as pd
pd.options.future.infer_string = True  # PyArrow string default
```

Plus: Bei `pd.read_csv` (deine D14, 8 Stellen offen) `dtype_backend="pyarrow"` nutzen:

```python
df = pd.read_csv(path, dtype_backend="pyarrow")
```

**Aufwand:** Sehr klein (Teil von D14, +1h für pandas-Setup-Konfig).

---

## 7. Code-Hygiene & Tooling

### 7.1 Ruff-Konfiguration aus großen Repos

**Was uns interessiert:**

NautilusTrader hat eine sehr strenge Ruff-Config mit vielen aktivierten Regeln. Lohnt sich, deren `pyproject.toml` als Vorlage zu nehmen.

**Empfehlung:**

Erweitere deine Ruff-Konfig schrittweise (du hast aktuell `select = ["E", "F", "NPY201"]`). Sukzessiv hinzufügen:

```toml
[tool.ruff.lint]
select = [
    "E", "W",       # pycodestyle
    "F",            # pyflakes
    "I",            # isort
    "B",            # flake8-bugbear (catches gotchas)
    "UP",           # pyupgrade (modernisiert Code)
    "SIM",          # flake8-simplify
    "PD",           # pandas-vet
    "NPY",          # numpy
    "RUF",          # ruff-specific
    "ASYNC",        # async issues
    "S",            # bandit (security)
    "PERF",         # perflint
]
ignore = [
    "E501",         # line too long (handled by black)
    "S101",         # asserts in code (we use them)
]
```

Fang mit 3-4 zusätzlichen Regelfamilien an, fix die Warnings, dann mehr aktivieren.

**Aufwand:** Inkrementell, pro Regel-Familie 2-4h Cleanup.

---

### 7.2 Pytest-Plugins, die wir noch nicht nutzen

Aus den großen Repos beobachtet:

- **pytest-xdist**: Tests parallel laufen lassen (`-n auto`). Bei 556 Tests potenziell 4× schneller in CI.
- **pytest-benchmark**: Performance-Regressions-Tracking. Würde dir bei C2 (iterrows-Performance) helfen, Verschlechterungen zu fangen.
- **pytest-randomly**: Tests in randomisierter Reihenfolge → fängt versehentliche Test-Reihenfolge-Abhängigkeiten.
- **hypothesis-jsonschema**: Property-Tests basierend auf JSON-Schema (relevant für deine API-Routen).

**Empfehlung:**

`pytest-xdist` und `pytest-randomly` einbauen. Beide sind drop-in:

```toml
[project.optional-dependencies]
dev = [
    "pytest-xdist>=3.5",
    "pytest-randomly>=3.15",
    ...
]
```

In CI: `pytest -n auto -p randomly`. Sollte deine 556 Tests in <2 Minuten laufen lassen.

**Aufwand:** 2-3h Setup + Test-Stabilisierung (manche Tests brechen mit randomisierter Reihenfolge → das ist ein Feature, nicht Bug).

---

### 7.3 GitHub Actions-Pattern aus dem `pysystemtrade`-Repo

Rob Carver hat seine Production-Pipeline minutiös dokumentiert. Sein Pattern: **scheduled health-checks** statt heartbeat-Commits.

Bei dir: C6-Befund mit 204 Heartbeat-Commits. Carver löst das mit:

```yaml
on:
  schedule:
    - cron: '0 6 * * *'  # 6 AM UTC daily

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - name: Send heartbeat to monitoring
        run: |
          curl -X POST $HEARTBEAT_WEBHOOK \
            -H "Content-Type: application/json" \
            -d '{"status": "alive", "ts": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}'
```

Kein Commit, kein Repo-Pollution, aber Monitoring sieht den Heartbeat. Healthcheck.io ist ein guter freier Service dafür.

**Aufwand:** 30 Minuten.

**Lizenz:** Konzept, kein Code-Issue.

---

## Priorisierter Adoption-Plan

Wenn ich alle 15 Empfehlungen nach **(Nutzen / Aufwand)** sortiere und auf deinen aktuellen Code-Stand normiere:

### Sofort einbauen (Hoher Nutzen, Geringer Aufwand)

| Empfehlung | Aufwand | Nutzen | Lizenz |
|---|---|---|---|
| **PyPortfolioOpt CovShrinkage** als Drop-in für `df.cov()` | 1-2h | hoch | MIT |
| **empyrical-reloaded** für kanonische Performance-Metriken | 3-4h | hoch | Apache 2.0 |
| **quantstats** für HTML-Reports in `daily_qa_report.py` | 2-3h | mittel-hoch | Apache 2.0 |
| **pytest-xdist + pytest-randomly** für CI-Speed | 2-3h | mittel | MIT |
| **Healthcheck-Pattern** statt Heartbeat-Commits | 30min | niedrig-mittel | konzept |

**Gesamt: ~10h für 5 substanzielle Verbesserungen.**

### Sprint einplanen (Mittlerer Nutzen, Mittlerer Aufwand)

| Empfehlung | Aufwand | Nutzen | Lizenz |
|---|---|---|---|
| **Anti-Leakage-Tool** (Lookahead + Recursive) inspiriert von Freqtrade | 8-12h | sehr hoch | konzept |
| **Bootstrap-CIs** für Performance-Metriken (PyBroker-Pattern) | 3-4h | hoch | konzept |
| **FinBERT** als Sentiment-Backend in News-Pipeline | 4-6h | hoch | Apache 2.0 |
| **Riskfolio-Lib Equivalence-Tests** für deine HRP/BL-Implementierungen | 4-6h | mittel-hoch | BSD |
| **CPCV** (Combinatorial Purged CV) als Erweiterung zu A9 | 6-8h | hoch | konzept |

**Gesamt: ~30h für 5 strategische Verbesserungen.**

### Architektur-Refactor (Hoher Nutzen, Hoher Aufwand)

| Empfehlung | Aufwand | Nutzen | Lizenz |
|---|---|---|---|
| **Broker-Adapter-Pattern** (Nautilus-Style) zur Mode-Branch-Eliminierung | 8-16h | sehr hoch | konzept |
| **DataBlob-Pattern** (pysystemtrade-Style) als zentrale Daten-Schnittstelle | 8-12h | hoch | konzept |
| **RiskModel-Chain** (QSTrader-Style) statt Monolith in `size_positions` | 6-10h | hoch | konzept |

**Gesamt: ~30h für 3 Architektur-Verbesserungen, idealerweise nach B3/B5-Konsolidierung.**

### Optional / Spekulativ (Nice-to-have)

- BERTopic für News-Clustering (8-12h)
- Polars für Hot-Paths (4-8h pro Path)
- Volle Riskfolio-Migration (12-20h, nur wenn Equivalence-Tests gut sind)

---

## Wichtige Lizenz-Übersicht

Was wir adaptieren dürfen und wie:

| Projekt | Lizenz | Was erlaubt |
|---|---|---|
| **NautilusTrader** | LGPL v3 | Konzept ✓, Code-Übernahme nur in separates Modul |
| **QSTrader** | MIT | Alles ✓ — sogar Copy-Paste mit Lizenz-Vermerk |
| **PyBroker** | Apache 2.0 | Alles ✓ — sogar Code-Übernahme |
| **pysystemtrade** | GPL v3 | Konzept ✓, Code-Übernahme NEIN (würde uns viral GPL-isieren) |
| **Freqtrade** | GPL v3 | Konzept ✓, Code-Übernahme NEIN |
| **Riskfolio-Lib** | BSD | Alles ✓ |
| **PyPortfolioOpt** | MIT | Alles ✓ |
| **quantstats** | Apache 2.0 | Alles ✓ |
| **empyrical-reloaded** | Apache 2.0 | Alles ✓ |
| **FinBERT** | Apache 2.0 | Alles ✓ |
| **BERTopic** | MIT | Alles ✓ |
| **polars** | MIT | Alles ✓ |

**Praktische Regel:** Bei MIT/BSD/Apache 2.0 darfst du Code direkt übernehmen, mit Lizenz-Vermerk im Header der Datei. Bei LGPL musst du in separate Datei/Modul packen. Bei GPL/AGPL nur Konzept-Adaption — die Idee neu implementieren.

---

## Schluss-Bemerkung: Was du vermeiden solltest

Bei meinem Durchgang durch viele Repos sind mir Anti-Patterns aufgefallen, die du **nicht** importieren solltest, auch wenn du auf sie stößt:

1. **Crypto-Bots mit hardcoded Strategie-Parameters** (viele Freqtrade-Strategy-Repos): Hyperopt-Output committed mit magischen Konstanten. Sieht impressiv aus, ist aber Overfitting-pur.

2. **"All-in-One"-Frameworks ohne Modularität** (vectorbt-pro): zwingen dich in ihre API. Dein Repo ist schon zu reif für sowas.

3. **Strategien aus Discord-Channels**: viele kursieren ohne Lookahead-Analyse. Bevor du eine kopierst, **prüfe sie mit deinem Anti-Leakage-Tool** (siehe 2.1).

4. **Hedge-Fund-Style "Reactive ML"**: Modelle, die täglich retrained werden auf deren n-1-Tag Daten — sehr leakage-anfällig, schwer zu validieren.

---

**Mein Vorschlag für dich:**

Fang mit den **5 Sofort-Items** an (~10h). Damit hast du bessere Cov-Estimates, kanonische Metriken, professionelle HTML-Reports, schnellere CI und keine Heartbeat-Pollution mehr. Das ist niedrig-hängende Frucht, die aber dein Repo unmittelbar professioneller macht.

Danach: **Anti-Leakage-Tool** (Punkt 2.1) als nächstes größeres Investment. Das ist die konkrete Versicherung gegen die ML-Korrektheits-Bugs, die A1 und A9 nur teilweise abdecken.

Architektur-Refactors (Broker-Adapter, DataBlob, RiskModel-Chain) am Ende — die machen erst Sinn, wenn die Konsolidierungs-Schulden aus B3 und B5 abgebaut sind.

Lass mich wissen, womit du anfangen willst. Bei jedem Punkt kann ich dir den konkreten Implementierungs-Code hinschreiben.
