# 30 — News-TA-Fusion-Architektur

**Zweck:** Der Kern der Systemarchitektur. News und TA werden nicht isoliert verwendet, sondern auf **drei Ebenen parallel integriert**, die sich gegenseitig ergänzen, nicht ersetzen.

Dieses Dokument ist die wichtigste einzelne Design-Entscheidung im ganzen Plan.

---

## Die drei Schichten im Überblick

```
┌──────────────────────────────────────────────────────────┐
│ Schicht 3 — 2D-Decision-Matrix                            │
│   TA-Score × News-Score → Size-Multiplier                 │
│   Double-Confirmation-Boost / Conflict-Skip               │
├──────────────────────────────────────────────────────────┤
│ Schicht 2 — Meta-Labeling-Gate                            │
│   Primary-Signal → Binary-Gate (take/skip)                │
│   News als M2-Context-Feature                             │
├──────────────────────────────────────────────────────────┤
│ Schicht 1 — News als 9. BaseSignal                        │
│   Composite-Feature mit Regime-Gewichtung                 │
└──────────────────────────────────────────────────────────┘

Die drei Schichten arbeiten PARALLEL:
Schicht 1 liefert kontinuierlichen Score-Beitrag
Schicht 2 filtert Grenzfälle aus
Schicht 3 skaliert Position-Size
```

**Warum nicht nur eine Schicht?** Jede hat andere Stärken:
- **Schicht 1** ist additiv, liefert kontinuierliches Signal.
- **Schicht 2** ist Binary-Gate, reduziert False-Positives aus dem Primary-Signal.
- **Schicht 3** ist multiplikativ, skaliert Conviction.

Die drei zusammen decken "continuous, binary, multiplicative" — drei verschiedene mathematische Integrationen von Information.

---

## Schicht 1: News als 9. BaseSignal

### Die sechs Sub-Features

Pro Ticker und pro Zeitpunkt ein einzelner Z-Score in [−3, +3]:

| Sub-Feature | Formel | Rolle |
|---|---|---|
| `sentiment_vw` | Volume-gewichteter FinBERT-Tone-Score, EWMA HL=2d | Richtungs-Signal |
| `novelty` | `1 − max(cos_sim zu Headlines letzter 7d)` | Fresh-News-Verstärker |
| `surprise` | `(actual − consensus) / σ(consensus)` bei Earnings | Post-Earnings-Drift |
| `event_volume_z` | 30d-Rolling-Z-Score der Artikelzahl pro Ticker | Abnormale Coverage |
| `velocity` | `Δ(sentiment_24h) − Δ(sentiment_7d)` | Sentiment-Momentum |
| `dispersion` | `std(sentiment)` über Quellen | Konsens vs Uneinigkeit |

### Aggregation zu einem News-Z-Score

```python
def news_z_score(features):
    # Weighted aggregate with capping
    w = {
        'sentiment_vw': 0.30,
        'novelty': 0.15,
        'surprise': 0.20,
        'event_volume_z': 0.10,
        'velocity': 0.15,
        'dispersion': -0.10  # high dispersion = penalty
    }
    raw = sum(w[k] * features[k] for k in w)
    # clip to [-3, +3]
    return max(-3.0, min(3.0, raw))
```

### Regime-Gewichtung im Composite

Die Gewichtung der News-Dimension im 9-dimensionalen Composite hängt vom Regime ab:

| Regime | News-Weight | Begründung |
|---|---|---|
| Calm (niedriger VIX, flache Curve) | 0.05 | News spielt kleine Rolle |
| Normal | 0.10 | Standard |
| Elevated (hoher VIX, invertierte Curve) | 0.20 | News wichtig |
| Crisis (VIX > 30, HY-Spread > 6%) | 0.30 | News dominiert |

Regime wird nicht per VIX allein bestimmt (30% False-Positives bei 1-Tages-Spikes), sondern aus VIX + Term-Slope + HY-Credit-Spreads. Ab Phase 2 über ein **Gaussian-HMM** (siehe `13_FREE_MODULE.md` §13.2).

### Das Base-Rate-Problem

Manche Ticker bekommen 40 Artikel/Tag, andere 3/Woche. **Lösung: per-Ticker Rolling-Z-Scores**:

- `event_volume_z`: 60-Tage-Rolling-Window (stabil)
- `sentiment_vw` und `velocity`: EWMA mit HL=30d (reaktiver)

**Kausale Normalisierung Pflicht** via `.shift(1)` — sonst Look-Ahead-Bias.

### Sentiment-Modelle

Siehe `11_FREE_MODELLE.md` §11.2 für Details.

- **Primary:** `yiyanghkust/finbert-tone` (FinBERT-Tone)
- **Fast-Path:** `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`
- **Event-Type-Klassifikation:** `facebook/bart-large-mnli` zero-shot für Ambiguitätsfälle
- **FOMC-Special:** `gtfintechlab/FOMC-RoBERTa`
- **Multi-Target:** Claude Haiku 4.5 (Paid, <10 EUR/Monat) für Mehrdeutigkeit

### Event-Type-Taxonomie

Pro Event-Type andere Half-Lives und Feature-Gewichte:

| Event-Type | Half-Life | Dominierendes Feature |
|---|---|---|
| Earnings | 3 Tage | surprise |
| M&A (Target) | 5 Tage | sentiment + volume |
| M&A (Acquirer) | 5 Tage | sentiment |
| Management-Changes | 2 Tage | sentiment (CEO>CFO>COO-Weight) |
| Regulatory | 4 Tage | sentiment + sector-spillover |
| Analyst-Rating | 1 Tag | (new_target − old_target) / price |
| Product-Launch | 2 Tage | velocity + novelty |
| Legal/Lawsuit | 3 Tage | sentiment (asymmetrisch negativ) |
| Macro | N/A (propagates) | separate Pipeline |

---

## Schicht 2: News als Meta-Labeling-Gate

### Das López-de-Prado-Pattern

Meta-Labeling (AFML Kapitel 3) ist zweistufig:
- **Primary-Model (M1)** produziert Side: Long/Short/Neutral
- **Secondary-Model (M2)** lernt `P(take trade | side, context)`

M1 optimiert Recall, M2 optimiert Precision. Zusammen maximieren Sharpe.

**News-Features sind ideale M2-Inputs**, weil sie Kontext liefern, den reine Preis-Features nicht haben.

### Library-Stack

Siehe `11_FREE_MODELLE.md`:
- **mlfinpy** (statt paywall-`mlfinlab`) für Triple-Barrier + CUSUM
- **skfolio.CombinatorialPurgedCV** für Validation
- **LightGBM** + `CalibratedClassifierCV` mit Isotonic-Regression

### Die 12-15 Meta-Features

```python
META_FEATURES = [
    'sentiment_z',      # aus Schicht 1
    'novelty_z',
    'surprise_z',
    'event_vol_z',
    'velocity_z',
    'dispersion_z',
    # Event-Type One-Hot (8 Dimensionen)
    'event_earnings', 'event_m_and_a', 'event_mgmt',
    'event_regulatory', 'event_analyst', 'event_product',
    'event_legal', 'event_macro',
    # Context
    'days_since_earnings',
    'days_to_next_earnings',
    'macro_shock_flag',     # FOMC/CPI ±1d
    'vix_level',
    'vix_regime_ord',
    'hy_oas',
    'corroboration_count',  # unabhängige Quellen
    'primary_strength',     # aus M1
    'news_vs_primary_agree'  # Sign-Match
]
```

### Dynamische Triple-Barrier unter News

Die Triple-Barrier-Parameter werden nach Kontext angepasst, nicht statisch:

| Kontext | Profit-Take × σ | Stop-Loss × σ | Vertical Barrier |
|---|---|---|---|
| Kein Event (Baseline) | 2.0 | 2.0 | 10 Bars |
| Earnings < 24h | 2.5 | **1.5** | **2 Bars** |
| M&A pending | 3.0 | 1.0 | 5 Bars |
| High Dispersion (std > 0.5) | 1.5 | 1.5 | 3 Bars |
| FOMC / CPI < 24h | 2.0 | 1.5 | 2 Bars |

**Begründung:** Vor Earnings ist Gap-Risiko asymmetrisch. Kürzere vertikale Barriere verhindert, dass ein profitabler Pre-Earnings-Trade im Earnings-Crash ausläuft.

### CUSUM-Events statt Zeitbars

```python
from mlfinpy.filters import cusum_filter

def sample_events(close_prices, threshold_multiplier=2.0):
    daily_vol = close_prices.pct_change().ewm(span=20).std()
    threshold = daily_vol * threshold_multiplier
    t_events = cusum_filter(close_prices, threshold)
    return t_events
```

Standard-Threshold: `2 × EWMA-Daily-Vol`.

### CPCV-Pipeline mit Event-Embargos

```python
from skfolio.model_selection import CombinatorialPurgedCV

cpcv = CombinatorialPurgedCV(
    n_folds=6,
    n_test_folds=2,
    purged_size=max_vertical_barrier,  # ≥ vertikale Barriere
    embargo_size=2  # Business-Days
)
```

**Standard-Bug in Tutorials:** purged_size < vertical_barrier → Leakage → fiktiver Sharpe. **Hältst du das ein, sparst du dir 80% der Anfänger-Fehler.**

### Die False-Positive-Reduktions-Regel

Mathematisch:

```
Trade ausführen ⇔ 
    P_meta(y=1 | x) ≥ θ_meta
  AND 
    ¬(sign(news_z) ≠ primary_side AND |news_z| > τ_veto)
```

**Konkrete Werte:**
- `θ_meta = 0.55` in Phase 1 (Tolerance)
- `θ_meta = 0.65` in Phase 3 (strenger)
- `τ_veto = 1.5` (News widerspricht Primary mit hoher Magnitude → Skip)

### Size-Kalibrierung (Kelly-artig)

```python
def size_from_meta(p_meta, theta_meta=0.55):
    if p_meta < theta_meta:
        return 0.0
    # equivalent to Kelly auf kalibrierter Posterior
    return max(0.0, min(1.0, (p_meta - theta_meta) / (1 - theta_meta)))
```

---

## Schicht 3: 2D-Decision-Matrix

### Das Pattern

Diese Schicht behandelt News als **unabhängige zweite Dimension** neben dem TA-Composite. **Parallel zum Meta-Labeling-Gate, nicht redundant.**

- Meta-Labeling wirkt als Binary-Gate (take/skip)
- 2D-Matrix wirkt als **Size-Multiplier**

### Die 5×5-Basis-Matrix

TA-Score und News-Z werden auf Quantile gebunden: `[−∞, −1]`, `[−1, −0.3]`, `[−0.3, 0.3]`, `[0.3, 1]`, `[1, ∞]`.

| TA ↓ \ News → | Strong − | Weak − | Neutral | Weak + | Strong + |
|---|---|---|---|---|---|
| **Strong −** | **Full-Short** | Full-Short | Half-Short | Skip | Skip |
| **Weak −** | Half-Short | Half-Short | Half-Short | Skip | Half-Long |
| **Neutral** | Half-Short | Skip | Skip | Skip | Half-Long |
| **Weak +** | Half-Short | Skip | Half-Long | Half-Long | Full-Long |
| **Strong +** | Skip | Skip | Half-Long | Full-Long | **Full-Long** |

**Interpretation:**
- **Diagonale** = Double-Confirmation → Full-Size
- **Anti-Diagonale** = Conflict → Skip
- **Ecken** = einseitig dominiertes Signal → Half-Size

### Bayesian-Updating (produktionsreife Live-Variante)

```python
import numpy as np

def bayesian_update(ta_score, news_z, kappa=10):
    """
    Beta-Binomial Update.
    ta_score: [-1, 1] aus Composite
    news_z:   [-3, +3]
    """
    # TA-Score zu Beta-Prior
    ta_prob = 1 / (1 + np.exp(-2 * ta_score))  # sigmoid
    alpha_prior = ta_prob * kappa
    beta_prior = (1 - ta_prob) * kappa
    
    # News als Binomial-Evidence
    news_prob = 1 / (1 + np.exp(-news_z))
    alpha_post = alpha_prior + news_prob
    beta_post = beta_prior + (1 - news_prob)
    
    posterior_mean = alpha_post / (alpha_post + beta_post)
    return posterior_mean  # [0, 1]
```

Latenz: ~50 µs pro Update. Produktionsreif.

### Signal-Agreement als Multiplier

```python
def agreement_multiplier(ta_score, news_z):
    sign_match = np.sign(ta_score) == np.sign(news_z)
    magnitude_avg = (abs(ta_score) + abs(news_z)/3) / 2  # normalized
    
    if sign_match:
        return 1.0 + 0.5 * magnitude_avg  # bis 1.5×
    elif abs(ta_score) < 0.3 or abs(news_z) < 0.3:
        return 1.0  # einseitig, neutral
    else:
        return 0.5  # Conflict
```

### Matrix-Kalibrierung via Optuna

```python
import optuna
from optuna.samplers import GPSampler

def objective(trial, ...):
    # 20-30 Parameter optimieren
    thresholds = [trial.suggest_float(f"t_{i}", -3, 3) for i in range(5)]
    matrix_mapping = [trial.suggest_categorical(f"action_{i}",
                      ["full_long", "half_long", "skip", "half_short", "full_short"])
                      for i in range(25)]
    
    # CPCV-aggregiertes Objective
    sharpe = backtest_with_cpcv(thresholds, matrix_mapping)
    calmar = ...
    turnover = ...
    
    return sharpe + 0.5 * calmar - 0.1 * turnover

study = optuna.create_study(sampler=GPSampler(), direction="maximize")
study.optimize(objective, n_trials=200)
```

**Ohne CPCV im Objective: massives Overfitting garantiert.** Das ist keine Theorie, das ist der Standard-Fehler in Retail-Quant-Repos.

---

## Cross-Impact-Graph (Zusatz-Layer)

Über die drei Schichten hinaus gibt es noch einen vierten, optionalen Layer: **Cross-Impact**. News über Ticker A propagieren auf korrelierte Tickers B, C, D.

### Entity-Linking

```python
# spaCy als Fast-Path
import spacy
nlp = spacy.load("en_core_web_lg")  # nicht _trf (20× langsamer)

def extract_entities(text):
    doc = nlp(text)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    # Plus Cashtag-Regex
    cashtags = re.findall(r'\$[A-Z]{1,5}\b', text)
    return orgs, cashtags

# Company-zu-Ticker hierarchisch
def map_to_ticker(company_name):
    # 1. Lokaler Alias-Cache
    if name in local_cache:
        return local_cache[name]
    # 2. OpenFIGI v3 (free, V2 Sunset 01.07.2026!)
    ticker = openfigi_lookup(company_name)
    if ticker:
        return ticker
    # 3. Wikidata SPARQL (P414/P249)
    ticker = wikidata_lookup(company_name)
    if ticker:
        return ticker
    # 4. yfinance-Fallback
    return yfinance_search(company_name)
```

### Primary vs Mentioned

Drei Heuristiken:
1. **Titel-Position:** Erstes Drittel der Headline → Primary; Body → Mentioned mit Weight 0.3
2. **Mention-Density:** `mention_count / total_mentions > 0.4` → Primary
3. **GDELT GKG V2.1 Character-Offsets** für räumliche Nähe zu Event-Keywords

Phase 2: LLM-Veto via Claude Haiku 4.5 für Grenzfälle (<0.01 EUR/Tag).

### Pearson-Correlation-Graph (Phase 1)

```python
import networkx as nx
import numpy as np
from sklearn.covariance import LedoitWolf

def build_cross_impact_graph(returns_df, window=60, threshold=0.5):
    lw = LedoitWolf()
    lw.fit(returns_df.tail(window))
    cov = lw.covariance_
    # Korrelation aus Kovarianz
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    
    G = nx.Graph()
    tickers = returns_df.columns
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i < j and abs(corr[i, j]) > threshold:
                G.add_edge(t1, t2, weight=corr[i, j])
    return G
```

Wenn News auf AAPL einschlägt, propagiert Sentiment mit Weight × Correlation auf TSM, QCOM, AVGO.

### GNN — warum nicht

Die Literatur 2020-2025 zeigt typische Accuracy-Gewinne von 1-4% gegenüber LSTM-Baselines — **die nach Transaktionskosten verschwinden**. Drei Reproduzierbarkeitsprobleme:

1. In-Sample-Graphen ohne Point-in-Time-Konstruktion führen zu 50%+ Performance-Einbruch live
2. Testsets <1.200 Artikel zu klein für robuste Schlüsse
3. Gemeldete Backtest-Gewinne überleben Transaktionskosten nicht

**Verdict:** Für Solo-Quant in Phase 1/2 Hype. Library-Status 2026: PyTorch Geometric 2.5+ ist Standard, stellargraph tot.

---

## Die Integration: wie die Schichten zusammenwirken

```python
def decide_trade(ticker, current_bars, news_features):
    # SCHICHT 1: Composite-Score mit News als 9. Dimension
    composite_score = compute_composite(
        ticker, current_bars, news_features,
        regime=classify_regime()
    )  # [-1, +1]
    primary_side = np.sign(composite_score)
    
    # SCHICHT 2: Meta-Labeling-Gate
    meta_features = build_meta_features(ticker, news_features, composite_score)
    p_meta = meta_model.predict_proba(meta_features)[0, 1]
    
    news_z = news_features['aggregate_z']
    if p_meta < 0.55:
        return {"action": "skip", "reason": "meta_below_threshold"}
    if np.sign(news_z) != primary_side and abs(news_z) > 1.5:
        return {"action": "skip", "reason": "news_veto"}
    
    # Basis-Size aus Meta
    base_size = (p_meta - 0.55) / 0.45  # [0, 1]
    
    # SCHICHT 3: 2D-Matrix-Multiplier
    multiplier = agreement_multiplier(composite_score, news_z)
    
    final_size = base_size * multiplier
    
    # SCHICHT 4 (optional): Cross-Impact aus Graph
    sector_sentiment = propagate_through_graph(ticker, news_features)
    if abs(sector_sentiment) > 0.7 and np.sign(sector_sentiment) != primary_side:
        final_size *= 0.5  # Sector-Headwind
    
    return {
        "action": "long" if primary_side > 0 else "short",
        "size": final_size,
        "composite_score": composite_score,
        "news_z": news_z,
        "p_meta": p_meta,
        "multiplier": multiplier,
    }
```

---

## Umsetzungs-Checkliste

**Schicht 1:**
- [ ] 6 Sub-Features pro Ticker berechnet
- [ ] EWMA- und Rolling-Z-Score kausal via `.shift(1)`
- [ ] FinBERT-Tone + DistilRoBERTa als Ensemble
- [ ] Event-Type-Klassifikation mit Taxonomie
- [ ] Regime-abhängige News-Gewichtung

**Schicht 2:**
- [ ] mlfinpy Triple-Barrier-Labeling + CUSUM-Events
- [ ] 12-15 Meta-Features-Set definiert
- [ ] Dynamische Triple-Barrier-Parameter nach Kontext
- [ ] LightGBM Meta-Model + CalibratedClassifierCV + Isotonic
- [ ] skfolio CombinatorialPurgedCV für Validation
- [ ] θ_meta=0.55 + τ_veto=1.5 konfiguriert

**Schicht 3:**
- [ ] 5×5-Matrix kalibriert via Optuna-GPSampler
- [ ] Bayesian-Beta-Update als Live-Pfad
- [ ] Agreement-Multiplier [0.5, 1.5]
- [ ] Shadow-Mode-Parallel-Run zur Matrix-Verifizierung

**Schicht 4 (optional):**
- [ ] OpenFIGI v3 Integration vor V2-Sunset Juli 2026
- [ ] Pearson-Correlation-Graph via Ledoit-Wolf
- [ ] Supply-Chain-JSON für Top-60 (manuell)
- [ ] Operator-Graph (CEOs via Wikidata)

---

## Ehrliche Einschätzung

**Die Drei-Schichten-Integration ist nicht über-engineered.** Sie ist die Antwort auf eine reale Lücke:

- Reine TA-Systeme verpassen Narrative-Shifts (z.B. FOMC-Surprise, M&A).
- Reine News-Systeme haben schlechtes Timing (Sentiment-Extremes sind oft Top/Bottom).
- Nur kombiniert ergibt sich robustes Verhalten.

**Die drei Schichten sind komplementär, nicht redundant:**
- Schicht 1 = was weiß ich insgesamt? (continuous)
- Schicht 2 = ist das ein guter Trade? (binary gate)
- Schicht 3 = wie groß positioniere ich? (multiplier)

**Sequentielle Einführung:**
- Phase 1: Schicht 1 live
- Phase 2 Anfang: Schicht 2 im Shadow-Mode 4-6 Wochen
- Phase 2 Mitte: Schicht 2 live mit θ=0.55, ohne Veto
- Phase 2 Ende: τ_veto-Regel aktivieren
- Phase 2 Ende: Schicht 3 live (Multiplier)
- Phase 3: Schicht 4 Cross-Impact-Graph

**Kritischer Erfolgsfaktor:** Jede Schicht ≥60 Tage im Shadow-Mode, bevor Live-Size angehoben wird.
