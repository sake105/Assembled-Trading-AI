# 11 — Free Modelle und Libraries (0 EUR/Monat)

**Zweck:** Alle Modelle und Libraries für Feature-Engineering, ML-Training, Validierung und NLP — ausschließlich Open-Source und lokal.

**Regel:** Nichts in diesem Dokument kostet laufend Geld. GPU ist optional — alle Modelle laufen auf CPU in akzeptabler Latenz.

---

## Module in diesem Dokument

| # | Modul | Kategorie |
|---|---|---|
| 11.1 | TA-Lib + pandas-ta-classic + talipp | Technische Indikatoren |
| 11.2 | FinBERT-Tone + DistilRoBERTa-Financial | Finanz-Sentiment (NLP) |
| 11.3 | BGE-small + all-MiniLM + Sentence-Transformers | Embeddings (Novelty, Similarity) |
| 11.4 | mlfinpy | López de Prado AFML-Methoden |
| 11.5 | skfolio CPCV + CombinatorialPurgedCV | Validierung |
| 11.6 | LightGBM + XGBoost | ML-Kern |
| 11.7 | MAPIE (Conformal Prediction) | Uncertainty |
| 11.8 | arch (GARCH-Familie) | Volatility-Modelle |
| 11.9 | hmmlearn + statsmodels MarkovRegression | Regime-Detection |
| 11.10 | ruptures | Change-Point-Detection |
| 11.11 | pykalman + filterpy | Kalman-Filter für Pairs |
| 11.12 | Riskfolio-Lib + skfolio | Portfolio-Optimization |
| 11.13 | stumpy + tslearn + dtaidistance | Pattern-Recognition |
| 11.14 | py_vollib + py_vollib_vectorized | Options-Greeks + IV |
| 11.15 | spaCy + GLiNER | Named Entity Recognition |
| 11.16 | HDBSCAN + hnswlib | News-Clustering |
| 11.17 | Evidently + NannyML | Drift-Monitoring |
| 11.18 | SHAP | Explainability |

---

## 11.1 TA-Lib + pandas-ta-classic + talipp

**Status 2026:** TA-Lib 0.6.8 (Okt 2025) liefert pre-built Wheels für Python 3.10-3.14 über `pip install TA-Lib` — **Windows-Pain ist vorbei**, keine Visual-Studio-Build-Tools mehr nötig.

**Library-Wahl:**

| Library | Verwendung |
|---|---|
| **TA-Lib 0.6.8** | Produktion, C-Backend, ~0.3-1 ms pro Indikator auf 100k Bars |
| **pandas-ta-classic 0.4.47** | Nischen-Indikatoren, xgboosted-Fork (aktiv März 2026) |
| **talipp 2.5+** | Incremental Indicators für Live-Streaming |

**Achtung:** Das Original `pandas-ta` von twopirllc ist seit PyPI-Wipe September 2025 mit unklarem Maintainer-Wechsel ein **Supply-Chain-Risiko**. **Nicht verwenden.** Lebende Forks: `pandas-ta-classic` (xgboosted) und `pandas-ta-openbb` (NumPy-2-kompatibel).

**Empfehlung:** TA-Lib für alle Standard-Indikatoren (RSI, MACD, BB, ADX etc.), pandas-ta-classic für Nischen (KAMA, ALMA, TTM-Squeeze), talipp für Live-Incremental.

---

## 11.2 Finanz-Sentiment-Modelle

**Primary: `yiyanghkust/finbert-tone`** (FinBERT-Tone)
- Trainiert auf 4.9 Mrd Tokens Analyst-Reports + 10-K/Q + Transcripts, fine-tuned auf 10.000 manuell annotierte Sätze.
- Label-Mapping: 0=neutral, 1=positive, 2=negative (**anders** als ProsusAI/finbert, Achtung).
- Größe: ~440 MB.
- Best-in-Class für Analyst-Reports und Earnings-Calls.

**Fast-Path: `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`**
- 82 M Params, ~10 ms/Doc CPU.
- ONNX-quantisierbar für <5 ms/Doc.
- Für News-Headlines-Volumen-Verarbeitung.

**SaaS-sicher: `ProsusAI/finbert`**
- Apache-2.0-Lizenz (FinBERT-Tone ist restriktiver).
- Nur noch für Legacy-Systeme verwenden, für Neuentwicklung nicht mehr — alte `pytorch_pretrained_bert`-Abhängigkeit.

**A/B-Test-Kandidat (Phase 2):** `tabularisai/ModernFinBERT`
- ModernBERT-Backbone, 8192-Token-Kontext.
- Claim "+48 % Accuracy" ist extern **nicht validiert** — A/B-Test Pflicht.

**FOMC/ECB-Spezialist:** `gtfintechlab/FOMC-RoBERTa`
- Trainiert auf FOMC-Statements + Minutes.
- Für Central-Bank-Divergence-Signal.

---

## 11.3 Embeddings für Novelty und Similarity

**Primary: `BAAI/bge-small-en-v1.5`**
- 384d, 33 M Params, MTEB ~63, ~4 ms/Headline CPU.
- Sweet-Spot für Novelty-Detection und News-Clustering.

**Alternative: `sentence-transformers/all-MiniLM-L6-v2`**
- 23 MB, 14k sent/s CPU.
- Guter Legacy-Standard, aber BGE-small ist besser kalibriert für Finanz-Text.

**Phase-2-Upgrade: `FinLang/finance-embeddings-investopedia`**
- Finance-Domain-Fit.
- A/B-Test empfohlen.

**Phase-2-Multilingual: `BAAI/bge-m3`**
- Multilingual, sehr stark — bei EU-Tickern relevant.

**Install:** `pip install sentence-transformers==3.3.1`

---

## 11.4 mlfinpy — López de Prado AFML

**Was:** Open-Source-Reimplementierung der López-de-Prado-Algorithmen aus *Advances in Financial Machine Learning*.
**Install:** `pip install mlfinpy==0.1.2`
**Lizenz:** MIT.

**Wichtige Module:**

| Modul | Zweck |
|---|---|
| `labeling.triple_barrier` | Triple-Barrier-Labeling |
| `labeling.add_vertical_barrier` | Time-based Vertical Barrier |
| `labeling.trend_scanning` | Trend-Labeling |
| `data_structures.standard` | Tick/Volume/Dollar/Run/Imbalance-Bars |
| `features.fracdiff` | Fractional Differentiation |
| `sample_weights.attribution` | Sample-Uniqueness-Weighting |

**Warum nicht `mlfinlab`:** Hudson & Thames hat auf "All Rights Reserved" umgestellt, die OSS-Version wird nicht mehr gepflegt, Commercial-Zugang via QuantConnect ~100 £/Monat. **mlfinpy ist der saubere freie Ersatz.**

**Verwendung im System:** Meta-Labeling, Triple-Barrier, CUSUM-Events, Fractional Diff — siehe `32_VALIDIERUNG.md`.

---

## 11.5 skfolio — CombinatorialPurgedCV

**Was:** Sklearn-contrib Library mit modernen Portfolio-Methoden UND Validation-Tools.
**Install:** `pip install skfolio`
**Lizenz:** BSD-3.
**Status:** sehr aktiv.

**Highlights:**
- `model_selection.CombinatorialPurgedCV` (der Goldstandard für Finanz-ML-Validation)
- `model_selection.WalkForward`
- HRP, NCO, CVaR-Opt
- Vine-Copula-Stress-Tests
- Hierarchical Portfolios

**Verwendung im System:** Meta-Labeling-Validation (`30_NEWS_TA_FUSION.md`), Portfolio-Construction (`13_FREE_MODULE.md`).

---

## 11.6 LightGBM + XGBoost

**LightGBM 4.6** als Primary für alle ML-Tasks:
- Schnelleres Training als XGBoost auf typischen Quant-Datensätzen.
- Native Support für `categorical_feature` (ideal für Sektor-Embeddings).
- `objective='quantile', alpha=0.05/0.95` für Quantile-Regression → 90%-Intervalle als Conformal-Alternative.

**XGBoost 2.1+** als Alternative:
- Robuster bei sehr kleinen Datensätzen.
- Für Ensemble-Stacking.

**Install:** `pip install lightgbm==4.6.0 xgboost==2.1.0`

**Für Meta-Labeling:** LightGBM + `CalibratedClassifierCV` mit Isotonic-Regression (≥1000 Samples) oder Platt-Sigmoid (<1000 Samples).

---

## 11.7 MAPIE — Conformal Prediction

**Was:** Verteilungsfreie Prediction-Intervalle mit marginaler Coverage-Garantie.
**Install:** `pip install mapie==0.9.2`
**Lizenz:** BSD-3.
**Roadmap 2026:** Exchangeability-Tests, adaptive CP, Risk-Control.

**Kern-Use-Case für Zeitreihen:** **EnbPI** (Xu/Xie 2021) — entfernt iid-Annahme, für Finanzreihen geeigneter als klassisches CP.

**Integration im System:**
- Auf Return-Forecast: liefert 90%-Intervall.
- Prediction-Interval-Width als Konfidenz-Proxy → Position-Sizing.
- Enge Bänder → Aufgewichtung; breite Bänder → Abregelung.

**Code-Skeleton:**
```python
from mapie.regression import MapieTimeSeriesRegressor
mapie = MapieTimeSeriesRegressor(estimator=lgb_model, cv="enbpi", alpha=0.1)
mapie.fit(X_train, y_train)
y_pred, y_pis = mapie.predict(X_test, alpha=0.1)  # 90%-Intervall
```

---

## 11.8 arch — GARCH-Familie

**Was:** One-Stop-Shop für Volatility-Modelling.
**Install:** `pip install arch==8.0.0`
**Lizenz:** NCSA.
**Status:** sehr aktiv, Maintainer Kevin Sheppard (Oxford).

**Modelle:**
- GARCH(1,1) — Baseline
- GJR-GARCH(1,1,1) mit skew-t-Innovationen — **De-facto-Standard für US-Equity-Daily**
- EGARCH, APARCH, FIGARCH, HARCH
- Realized-Volatility-Toolkit
- Bootstrap (Politis-Romano Stationary Bootstrap für Konfidenzintervalle)

**Use-Case:**
```python
from arch import arch_model
model = arch_model(returns*100, vol='GARCH', p=1, o=1, q=1, dist='skewt')
res = model.fit(disp='off')
sigma_forecast = res.forecast(horizon=5).variance.iloc[-1, :]
```

**Verwendung:** Vol-Feature für ML-Models, Vol-Skalierung für Triple-Barrier, GJR als Vol-Prognose-Baseline.

---

## 11.9 hmmlearn + statsmodels.MarkovRegression

**hmmlearn 0.3.3** — Gaussian-HMM für Regime-Detection.
**statsmodels.MarkovRegression** — Hamilton-1989-Baseline, interpretabel.

**Install:** `pip install hmmlearn statsmodels`

**Pattern für Regime-Feature:**
```python
from hmmlearn.hmm import GaussianHMM
# Features: log-Returns + 20d-Realized-Vol
X = np.column_stack([log_returns, realized_vol_20d])
hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
hmm.fit(X)
regimes = hmm.predict(X)  # Bull / Bear / High-Vol
```

**Regel:** Nicht mehr als 3-4 Regimes — Overfitting-Risiko + Label-Flipping bei kleinen Samples.

**Retraining:** wöchentlich (Walk-Forward alle 63 Bars).

---

## 11.10 ruptures — Change-Point-Detection

**Was:** BSD-2-Lizenz, Charles Truong, sehr aktiv (neuer PhD seit 2025).
**Install:** `pip install ruptures==1.1.10`

**Algorithmen:**
- **PELT** (Pruned Exact Linear Time) — schnellster exakter Algorithmus
- **BinSeg** (Binary Segmentation)
- **KernelCPD** (Kernel-basiert für nicht-parametrische Change-Points)

**Verwendung:** offline Regime-Break-Detection als Ergänzung zum HMM.

---

## 11.11 pykalman + filterpy — Kalman-Filter

**Status 2026:** `pykalman 0.9.7` (BSD-3, Community-Revival Jan 2026) ist zurück. Alternativ `filterpy 1.4.4` (MIT, Roger Labbe, exzellente Doku).

**Install:** `pip install pykalman filterpy`

**Kern-Use-Case: Pairs-Trading:**
```python
from pykalman import KalmanFilter
# State: [β_t, α_t]
# Observation: y_t = β_t·x_t + α_t + ε_t
kf = KalmanFilter(
    transition_matrices=np.eye(2),
    observation_matrices=[[x_t, 1]],  # zeitvariant
    observation_covariance=1.0,
    transition_covariance=np.eye(2)*0.01,
)
```

**Spread + Z-Score:** `spread = y - β̂·x` → Z-Score auf Rolling-Mean/Std oder GARCH-Vol.
**Entry:** |z|>2, Exit |z|<0.5, Stop |z|>4.

**Verwendung:** Pairs-Trading-Modul (falls gebaut, siehe `13_FREE_MODULE.md`).

---

## 11.12 Riskfolio-Lib + skfolio — Portfolio-Optimization

**Riskfolio-Lib 7.2.1** für 24 Risiko-Maße (CVaR, EVaR, GMD, MAD) und FactorModel.
**skfolio** für sklearn-kompatible HRP, NCO, CombinatorialPurgedCV.
**cvxpy 1.8.2** mit CLARABEL-Default-Solver (Rust-IPM).

**Install:**
```bash
pip install riskfolio-lib==7.2.1 cvxpy==1.8.2
```

**Quick-Start:**
```python
import riskfolio as rp
port = rp.Portfolio(returns=returns_df)
port.assets_stats(method_mu='hist', method_cov='ledoit')
w = port.optimization(model='Classic', rm='CVaR', obj='Sharpe')
```

**Verwendung:** Portfolio-Konstruktion nach Composite-Score-Selection. Siehe `13_FREE_MODULE.md`.

---

## 11.13 Pattern-Recognition

**stumpy 1.14+** — Matrix-Profile für Motif/Discord-Discovery.
- Sean Law / TD Ameritrade.
- `stumpy.stumpi()` für inkrementelles O(n)-Update pro neuer Bar — **ideal für Live-FastAPI**.

**tslearn 0.6.x** oder **dtaidistance 2.3.x** — Dynamic Time Warping.
- `dtaidistance` ist **schnellster CPU-Benchmark** via C-Backend.
- `fastdtw` paradox langsamer bei kleinen n.

**tslearn.clustering.KShape** — Shape-Based-Clustering (Paparrizos/Gravano 2015, SIGMOD Test-of-Time 2025).

**Install:** `pip install stumpy==1.14.0 tslearn==0.6.3 dtaidistance==2.3.13`

**Verwendung:** Chart-Pattern-Dimension (siehe `31_COMPOSITE_SCORE.md` Dim-5). Matrix-Profile schlägt Head-and-Shoulders-Detection in den meisten Tests.

---

## 11.14 py_vollib + py_vollib_vectorized

**Was:** Options-Greeks + Implied Volatility via Peter Jäckels "Let's-be-rational" (IV-Inversion in 2 Iterationen).
**Install:** `pip install py_vollib==1.0.1 py_vollib_vectorized==0.1.1`
**Lizenz:** MIT.

**Vermeiden:** `mibian` — veraltet seit 2018.

**Verwendung:**
- IV-Rank + Skew + Term-Structure-Features (Vol-Surface-Dimension)
- Greeks für Options-Overlay-Strategien

---

## 11.15 spaCy + GLiNER — NER

**spaCy 3.7+** mit `en_core_web_lg` (**nicht** `_trf` — 20× langsamer, 10 GB RAM).
**GLiNER 0.2+ / GLiNER2** (Juli 2025) als Zero-Shot für niedrig-konfidente spaCy-Calls.

**Install:**
```bash
pip install spacy gliner
python -m spacy download en_core_web_lg
```

**Pattern:** spaCy als Fast-Path, Cashtag-Regex `\$[A-Z]{1,5}\b` als Zusatz, GLiNER als Zweitrunde. Nicht GLiNER als Primary — 3-5× langsamer.

**Company-zu-Ticker-Mapping:** hierarchisch
1. Lokaler Alias-Cache
2. **OpenFIGI v3** (free, MIT) — V2 Sunset 01.07.2026
3. Wikidata SPARQL (P414 = ISIN, P249 = Ticker)
4. yfinance als letzter Fallback

---

## 11.16 HDBSCAN + hnswlib — News-Clustering

**HDBSCAN 0.8.38+** für News-Event-Clustering.
```python
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='cosine')
labels = clusterer.fit_predict(embeddings)
```

**hnswlib 0.8** für semantische Dedup-Suche (inkrementelle Inserts, besser als FAISS für Streaming).
```python
import hnswlib
p = hnswlib.Index(space='cosine', dim=384)
p.init_index(max_elements=100_000, ef_construction=200, M=16)
# Dedup-Threshold ~0.92 Kosinus-Similarität
```

**Install:** `pip install hdbscan==0.8.38 hnswlib==0.8.0`

---

## 11.17 Evidently + NannyML — Drift-Monitoring

**Evidently 0.7+** für Feature-Drift (PSI/KS/JS/Wasserstein, 100+ Metriken, HTML-Reports + UI).
**NannyML** für Performance-Estimation **ohne Ground-Truth** (CBPE, DLE — einzigartig).

**Install:** `pip install evidently nannyml`
**Lizenz:** Apache-2.0.

**Prometheus-Alert-Regel:**
```
psi_drift_score > 0.25 for 2d → Slack + auto-throttle auf 25% Size
psi_drift_score > 0.35 for 1d → Auto-Pause Signal
```

**Verwendung:** Daily-Cronjob, PSI pro Feature vs. Training-Baseline.

---

## 11.18 SHAP — Explainability

**Install:** `pip install shap==0.48`

**Pattern:**
- `TreeExplainer` auf LightGBM/XGBoost-Meta-Labeler (schnell, exakt).
- Pro Trade Shapley-Values speichern (~KB/Trade) → Feature-Level-Drawdown-Attribution.
- `KernelExplainer` 100× langsamer — nur für Blackbox.

**Verwendung:** P&L-Attribution pro Signal, Dashboard-Waterfall.

---

## Umsetzungs-Checkliste

- [ ] TA-Lib 0.6.8 installiert und funktioniert
- [ ] FinBERT-Tone lokal geladen, Label-Mapping dokumentiert
- [ ] BGE-small für Embeddings, Cache-Strategy klar
- [ ] mlfinpy statt mlfinlab im gesamten Code
- [ ] skfolio.CombinatorialPurgedCV als Standard-CV für Meta-Labeling
- [ ] LightGBM + CalibratedClassifierCV-Pattern dokumentiert
- [ ] MAPIE mit EnbPI auf eine Beispiel-Strategie
- [ ] arch GJR-GARCH als Vol-Feature in Feature-Store
- [ ] hmmlearn-Regime-Classifier mit wöchentlichem Retraining
- [ ] pykalman-Pairs-Trading-Template verfügbar (optional Phase 3)
- [ ] Riskfolio + cvxpy für Portfolio-Opt
- [ ] stumpy-Matrix-Profile + dtaidistance-DTW in Pattern-Pipeline
- [ ] py_vollib für Vol-Surface-Features
- [ ] spaCy `en_core_web_lg` + OpenFIGI-Mapping
- [ ] HDBSCAN + hnswlib für News-Clustering
- [ ] Evidently-Daily-Cronjob für PSI-Drift
- [ ] SHAP-Waterfall pro Trade im Dashboard

---

## Was NICHT in diesem Katalog ist

Explizit nicht empfohlen für Solo-Quant-System:

- **TensorTrade:** Projekt tot.
- **FinRL / RL-Trading:** Research-only, nicht produktiv.
- **FinGPT v3 13B:** HW-Anforderungen sprengen Budget (>16 GB VRAM).
- **pytorch-forecasting (primär):** nur für TFT-Interpretability-Use-Cases. `neuralforecast` (Nixtla) ist besser für Neural-TS.
- **GNN (pytorch-geometric, DGL):** Phase-3-Experiment, nicht für Alpha in Phase 1/2.
- **stellargraph:** seit 12+ Monaten kein Release, tot.
- **Deep Learning auf Candle-Bildern:** Overfitting-Minenfeld.
- **BloombergGPT-Alternativen (FinMA, PIXIU):** Research-oriented, nicht deployment-ready.
