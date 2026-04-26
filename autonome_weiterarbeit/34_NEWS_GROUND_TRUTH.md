# 34 — News-Ground-Truth-Validation

**Zweck:** Beweise dass die 6 News-Sub-Features aus `30_NEWS_TA_FUSION.md` echt Alpha liefern und nicht nur Rauschen sind, das im Composite-Score fettgewichtet wird. Du hast Plan + Code gebaut — dieses Dokument prüft, ob der gebaute Teil **wirklich funktioniert**.

**Scope:** Rang 2 aus der Gap-Analyse. Voraussetzung für echten Einsatz von Plan-Layer 3.

**Kern-Problem:** News-Sentiment-Analyse ist ein Bereich mit extrem hohem Hype-Potenzial. Jeder Pre-Print verspricht Sharpe 2+, real-live bleibt 0.3-0.7 übrig. Ohne strikte Validierungs-Pipeline baust du auf Sand.

---

## 0. Die ungemütlichen Wahrheiten aus der Literatur

Bevor wir ins Methodische gehen — vier Befunde aus 2024/2025, die dein Erwartungsmanagement kalibrieren sollten:

### 0.1 FinBERT ist gut, aber nicht magisch

- Financial PhraseBank (der Standard-Benchmark): fine-tuned FinBERT erreicht **Accuracy 0.88 / F1 0.87** (Shen et al. 2024).
- Auf einem **unabhängigen** 1500-Headline-Corpus aus H1 2025 (MDPI/Electronics, Nov 2025): Zero-shot FinBERT nur **Macro-F1 0.555**; fine-tuned auf Gold-Data 0.707. **Out-of-sample bricht die Performance um ~20-25 Punkte ein.**
- FinBERT macht 73 % der Fehler zwischen positiv und neutral, nur 5 % zwischen positiv und negativ. Das heißt: die Neutralitäts-Entscheidung ist fragil, die Richtungs-Entscheidung stabil.

**Konsequenz für deinen Plan:** Nicht die rohe FinBERT-Wahrscheinlichkeit als Signal verwenden. Stattdessen Delta zur Baseline (z.B. +1 wenn positiv wahrscheinlich > 0.6, -1 wenn negativ > 0.6, 0 sonst). Tri-class mit harten Schwellen, nicht continuous.

### 0.2 LLMs (GPT-4o, Claude, DeepSeek) schlagen FinBERT manchmal — aber nicht in allen Settings

- Kirtac & Germano (2024): ChatGPT outperformed FinBERT by **35 %** in Forex-Headline-Classification.
- Shen et al. (2024): Fine-tuned FinBERT 0.88, GPT-4o zero-shot 0.86, few-shot 0.85. **Knappes Kopf-an-Kopf**, aber FinBERT kostet 0.
- CLiC-it 2025 Benchmark: GPT-4o und DeepSeek-R1 schlagen FinBERT auf **target-level** Analyse (welche Firma ist gemeint bei Multi-Firm-News). 
- Aber: Reasoning-Prompts (Chain-of-Thought) **verschlechtern** die Performance konsistent (ACM AIF 2025). Einfaches "Classify positive/negative/neutral" war besser als "Think step by step".

**Konsequenz für Haiku 4.5 in deinem Plan:** Zero-shot mit knappem Prompt, kein CoT. Nur als **Second-Stage-Veto**, nachdem FinBERT-Vorfilter gelaufen ist. Nicht für jede Headline direkt.

### 0.3 News-Daily-Sentiment hat schwache Predictive Power

- Großer MDPI-Survey 2025 (1.86 Mio Headlines): "News content predominantly consists of objective or neutral information, with only a small portion carrying subjective or emotive weight." **Die meiste Nachricht ist Rauschen.**
- Forward-looking implied sentiment (VIX) erklärt ~45-50 % der Return-Varianz. Explicit-Sentiment-Scores (FinBERT, VADER, TextBlob): **"lack robust predictive power"** nach Permutation-Importance.
- Aber: Weekend-/Holiday-News haben "modest yet valuable market signals" — Signal-Wirkung ist **zeitlich konzentriert** um Nicht-Handels-Zeiten.

**Konsequenz:** Die Hypothese "News-Sentiment ist ein gleichmäßig verteiltes Alpha-Signal" ist falsch. Die richtige Hypothese: "News-Sentiment ist episodisch relevant — bei großen Events, News-Floods, oder außerhalb der Handelszeiten." Dein Validation muss diese Heterogenität testen.

### 0.4 Look-Ahead-Bias bei LLM-basierten News-Strategien ist systemisch

- Glasserman & Lin (Columbia, 2023): LLMs sind auf **jahrelangen** Daten trainiert. Wenn du auf 2022er-Headlines backtestest, "weiß" der LLM bereits, was passiert ist.
- Ihr Test: Entity-Anonymization (Firma-Namen in Headlines maskieren). **Ergebnis: In-sample Performance steigt**, was beweist dass der LLM nicht die Sentiment-Analyse macht, sondern von seinem Pre-Training "weiß". Live-Performance divergiert.

**Konsequenz:** Wenn du LLMs für historische Validierung nutzt (Haiku 4.5 für 2022-Backtests), läuft ein Teil der "Performance" aus Leakage. Du musst Entity-Anonymization als Sanity-Check einbauen.

---

## 1. Gold-Standard-Datasets: das Fundament

### 1.1 Was die Community nutzt (und warum)

| Dataset | Umfang | Typ | Nutzen für dich |
|---|---|---|---|
| **Financial PhraseBank** (Malo et al. 2014) | 4.846 Sätze, 16 Annotatoren | Sentence-level 3-class | **Pflicht**. Community-Standard. Hugging Face: `takala/financial_phrasebank` |
| **FiQA Task 1** (2018) | 498 Headlines + 675 Posts | Aspect-based, continuous scores | Nutzt du für **target-based** Validation (welche Firma ist gemeint) |
| **SEntFiN** (2024) | ~10.000 Microblog-Posts | 3-class mit Fine-grained | Social-Media-spezifisch. Für Twitter/StockTwits-Teil |
| **Twitter Financial News** (Zeroshot) | 11.932 Tweets | 3-class | Alternative zu SEntFiN, weniger kuratiert |
| **Business Insider + FPB** (Omarzadeh, IEEE DataPort 2025) | FPB + 1.000 BI headlines | 3-class | Neue, moderne Ground-Truth aus 2024/2025 |

**Für deinen Plan: Download alle 5, stecke sie in `tests/news_gold/`. Total ~30 MB, ein Nachmittag Arbeit.**

### 1.2 Dein eigenes Gold-Dataset

Aus Erfahrung (MDPI-Electronics Nov 2025, Kang & Choi): **ein eigenes, manuell gelabeltes Test-Set ist notwendig**. Die öffentlichen Datasets sind pre-2020 dominiert — deine Live-Daten sind 2026+. Drift bis zur Unkenntlichkeit.

**Rezept:**

1. Sammle 500-1000 Headlines aus deinen echten News-Quellen (Finnhub, GDELT, RSS) aus 3 verschiedenen Wochen (z.B. Q1 2026, ein Earnings-Peak, eine ruhige Phase).
2. Stratifiziere nach Sektor (Tech, Financials, Energy, Healthcare) und Headline-Typ (Earnings, M&A, Regulierung, Makro).
3. Label selbst. Regel: "Würde diese Headline, unabhängig von Preisen, einen professionellen Trader positiv/negativ/neutral auf die Firma einstimmen?" Tri-class.
4. **Double-check mit Claude Sonnet 4.7**, aber du bist die Endautorität. Wo Claude und du disagreed: **das sind die lehrreichen Fälle** — häufig subtile Sprache, die dein System lernen muss.
5. Committe als `tests/news_gold/ata_headlines_2026q1.jsonl`.

**Aufwand:** 6-8 Stunden für 500 Headlines. Das ist die **teuerste** aber **wertvollste** Einzelinvestition in der News-Validation.

---

## 2. Die drei Validierungs-Ebenen

### Ebene A: Classification-Accuracy (is it correct?)
### Ebene B: Economic Relevance (does it predict returns?)
### Ebene C: Tradability (does it survive costs?)

Ein Feature muss **alle drei** bestehen. Viele Papers validieren nur A, manche A+B, wenige alle drei.

---

## 3. Ebene A — Classification-Accuracy

### 3.1 Metriken (Standard)

```python
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class_f1": f1_score(y_true, y_pred, average=None).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
```

**Schwellwerte** (aus der Literatur 2024-2025):

| Metrik | Schwelle | Bedeutung |
|---|---|---|
| Accuracy auf FPB allagree | ≥ 0.85 | Community-Standard für Production |
| Macro-F1 auf FPB allagree | ≥ 0.80 | Ungewichtet, fair über Klassen |
| Macro-F1 auf eigenem Gold | ≥ 0.65 | Realistisch für out-of-domain |
| Confusion: Positiv↔Negativ-Fehler | ≤ 5 % | Kritisch: Vorzeichen-Fehler |
| Confusion: Neutral-Verwechslung | ≤ 20 % | Toleriert, weil fuzzy |

**Wenn du unter diesen Zahlen bleibst:** FinBERT ist nicht richtig eingesetzt oder deine Labels sind verzerrt.

### 3.2 Der Baseline-Test

```python
# scripts/news_validation/level_a_classification.py
"""
Läuft FinBERT-Tone und Claude Haiku 4.5 gegen alle 5 Gold-Datasets + dein eigenes.
Output: Ein Markdown-Report mit den Metriken pro Dataset.
"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from anthropic import Anthropic

DATASETS = {
    "fpb_allagree":        "tests/news_gold/fpb_allagree.jsonl",
    "fpb_75agree":         "tests/news_gold/fpb_75agree.jsonl",
    "fiqa_headlines":      "tests/news_gold/fiqa_headlines.jsonl",
    "twitter_fin":         "tests/news_gold/twitter_financial_news.jsonl",
    "sentfin":             "tests/news_gold/sentfin.jsonl",
    "ata_2026q1":          "tests/news_gold/ata_headlines_2026q1.jsonl",  # dein eigenes
}

def run_finbert_tone(texts):
    pipe = pipeline("text-classification", 
                    model="yiyanghkust/finbert-tone",  # empfehlenswert, Yang et al 2020
                    tokenizer="yiyanghkust/finbert-tone",
                    max_length=128, truncation=True)
    return [pipe(t)[0]["label"].lower() for t in texts]

def run_haiku_zeroshot(texts, anthropic_client):
    """Simple 3-class prompt. KEIN CoT (siehe ACM AIF 2025)."""
    prompt_template = """Classify the sentiment of this financial headline toward the mentioned company as one of: positive, negative, neutral. Reply with only that word.

Headline: {headline}

Sentiment:"""
    results = []
    for t in texts:
        resp = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": prompt_template.format(headline=t)}]
        )
        label = resp.content[0].text.strip().lower()
        # normalisieren: alles was nicht exakt passt → neutral
        if label not in ("positive", "negative", "neutral"):
            label = "neutral"
        results.append(label)
    return results

# Vollständiges Script siehe scripts/news_validation/
```

### 3.3 Spezifische Anti-Patterns beim FinBERT-Einsatz

**Fehler 1 — falsche Checkpoint-Wahl:**
- `ProsusAI/finbert` (der Original-Checkpoint): Accuracy ~0.85 auf FPB, aber veraltet (2019).
- `yiyanghkust/finbert-tone`: 2020, auf Analyst-Reports fine-tuned, **robuster** für News. **Empfehlung.**
- `ahmedrachid/FinancialBERT-Sentiment-Analysis`: weniger bekannt, schlägt `finbert-tone` auf einigen Benchmarks — als Vergleich testen.

**Fehler 2 — Headline-only vs. Full-Article:**
- Headline-only: schnell, günstig, 60-80 % der Trefferqualität.
- Full-Article (Headline + Body): 5-10 Punkte F1 besser, aber 20-30× langsamer.
- **Empfehlung:** Headline-only für Real-Time, Full-Article für Batch-Nacht-Jobs.

**Fehler 3 — Rohe Softmax-Scores als Signal-Strength:**
- FinBERT liefert Softmax-Probs über 3 Klassen.
- Version A (naiv): `sentiment_score = p(pos) - p(neg)`. Problem: neutrale Nachrichten mit knappem Pos-Edge werden als "schwach positiv" interpretiert.
- Version B (besser): Nur wenn `max(p) > 0.6`, verwende `sign(pos-neg) * (max(p) - 0.33)`. Sonst Signal = 0.
- Version C (robust): Klassenentscheidung {−1, 0, +1}, dann Aggregation über **mehrere** Artikel zu einer Ticker-Ebene-Score. Einzelne Confidence ist unbedeutend.

**Empfehlung für deinen Plan:** Version C. FinBERT-Wahrscheinlichkeit wird nicht einzeln gewichtet; nur Mehrheits-Urteil über Artikel-Cluster zählt.

---

## 4. Ebene B — Economic Relevance (Event-Study)

Hier trennt sich die Spreu vom Weizen. 95 % der "News-Sentiment-Predictors" überleben diesen Test nicht.

### 4.1 Event-Study-Framework (MacKinlay 1997 Standard)

**Konzept:** Für jede News-Event (Timestamp + Ticker + Sentiment-Label):
1. Definiere ein **Estimation-Window** (z.B. T-250 bis T-11, also ~1 Jahr vor).
2. Definiere ein **Event-Window** (z.B. T-1 bis T+5).
3. Schätze Expected-Return im Estimation-Window via Market-Model (Regression auf Market-Return).
4. Abnormal-Return (AR) = Actual - Expected im Event-Window.
5. Cumulative AR (CAR) = Summe der täglichen ARs im Event-Window.

**Wenn deine News-Sentiment-Klassifizierung echt ist:** Positive News → CAR > 0, negative News → CAR < 0, statistisch signifikant.

### 4.2 Python-Implementation

```python
# scripts/news_validation/level_b_event_study.py
"""
Event-Study für eine Sentiment-Klassifizierung.
Input: news_events (Ticker, Timestamp, Label)
Output: Mean CAR pro Klasse, T-Statistik, Signifikanz
"""

import pandas as pd
import numpy as np
from scipy import stats

def compute_market_model(returns_ticker, returns_market, estimation_start, estimation_end):
    """Lineare Regression R_it = α_i + β_i R_mt + ε_it über Estimation-Window."""
    mask = (returns_ticker.index >= estimation_start) & (returns_ticker.index <= estimation_end)
    y = returns_ticker[mask].dropna()
    x = returns_market.loc[y.index]
    
    if len(y) < 100:  # mindestens 100 Tage für stabile Schätzung
        return None, None, None
    
    beta, alpha = np.polyfit(x, y, 1)
    residual_std = np.std(y - (alpha + beta * x))
    return alpha, beta, residual_std

def compute_abnormal_return(returns_ticker, returns_market, alpha, beta, 
                             event_start, event_end):
    """AR_it = R_it - (α + β R_mt) im Event-Window."""
    event_mask = (returns_ticker.index >= event_start) & (returns_ticker.index <= event_end)
    r_t = returns_ticker[event_mask]
    r_m = returns_market.loc[r_t.index]
    expected = alpha + beta * r_m
    return r_t - expected

def event_study(events_df, prices_df, benchmark_returns, 
                est_window=(-250, -11), event_window=(-1, 5)):
    """
    events_df: columns = [ticker, event_date, sentiment_label]
    prices_df: DataFrame mit Ticker-Returns als Columns, Date als Index
    benchmark_returns: Series mit SPY/Index-Returns
    """
    results = []
    
    for idx, row in events_df.iterrows():
        ticker = row["ticker"]
        event_date = pd.Timestamp(row["event_date"])
        label = row["sentiment_label"]
        
        if ticker not in prices_df.columns:
            continue
        
        ticker_returns = prices_df[ticker]
        
        est_start = event_date + pd.Timedelta(days=est_window[0])
        est_end = event_date + pd.Timedelta(days=est_window[1])
        evt_start = event_date + pd.Timedelta(days=event_window[0])
        evt_end = event_date + pd.Timedelta(days=event_window[1])
        
        alpha, beta, resid_std = compute_market_model(
            ticker_returns, benchmark_returns, est_start, est_end)
        if alpha is None:
            continue
        
        ars = compute_abnormal_return(
            ticker_returns, benchmark_returns, alpha, beta, evt_start, evt_end)
        
        car = ars.sum()
        
        results.append({
            "ticker": ticker,
            "event_date": event_date,
            "label": label,
            "car": car,
            "n_event_days": len(ars),
            "resid_std": resid_std,
        })
    
    return pd.DataFrame(results)

def test_significance(event_study_df):
    """Test ob Mean CAR pro Label signifikant von 0 verschieden."""
    by_label = event_study_df.groupby("label")
    
    report = {}
    for label, group in by_label:
        car_series = group["car"]
        t_stat, p_val = stats.ttest_1samp(car_series, 0.0)
        report[label] = {
            "n": len(car_series),
            "mean_car": car_series.mean(),
            "median_car": car_series.median(),
            "std_car": car_series.std(),
            "t_stat": t_stat,
            "p_value": p_val,
            "significant_5pct": p_val < 0.05,
        }
    return report
```

### 4.3 Was dabei rauskommen sollte

**Healthy-Pattern (Sentiment ist real):**
```
Label: positive
  n=2431, mean_CAR=+0.48%, t_stat=+3.21, p=0.001  ✓
Label: negative
  n=1987, mean_CAR=-0.67%, t_stat=-4.15, p<0.001  ✓
Label: neutral
  n=5312, mean_CAR=+0.02%, t_stat=+0.18, p=0.85   ✓ (nicht signifikant = korrekt)
```

**Warnzeichen 1 — Asymmetrie:**
```
positive: +0.48%, p=0.001
negative: -0.09%, p=0.62
```
Das bedeutet: Dein Sentiment ist nur für Pos-News präzise, negative Einstufungen sind Rauschen. Mögliche Ursache: Trainingsdaten-Bias (FinBERT neigt zu Pos-/Neu-Verwechslung). Nicht bedingungslos als Long-Short-Signal nutzen.

**Warnzeichen 2 — Neutral mit Signal:**
```
neutral: +0.35%, p=0.02
```
Das heißt: Deine "neutralen" News sind gar nicht neutral. Labels sind falsch oder Classifier ist miskalibriert.

**Warnzeichen 3 — CARs winzig:**
```
positive: +0.03%, p=0.04
negative: -0.04%, p=0.03
```
Statistisch signifikant, aber ökonomisch bedeutungslos. Nach Slippage + Spread (5-20 bps) wird das Signal aufgefressen. **Das ist der häufigste Fall in der Literatur.**

### 4.4 Langtermige Drift: PEAD als Sanity-Check

PEAD (Post-Earnings-Announcement-Drift) ist die **am besten dokumentierte** News-bezogene Anomalie seit Ball & Brown 1968. Wenn dein News-System auf Earnings-News keine PEAD-Drift nachweist, ist etwas grundlegend kaputt.

**Konkreter Test:**

```python
# Filter events_df auf earnings_announcement-Events
earnings_events = events_df[events_df["event_type"] == "earnings"].copy()

# Bucketize nach SUE (Standardized Unexpected Earnings) oder als Proxy:
# Sentiment-Score als Surprise-Proxy
# Top-Decile (stärkste positive) vs Bottom-Decile (stärkste negative)

# Event-Window: T+1 bis T+60 (60 Handelstage Drift)
drift_study = event_study(earnings_events, prices_df, benchmark_returns,
                           est_window=(-250, -11), event_window=(1, 60))

# Erwartung: Top-Decile CAR +4-8% über 60 Tage, Bottom-Decile -4-8%
```

**Bekannte Zahlen:**
- Bernard & Thomas (1989/1990): 10-25 % Annual-Return für Top-minus-Bottom-PEAD-Portfolio.
- Post-2010-Studien (Martineau 2021, Columbia 2024): **PEAD has declined** durch mehr Algo-Trading und schnellere Information Processing. Erwartung heute: 3-8 %/Jahr statt 15-25 %.
- Wenn dein Sentiment-System PEAD **nicht** reproduzieren kann (Top-Decile outperformance auf 60d): Problem mit Labeling oder mit Event-Detection.

### 4.5 Kontrolle auf Look-Ahead-Bias

**Der kritische Test nach Glasserman & Lin (2023):**

```python
def entity_anonymize(headline: str, ticker: str, company_name: str) -> str:
    """Ersetze Ticker und Firma durch neutrale Platzhalter."""
    anonymized = headline
    anonymized = anonymized.replace(ticker, "XYZ")
    anonymized = anonymized.replace(company_name, "Company XYZ")
    # Auch CEO-Namen, Produktnamen etc. wenn bekannt
    return anonymized

# Klassifiziere beide Versionen
original_labels = run_finbert_tone(headlines)
anonymized_labels = run_finbert_tone([entity_anonymize(h, t, c) 
                                        for h, t, c in zip(headlines, tickers, companies)])

# Vergleich-Metriken
agreement = sum(o == a for o, a in zip(original_labels, anonymized_labels)) / len(original_labels)
print(f"Label-Agreement nach Anonymisierung: {agreement:.1%}")
```

**Erwartung für FinBERT:** Agreement **≥ 95 %**. FinBERT ist semantisch, nicht wissensbasiert.
**Erwartung für Claude Haiku:** Agreement **≥ 85 %**. LLM kennt Firma, aber wenn Prompt fokussiert auf Sentiment-Ausdruck, nicht Prior-Knowledge abrufen.

**Warnung:** Wenn Agreement < 80 %, dann klassifiziert dein LLM nach Firmen-Prior, nicht nach Text. Das heißt: Live-Einsatz auf neuen Firmen = Rauschen.

---

## 5. Ebene C — Tradability (IC, Turnover, Costs)

Classification-Accuracy und CAR-Significance beweisen: Sentiment **korreliert** mit Returns. Das bedeutet **nicht** dass du damit Geld verdienen kannst. Ebene C testet das.

### 5.1 Alphalens-Integration

`Alphalens` (Quantopian-Library, noch maintained als `alphalens-reloaded`) ist der De-facto-Standard für Factor-Validation.

```bash
uv pip install alphalens-reloaded==0.5.0
```

```python
# scripts/news_validation/level_c_alphalens.py
import alphalens
import pandas as pd

def build_factor_series(news_events_df, tickers_universe, dates):
    """
    Baut MultiIndex (date, asset) → factor_value Series.
    
    Aggregation pro Tag × Ticker:
      - Alle Headlines des Tages für Ticker X sammeln
      - Average Sentiment-Score (Soft-Label: {+1, 0, -1} Mapping)
      - Wenn keine News: NaN (→ Alphalens droppt dann automatisch)
    """
    agg = news_events_df.groupby(["ticker", pd.Grouper(key="timestamp", freq="D")])
    agg_score = agg["sentiment_numeric"].mean()  # -1..+1
    agg_count = agg.size().rename("news_count")
    
    # Reindex auf komplettes (date, ticker) Grid
    idx = pd.MultiIndex.from_product([dates, tickers_universe], names=["date", "asset"])
    factor = agg_score.reindex(idx)
    return factor

# Nutzung
factor_series = build_factor_series(news_events_df, sp500_tickers, dates)
prices = get_prices_dataframe(sp500_tickers, dates)

factor_data = alphalens.utils.get_clean_factor_and_forward_returns(
    factor=factor_series,
    prices=prices,
    quantiles=5,
    periods=(1, 5, 10, 20),
)

# Full tear sheet
alphalens.tears.create_full_tear_sheet(factor_data)
```

### 5.2 Die Kernzahlen (IC und IR)

**Information Coefficient (IC)** = Spearman-Rank-Correlation zwischen Factor-Value zu T und Forward-Return (T zu T+N).

- IC per day/week/month
- Mean IC über lange Periode
- IR (Information Ratio) = Mean_IC / Std_IC — die Konsistenz

**Benchmarks aus der Literatur (AQR, Robeco, 2023-2025):**

| IC-Range | Bedeutung |
|---|---|
| IC > 0.10 | **Sehr gut.** Selten, meist Multi-Factor-Ensembles |
| IC 0.05-0.10 | **Gut.** Single-Factor-Edge, tradable |
| IC 0.02-0.05 | **Marginal.** Nur mit großer Breite (100+ Ticker) profitable |
| IC < 0.02 | **Rauschen.** Nicht tradable |
| IC > 0.20 | **Rot-Flag.** Vermutlich Look-Ahead-Bias oder Overfitting |

**Für News-Sentiment spezifisch:** Studien zeigen typisch IC 0.01-0.04 auf täglicher Basis. Das ist niedrig, aber verwertbar wenn dein Universum groß genug ist und Turnover kontrollierbar.

### 5.3 Quantile-Return-Spread

Aus Alphalens: Durchschnittliche Forward-Return pro Quantile-Bucket (Q1 bis Q5).

**Gesundes Pattern:**
```
5-day forward return by quantile:
Q1 (most negative):  -0.08%
Q2:                  -0.03%
Q3 (neutral):         0.01%
Q4:                   0.04%
Q5 (most positive):  +0.11%
```

Monoton von Q1 bis Q5 — **das ist das Benchmark.** Nicht-monoton oder wilde Swings = Signal ist instabil.

### 5.4 Turnover

Wenn dein News-Sentiment dich täglich zwingt, die komplette Portfolio-Allokation zu drehen: Signal ist zu kurzlebig.

```python
# In Alphalens automatisch:
turnover = factor_data.groupby(level="date")["factor_quantile"].apply(
    lambda x: (x != x.shift(1)).sum() / len(x)
)
print(f"Daily quantile turnover: {turnover.mean():.1%}")
```

**Benchmarks:**
- < 20 % / Tag: **OK**, Signal ist persistent genug
- 20-50 % / Tag: **Problematisch**, Kosten essen Alpha
- > 50 % / Tag: **Untradable**, auch mit perfekter Signal-Quality

### 5.5 Net-Edge nach Kosten

```python
def net_edge_after_costs(mean_return_by_quantile, turnover_rate, 
                         cost_bps_per_side=8):
    """
    Cost-Modell: 8 bps per side (realistisch für US-Equity Mid-Cap).
    Round-trip bei voller Quantile-Rotation: 2 × 8 = 16 bps.
    """
    q5_minus_q1 = mean_return_by_quantile["Q5"] - mean_return_by_quantile["Q1"]
    daily_cost = 2 * cost_bps_per_side * turnover_rate / 10000
    net_q5_q1 = q5_minus_q1 - daily_cost
    annualized = net_q5_q1 * 252  # daily to yearly
    return {
        "gross_edge_annual_pct": q5_minus_q1 * 252 * 100,
        "cost_annual_pct": daily_cost * 252 * 100,
        "net_edge_annual_pct": annualized * 100,
    }
```

**Go/No-Go-Kriterium:** Net-Edge annual > 3 % nach Kosten. Darunter: das Feature ist nicht tradable im aktuellen Form.

---

## 6. Die sechs News-Sub-Features aus `30_NEWS_TA_FUSION.md` systematisch prüfen

Dein Plan spezifiziert 6 News-Sub-Features für die Composite-Dimension. Jedes einzeln validieren:

### 6.1 Feature 1: FinBERT-Sentiment (headline-level)

**Test-Protokoll:**
1. Ebene A: FPB allagree, FPB 75%, eigenes Gold-Set → F1 ≥ 0.80, 0.65, 0.65
2. Ebene B: Event-Study auf 1000+ Earnings-Events → Pos-CAR > +0.3 %, Neg-CAR < -0.3 %, beide p < 0.05
3. Ebene C: Alphalens IC 5-day ≥ 0.02, Quantile-Spread ≥ 15 bps

**Falls durchgefallen:** Checkpoint wechseln (`finbert-tone` statt `finbert`), Full-Article statt Headline-only, Tri-class statt soft-score.

### 6.2 Feature 2: News-Volume-Spike (Anzahl Artikel pro Ticker pro Tag)

**Hypothese:** Plötzlicher Anstieg der News-Coverage = Event, oft mit Markt-Reaktion.

**Test-Protokoll:**
1. **Keine Ebene A** nötig (kein Sentiment, reines Count)
2. Ebene B: Event-Study auf "Volume-Spike-Events" (Tag mit >3× 20-Tage-Durchschnitt) → |CAR| > 0.5 %
3. Ebene C: IC auf forward vol (nicht return) → persistenter Volatility-Boost

**Falls durchgefallen:** Normalize by sector or by market-cap. Bei kleinen Firmen ist 1 zusätzliche News bereits 10× Spike.

### 6.3 Feature 3: News-Source-Quality-Weight (Reuters > Seeking Alpha)

**Hypothese:** Nicht jede Quelle ist gleich stark. Market reagiert mehr auf Reuters als auf Seeking-Alpha-Blog.

**Test-Protokoll:**
1. Stratifiziere Event-Study nach Quelle
2. Für jede Quelle: Mean |CAR| und statistische Signifikanz
3. Build Source-Weight-Ranking basierend auf CAR-Magnitude

**Warnung:** Reuters-Events sind oft schon vor der News im Markt (Informationsleck, Analysten-Leaks). In der Literatur hat **Benzinga** überraschend starke CARs gezeigt, weil die breaking vor Mainstream publizieren.

### 6.4 Feature 4: Headline-Uncertainty-Score (LLM-basiert)

**Hypothese:** Nicht nur Pos/Neg zählt, sondern Unsicherheit. "Revenue up 5 % beat estimates" ist klar; "Revenue guidance withdrawn" ist klar-negativ-aber-unsicher.

**Test-Protokoll:**
1. LLM (Haiku 4.5) klassifiziert headlines in Certainty-Score {low, medium, high}
2. Event-Study: High-Uncertainty-Events → höhere Vol-Post-Event, möglicherweise kleinere Mean-CAR aber größere Std
3. Tradable via Options-Vol, nicht via Direction-Play

### 6.5 Feature 5: Topic-Cluster-Signal (über HDBSCAN)

**Hypothese:** Wenn mehrere Ticker gleichzeitig ähnliche News bekommen (z.B. "chip shortage"), ist das ein Sektor-Signal.

**Test-Protokoll:**
1. Embed headlines via `sentence-transformers/all-MiniLM-L6-v2`
2. HDBSCAN-Cluster pro Tag
3. Für jeden Cluster > 5 Headlines: Mean sentiment pro beteiligte Ticker
4. Event-Study: Tickers in "hot topic cluster" → statistisch signifikante Gruppen-CAR

**Schwierigkeit:** HDBSCAN-Parameter (`min_cluster_size`, `min_samples`) müssen out-of-sample stabil sein. Setze Pipeline auf rollierender 30-Tage-Basis.

### 6.6 Feature 6: Cross-Source-Corroboration-Boost

**Hypothese:** Wenn dieselbe Story von 3+ unabhängigen Quellen berichtet wird, stärkeres Signal.

**Test-Protokoll:**
1. Semantic-Dedup via `hnswlib` mit Cosine-Similarity > 0.85
2. Für jede "Story" (= Cluster von Duplikaten): Anzahl unique Quellen
3. Event-Study: Stories mit N=1 vs. N=3+ vs. N=5+ Quellen → steigende CAR-Magnitude?

**Das ist der wichtigste Corroboration-Test.** Aus der Literatur: 1-Quellen-News sind oft Spekulation, 3+ sind konfirmiert.

---

## 7. Das Gesamt-Gate für News-Features

Bevor ein News-Feature in den Composite-Score eingeht:

```python
def news_feature_production_ready(feature_name, validation_results):
    """Gate: alle drei Ebenen müssen bestanden sein."""
    criteria = {
        "level_a_fpb_macro_f1": 0.70,
        "level_a_own_gold_macro_f1": 0.60,
        "level_b_car_significance_p": 0.05,
        "level_b_car_magnitude_bps": 30,   # mindestens 30 bps Effekt
        "level_c_ic_mean": 0.02,
        "level_c_quantile_spread_bps": 15,
        "level_c_net_edge_after_costs_pct": 3.0,
        "look_ahead_anonymization_agreement": 0.85,
    }
    passed = {}
    for k, thresh in criteria.items():
        val = validation_results.get(k)
        if val is None:
            passed[k] = False
            continue
        if "p" in k and "significance" in k:
            passed[k] = val < thresh
        else:
            passed[k] = val >= thresh
    
    all_passed = all(passed.values())
    return all_passed, passed
```

**Production-Gate-Regel:** Wenn nicht alle 8 Kriterien erfüllt: Feature bleibt in **Shadow-Mode** (kein P&L-Einfluss, aber weiter beobachtet), nicht in Production.

---

## 8. Walk-Forward-Validation

Alle obigen Tests nur auf festem Train/Test-Split = anfällig für Period-Specific-Overfitting. **Walk-Forward** ist Pflicht.

### 8.1 Das Setup

```python
# scripts/news_validation/walk_forward.py
"""
Rolling-Window-Evaluation:
- Train-Window: 12 Monate
- Shift-Stride: 1 Monat
- Test-Window: 1 Monat (out-of-sample)
- Über 3-Jahres-Historie → 24 rolling windows
"""

import pandas as pd
from datetime import timedelta

def walk_forward_eval(news_df, prices_df, 
                       train_months=12, test_months=1,
                       start_date="2023-01-01", end_date="2026-03-01"):
    """Rolling evaluation."""
    results = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    while current + pd.DateOffset(months=train_months + test_months) <= end:
        train_start = current
        train_end = current + pd.DateOffset(months=train_months)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = train_end + pd.DateOffset(months=test_months)
        
        # Für jedes Feature pro Fenster
        train_news = news_df.loc[train_start:train_end]
        test_news = news_df.loc[test_start:test_end]
        
        # Evaluate: Event-Study + IC auf Test-Window
        test_metrics = evaluate_features(test_news, prices_df)
        test_metrics["train_start"] = train_start
        test_metrics["train_end"] = train_end
        test_metrics["test_start"] = test_start
        test_metrics["test_end"] = test_end
        
        results.append(test_metrics)
        current = current + pd.DateOffset(months=1)
    
    return pd.DataFrame(results)
```

### 8.2 Was du dir anschaust

- **Stability:** IC über die 24 Windows. Constant? Oder hat es 2024 funktioniert und 2025 nicht mehr?
- **Regime-Dependency:** IC in Bull-Markets (S&P > 20d-MA) vs. Bear-Markets? Historisch: News-Sentiment ist **stärker in Bear-Markets** (höhere Relevanz negativer News).
- **Seasonal:** IC in Earnings-Wochen vs. Non-Earnings? Erwartung: starkes IC nur in Earnings-Wochen.

**Entscheidungsregel:** Feature geht in Production nur wenn **20+ von 24 Windows** ein positives IC zeigen. Sonst: Signal ist fragil.

---

## 9. Benchmarks, an denen du dich messen solltest

Von Papers & Practitioners 2024/2025, für **News-only-basierte Strategien** auf US-Equity:

| Quelle | Zeitraum | Sharpe | Net-Alpha p.a. |
|---|---|---|---|
| Kargarzadeh 2024 (GPT-4 + Russell 2000) | 2022, 2023 | 3.64, 5.10 | starke Claims, teilweise Look-Ahead-verdächtig |
| Shobayo et al. 2024 (FinBERT + Logistic) | NGX Nigeria | — | ~15 % Accuracy-Advantage, kein P&L ausgewiesen |
| Kirtac & Germano 2024 (GPT + Pessimism) | S&P 500 2021-2024 | ~1.2 | 4-7 % p.a. post-cost |
| MDPI 2025 (weekly sentiment + NASDAQ mini) | 2022-2024 | verbessert vs. baseline | marginales Plus |
| Gómez-Martínez et al. 2025 (Sentiment + Futures) | 2022-2024 | verbessert | Strategie-abhängig |

**Realistische Erwartung:** Sharpe 0.5-1.2 nach Kosten, Net-Alpha 3-8 % p.a. Alles darüber ist Look-Ahead-verdächtig.

**Wenn dein System im Backtest Sharpe > 2 zeigt:** Anonymisiere Entity-Namen und teste erneut. Wenn Sharpe bleibt → evtl. echtes Signal. Wenn Sharpe einbricht → Look-Ahead.

---

## 10. Umsetzungs-Checkliste

**Phase 1 — Datensätze (Woche 1):**
- [ ] Financial PhraseBank (all 4 config) heruntergeladen → `tests/news_gold/`
- [ ] FiQA Task 1 Headlines
- [ ] Twitter Financial News (Zeroshot)
- [ ] SEntFiN
- [ ] **Dein eigenes 500-Headline-Gold-Set** gelabelt

**Phase 2 — Ebene A (Woche 2):**
- [ ] FinBERT-tone Baseline auf allen 5 Datasets
- [ ] Claude Haiku 4.5 zero-shot Baseline
- [ ] Comparison-Report mit Confusion-Matrices
- [ ] Entscheidung: Welcher Classifier wird Primary?

**Phase 3 — Ebene B (Woche 3-4):**
- [ ] Event-Study-Pipeline (3-12 Monate Historie, 1000+ Events)
- [ ] Market-Model auf S&P 500 tickers
- [ ] CAR-Distribution pro Label
- [ ] Anonymization-Test für Look-Ahead

**Phase 4 — Ebene C (Woche 5-6):**
- [ ] Alphalens-Integration
- [ ] IC/IR pro Feature
- [ ] Quantile-Spreads
- [ ] Turnover-Analyse
- [ ] Net-Edge-After-Costs

**Phase 5 — Walk-Forward (Woche 7-8):**
- [ ] 24 rolling windows über 3 Jahre
- [ ] Stability-Plot
- [ ] Regime-Decomposition (Bull/Bear, Earnings/Non-Earnings)

**Phase 6 — Production-Gate (Woche 9):**
- [ ] Pro Feature: 8-Kriterien-Check
- [ ] Feature-Catalog (siehe Gap-Analyse Neu #6) mit jedem News-Feature dokumentiert
- [ ] Shadow-Mode Aktivierung im Dispatcher (parallel zu aktueller Pipeline)
- [ ] 30-Tage-Shadow-Run
- [ ] Cutover-Entscheidung pro Feature

---

## 11. Quellen (die wichtigsten)

**Benchmarks und Datasets:**
- Malo et al. (2014): Financial PhraseBank. [Hugging Face](https://huggingface.co/datasets/takala/financial_phrasebank)
- Yang et al. (2020): FinBERT-Tone. [Hugging Face](https://huggingface.co/yiyanghkust/finbert-tone)
- Cortis et al. (2017): FiQA Task 1. [Website](https://sites.google.com/view/fiqa/home)
- Omarzadeh (2025): Business Insider + FPB. [IEEE DataPort](https://ieee-dataport.org/documents/business-insider-dataset-financial-phrasebank)

**LLM vs FinBERT:**
- Shen & Zhang (2024): Financial Sentiment Analysis on News and Reports Using LLMs and FinBERT. [ICPICS 2024](https://arxiv.org/pdf/2410.01987)
- Fatouros et al. (2024): Transforming sentiment analysis in financial domain with ChatGPT. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666827023000610)
- Kang & Choi (2025): GPT vs FinBERT Sector-Specific Comparison. [MDPI Electronics](https://www.mdpi.com/2079-9292/14/6/1090)
- ACM AIF 2025: Reasoning or Overthinking — CoT schadet bei Sentiment. [ACM DL](https://dl.acm.org/doi/10.1145/3768292.3770341)
- CLiC-it 2025: Target-Based Financial Sentiment LLM Benchmark. [ACL Anthology](https://aclanthology.org/2025.clicit-1.74.pdf)

**Event Study Methodology:**
- MacKinlay (1997): Event Studies in Economics and Finance. Journal of Economic Literature.
- eventstudy.de Blog: [Cumulative Abnormal Return](https://eventstudy.de/blog/cumulative-abnormal-return)
- Nature Scientific Reports 2021: [Sentiment correlation in financial news networks](https://www.nature.com/articles/s41598-021-82338-6)

**PEAD:**
- Ball & Brown (1968) — das Originalpaper
- Martineau (2021): Rest in Peace PEAD — warum PEAD schwächer wird. Critical Finance Review.
- Columbia PEAD Declined Paper 2024. [Columbia Business School](https://business.columbia.edu/sites/default/files-efs/imce-uploads/CEASA/Events%20Page/PEAD_Declined_over_time.pdf)
- Wikipedia: [PEAD](https://en.wikipedia.org/wiki/Post%E2%80%93earnings-announcement_drift)

**Alphalens / IC:**
- [alphalens-reloaded](https://github.com/stefan-jansen/alphalens-reloaded) — maintained Fork
- PyQuant News: [How to use the information coefficient](https://www.pyquantnews.com/the-pyquant-newsletter/information-coefficient-measure-your-alpha)
- PyQuant News 2026: [Real Factor Alpha](https://www.pyquantnews.com/free-python-resources/real-factor-alpha-how-to-measure-it-with-information-coefficient-and-alphalens-in-python) — realistische IC-Ranges

**Look-Ahead Bias:**
- Glasserman & Lin (2023): Assessing Look-Ahead Bias in LLM Stock Return Predictions. [Columbia DSI](https://datascience.columbia.edu/wp-content/uploads/2024/01/P021_FB_PosterSession_Fall2023.pdf)
- Quant Journey 2024: [Advanced Look-Ahead Bias Prevention](https://quantjourney.substack.com/p/advanced-look-ahead-bias-prevention)

**Backtest-Studien:**
- arxiv 2507.03350: Backtesting Sentiment Signals for Trading. Dow Jones 30, 28 Monate.
- MDPI JRFM 2025 18/412: News Sentiment and Stock Market Dynamics (1.86 Mio Headlines).

---

## 12. Ehrliche Einschätzung

**Die harte Wahrheit:** Von den 6 News-Features im Plan werden wahrscheinlich **2-4** den Production-Gate passieren. Mein Tipp:

- **Feature 1 (FinBERT-Sentiment)**: sehr wahrscheinlich passes, sobald Tri-class statt continuous
- **Feature 2 (News-Volume-Spike)**: wahrscheinlich passes, wenn sektor-normalisiert
- **Feature 3 (Source-Quality)**: wahrscheinlich passes, aber niedriger IC
- **Feature 4 (Uncertainty-Score)**: unklar, braucht eigenes Validation-Set
- **Feature 5 (Topic-Cluster)**: technisch anspruchsvoll, oft fragil
- **Feature 6 (Cross-Source-Corroboration)**: konzeptionell stark, aber rare Events → statistische Power niedrig

**Das bedeutet nicht "News-Features sind nutzlos".** Es bedeutet: Du brauchst Validation-Disziplin, um die Spreu vom Weizen zu trennen. Ohne dieses Playbook baust du einen Composite-Score, in dem möglicherweise 4 von 6 Dimensionen Rauschen beitragen — das verwässert die echten Signale.

**Die dreizehn Wochen Aufwand sind nicht optional.** Wenn du sie abkürzt, baust du auf ungeprüftem Fundament. Jede spätere Performance-Debuggin-Session ohne diese Basisdaten ist Fehlersuche im Dunkeln.

**Die wichtigsten drei Sachen, die du nicht auslassen darfst:**
1. **Eigenes Gold-Dataset** — FPB ist pre-2020, deine Live-Daten sind post-2025
2. **Entity-Anonymization-Test** — wenn LLM von Firmen-Prior profitiert, live ist es tot
3. **Net-Edge-After-Costs** — Gross-IC ist akademisch, Net-Edge ist real
