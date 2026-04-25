# 13 — Free Trading-Module (0 EUR/Monat)

**Zweck:** Konkrete Module, die du ins System einbaust, um echten Alpha zu generieren. Alle mit Free-Daten und Free-Libraries umsetzbar. Jedes Modul mit Priorität, Aufwand, und erwartetem Edge.

---

## Module in diesem Dokument — Priorisiert nach Signal-Edge-nach-Kosten

| # | Modul | Phase | Aufwand | Literatur-Edge |
|---|---|---|---|---|
| 13.1 | **Liquidity-Condition-Index** | **P1 sofort** | **L** | Regime-Gate, ~30 LOC |
| 13.2 | **Regime-Switching HMM** | **P1** | M | Sharpe +0.2–0.5 |
| 13.3 | **GARCH Vol-Targeting** | **P1** | M | +0.1–0.3 Sharpe, −30-50% MaxDD |
| 13.4 | **Insider Form-4 Signal** | **P1** | M | 82 bps/Monat (Cohen/Malloy/Pomorski 2012) |
| 13.5 | **Analyst Revisions Signal** | **P1** | L | IC 0.02–0.05 |
| 13.6 | **PEAD (SUE)** | **P1** | M | IC 0.02–0.04 auf 20d bei Mid-Caps |
| 13.7 | **Residual Momentum FF5** | **P1** | M | 2× Sharpe vs Total-Return-Mom |
| 13.8 | Macro-Regime 4-Quadrant | P2 | M | Categorical Feature |
| 13.9 | Recession-Probability | P2 | M | Binary Risk-Off-Signal |
| 13.10 | Sentiment-Panel | P2 | M | Fear&Greed-Replikation |
| 13.11 | Short-Interest FINRA | P2 | L | Short-Squeeze-Indicator |
| 13.12 | Buyback-Drift | P2 | M | 3-6% abnormal 12-24 Mo |
| 13.13 | ETF-Flow self-computed | P2 | L | Sector-Rotation-Signal |
| 13.14 | Wikipedia Page Views | P2 | L | Attention Sharpe ~0.3 |
| 13.15 | Cross-Asset-Carry | P3 | M | Macro-Overlay |
| 13.16 | Tail-Risk-Hedge | P3 | M | Systematischer OTM-Put |

---

## 13.1 Liquidity-Condition-Index (LCI) — der Quick-Win

**Warum:** Bestes Cost/Benefit unter allen Spezialmodulen. **~30 LOC**, sofort live, Regime-Gate für alle anderen Signale.

**Konstruktion (Z-Score-Aggregat):**

```python
def compute_lci(lookback_days=252):
    # Rolling Z-Scores, je positiver desto riskier
    hyg_lqd_ratio = fred.get_series("BAMLH0A0HYM2") / fred.get_series("BAMLC0A0CM")
    dxy_z = zscore(fred.get_series("DTWEXBGS"), lookback_days)
    vix_z = zscore(fred.get_series("VIXCLS"), lookback_days)
    curve_z = zscore(fred.get_series("T10Y2Y"), lookback_days) * -1  # Inversion → Risk
    
    lci = (
        0.3 * zscore(hyg_lqd_ratio, lookback_days) +
        0.2 * dxy_z +
        0.3 * vix_z +
        0.2 * curve_z
    )
    return lci
```

**Interpretation:**
- `LCI < -1`: Risk-On, Momentum-Signale aktivieren
- `-1 ≤ LCI ≤ +1`: Normal
- `LCI > +1`: Risk-Off, Defensive-Modus
- `LCI > +2`: Crisis, nur Long-Vol und Cash

**Datenquellen:** alle FRED, siehe `10_FREE_DATEN.md` §10.2.

**Integration:** Multiplier auf alle Composite-Scores.

---

## 13.2 Regime-Switching HMM

**Warum:** Momentum-Crash-Vermeidung. Sharpe +0.2–0.5 historisch (Ang/Bekaert 2002, Chen et al. 2017).

**Library:** `hmmlearn==0.3.3` (siehe `11_FREE_MODELLE.md` §11.9).

**Input-Features:** log-Returns + 20d-Realized-Vol + VIX + Term-Slope.

**Pattern:**
```python
from hmmlearn.hmm import GaussianHMM
import numpy as np

def fit_regime_hmm(returns, vol_20d, vix, term_slope, n_states=3):
    X = np.column_stack([returns, vol_20d, vix, term_slope])
    X = (X - X.mean(0)) / X.std(0)  # standardize
    
    hmm = GaussianHMM(n_components=n_states,
                       covariance_type="full",
                       n_iter=200,
                       random_state=42)
    hmm.fit(X)
    states = hmm.predict(X)
    return hmm, states
```

**Labels zuweisen (manuell nach State-Statistiken):**
- State mit höchstem mean-return + niedrigster Vol → `bull_trend`
- State mit niedrigem mean-return + hoher Vol → `bear_hv`
- Rest → `ranging` oder `transition`

**Retraining:** **wöchentlich**, Walk-Forward alle 63 Bars. **Label-Flipping** prüfen (HMM liebt State-Swaps, Mapping-Table pflegen).

**Regel:** Nicht mehr als 3-4 Regime — Overfitting-Risiko steigt exponentiell.

**Integration:** Regime-State ist der Dispatcher für die 9-dimensionale Gewichtungsmatrix im Composite-Score.

---

## 13.3 GARCH Vol-Targeting

**Warum:** Drawdown-Kontrolle. Vol-Target-Portfolios haben historisch 30-50% niedrigere Max-DD bei gleichem Return (Harvey/Hoyle/Korgaonkar 2018).

**Library:** `arch==8.0.0` (siehe `11_FREE_MODELLE.md` §11.8).

**Vol-Target-Formel:**
```python
def size_vol_target(asset_vol_forecast, target_vol=0.15, max_leverage=1.5):
    size = min(target_vol / asset_vol_forecast, max_leverage)
    return size
```

**Vol-Forecast via GJR-GARCH:**
```python
from arch import arch_model

def forecast_vol(returns, horizon=5):
    model = arch_model(returns * 100, vol='GARCH', p=1, o=1, q=1, dist='skewt')
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=horizon)
    sigma = np.sqrt(forecast.variance.iloc[-1].mean()) / 100
    # annualisieren
    sigma_annual = sigma * np.sqrt(252)
    return sigma_annual
```

**Pattern:** Target-Portfolio-Vol von 15% annualisiert. Jede Position wird entsprechend skaliert: `position_size = target_vol / asset_vol_forecast`.

**Integration:** Multiplier auf Composite-Score → Position-Size.

---

## 13.4 Insider Form-4 Signal

**Warum:** 82 bps/Monat Alpha bei "opportunistic insiders" (Cohen/Malloy/Pomorski 2012, JF 67(3)).

**Datenquelle:** SEC EDGAR via `edgartools` (siehe `10_FREE_DATEN.md` §10.1).

**Feature-Engineering:**
```python
# Cluster-Buy-Score
def cluster_buy_score(ticker, lookback_days=30):
    filings = get_form4_filings(ticker, days=lookback_days)
    purchases = [f for f in filings if f.transaction_code == 'P']
    unique_insiders = len(set(f.reporting_person for f in purchases))
    return unique_insiders

# Cluster-Buy: ≥3 distinct Insider mit Code=P in 30d = bullish
# Net-Officer-USD: CEO/CFO/COO Purchases minus Sales in 90d
def net_officer_usd(ticker, lookback_days=90):
    filings = get_form4_filings(ticker, days=lookback_days)
    officers = [f for f in filings if f.is_officer]
    net = sum(f.value if f.transaction_code == 'P' else -f.value
              for f in officers)
    return net
```

**Filter:**
- Nur `P` (Purchase) und `S` (Sell) — **nicht** `A` (Award), `M` (Option-Exercise).
- **10b5-1-Plan-Dispositions filtern** (Footnote-Markierung) — nicht-informativ, vorab geplant.

**Sentiment:**
- `cluster_buy >= 3` UND `net_officer_usd > 250_000`: **Bullish** (sehr starkes Signal)
- `cluster_buy >= 2`: Weak-Bullish
- Cluster-Sell analog bearish

**Integration:** Als eigenes BaseSignal oder als Feature im News-Signal-Composite.

---

## 13.5 Analyst Revisions Signal

**Warum:** Blitz/Hanauer/Honarvar 2023, IC 0.02–0.05. Billig, effektiv.

**Datenquelle:** Finnhub Free `/stock/recommendation` (siehe `10_FREE_DATEN.md` §10.9).

**Feature:**
```python
def analyst_revision_score(ticker, finnhub_client):
    recs = finnhub_client.recommendation_trends(ticker)
    if len(recs) < 2:
        return 0.0
    current = recs[0]
    prior = recs[1]
    
    # Bullish Score: (Buy + StrongBuy) − (Sell + StrongSell)
    def score(r):
        return (r['buy'] + 2*r['strongBuy']) - (r['sell'] + 2*r['strongSell'])
    
    delta = score(current) - score(prior)
    total = score(current) + score(prior) + 1e-6
    return delta / abs(total)  # normalized revision
```

**Integration:** Analyst-Revision-Score als 8-Feature-Composite.

---

## 13.6 PEAD — Post-Earnings-Announcement-Drift

**Warum:** Bernard-Thomas 1989, rezent 5-8% annualisiert. Decile-Long-Short auf SUE.

**Datenquelle:** Finnhub `/stock/earnings` für EPS-Actual + Estimate.

**SUE-Berechnung:**
```python
def compute_sue(ticker, finnhub_client, lookback_quarters=8):
    earnings = finnhub_client.company_earnings(ticker, limit=lookback_quarters)
    # Earnings Surprise: actual vs consensus
    surprises = [(e['actual'] - e['estimate']) for e in earnings]
    # SUE = latest surprise / σ(historical surprises)
    latest = surprises[0]
    historical_std = np.std(surprises[1:])
    return latest / (historical_std + 1e-6)
```

**Drift-Window:** 25-30% des Drifts in 3-Tages-Fenstern um folgende Quartals-Earnings.

**Pre-Trade-Hook:** `if days_to_earnings(ticker) <= 2: reduce_size() or block_short`.

**Integration:** PEAD-Signal als kurzfristiger Momentum-Booster nach Earnings-Release.

---

## 13.7 Residual Momentum FF5

**Warum:** Blitz et al. 2011, ~2× Sharpe vs. Total-Return-Momentum.

**Libraries:** `pandas-datareader.famafrench` + `statsmodels.OLS`.

**Setup:**
```python
from pandas_datareader import famafrench as ff
import statsmodels.api as sm

# Ken French FF5 + Momentum Factors (free)
ff5_data = ff.FamaFrenchReader('F-F_Research_Data_5_Factors_2x3_daily').read()[0]
mom_data = ff.FamaFrenchReader('F-F_Momentum_Factor_daily').read()[0]
factors = pd.concat([ff5_data, mom_data], axis=1)

# Residual Momentum pro Ticker
def residual_momentum(ticker_returns, factors, window=252):
    results = []
    for i in range(window, len(ticker_returns)):
        # Rolling Regression
        y = ticker_returns.iloc[i-window:i]
        X = factors.iloc[i-window:i]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        residuals = model.resid
        # 12m Momentum der Residuen, skipping last month (11-1-Momentum)
        mom_11_1 = residuals.iloc[-252:-21].mean()
        results.append(mom_11_1)
    return pd.Series(results, index=ticker_returns.index[window:])
```

**Integration:** Residual-Momentum ersetzt Total-Return-Momentum in Cross-Sectional-Rankings.

---

## 13.8 Macro-Regime 4-Quadrant

**Warum:** Framework nach Hedgefundie/Dalio. Kategorisches Feature für ML.

**Input:** FRED + yfinance.

**Quadranten:**
- Growth↑ Inflation↑ → Commodities, Emerging, Value
- Growth↑ Inflation↓ → Growth, Large-Cap-Tech
- Growth↓ Inflation↑ → Gold, Defensive-Value
- Growth↓ Inflation↓ → Treasuries, Cash

**Messung:**
- Growth: `growth_z = zscore(ISM_PMI - 50) + zscore(NFP_3m_change)`
- Inflation: `infl_z = zscore(CPI_YoY) + zscore(5y5y_forward_breakeven)`

**Integration:** als 1-of-4-Kategorie-Feature in alle ML-Modelle.

---

## 13.9 Recession-Probability

**Warum:** Hamilton 2022 MarkovRegression auf T10Y3M + NFCI. Binary Risk-Off-Timing.

**Library:** `statsmodels.tsa.regime_switching.MarkovRegression`.

**Setup:**
```python
from statsmodels.tsa.regime_switching import MarkovRegression

def recession_prob(t10y3m, nfci):
    X = np.column_stack([t10y3m, nfci])
    model = MarkovRegression(
        t10y3m,  # T10Y3M als Endogen
        k_regimes=2,
        trend='c',
        switching_variance=True,
        exog_tvtp=nfci  # zeitvariable Transition
    )
    res = model.fit()
    recession_probability = res.smoothed_marginal_probabilities[:, 1]
    return recession_probability
```

**Integration:** Wenn `recession_prob > 0.5` → alle Long-Signale mit 0.5-Multiplier skalieren.

---

## 13.10 Sentiment-Panel (Fear&Greed-Replikation)

**Warum:** Contrarian-Overlay. Wenn Retail panisch → accumulate.

**Komponenten (alle free):**
- CBOE Put/Call-Ratio (siehe 10.4)
- HYG-LQD-Spread (FRED)
- VIX (FRED)
- 127d-SPY-Momentum (yfinance)
- AAII wöchentlich (API umständlich, manuell einmal/Woche)
- UMich Consumer Sentiment (FRED: `UMCSENT`)

**Aggregation:**
```python
def sentiment_panel_score():
    components = {
        "put_call": zscore(cboe_put_call),
        "hyg_lqd": zscore(hyg_lqd_spread),
        "vix": zscore(vix),
        "spy_momentum": zscore(spy_127d_return),
        "aaii": zscore(aaii_bull_bear_spread),
        "umich": zscore(umich_sentiment),
    }
    # Fear-Score (0-100, 100=extreme fear)
    return 50 + 10 * sum(components.values()) / len(components)
```

**Integration:** Contrarian-Multiplier. Wenn Score > 80 (Extreme Fear) → Long-Bias. Wenn < 20 (Extreme Greed) → Short-Bias oder Cash.

---

## 13.11 Short-Interest FINRA

**Datenquelle:** FINRA-API (siehe `10_FREE_DATEN.md` §10.3).

**Features:**
```python
def short_interest_features(ticker):
    si = finra.get_short_interest(ticker)  # bi-monthly
    days_to_cover = si['short_interest'] / si['avg_daily_volume']
    short_ratio_change = si['short_interest_pct'].pct_change()
    return {
        'days_to_cover': days_to_cover,
        'si_change_pct': short_ratio_change,
        'si_pct_float': si['short_interest'] / si['float'],
    }
```

**Signale:**
- `days_to_cover > 5`: Short-Squeeze-Potenzial bei positivem Catalyst
- `si_change_pct > 20%` in einer Periode: Aggressive Short-Build-up (Bearish-Fundamentals)

**Integration:** Feature im Composite, besonders für Meta-Labeling-Gate.

---

## 13.12 Buyback-Drift

**Warum:** Peyer/Vermaelen 2009, 3-6% abnormal Returns 12-24 Monate.

**Datenquelle:** SEC EDGAR 8-K-Parser via `edgartools`.

**Detection:**
```python
def detect_buyback_announcement(ticker, days=30):
    filings = get_8k_filings(ticker, days=days)
    for f in filings:
        if f.item_code == '8.01' and 'repurchase' in f.text.lower():
            # extract authorized amount via regex
            amount = extract_buyback_amount(f.text)
            return {
                'announcement_date': f.filing_date,
                'amount_usd': amount,
                'pct_market_cap': amount / market_cap(ticker),
            }
    return None
```

**Signal:**
- `pct_market_cap > 5%`: Starkes Bullish-Signal für 12-24 Monate
- Kombination mit Insider-Buying: sehr starkes Signal

---

## 13.13 ETF-Flow Self-Computed

**Warum:** ETF-Flows sind Proxy für Institutional-Sector-Rotation. Kein API nötig.

**Berechnung:**
```python
def etf_flow(etf_ticker, lookback=5):
    # Shares Outstanding Δ × NAV
    shares_out_series = yf.Ticker(etf_ticker).history(period=f"{lookback}d")['sharesOutstanding']
    price_series = yf.Ticker(etf_ticker).history(period=f"{lookback}d")['Close']
    
    delta_shares = shares_out_series.diff()
    flow_usd = delta_shares * price_series
    return flow_usd.sum()
```

**Integration:** Sector-Rotation-Dashboard. XLK-Flow positiv + XLE negativ → Tech-Over-Energy-Rotation.

---

## 13.14 Wikipedia Page Views

**Warum:** Moat et al. 2013, long-short Sharpe ~0.3.

**Library:** `mwviews==0.3` (siehe `10_FREE_DATEN.md` §10.14).

**Pattern:**
```python
from mwviews.api import PageviewsClient

def wikipedia_attention(company_names: list, days=30):
    client = PageviewsClient(user_agent="TradingBot/1.0")
    views = client.article_views('en.wikipedia', 
                                  company_names,
                                  granularity='daily',
                                  start=start_date,
                                  end=end_date)
    return views
```

**Feature:** `views_z = zscore(views_7d_mean / views_90d_mean)` — 7-Tages-Views relativ zu 90-Tage-Baseline.

**Integration:** Attention-Feature, besonders wertvoll für Small/Mid-Caps ohne News-Coverage.

---

## 13.15 Cross-Asset-Carry

**Warum:** Klassische CTA-Faktoren. Breite Faktor-Diversifikation.

**Proxy (via ETFs):**
- Equity-Carry: SPY - T-Bill-Rate (Dividend-Yield)
- Bond-Carry: TLT-Yield - SHY-Yield (Term-Premium)
- FX-Carry: UUP (USD) vs FXE (EUR) vs FXY (JPY)
- Commodity-Carry: Roll-Yield-Proxy via USO / UNG Front-Spread

**Integration:** Macro-Overlay-Feature.

---

## 13.16 Tail-Risk-Hedge (SPY OTM-Puts)

**Warum:** Systematischer OTM-Put-Buying-Sleeve (Universa-Style). 2-5% OTM, 30-45 DTE.

**Implementierung (erst mit Alpaca-Options-Live):**
```python
def tail_hedge_rules():
    return {
        'allocation_pct': 0.02,  # 2% of portfolio
        'strike_otm_pct': 0.05,  # 5% OTM
        'dte_target': 35,  # 30-45 days
        'roll_trigger': {
            'dte_remaining': 15,  # roll when <15d
            'delta_threshold': -0.05,  # or when delta > -0.05
        }
    }
```

**Integration:** Erst wenn Alpaca Options live und IV-Rank-Feature stabil ist.

---

## Umsetzungs-Checkliste Phase 1

- [ ] 13.1 Liquidity-Condition-Index produktiv (30 LOC, sofort)
- [ ] 13.2 Regime-Switching HMM trainiert + wöchentliches Retraining
- [ ] 13.3 GJR-GARCH Vol-Forecast pro Ticker
- [ ] 13.4 Form-4 Cluster-Buy-Score in Pipeline
- [ ] 13.5 Analyst-Revisions via Finnhub
- [ ] 13.6 PEAD/SUE für Earnings-Event-Trading
- [ ] 13.7 Residual-Momentum FF5 als Momentum-Feature

## Umsetzungs-Checkliste Phase 2

- [ ] 13.8 Macro-4-Quadrant als Kategorie-Feature
- [ ] 13.9 Recession-Probability MarkovRegression
- [ ] 13.10 Sentiment-Panel Fear&Greed-Replikation
- [ ] 13.11 FINRA Short-Interest + Days-to-Cover
- [ ] 13.12 Buyback-8-K-Parser
- [ ] 13.13 ETF-Flow per Sektor
- [ ] 13.14 Wikipedia-Attention für Top-100-Ticker

## Umsetzungs-Checkliste Phase 3

- [ ] 13.15 Cross-Asset-Carry-Overlay
- [ ] 13.16 Tail-Risk-Hedge via Alpaca Options (wenn Live-Options aktiv)

---

## Was NICHT in diesem Katalog ist

**Explizit nicht empfohlen für Solo-Quant-System:**

- **Klassisches Pairs-Trading:** Sharpe 1.5→0.8 post-2010, brauchbar nur bei >200 Namen. Für 60 Ticker nicht.
- **Merger-Arb / Spin-Off:** Deal-Spreads brauchen Legal-Analyse, negative Skew.
- **Options Delta-Neutral / Vol-Arb:** Vol-Surface-Modellierung (Heston, SABR) ist eigene Karriere. Überspringen.
- **Satellite-Data für Retail-Equity:** TB-Storage + Vision-Models unerreichbar.
- **Credit-Card-Panel-Daten:** unerreichbar.
- **Google-Trends-Signale:** pytrends archiviert April 2025, gefälschte Daten.
- **Options-Flow als Signal (free):** yfinance-Chain-Scraping ToS-grau, Unusual-Whales 48-95 USD/Monat außer Budget.

---

## Erwartungsmanagement

**Literatur-Sharpe-Werte sind In-Sample und pre-Kosten.** Live-Sharpe typischerweise 30-50% niedriger.

**Bei Alpaca Paper IEX Free, überlebensfähige Signale:**
- ✅ Residual-Momentum
- ✅ Regime-Switching-Gewichtung
- ✅ Insider-Cluster-Buying
- ✅ PEAD bei Mid-Caps
- ✅ Liquidity-Condition-Index

**Grenzwertig:**
- ⚠ Analyst-Revisions
- ⚠ Buyback-Drift
- ⚠ Wikipedia-Attention

**Nicht survival-fähig oder unerreichbar:**
- ❌ Options-Flow free
- ❌ Satellite-Alpha
- ❌ Klassisches Statarb bei 60 Namen

**Also: erst die Phase-1-Module (13.1-13.7) stabil und CPCV-validiert, bevor du Phase 2 angehst.**
