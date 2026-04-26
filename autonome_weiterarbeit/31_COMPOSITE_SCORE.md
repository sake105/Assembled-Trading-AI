# 31 — Composite Score (9 Dimensionen)

**Zweck:** Der Composite-Score ist das Herz des Systems. Hier sind die 9 Signal-Dimensionen mit Formeln, Regime-Gewichtung und Implementierungs-Pattern.

**Ausgabe:** Ein Score in `[-1, +1]` pro Ticker pro Zeitpunkt.

---

## Die 9 Dimensionen

| # | Dimension | Primäres Feature | Regime-Relevanz |
|---|---|---|---|
| 1 | Multi-Timeframe-Alignment | Top-Down-Bias (D→15m→5m) | Trending hoch |
| 2 | Klassische TA | RSI+MACD+BB mit Regime-Params | Alle |
| 3 | Microstructure | Amihud, OFI (bei Tick), Kyle-Lambda | Intraday |
| 4 | Volume-/Market-Profile | POC, VAH/VAL, AVWAP-Dev | Ranging |
| 5 | Chart-Pattern-ML | Matrix-Profile, DTW, HS-Detection | Alle |
| 6 | Volatility-Surface | IV-Rank, Skew, VIX-Term, VRP | High-Vol |
| 7 | Breadth/Intermarket | McClellan, Risk-On/Off, Correlation | Crisis hoch |
| 8 | Seasonality | Overnight-Gap, ORB, Turn-of-Month | Konstant |
| 9 | News | sentiment_vw + novelty + surprise + velocity | Crisis sehr hoch |

---

## Dimension 1: Multi-Timeframe-Alignment

**Konzept:** Top-Down-Ansatz — langfristige Trend-Richtung dominiert, kurzfristige Indikatoren liefern Entry-Signale.

**Timeframes:**
- Daily: Trend-Richtung (Bull/Bear/Sideways)
- 15-min: Setup-Identifikation
- 5-min: Entry-Timing

**Features:**
```python
def mtf_alignment_score(bars_daily, bars_15m, bars_5m):
    # Daily: SMA-50 vs SMA-200 für Regime, ADX für Stärke
    d_trend = np.sign(bars_daily.sma_50.iloc[-1] - bars_daily.sma_200.iloc[-1])
    d_adx = bars_daily.adx.iloc[-1]
    d_strength = min(d_adx / 25.0, 2.0)  # cap at 2
    
    # 15m: MACD-Histogram-Sign
    mtf_macd = np.sign(bars_15m.macd_hist.iloc[-1])
    
    # 5m: RSI-Position
    rsi_5m = bars_5m.rsi_14.iloc[-1]
    entry_bias = (rsi_5m - 50) / 50  # [-1, +1]
    
    # Alignment-Score
    aligned_signs = [d_trend, mtf_macd, np.sign(entry_bias)]
    n_aligned = sum(s > 0 for s in aligned_signs) - sum(s < 0 for s in aligned_signs)
    # n_aligned in [-3, +3]
    
    score = (n_aligned / 3) * d_strength  # weight by daily trend strength
    return max(-1.0, min(1.0, score))
```

**Regime-Sensitivity:**
- Trending (ADX > 25): Gewicht 1.5× erhöhen
- Ranging (ADX < 20): Gewicht 0.5× reduzieren

---

## Dimension 2: Klassische TA mit Regime-Parametern

**Kern:** RSI + MACD + Bollinger-Bands, aber **Parameter wechseln mit Regime** — das ist der Unterschied zu jedem Retail-Indikator-Aggregat.

```python
TA_PARAMS_BY_REGIME = {
    'bull_trend': {
        'rsi_overbought': 75,  # erlaube Overbought länger
        'rsi_oversold': 35,
        'bb_std': 2.0,
        'macd_signal_length': 9,
    },
    'bear_trend': {
        'rsi_overbought': 65,
        'rsi_oversold': 25,
        'bb_std': 2.0,
        'macd_signal_length': 9,
    },
    'ranging': {
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'bb_std': 1.5,  # enger
        'macd_signal_length': 9,
    },
    'high_vol': {
        'rsi_overbought': 80,
        'rsi_oversold': 20,  # Extrem-Thresholds
        'bb_std': 2.5,  # weiter
        'macd_signal_length': 13,
    }
}

def classical_ta_score(bars, regime):
    params = TA_PARAMS_BY_REGIME[regime]
    rsi = bars.rsi_14.iloc[-1]
    macd_hist = bars.macd_hist.iloc[-1]
    bb_percent = bars.bb_percent.iloc[-1]  # [0, 1]
    
    # RSI-Component
    rsi_score = 0
    if rsi > params['rsi_overbought']:
        rsi_score = -1 * (rsi - params['rsi_overbought']) / (100 - params['rsi_overbought'])
    elif rsi < params['rsi_oversold']:
        rsi_score = (params['rsi_oversold'] - rsi) / params['rsi_oversold']
    
    # MACD-Component
    macd_score = np.sign(macd_hist) * min(abs(macd_hist) * 10, 1.0)
    
    # BB-Position
    bb_score = -1 * (bb_percent - 0.5) * 2  # Mean-Reversion in Ranging
    if regime in ['bull_trend', 'bear_trend']:
        bb_score = 0  # BB-Mean-Reversion nicht in Trends
    
    return (rsi_score + macd_score + bb_score) / 3
```

---

## Dimension 3: Microstructure

**Warnung:** Mit Alpaca IEX (2% Marktvolumen) sind Microstructure-Features stark verzerrt. Für EOD-System OK, für ernsthafte Intraday-Microstructure → Paid (Polygon, Databento).

**Features (auch mit IEX approximativ):**

```python
def microstructure_score(bars):
    # Amihud Illiquidity (nur bei ADV < 1M anwendbar)
    amihud = abs(bars.returns) / (bars.dollar_volume + 1e-6)
    amihud_z = zscore(amihud, lookback=20)
    illiquidity_penalty = -min(amihud_z.iloc[-1], 2.0)  # capped
    
    # Order Flow Imbalance Proxy (aus 1-min-Bars)
    up_vol = bars[bars.close > bars.open].volume.sum()
    down_vol = bars[bars.close < bars.open].volume.sum()
    ofi_proxy = (up_vol - down_vol) / (up_vol + down_vol + 1e-6)
    
    # Volume-Price-Correlation
    vp_corr = bars.volume.rolling(20).corr(bars.close).iloc[-1]
    
    return (0.3 * illiquidity_penalty + 0.5 * ofi_proxy + 0.2 * vp_corr)
```

**Regime-Sensitivity:** Nur in Intraday-Regimes aktiv. Bei EOD-Trading auf 0 setzen.

---

## Dimension 4: Volume-/Market-Profile

**Konzept:** Point-of-Control (POC), Value-Area-High/Low (VAH/VAL), Anchored-VWAP.

```python
def volume_profile_score(bars, profile_window=20):
    # Rolling Volume-Profile
    price_bins = np.linspace(bars.low.min(), bars.high.max(), 50)
    profile = np.zeros(49)
    for i in range(len(bars)):
        bar = bars.iloc[i]
        bin_idx = np.searchsorted(price_bins, (bar.high + bar.low) / 2)
        if 0 <= bin_idx < 49:
            profile[bin_idx] += bar.volume
    
    poc_idx = np.argmax(profile)
    poc_price = (price_bins[poc_idx] + price_bins[poc_idx+1]) / 2
    
    current_price = bars.close.iloc[-1]
    dist_from_poc_pct = (current_price - poc_price) / poc_price
    
    # Mean-Reversion-Score
    # Zu weit weg von POC → Mean-Reversion
    mr_score = -np.tanh(dist_from_poc_pct * 10)  # bis ±1
    
    # AVWAP-Deviation von Session-Open
    vwap_session = (bars.close * bars.volume).cumsum() / bars.volume.cumsum()
    avwap_dev = (current_price - vwap_session.iloc[-1]) / vwap_session.iloc[-1]
    avwap_score = -np.tanh(avwap_dev * 10)
    
    return 0.6 * mr_score + 0.4 * avwap_score
```

**Regime-Sensitivity:**
- Ranging: voll (Mean-Reversion)
- Trending: halbiert (kann überzogen werden)

---

## Dimension 5: Chart-Pattern-ML

**Library:** `stumpy` (Matrix-Profile), `dtaidistance` (DTW).

```python
import stumpy

def chart_pattern_score(bars, window=30):
    # Matrix-Profile zu historischen Winning-Patterns
    close_series = bars.close.values[-200:]  # letzte 200 Bars
    mp = stumpy.stump(close_series, m=window)
    
    # Niedrigster Distance = ähnlichstes Pattern
    min_distance = mp[:, 0].min()
    similarity = 1 / (1 + min_distance)  # [0, 1]
    
    # Labeling: Winning-Patterns sind vorher manuell markiert
    # Für MVP: Proxy via Head-and-Shoulders-Detection
    hs_score = detect_head_and_shoulders(close_series)
    
    # Doppel-Boden/-Top
    db_score = detect_double_bottom_or_top(close_series)
    
    return np.clip(0.5 * hs_score + 0.5 * db_score, -1, 1)
```

**Regime-Sensitivity:** Alle Regimes, konstant.

---

## Dimension 6: Volatility-Surface

**Features:** IV-Rank, Skew (25-Delta Put/Call), VIX-Term-Structure, Variance Risk Premium (VRP).

```python
def vol_surface_score(ticker, options_chain_snapshot, vix_term):
    # IV-Rank (wenn Options verfügbar)
    iv_history = get_iv_30d_history(ticker, days=252)
    iv_current = options_chain_snapshot.atm_iv_30d
    iv_rank = (iv_current - iv_history.min()) / (iv_history.max() - iv_history.min() + 1e-6)
    
    # Niedrig IV-Rank = Long-Opportunity (Vol wird steigen)
    iv_rank_score = 1 - 2 * iv_rank  # [-1, +1]
    
    # Skew: 25-Delta-Put vs 25-Delta-Call IV
    put_25d_iv = options_chain_snapshot.put_25d_iv
    call_25d_iv = options_chain_snapshot.call_25d_iv
    skew = put_25d_iv - call_25d_iv
    # Hoher Skew = Panik = Contrarian-Long
    skew_score = np.tanh(-skew * 10)
    
    # VIX-Term: VIX_9D vs VIX_30D (Contango = Risk-On, Backwardation = Risk-Off)
    vix_term_score = -np.sign(vix_term.vix_9d - vix_term.vix_30d)
    # Mild Risk-Off-Signal
    
    # Variance-Risk-Premium (IV - Realized-Vol)
    realized_vol = compute_realized_vol(ticker, days=30)
    vrp = iv_current - realized_vol
    vrp_score = np.tanh(-vrp * 5)  # hohe VRP = Short-Vol-Opportunity
    
    return 0.4*iv_rank_score + 0.2*skew_score + 0.2*vix_term_score + 0.2*vrp_score
```

**Regime-Sensitivity:** High-Vol-Regime: stark erhöht.

---

## Dimension 7: Breadth/Intermarket

**Features:** McClellan-Oscillator, Advance-Decline-Line, Risk-On/Off-Ratios, Intermarket-Correlation.

```python
def breadth_intermarket_score(spy, xlk, xly, xlp, xlu, tlt, hyg, dxy):
    # McClellan-Oscillator (19-day EMA of AD − 39-day EMA)
    ad_line = compute_ad_line()
    ema_19 = ad_line.ewm(span=19).mean()
    ema_39 = ad_line.ewm(span=39).mean()
    mcclellan = (ema_19 - ema_39).iloc[-1]
    mc_score = np.tanh(mcclellan / 50)
    
    # Risk-On-Ratios (XLY/XLP für Zyklisch vs Defensiv)
    risk_on_ratio = (xly.close / xlp.close).iloc[-1] / (xly.close / xlp.close).iloc[-60]
    risk_on_score = np.tanh((risk_on_ratio - 1) * 5)
    
    # HYG/TLT (Credit-Stress)
    hyg_tlt = (hyg.close / tlt.close).iloc[-1] / (hyg.close / tlt.close).iloc[-60]
    credit_score = np.tanh((hyg_tlt - 1) * 5)
    
    # Dollar-Trend (DXY)
    dxy_change = (dxy.close.iloc[-1] / dxy.close.iloc[-20] - 1)
    # Starker Dollar = Equity-Headwind
    dxy_score = -np.tanh(dxy_change * 10)
    
    return 0.3*mc_score + 0.3*risk_on_score + 0.2*credit_score + 0.2*dxy_score
```

**Regime-Sensitivity:** Crisis-Regime: stark erhöht.

---

## Dimension 8: Seasonality

**Features:** Overnight-Gap, Opening-Range-Breakout (ORB), Turn-of-Month.

```python
def seasonality_score(bars, date):
    # Overnight-Gap: vorheriges Close → heutiges Open
    gap = (bars.open.iloc[-1] / bars.close.iloc[-2]) - 1
    gap_z = (gap - bars.overnight_gap_30d_mean) / bars.overnight_gap_30d_std
    # Gap-Fade-Hypothese: extreme Gaps werden gefüllt
    gap_score = -np.tanh(gap_z * 2)
    
    # Turn-of-Month-Effekt: letzte 2 Tage und erste 2 Tage = bullish bias
    day_of_month = date.day
    days_in_month = pd.Timestamp(date).days_in_month
    is_turn = day_of_month <= 2 or day_of_month >= (days_in_month - 1)
    tom_score = 0.3 if is_turn else 0
    
    # Monday-Effect (historisch schwach, aber sign-konsistent)
    day_of_week = date.weekday()
    monday_score = -0.1 if day_of_week == 0 else 0
    
    # Friday-Effect (leicht bullish historisch)
    friday_score = 0.1 if day_of_week == 4 else 0
    
    return gap_score + tom_score + monday_score + friday_score
```

**Regime-Sensitivity:** Konstant über Regimes.

---

## Dimension 9: News

Siehe `30_NEWS_TA_FUSION.md` Schicht 1.

```python
def news_score(ticker, news_features):
    weights = {
        'sentiment_vw': 0.30,
        'novelty': 0.15,
        'surprise': 0.20,
        'event_volume_z': 0.10,
        'velocity': 0.15,
        'dispersion': -0.10
    }
    raw = sum(weights[k] * news_features[k] for k in weights)
    return max(-3.0, min(3.0, raw)) / 3  # normalize to [-1, 1]
```

---

## Regime-Gewichtungs-Matrix

Die Gewichtung aller 9 Dimensionen hängt vom Regime ab:

```python
COMPOSITE_WEIGHTS_BY_REGIME = {
    'calm': {
        'mtf': 0.15, 'classical_ta': 0.20, 'microstructure': 0.05,
        'volume_profile': 0.15, 'chart_pattern': 0.10,
        'vol_surface': 0.05, 'breadth': 0.10, 'seasonality': 0.15,
        'news': 0.05
    },
    'normal': {
        'mtf': 0.15, 'classical_ta': 0.15, 'microstructure': 0.05,
        'volume_profile': 0.15, 'chart_pattern': 0.10,
        'vol_surface': 0.10, 'breadth': 0.10, 'seasonality': 0.10,
        'news': 0.10
    },
    'elevated': {
        'mtf': 0.15, 'classical_ta': 0.10, 'microstructure': 0.05,
        'volume_profile': 0.10, 'chart_pattern': 0.10,
        'vol_surface': 0.15, 'breadth': 0.15, 'seasonality': 0.05,
        'news': 0.15
    },
    'crisis': {
        'mtf': 0.10, 'classical_ta': 0.05, 'microstructure': 0.05,
        'volume_profile': 0.05, 'chart_pattern': 0.10,
        'vol_surface': 0.20, 'breadth': 0.20, 'seasonality': 0.05,
        'news': 0.20
    }
}
```

**Konstruktion des Composite-Scores:**
```python
def composite_score(ticker, all_features, regime):
    weights = COMPOSITE_WEIGHTS_BY_REGIME[regime]
    scores = {
        'mtf': mtf_alignment_score(all_features['bars_mtf']),
        'classical_ta': classical_ta_score(all_features['bars'], regime),
        'microstructure': microstructure_score(all_features['bars']),
        'volume_profile': volume_profile_score(all_features['bars']),
        'chart_pattern': chart_pattern_score(all_features['bars']),
        'vol_surface': vol_surface_score(ticker, all_features['options'], all_features['vix_term']),
        'breadth': breadth_intermarket_score(all_features['macro']),
        'seasonality': seasonality_score(all_features['bars'], date),
        'news': news_score(ticker, all_features['news']),
    }
    
    composite = sum(weights[k] * scores[k] for k in weights)
    return np.clip(composite, -1, 1), scores
```

---

## Umsetzungs-Checkliste

- [ ] Alle 9 Dimension-Funktionen implementiert
- [ ] Regime-Classifier (HMM oder Rule-Based) produktiv
- [ ] Regime-Gewichtungs-Matrix konfiguriert
- [ ] Composite-Score-Funktion getestet
- [ ] Pro-Dimension-Rolling-IC tracken
- [ ] Dashboard mit Waterfall-Plot (9 Sub-Scores → Composite)

---

## Ehrliche Einschätzung

**Die 9-dimensionale Struktur ist keine Hype-Architektur.** Sie löst ein konkretes Problem: keine einzelne Informationsquelle liefert konsistenten Edge über Regime hinweg.

**Was jede Dimension gewinnt:**
- MTF + Klassische TA = Baseline (60% vom Edge)
- Microstructure = Intraday-Details
- Volume-Profile = Mean-Reversion-Levels
- Chart-Pattern = Technische Figuren (Phase 3 ML-Training)
- Vol-Surface = Optionsmarkt-Information
- Breadth = Marktbreite (wichtig in Crisis)
- Seasonality = Kalender-Effekte
- News = Narrative-Shift-Detection

**In Phase 1 reichen 4-5 Dimensionen.** Nicht alle gleichzeitig implementieren.

**Prioritäts-Reihenfolge für MVP:**
1. Klassische TA (schnell, verifiziert)
2. MTF-Alignment (einfach, stabil)
3. News (9. Dim, Kern-Innovation)
4. Breadth (Regime-Filter-Basis)
5. Volume-Profile

Rest (Microstructure, Vol-Surface, Chart-Pattern-ML, Seasonality) in Phase 2-3 dazu.
