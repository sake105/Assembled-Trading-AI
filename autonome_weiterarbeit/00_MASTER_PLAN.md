# 00 — Master-Plan: Gesamtarchitektur

**Dieses Dokument ist dein Startpunkt.** Es erklärt, was das System am Ende sein soll, wie es aufgebaut ist, und in welcher Reihenfolge du es baust.

---

## Das Zielsystem in einem Absatz

Ein Python-FastAPI-Backend, das pro Ticker einen **Composite-Score von 0–100** aus neun Signal-Dimensionen berechnet (Multi-Timeframe-TA + klassische Indikatoren + Microstructure + Volume-Profile + Chart-Pattern + Volatility-Surface + Breadth/Intermarket + Seasonality + News), regime-abhängig gewichtet wird, durch ein Meta-Labeling-Gate gefiltert und mit Conformal-Prediction-Intervallen die Positionsgröße bestimmt. Das System läuft auf Alpaca Paper Trading, deckt US-Core-Universum + Europa + ETFs ab, validiert jede Strategie mit CPCV und Deflated Sharpe Ratio, kostet zwischen 0 und 65 EUR pro Monat.

---

## Die vier Architektur-Layer

```
┌─────────────────────────────────────────────────────┐
│ Layer 4: Entscheidung & Execution                   │
│   Meta-Labeling Gate → Size → Orders → Alpaca       │
├─────────────────────────────────────────────────────┤
│ Layer 3: Composite-Scoring                          │
│   9 Signal-Dimensionen → Regime-Gewichtung → Score  │
├─────────────────────────────────────────────────────┤
│ Layer 2: Feature-Engineering                        │
│   TA + Microstructure + News-NLP + Macro + Events   │
├─────────────────────────────────────────────────────┤
│ Layer 1: Daten-Ingest                               │
│   Alpaca + EODHD + SEC + FRED + GDELT + RSS         │
└─────────────────────────────────────────────────────┘
```

Jeder Layer hat eigene Dateien im Plan:

- Layer 1 → `10_FREE_DATEN.md` + `20_PAID_DATEN.md` + `14_FREE_UNIVERSUM.md` + `23_PAID_UNIVERSUM.md`
- Layer 2 → `11_FREE_MODELLE.md` + `21_PAID_MODELLE.md` + Teile von `13_FREE_MODULE.md`
- Layer 3 → `30_NEWS_TA_FUSION.md` + `31_COMPOSITE_SCORE.md`
- Layer 4 → `32_VALIDIERUNG.md`

---

## Die drei Entscheidungen, die du treffen musst

### Entscheidung 1: Komplett free oder Paid-Minimum?

| Option | Kosten | Du bekommst | Du verlierst |
|---|---|---|---|
| **Free-Only** | 0 EUR | Funktionierenden MVP mit US-Core + 60d News-History | Delisted-Survivorship-Schutz, EU-Ticker, sauberes LLM-Reasoning für Headlines |
| **Paid-Minimum** | ~22 EUR | EODHD All-World EOD, Hetzner CX22 | — |
| **Paid-Optimal** | ~65 EUR | + Claude Haiku LLM, + Finnhub Premium | — |

**Empfehlung:** 3 Monate komplett free laufen lassen. Wenn die Pipeline stabil ist und du echte Backtests fährst, auf Paid-Minimum upgraden. EODHD ist der einzige Paid-Posten mit klarem ROI (Survivorship-Bias-Schutz kann Sharpe um 0.1–0.3 verzerren).

### Entscheidung 2: Welche Reihenfolge beim Bauen?

```
Monat 1-3:   News-Pipeline + 9. BaseSignal + Regime-HMM + Liquidity-Index
Monat 4-6:   Meta-Labeling-Gate + Tier-2-Universum + Sektor-Modelle
Monat 7-9:   2D-Decision-Matrix + Conformal Prediction + Canary-Deployment
Monat 10-12: FinBERT-Fine-Tune + Tier-3 event-driven + optional Experiments
```

Volldetail in `40_ROADMAP.md`.

### Entscheidung 3: Wo hosten?

| Phase | Wo | Wann wechseln |
|---|---|---|
| MVP (Monat 1-3) | Windows lokal | Wenn FastAPI 24/7 laufen muss |
| Cloud-Start | Hetzner CX22 (4 EUR) oder Oracle Always-Free | Sobald Live-Paper mehr als 4 h/Tag läuft |
| Skalierung | Hetzner CX32 (9 EUR) | Wenn Tier-2 voll aktiv ist |

---

## Die wichtigsten Regeln

Diese Regeln ziehen sich durch alle Module. Hältst du dich daran, sparst du dir die klassischen Retail-Quant-Fallen.

1. **Nie ohne Purged-CV.** `sklearn.KFold` bei Zeitreihen = Leakage = Fake-Sharpe. Immer `skfolio.CombinatorialPurgedCV`.
2. **Nie Meta-Labeling ohne vertical barrier ≤ purged_size.** Sonst Label-Overlap in Train und Test.
3. **Nie einzelne Signale trauen.** Minimum drei unabhängige Signale als Composite, sonst Daten-Snooping.
4. **Nie yfinance in Produktion (Richtung B).** ToS verbietet kommerzielle Weitergabe. Für private Forschung ok.
5. **Nie Regime-Parameter statisch.** ADX > 25 aktiviert Momentum-Signale, ADX < 20 aktiviert Mean-Reversion.
6. **Nie Backtests ohne Survivorship-Bias-Schutz.** 0.1-0.3 Sharpe Overstatement typisch.
7. **Nie Signale live, die nicht ≥60 Tage im Shadow-Modus gelaufen sind.**
8. **Immer Secrets in SOPS+age oder `.env`-gitignored.** Niemals in Code.

---

## Die neun Signal-Dimensionen im Composite-Score

Volldetail in `31_COMPOSITE_SCORE.md`.

| # | Dimension | Primäres Feature | Regime-Relevanz |
|---|---|---|---|
| 1 | Multi-Timeframe-Alignment | Top-Down-Bias (D→15m→5m) | Trending hoch |
| 2 | Klassische TA | RSI+MACD+BB mit Regime-Params | Alle |
| 3 | Microstructure | Amihud, OFI (bei Tick), Kyle-Lambda | Intraday |
| 4 | Volume-/Market-Profile | POC, VAH/VAL, AVWAP-Dev | Ranging |
| 5 | Chart-Pattern-ML | Matrix-Profile, DTW, HS-Detection | Alle |
| 6 | Volatility-Surface | IV-Rank, Skew, VIX-Term, VRP | High-Vol hoch |
| 7 | Breadth/Intermarket | McClellan, Risk-On/Off, Correlation | Crisis hoch |
| 8 | Seasonality | Overnight-Gap, ORB, Turn-of-Month | Konstant |
| 9 | News | sentiment_vw + novelty + surprise + velocity | Crisis sehr hoch |

Die Gewichte pro Dimension sind regime-abhängig, nicht statisch. Ein HMM auf VIX + Term-Slope + HY-Spread klassifiziert Calm/Normal/Elevated/Crisis und aktiviert die passende Gewichts-Matrix.

---

## Was du am Ende hast

Nach 12 Monaten sauberer Umsetzung:

- **~1 800 Ticker aktiv** (S&P 500 + EuroStoxx 50 + STOXX 600 + S&P 400/600 + 80 ETFs + ADRs)
- **9 Signal-Dimensionen** im Composite-Score, alle regime-kalibriert und shadow-validiert
- **Meta-Labeling-Gate** mit LightGBM + Isotonic-Calibration + CPCV-Validation
- **Conformal-Prediction-Intervalle** als Size-Discount
- **News-Pipeline** mit GDELT-Deep + SEC-8-K real-time + FinBERT-Tone + optional Haiku-LLM-Zweitrunde
- **Validierungs-Stack** mit Deflated-Sharpe, PBO, Walk-Forward, Canary-Deployment
- **Budget:** 0-65 EUR/Monat

Wenn alle neun Dimensionen validiert und live sind, ist das System bereit für Phase 2 (News-Signal-SaaS an B2B-Kunden) oder für seriöses Eigenkapital-Deployment (Richtung A).

---

## Wo du jetzt anfängst

Lies in dieser Reihenfolge:

1. **`40_ROADMAP.md`** — der 12-Monats-Plan mit Phasen
2. **`30_NEWS_TA_FUSION.md`** — die Kern-Innovation
3. **`13_FREE_MODULE.md`** — die ersten konkreten Module
4. **`14_FREE_UNIVERSUM.md`** — die Ticker-Liste für deinen MVP
5. **`99_STACK_LOCKFILE.md`** — die Libraries zum Installieren

Der Rest ist Nachschlagewerk, je nach dem, woran du gerade arbeitest.
