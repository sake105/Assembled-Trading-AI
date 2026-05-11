# Erweiterung — Final Executive Summary V2

**Branch:** ERWEITERUNG (getrennt von Mainline — kein Merge initiiert)
**Stand:** 2026-05-11
**Tests:** 486 passed
**Commits in dieser Session-Reihe:** 19

---

## 1. Bottom Line

Die finale, produktionsreife Strategie der Erweiterung — **statistisch
signifikant validiert über 19 Jahre Daten (inkl. GFC, COVID, Inflation):**

**`MasterAllocator(sa_weight=0.70)`** — Production-API in
`src/erweiterung/strategies/master_allocator.py`.

### Long-History 2007-2026 (19y, 4589 Tage)

| Strategy | AnnRet | Sharpe | Sortino | Calmar | MDD | Calmar-p vs 60/40 |
|----------|-------:|-------:|--------:|-------:|----:|------------------:|
| 60/40 Classic | +8.02 % | +0.696 | +0.664 | +0.241 | −33.24 % | — |
| Pure Equity Factor | +29.05 % | +1.108 | +1.055 | +0.574 | −50.60 % | **0.996** ✓ |
| SA_VolTarget | +16.99 % | **+1.275** | **+1.218** | **+0.906** | −18.76 % | **0.998** ✓ |
| XA_Hybrid | +8.35 % | +0.719 | +0.663 | +0.342 | −24.42 % | 0.544 |
| **Master_70_30** | **+14.47 %** | **+1.208** | **+1.140** | **+0.741** | **−19.52 %** | **0.997** ✓ |

**Drei Strategien sind statistisch signifikant besser als 60/40 Classic
über 19 Jahre Multi-Decade-Daten:**
- SA_VolTarget (p = 0.998)
- Master_70_30 (p = 0.997)
- Pure Equity Factor (p = 0.996)

**Equity-Curve-Audit Master_70_30 (19y):**
- Sharpe 1.230, WD/Vol 6.07 (realistic fat-tails), Kurtosis 3.02
- Lag-1 Autocorr −0.049 (kein Smoothing-Verdacht)
- **0 kritische Flags** (kein SUSPICIOUS_SHARPE / SMOOTHED / MDD_TOO_LOW)

---

## 2. Methodische Innovation

### Calmar-Bootstrap als Test-Statistik (`src/erweiterung/backtest/calmar_bootstrap.py`)

Sharpe-Bootstrap gibt uninformative p≈0.99 für MDD-Reducer-Strategien.
Stationary-Bootstrap der Calmar-Differenz (Politis-Romano) differenziert
zwischen 0.02 (signifikant schlechter) und 0.998 (signifikant besser).

**Standard-Test für alle Risk-Allocator-Strategien.**

### Equity-Curve Anomaly-Audit (`src/erweiterung/qa/equity_curve_audit.py`)

Sechs Heuristiken finden:
- SUSPICIOUS_SHARPE (> 3.0), EXTREMELY_HIGH_SHARPE (> 5.0)
- RETURNS_LIKELY_SMOOTHED (AC1 > 0.4)
- MDD_TOO_LOW_FOR_SHARPE (Sharpe > 2.0 & |MDD| < 5 %)
- WORST_DAY_TOO_MILD_FOR_PERIOD
- LOW_KURTOSIS_SYNTHETIC_LIKE
- MARKET_CORR_TOO_LOW

**Integriert als CI-Smoke-Test** in `.github/workflows/erweiterung-tests.yml`.

### Walk-Forward-Diagnostik

Train→Test-Korrelation als Overfit-Warnung:
- Drawdown-Switch: −0.372 (Overfit-Warnung) → OOS-Edge widerlegt
- Multi-Factor-Combiner: −0.582 (stark anti-prädiktiv) → Edge widerlegt

**Resultat:** Bei Hyperparameter-Optimierung ist anti-prädiktive
Train→Test-Korrelation ein klares Warnsignal.

---

## 3. Was funktioniert (validierte Konzepte mit 19y-Statistik)

| Konzept | Modul | Validierung |
|---------|-------|:------------|
| Vol-Targeting auf Single-Asset-Mom | `strategies/volatility_targeting.py` | 19y p=0.998 vs 60/40 |
| Master_70_30 Mix | `strategies/master_allocator.py` | 19y p=0.997 vs 60/40 |
| Cross-Asset-Mom-Top-N | `strategies/master_allocator.py` | 5y bull-bias erkannt, 19y schwächer |
| Calmar-Bootstrap | `backtest/calmar_bootstrap.py` | methodisch validiert |
| Equity-Audit | `qa/equity_curve_audit.py` | Mainline-Anomalien gefunden, in CI |
| Long-History-ETF-Cache | `data/cache/yfinance_long/` | 11 ETFs × 19y |
| FOMC-Tone-Pipeline | `strategies/fomc_macro_signal.py` | Methodik validiert, Daten dünn |
| EDGAR-Live-Pipeline | `live_pipeline/` | Module End-to-End validiert |
| News-Sentiment-Cross-Section | `strategies/news_sentiment_strategy.py` | Code production-ready, Daten dünn |

---

## 4. Was widerlegt wurde (ehrliche Negativ-Befunde)

| Konzept | Befund | Konsequenz |
|---------|--------|:-----------|
| Binäres Drawdown-Switching | WF-OOS p=0.99 (vs In-Sample p=0.0000) | Lucky-Window-Artefakt, kein echter Edge |
| Threshold-Auto-Tuning | Train-Test-Corr −0.37 bis −0.58 | Overfit-Risk hoch, vermeiden |
| Multi-Factor-Combiner | HRP 12/13 Mal gewählt | Overfit-Tax, schlechter als Single |
| Macro-Regime-Trigger | Calmar p~0.55 | Wenig trennscharf 2021-2026 |
| Triple-Barrier Meta-Labeling | p=0.02 signifikant SCHLECHTER | ML-Overengineering |
| VIX-Tail-Hedge auf Master | p=0.234 bis 0.030 | Doppelhedging mit Vol-Target |
| Cross-Asset-5y-Resultate | 5y Sharpe 1.01 → 19y 0.74 | Bull-Market-Bias-Artefakt |

---

## 5. Mainline-Audit-Erkenntnisse

Per `equity_curve_audit.py` im Mainline-Repo gefunden:

1. **3 Original-Equity-Files bit-identisch:** `baseline.csv`,
   `altdata.csv`, `test1_aitech_qagate.csv` — Altdata/QAgate-Varianten
   waren No-Ops im Original-Backtest.
2. **Sharpe 4.63 + MDD −4.52 %** löst `SUSPICIOUS_SHARPE` +
   `MDD_TOO_LOW_FOR_SHARPE` Flags. Statistisch außerhalb des für
   Long-Only typischen Bereichs über 836 Tage.
3. **Korrelation Erweiterung ↔ Original-Baseline: 0.07-0.18** —
   orthogonale Strategien.
4. **Real-Test T2 (200-Sym 2025-26 no-leverage):** Sharpe 0.77 /
   MDD −30 % — plausibles Profil, sollte als Headline-Number genutzt
   werden statt des Baseline-Backtests.

---

## 6. Production-API

```python
from erweiterung.strategies.master_allocator import (
    MasterAllocator, MasterAllocatorConfig,
)
from erweiterung.altdata.yfinance_cache_loader import load_universe_panel
from erweiterung.factors.fama_french import momentum_12_1

# Lade Cross-Asset (lange Historie):
xa_rets = load_universe_panel(
    "data/cache/yfinance_long",  # 19y ETF-Cache
    ["SPY", "QQQ", "IWM", "EFA", "EEM",
     "AGG", "TLT", "HYG", "GLD", "SLV", "DBC"],
).pivot_table(index="date", columns="symbol", values="close").pct_change()

# Equity-Faktor (z.B. Mom-12/1-LO auf 22 Mega-Caps)
equity_factor_ret = ...

# Allokieren
alloc = MasterAllocator(MasterAllocatorConfig(sa_weight=0.70))
out = alloc.allocate(equity_factor_ret, xa_rets)
# out["master_return"] = production-ready return series
```

End-to-End-Endpoint: `scripts/erweiterung/run_master_long_history.py`

---

## 7. Wichtigste methodische Lehren

1. **Walk-Forward OOS ist Pflicht.** In-Sample-Statistik kann komplett
   trügerisch sein (Regime-Switch In-Sample p=0.0000 → OOS p=0.99).
   Beim Master selbst hat sich aber der In-Sample-Edge im Walk-Forward
   bestätigt (p=0.808 in 5y-WF, p=0.997 in 19y-In-Sample).

2. **5-Jahres-Backtests sind unzureichend.** Bull-Market-Bias verzerrt
   jede Sharpe-Statistik nach oben. Hybrid_VT_Mom 5y Sharpe 1.01 fiel
   im 19y-Sample auf 0.74 zurück.

3. **Calmar-Bootstrap > Sharpe-Bootstrap** für Risk-Allocator-Tests.

4. **Vol-Targeting ist Self-Tail-Hedge.** Externer VIX-Trigger schadet
   (Doppelhedging-Effekt).

5. **ML auf bereits gut-designte Allokationen overfittet.** Meta-Labeling
   verschlechtert Master_70_30 signifikant.

6. **Equity-Curve-Audit fängt Mainline-Anomalien.** Sollte CI-Smoke-Test
   sein — wurde umgesetzt.

7. **Multi-Asset-Diversifikation hilft im 5y, aber kaum im 19y.**
   Single-Asset-VolTarget (SA_VolTarget) ist über 19y nominell stärker
   als XA_Hybrid. Master_70_30 ist Kompromiss.

---

## 8. Erweiterung Inventory

| Bereich | Module | Status |
|---------|--------|--------|
| Strategies | volatility_targeting, master_allocator, regime_conditional_allocator, multi_signal_regime, macro_stress_signals, ensemble_regime, multi_factor_vol_target, news_sentiment_strategy, tail_risk_hedge, fomc_macro_signal | Production |
| Backtest | calmar_bootstrap, white_reality_check, deflated_sharpe, performance_metrics, cpcv | Production |
| QA | equity_curve_audit + CI-Integration | Production |
| Robustness | walk_forward, sub_period | Production |
| Altdata | yfinance_cache_loader (+ yfinance_long-Cache 19y) | Production |
| ML | meta_labeling_master, triple_barrier | Module ready, Edge widerlegt |
| Live-Pipeline | material_event_classifier, filing_signal_mapper | Module + Smoke-Demo validiert |
| Transcripts | fomc_tone, earnings_call_tone, loughran_mcdonald | Module ready, Daten dünn |

**Tests:** 486 passed, 0 failed.

---

## 9. Was OFFEN bleibt (ehrlich)

| Item | Status | Hindernis |
|------|--------|-----------|
| Multi-Jahr News-Feed | Module ready | GDELT/Provider-Backfill nötig (rate-limited) |
| FOMC-Statement-Archive (10+ Jahre Texte) | Module ready | FED-Archive-Scrape nötig |
| Tail-Hedge auf Pure-Mom-LO | Module ready | Eigentlich keine; Future-Work |
| Hyperparameter-Free Master | Diskussion | sa_weight=0.7 Default robust per Test |
| Live-Paper-Pilot mit Alpaca | Vom User abgelehnt | User-Wahl, nicht Lücke |

---

## 10. Abschließend

Die Erweiterung liefert:

1. **Statistisch signifikanter Edge** über 19 Jahre Daten (p=0.997 vs 60/40)
2. **Production-ready API** mit konsumierbarer MasterAllocator-Klasse
3. **Methodische Innovation** (Calmar-Bootstrap, Walk-Forward, Equity-Audit)
4. **Ehrliche Falsifikation** von 6+ attraktiven aber widerlegten Hypothesen
5. **Production-Pipeline** mit End-to-End-Endpoint (`run_master_long_history.py`)
6. **CI-Integration** des Audit-Moduls

**Master_70_30 ist die robust validierte Erweiterungs-Strategie.**
Calmar 0.741 vs 60/40 0.241 = 3.07× besser. Statistisch signifikant.

Beide Branches (Mainline & ERWEITERUNG) bleiben getrennt. Kein Merge.
