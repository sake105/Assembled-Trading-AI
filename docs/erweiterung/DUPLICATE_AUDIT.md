# DUPLICATE_AUDIT — Erweiterung vs. Mainline

Audit-Stand: 2026-05-10. Dieser Bericht ist autoritativ — bei zukünftigen Erweiterungen referenzieren.

## Methode

Es wurde geprüft, welche Module in `src/erweiterung/` semantisch oder funktional bereits in `src/assembled_core/` existieren. Quelle: Filesystem-Scan + grep-basierter API-Vergleich.

## Drei Kategorien

| Kategorie | Aktion | Anzahl |
|-----------|--------|--------|
| **A** — Echtes Duplikat, mainline ist klar überlegen | Datei gelöscht | 12 |
| **B** — Behalten als Forschungs-Variante (NumPy-only, andere API) | Doku-Hinweis ergänzt | 8 |
| **C** — Kein Mainline-Pendant — echtes Add-On | Behalten | ~80 Module |

## A — Gelöschte Duplikate

| Erweiterungs-Pfad (gelöscht) | Mainline-Ersatz | Mainline-LoC |
|------------------------------|-----------------|--------------|
| `altdata/sec_edgar.py` | `src/assembled_core/data/sources/edgar_source.py` | 155 |
| `altdata/finra_short_interest.py` | `src/assembled_core/data/sources/finra_source.py` | 147 |
| `altdata/wikipedia_pageviews.py` | `src/assembled_core/data/sources/wikipedia_views_source.py` | 186 |
| `nlp/finbert_sentiment.py` | `src/assembled_core/intel/finbert_sentiment.py` | 366 |
| `nlp/news_dedup.py` | `src/assembled_core/intel/news_dedupe.py` | 391 |
| `live/alpaca_paper_client.py` | `src/assembled_core/execution/broker_adapter.py` | 878 |
| `execution/almgren_chriss.py` | `src/assembled_core/execution/almgren_chriss.py` | 361 |
| `events/event_study.py` | `src/assembled_core/qa/event_study.py` | 446 |
| `signals/earnings_drift_v2.py` | `src/assembled_core/signals/pead_sue.py` | 119 |
| `signals/statistical_arbitrage.py` | `src/assembled_core/signals/pairs_trading.py` | (existiert) |
| `live/__init__.py`, `events/__init__.py` | (Subpackages entfernt) | — |

Mit dem Cleanup wurden 12 Dateien (~1300 LoC Duplikat-Code) entfernt, davon 2 leere `__init__.py`.

## B — Behaltene Forschungs-Varianten (mit Doku-Hinweis)

Diese Module **haben** ein mainline-Pendant, wurden aber bewusst behalten weil sie
**NumPy-only** sind (keine externen Libs wie scipy/skfolio/mlfinpy/arch nötig)
oder substantiell andere APIs anbieten. Jede Datei hat einen `DUPLIKAT-HINWEIS`-
Doku-Block, der auf den mainline-Pfad verweist.

| Erweiterungs-Modul | Mainline-Pendant | Begründung Behaltung |
|--------------------|------------------|----------------------|
| `portfolio/hierarchical_risk_parity.py` | `assembled_core/portfolio/hierarchical_risk_parity.py` (313 LoC, scipy) | NumPy-only |
| `portfolio/black_litterman.py` | `assembled_core/portfolio/black_litterman.py` (412 LoC) | Kompakter, Demo-tauglich |
| `portfolio/kelly_sizing.py` | `assembled_core/portfolio/kelly_uncertainty.py` (138 LoC) | Multi-Asset + Confidence-Discount |
| `backtest/cpcv.py` | `assembled_core/qa/cpcv_validation.py` (236 LoC, skfolio) | NumPy-only |
| `backtest/deflated_sharpe.py` | `assembled_core/qa/deflated_sharpe.py` (222 LoC) | Demo-Variante |
| `ml/triple_barrier.py` | `assembled_core/features/triple_barrier.py` (364 LoC, mlfinpy) | NumPy-only |
| `volatility/garch_models.py` | `assembled_core/risk/garch_vol.py` (187 LoC, arch) | Allgemeiner (GARCH/EGARCH/GJR), NumPy-Fallback |
| `risk/correlation_breakdown.py` | `assembled_core/risk/correlation_guard.py` (317 LoC) | APC+Eigenvalue-Composite-Score |
| `signals/cross_sectional_residuals.py` | `assembled_core/features/residual_momentum.py` (171 LoC) | Long-format-API + Reversal/Vol-Variante |

## C — Echte Add-Ons (kein Mainline-Pendant)

Folgende Subpackages enthalten Funktionalität, die im Mainline **nicht** existiert
und durch die Erweiterung ergänzt wird:

### Datenquellen (5 frei, kein Mainline-Pendant)
- `altdata/google_trends.py` — pytrends-Wrapper für SVI-Signale
- `altdata/cftc_cot.py` — CFTC Commitments of Traders (Disaggregated)
- `altdata/fred_md.py` — McCracken/Ng FRED-MD-Macro-Panel + PCA-Faktoren
- `altdata/yahoo_options.py` — Yahoo-Options-Chains (Skew/PC-Ratio/Term-Structure)
- `altdata/gdelt_extended.py` — GDELT 2.0 GKG (Geo/Theme-Aggregat)
- `altdata/reddit_pushshift.py` — Reddit Mentions via arctic-shift
- `altdata/coingecko_crypto.py` — Crypto-Risk-On/Off-Indikator
- `altdata/worldbank_macro.py` — World Bank Open-Data
- `economic_data/bls.py`, `economic_data/ecb_oecd_imf.py` — BLS/ECB/IMF-APIs

### Signale (4)
- `signals/options_implied.py` — IV-Skew, VRP, Garman-Klass-Vol
- `signals/attention.py` — Composite Wikipedia + Trends + Reddit
- `signals/lead_lag_network.py` — Granger-Causality-Netzwerk
- `signals/macro_nowcast.py` — Sahm-Rule + composite Recession-Score

### ML (2 echte + 1 Variante)
- `ml/conformal_prediction.py` — Split-Conformal + ACI (Gibbs/Candès 2021)
- `ml/stacking_ensemble.py` — Out-of-fold Stacking
- `ml/triple_barrier.py` — siehe B

### Portfolio (2)
- `portfolio/cvar_optimizer.py` — Rockafellar/Uryasev CVaR-LP
- `portfolio/risk_parity.py` — ERC (Newton)

### Risk (2)
- `risk/tail_risk_evt.py` — Generalized-Pareto-Distribution-Fit (Method-of-Moments)
- `risk/dynamic_drawdown_control.py` — CPPI-Floor + Vol-Targeting

### Backtest (3 echte + 2 Varianten)
- `backtest/white_reality_check.py` — White (2000) + Hansen-SPA-Test
- `backtest/performance_metrics.py` — Sharpe/Sortino/Calmar/Tail-Ratio
- `backtest/walk_forward.py` — strikter rollierender Walk-Forward

### Microstructure (2)
- `microstructure/liquidity_proxies.py` — Amihud, Roll, Corwin-Schultz, Kyle-Lambda
- `microstructure/vpin.py` — Easley/Lopez/O'Hara 2012

### Factors (4)
- `factors/fama_french.py` — SMB/HML/RMW/CMA/MOM-Konstruktion
- `factors/low_vol.py` — Betting-Against-Beta
- `factors/factor_ic.py` — IC + Alpha-Decay
- `factors/factor_neutralize.py` — Sektor-/Multi-Faktor-Neutralisierung

### Volatility (1 echte + 1 Variante)
- `volatility/har_rv.py` — Heterogeneous AR Realized Volatility (Corsi 2009)
- `volatility/garch_models.py` — siehe B

### Time-Series-Tools (4)
- `timeseries_tools/hurst_dfa.py` — Hurst (R/S) + DFA + Variance-Ratio
- `timeseries_tools/fractional_diff.py` — Lopez de Prado FFD
- `timeseries_tools/entropy.py` — Sample-/Approximate-/Shannon-Entropy
- `timeseries_tools/change_points.py` — CUSUM-Filter + Binary-Segmentation

### DL (3 + 2 advanced)
- `dl/patch_tst.py` — PatchTST (Nie 2023)
- `dl/lstm_returns.py` — LSTM-Univariate-Forecasting
- `dl/autoencoder_anomaly.py` — Cross-Section-Anomaly-Detection
- `dl_advanced/nbeats.py` — N-BEATS (Oreshkin 2020)
- `dl_advanced/reservoir_computing.py` — Echo-State-Network (NumPy-only)

### RL (2)
- `rl/portfolio_env.py` — Gym-style mit Differential-Sharpe-Reward
- `rl/ppo_agent.py` — PPO-Actor-Critic

### Bayesian, Cross-Asset, Survival, Discovery (alle 1+ Module)
- `bayesian/bayesian_linear.py` — NIG-Conjugate-LinReg + Sharpe-Posterior
- `crossasset/spreads.py` — GSR/Dollar-Equity/HYG-LQD/VIX-Term
- `survival/hazard_models.py` — Kaplan-Meier + Cox-PH (NumPy-only)
- `discovery/genetic_programming.py` — GP-Tree-Search

### Risk-Metrics (1 mit 8 Funktionen)
- `risk_metrics/advanced_metrics.py` — Omega/Treynor/Burke/Pain/Ulcer/Stutzer/M²/Upside-Potential

### Stress-Test (2)
- `stress_test/monte_carlo.py` — Bootstrap/Block/Stationary/Normal-Path-Simulation
- `stress_test/historical_replay.py` — 10 Standard-Crisis-Windows

### Strategien & Pipelines & Reports (3)
- `strategies/templates.py` — Trend/Low-Vol/Vol-Premium/Regime-Switching
- `pipelines/research_pipeline.py` — End-to-End-Forschungs-Pipeline
- `report/html_report.py` — HTML-Backtest-Report-Generator

### Meta (2)
- `meta/strategy_orchestrator.py` — Equal-Weight/Inverse-Vol/Hedge-Algo/Regime-Aware
- `meta/regime_router.py` — Vol/Trend/Composite-Regime

### Classical-ML (1)
- `classical_ml/boost_wrappers.py` — XGB/LGBM/CatBoost/RF + Optuna-Tuning

## Auswirkung des Cleanups

- **Vor Cleanup:** 106 Module, 153 Tests, 12 echte Duplikate.
- **Nach Cleanup:** 94 Module (-12), 136 Tests (-17), 0 unmarkierte Duplikate.
- **Behaltene Variants:** 9 Module mit klarem `DUPLIKAT-HINWEIS`-Doku-Block.
- **Echte Add-Ons:** ~80 Module, voll dokumentiert.

## Reproduktion des Audits

Liste aller mainline-Pfade die mit erweiterung-Modulen überlappen:

```bash
# Gibt mainline-Module mit erweiterung-Pendant
grep -l "Hierarchical Risk Parity\|HRP\|Black-Litterman\|deflated.sharpe\|triple.barrier\|CPCV\|GARCH\|VPIN\|Conformal" src/assembled_core/**/*.py
```

## Maintainer-Empfehlung

1. **Production-Pipelines:** ausschließlich mainline-Versionen importieren.
2. **Forschungs-Pipelines:** beide Welten OK; Doku-Hinweise lesen.
3. **Bei API-Änderungen** in mainline: prüfen ob erweiterung-Variante drift hat.
4. **Bei neuem Modul** in `assembled_core/`: prüfen ob es ein erweiterung-Pendant
   gibt und ggf. konsolidieren.
