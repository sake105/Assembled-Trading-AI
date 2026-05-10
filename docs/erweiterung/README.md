# ERWEITERUNG — Forschungs-Erweiterung des Assembled-Trading-Cores

Diese Erweiterung sitzt in `src/erweiterung/` und ist **vollständig isoliert** vom Mainline-Core. Sie ist als Forschungs-/Wissenschafts-Layer gedacht: moderne Methoden, kostenlose alternative Datenquellen, rigorose Backtesting-Statistik, ohne den bestehenden Code anzufassen.

## Designprinzipien

1. **Einbahn-Import:** `assembled_core` importiert nicht aus `erweiterung`. Umgekehrt nur explizit, falls überhaupt nötig.
2. **PIT-Sicherheit:** Jedes Signal/Feature dokumentiert seine Verzögerung explizit.
3. **Fail-soft:** Fehlende optionale Pakete oder API-Ausfälle führen zu leeren Resultaten, nicht zu Crashes.
4. **Keine bezahlten Quellen** im Code. Wünsche & Spec siehe [`PAID_DATA_WISHLIST.md`](./PAID_DATA_WISHLIST.md).
5. **Deterministisch reproduzierbar** mit Seed.

## Module

| Pfad | Inhalt |
|------|--------|
| `altdata/wikipedia_pageviews.py` | Wikimedia REST-API für Pageviews; Attention-Score nach Moat et al. (2013) |
| `altdata/google_trends.py` | pytrends-Wrapper mit Cache, Z-Skalierung |
| `altdata/sec_edgar.py` | SEC EDGAR Filings (Form 4, 8-K, 13D/G, 10-K/Q) + Form-4-Parser |
| `altdata/finra_short_interest.py` | FINRA Reg-SHO daily files; short_pressure-Signal (Boehmer/Jones/Zhang 2008) |
| `altdata/cftc_cot.py` | CFTC Commitments of Traders (Disaggregated); net-position z-score |
| `altdata/fred_md.py` | McCracken/Ng FRED-MD-Panel; Macro-PCA-Faktoren |
| `altdata/yahoo_options.py` | Yahoo-Options-Chains; Put/Call-Ratio, IV-Skew, Term-Structure |
| `altdata/gdelt_extended.py` | GDELT 2.0 GKG (Tagesaggregat Tone, Themen, Locations) |
| `altdata/reddit_pushshift.py` | Reddit-Mentions (arctic-shift Mirror); Ticker-Extraktion |
| `altdata/coingecko_crypto.py` | Crypto-Risk-On/Off-Indikator + Stablecoin-Liquiditätsproxy |
| `altdata/worldbank_macro.py` | World Bank Open Data (Cross-Country Macro) |
| `signals/cross_sectional_residuals.py` | Sektor-/Markt-neutralisierte Residuen, Momentum, Reversal, Vol |
| `signals/options_implied.py` | VRP, Skew, Garman-Klass-Volatility |
| `signals/attention.py` | Composite Wikipedia + Trends + Reddit Score |
| `signals/lead_lag_network.py` | Granger-Causality-Netzwerk; Lead-Signal-Propagation |
| `signals/statistical_arbitrage.py` | Engle-Granger-Cointegration + Pairs-Z-Score |
| `signals/earnings_drift_v2.py` | PEAD-v2: SUE mit Estimate-Dispersion, Conditional Drift |
| `signals/macro_nowcast.py` | Sahm-Rule, Yield-Curve, Credit-Spread → Recession-Prob |
| `portfolio/hierarchical_risk_parity.py` | HRP nach Lopez de Prado (2016) |
| `portfolio/black_litterman.py` | BL mit Idzorek-Konfidenz-Heuristik |
| `portfolio/cvar_optimizer.py` | CVaR-LP via Rockafellar/Uryasev |
| `portfolio/risk_parity.py` | Equal-Risk-Contribution (Newton) |
| `portfolio/kelly_sizing.py` | Single + Multi-Asset Kelly mit Confidence-Discount |
| `ml/conformal_prediction.py` | Split- und Adaptive-Conformal-Inference |
| `ml/triple_barrier.py` | Lopez de Prado Triple-Barrier + Meta-Labeling + Sample-Uniqueness |
| `ml/stacking_ensemble.py` | OOF-Stacking |
| `risk/tail_risk_evt.py` | Generalized-Pareto-Fit (Method-of-Moments); VaR/CVaR-EVT |
| `risk/dynamic_drawdown_control.py` | CPPI + Vol-Targeting-Composite |
| `risk/correlation_breakdown.py` | APC + Eigenvalue-Concentration → Crisis-Score |
| `backtest/cpcv.py` | Combinatorial Purged CV (Lopez de Prado) |
| `backtest/deflated_sharpe.py` | Deflated Sharpe + Probabilistic SR |
| `backtest/white_reality_check.py` | White (2000) + Hansen SPA (2005) |
| `backtest/performance_metrics.py` | Sharpe, Sortino, Calmar, MDD, Tail-Ratio, IR |
| `backtest/walk_forward.py` | Strikt rollierender Walk-Forward |
| `execution/almgren_chriss.py` | AC-2000 optimaler Trade-Schedule |
| `execution/adaptive_slippage.py` | Half-Spread + Linear + Sqrt-Impact |
| `meta/strategy_orchestrator.py` | Equal-Weight, Inverse-Vol, Hedge-Algo, Regime-Aware |
| `meta/regime_router.py` | Vol/Trend/Composite-Regime |

## Schnellstart

```bash
# Tests
.venv/Scripts/python.exe -m pytest tests/erweiterung -v

# Demo-Backtest (rein synthetisch, keine API-Calls)
.venv/Scripts/python.exe scripts/erweiterung/run_demo_backtest.py --n-days 1260
```

## Demo-Output (Beispiel, synthetische Daten)

```
  equal_weight       Sharpe=-0.564  Calmar=-0.301  MDD=-25.478%
  inverse_vol        Sharpe=-0.494  ...
  residual_volatility Sharpe=+0.356 (low-vol anomaly)
  ...
Reality-Check best=residual_volatility, p=0.650   <-- korrekt nicht signifikant
Hansen-SPA      best=residual_volatility, p=0.642
```

**Wichtig:** auf zufällig generierten Renditen darf keines der Signale signifikant sein, und White's Reality Check + Hansen's SPA bestätigen genau das. Ein Framework, das auf zufälligen Daten "Edges" findet, wäre kaputt.

## Wo paid-source einen echten Mehrwert bringt

Siehe [`PAID_DATA_WISHLIST.md`](./PAID_DATA_WISHLIST.md). Kurzform: OptionMetrics für historische IV-Surfaces, OptionMetrics-Style historischer Skew/Smile, RavenPack/Bloomberg-News-Feed mit präzisem PIT, Compustat für saubere Fundamentaldaten ab 1990.

## Status & Ehrlichkeit

- **Tests grün:** 91/91 lokal (`pytest tests/erweiterung`).
- **CI:** noch nicht verifiziert auf Ubuntu/Windows-Workflow.
- **End-to-End-Demo:** läuft auf synthetischen Daten; auf echten Daten noch zu validieren.
- **Nicht enthalten:** Live-Trading-Adapter, Realtime-API, vollständige Integration mit `assembled_core`-Pipeline. Das ist beabsichtigt — diese Erweiterung ist als Pull-Request-Material gedacht, das vom Maintainer geprüft und ggf. integriert wird.

## Architektur-Diagramm

```
┌──────────────────────────────────────────────────────────────────┐
│                         erweiterung                              │
├──────────────────────────────────────────────────────────────────┤
│  altdata  ─→  signals  ─→  ml/portfolio  ─→  meta  ─→  backtest │
│     ↓            ↓               ↓             ↓          ↑     │
│  Cache       PIT-Shift       Risk/Tail     Regime     Validation│
│ (Parquet)   per Quelle       Overlays      Router      (CPCV)   │
└──────────────────────────────────────────────────────────────────┘
                          ↓
                ┌─────────────────┐
                │  assembled_core │  (unverändert)
                └─────────────────┘
```

## Beitrag

Dies ist ein **Vorschlag** auf separater Branch (`ERWEITERUNG`). Über die Integration entscheidet ausschließlich der Repo-Maintainer.
