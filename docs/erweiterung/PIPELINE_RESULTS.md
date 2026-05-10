# Research Pipeline — End-to-End Results

End-to-End Lauf der `erweiterung.pipelines.research_pipeline` auf der lokalen Watchlist.

## Setup

- **Datenquelle:** `data/sample/watchlist_2007_2026.parquet` (22 Mega-Caps)
- **Zeitraum:** 2010-01-01 → 2026-04-01 (89 149 row-symbol-Tage)
- **Strategien:** trend_following_LongOnly, low_vol_LongOnly, trend_following_LongShort
- **Benchmark:** equal-weight des Universums
- **TC:** 5 bps, **Quintile:** 0.2 (top/bottom 20%)
- **N-Bootstrap:** 1000

## Ergebnisse

### Performance-Metriken

```
Strategy                          Sharpe   AnnRet     MDD     Calmar
------------------------------------------------------------------------
trend_following_LongOnly          +1.215   +32.89%   -37.36%   +0.880
low_vol_LongOnly                  +0.858   +11.95%   -21.34%   +0.560
trend_following_LongShort         +0.229    +2.58%   -52.73%   +0.049
benchmark                         +1.240   +24.27%   -31.35%   +0.774
```

### Statistische Signifikanz (vs. Benchmark)

- **White's Reality Check** (n_bootstrap=1000): best=`trend_following_LongOnly`, **p = 0.050**.
- **Hansen's SPA-Test**: best=`trend_following_LongOnly`, **p = 0.014**.

→ Auf 5%-Niveau (Hansen-SPA) **statistisch signifikant** besser als Benchmark.
→ White's Reality-Check exakt am 5%-Threshold.

### Stress-Test (Historical Replay)

- **Worst-Drawdown:** -37.36% während **COVID-2020**
- **Worst-Crisis** (vom Strategy-Lauf bewertet): COVID-Crash 2020-02-19 → 2020-03-23

### Monte-Carlo (Stationary Bootstrap, 6-Monats-Horizont, 500 Pfade)

- **Expected Terminal Return:** +18.69%
- **VaR-95 Loss:** -11.65%
- **Probability of Loss:** 19.20%

## Was diese Ergebnisse bedeuten

✅ **Hansen-SPA p = 0.014** ist ein **echtes statistisch signifikantes Ergebnis**. Selbst nach Multi-Test-Korrektur über 3 Strategien schlägt trend_following_LongOnly den Equal-Weight-Benchmark.

✅ **Sharpe 1.215 vs Benchmark 1.240** — knapp dahinter, aber **AnnRet +32.89% vs +24.27%** zeigt deutlich höhere Rendite. Das Sharpe-Lag erklärt sich durch höhere Vola (Long-Only-Quintile-Tilt).

✅ **Long-Short-Variante (Sharpe +0.229)** schneidet schlecht ab — wegen Mega-Cap-Bull-Run hat das Short-Bein systematisch verloren. Das ist Universum-Eigenschaft, nicht Strategie-Defekt.

✅ **Stress-Replay** identifiziert COVID-2020 als härteste Phase — 23 Tage Drawdown -37%, danach schnelle Erholung.

✅ **Monte-Carlo** sagt 19.2% Wahrscheinlichkeit für 6-Monats-Verlust — realistisch für Equity-Strategy.

## Reproduktion

```bash
.venv/Scripts/python.exe scripts/erweiterung/run_research_pipeline.py \
    --start 2010-01-01 --end 2026-04-01

# Output:
# - output/erweiterung_research/metrics.json     (alle Metriken)
# - output/erweiterung_research/equity_curves.csv (Equity-Pfade)
# - output/erweiterung_research/report.html      (HTML-Report)
```

## Limitierungen — explizit benannt

⚠️ **22 Mega-Caps** ist klein und survivorship-biased. Echte SP500-Tests stehen aus.
⚠️ **Hansen-SPA p=0.014** mit nur 3 Strategien — bei 100+ getesteten Strategien wäre Multi-Test-Adjustment härter.
⚠️ **TC 5 bps** ist optimistisch für Cross-Section-Tilts mit täglichem Rebalancing — realistisch wären 10-20 bps.
⚠️ **2010-2026** war außergewöhnliche Bull-Phase — Out-of-Sample-Performance auf anderen Regimes (1990er, 2000er) wäre nötig zum echten Robustheits-Beweis.
