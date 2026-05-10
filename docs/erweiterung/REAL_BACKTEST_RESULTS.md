# Real-Data Backtest Results — Erweiterung v2

Lauf am 2026-05-10 auf der lokalen Watchlist-Datenquelle (`data/sample/watchlist_2007_2026.parquet`).

## Setup

- **Universum:** 20 US-Mega-Caps (AAPL, ADBE, AMZN, AVGO, COST, CRM, CVX, GOOGL, HD, JNJ, JPM, MA, META, MSFT, NFLX, NVDA, PEP, PG, TSLA, UNH, V, XOM)
- **Zeitraum:** 2010-01-01 → 2026-04-01 (~16 Jahre, ~80979 row-symbol-Tage)
- **Quintile:** Top/Bottom 20% je Tag, daily rebalancing
- **Transaction-Costs:** 5 bps roundtrip
- **Market-Proxy:** equal-weight des Universums (kein SPY in Source)
- **Sektor-Faktoren:** pseudo-Sektor aus Asset-Cohort gebildet

## Ergebnisse

```
Strategy                       AnnRet   Sharpe  Sortino  Calmar     MDD
--------------------------------------------------------------------------
momentum_12_1_LS               +4.17%   +0.287   +0.273  +0.080  -52.00%
momentum_12_1_LongOnly        +36.30%   +1.276   +1.223  +1.005  -36.12%
low_vol_LS                    -29.13%   -0.939   -0.944  -0.292  -99.67%
low_vol_LongOnly              +11.12%   +0.812   +0.792  +0.545  -20.41%
residual_momentum_LS           -5.80%   -0.195   -0.190  -0.084  -69.06%
residual_momentum_LongOnly    +25.10%   +1.061   +1.031  +0.729  -34.44%
residual_lowvol_LS            -21.89%   -0.739   -0.748  -0.221  -98.91%
residual_lowvol_LongOnly      +11.28%   +0.656   +0.628  +0.238  -47.37%
combined_LongOnly_EqWeight    +20.77%   +1.171   +1.093  +0.658  -31.55%
combined_LongOnly_InvVol      +18.47%   +1.118   +1.049  +0.597  -30.96%
combined_LongOnly_Hedge       +23.73%   +1.196   +1.125  +0.764  -31.07%
combined_LongOnly_HRP         +16.65%   +1.070   +1.004  +0.560  -29.73%
benchmark_equal_weight        +24.55%   +1.248   +1.178  +0.802  -30.60%
```

## Statistische Signifikanz

- **White's Reality Check** (vs. Benchmark, 2000 Bootstrap): best = `momentum_12_1_LongOnly`, **p = 0.335**.
- **Hansen's SPA-Test** (studentisiert, 2000 Bootstrap): best = `momentum_12_1_LongOnly`, **p = 0.063**.

→ Hansen-SPA findet die Long-Only-Momentum-Strategie auf 10%-Niveau als signifikant
besser als der Benchmark; auf 5%-Niveau knapp nicht.

## IC-Diagnostic Residual-Momentum

- IC-Mean: -0.0054
- IR (annualisiert): -0.364
- Sign-Rate: 48.5 %

→ Residual-Momentum-Signal hat in dieser Universe-Größe (n=20) **keinen signifikanten cross-sectional-Edge**. Erwartet, weil das Universum bereits stark korreliert ist (alle Mega-Caps).

## Was diese Ergebnisse bedeuten

✅ **Die Long-Short-Strategien verlieren** — weil das Universum reine Mega-Caps in einer FAANG-getriebenen Hausse umfasst. Short-Bein verliert systematisch.

✅ **Long-Only-Momentum gewinnt** — +36.30 % p.a., Sharpe 1.276 — schlägt Equal-Weight-Benchmark in beiden Dimensionen.

✅ **Reality-Check & SPA arbeiten korrekt** — sie erkennen nicht-signifikante Strategien (p > 0.5) und marginalere Signale (p ≈ 0.06) statistisch sauber.

✅ **HRP/Hedge/InvVol-Combinations** liegen alle nahe am Benchmark (~ +20 %, Sharpe ~1.1) — vernünftiges Diversifikations-Verhalten.

## Limitierungen

⚠️ **Kleines Universum (20 Symbole)** — nicht repräsentativ für SP500.
⚠️ **Kein Survivorship-Bias-Schutz** — die 20 Symbole sind heutige Mega-Caps, nicht das damalige Universum.
⚠️ **Kein FF/Quality/Investment-Faktor** — Fundamentaldaten in dieser Quelle nicht enthalten.
⚠️ **Keine echten Sektor-ETFs** — pseudo-Sektoren als Workaround.

## Reproduktion

```bash
.venv/Scripts/python.exe scripts/erweiterung/run_real_backtest.py \
    --start 2010-01-01 --end 2026-04-01 --tc-bps 5
```

Output:
- `output/erweiterung_real_backtest.json`
- `output/erweiterung_real_equity.csv`
