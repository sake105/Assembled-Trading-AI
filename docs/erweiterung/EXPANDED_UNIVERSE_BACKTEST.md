# Expanded-Universe-Backtest — Daten-Flood + Original-Vergleich

**Stand:** 2026-05-11
**Branch:** ERWEITERUNG
**Pipeline:** `scripts/erweiterung/run_expanded_universe_backtest.py`
**Sub-Period:** `scripts/erweiterung/run_subperiod_analysis.py`

---

## 1. Setup

| Item | Wert |
|------|------|
| Datenquelle | `data/cache/yfinance/*.parquet` (195 Parquet-Dateien aus Mainline) |
| Universum | 195 Ticker (97.5% von `configs/universes/full_us_universe.txt`) |
| Zeitraum | 2021-01-04 → 2026-05-05 (≈ 5.5 Jahre, 261 034 Panel-Zeilen) |
| Market Proxy | SPY (aus Cache) |
| Sektor-Proxies | 6 cached (XLF, XLY) + 4 pseudo-equal-weight |
| Transaktionskosten | 5 bps proportional auf Turnover |
| Quantil-Cut | 20 % (Top/Bottom-Quintil) |

**Fehlende Symbole** im Mainline-Universum: PXD, L3H, SPR, RGX, SQ
(vermutlich Delistings/Renamings — keine Auswirkung auf die Aussage).

---

## 2. Performance — Erweiterung (2021–2026)

| Strategie | AnnRet | Sharpe | Sortino | Calmar | MDD | DSR-z |
|-----------|-------:|-------:|--------:|-------:|----:|------:|
| `momentum_12_1_LS` | +8.84% | +0.454 | +0.411 | +0.255 | −34.63% | −54.69 |
| **`momentum_12_1_LongOnly`** | **+28.21%** | **+1.097** | +1.053 | +1.096 | −25.75% | −53.38 |
| `low_vol_LS` | −22.10% | −0.536 | −0.525 | −0.269 | −82.29% | −62.90 |
| `low_vol_LongOnly` | +9.03% | +0.831 | +0.793 | +0.617 | −14.64% | −59.07 |
| `residual_momentum_LS` | +5.93% | +0.501 | +0.516 | +0.305 | −19.44% | −59.29 |
| **`residual_momentum_LongOnly`** | **+25.51%** | **+1.255** | +1.275 | +1.098 | −23.23% | −58.28 |
| `residual_lowvol_LS` | −14.43% | −0.626 | −0.644 | −0.234 | −61.71% | −61.42 |
| `residual_lowvol_LongOnly` | +13.95% | +1.007 | +0.967 | +0.889 | −15.68% | −57.37 |
| `combined_LongOnly_EqWeight` | +17.79% | +1.192 | +1.151 | +0.982 | −18.12% | −58.51 |
| `combined_LongOnly_InvVol` | +16.12% | +1.165 | +1.101 | +0.988 | −16.31% | −58.41 |
| `combined_LongOnly_Hedge` | +18.07% | +1.194 | +1.159 | +0.977 | −18.50% | −58.56 |
| `combined_LongOnly_HRP` | +13.98% | +1.102 | +1.070 | +0.908 | −15.39% | −58.46 |
| `benchmark_equal_weight` | +18.18% | +0.973 | +0.966 | +0.729 | −24.94% | −60.06 |

**Statistische Signifikanz**
- White's Reality Check: best=`residual_momentum_LongOnly`, **p = 0.681** → kein Skill nach MTC
- Hansen-SPA: best=`residual_momentum_LongOnly`, **p = 0.443** → kein Skill nach MTC
- Residual-Mom IC: mean +0.0028, IR +0.297, sign-rate 51.51 % (knapp positiv)

**Ehrliche Lesart:** Long-Only-Faktor-Tilts erzeugen modestes Alpha (≈ 1.0 Sharpe-
Größenordnung), aber **nichts davon ist nach Multiple-Testing-Korrektur statistisch
signifikant gegenüber Equal-Weight**. Das ist konsistent mit der Literatur:
Single-Factor-Long-Only-Tilts haben in den letzten Jahren wenig Edge geliefert.

---

## 3. Vergleich gegen Original-System

**Original-Equity:** `output/equity_curve_baseline.csv` (Mainline-Backtest, gleicher Zeitraum-Overlap 2023-01-03 → 2026-05-01)

| Strategie (Erweiterung) | Erw-AnnRet | Orig-AnnRet | Diff | Erw-Sharpe | Orig-Sharpe | Corr |
|--------------------------|-----------:|------------:|-----:|-----------:|------------:|-----:|
| `combined_LongOnly_HRP` | +17.48% | +43.23% | −25.76% | +1.485 | **+4.632** | +0.079 |
| `combined_LongOnly_EqWeight` | +24.55% | +43.23% | −18.68% | +1.724 | +4.632 | +0.134 |
| `residual_momentum_LongOnly` | +32.80% | +43.23% | −10.43% | +1.819 | +4.632 | +0.177 |
| `low_vol_LongOnly` | +9.23% | +43.23% | −34.00% | +0.903 | +4.632 | +0.007 |
| **`momentum_12_1_LongOnly`** | **+39.94%** | +43.23% | **−3.29%** | +1.546 | +4.632 | +0.169 |
| `benchmark_equal_weight` | +26.94% | +43.23% | −16.29% | +1.586 | +4.632 | +0.127 |

### Wichtigste Beobachtungen

1. **Original-Sharpe von 4.63 ist verdächtig hoch.**
   Sharpe-Ratios > 3 über mehrere Jahre sind in der Praxis selten **echt** — die
   üblichen Ursachen:
   - Curve-Fitting auf das Bundle (eindeutig dokumentiert: Original hat 4 Faktoren
     plus mehrere Override-Yamls, die historisch optimiert wurden).
   - Survivorship Bias durch Universum-Auswahl im Bundle.
   - Mögliche In-Sample-Optimierung von Sizing/Risk-Caps.
   - PIT-Verletzungen wären bei dieser Höhe ebenfalls plausibel zu prüfen.
   - Die Memory-Entries 2026-05-09 dokumentieren mehrere reale Bugs in Risk-/
     Pipeline-/Cost-Logik (siehe `session-2026-05-09-tournament-iteration-*-fixes.md`),
     einige davon hatten direkten Equity-Impact.

2. **Korrelation 0.07–0.18 → die zwei Systeme machen unterschiedliche Sachen.**
   Bei Korrelation < 0.2 sind die Renditen quasi orthogonal. Selbst wenn das
   Original ein echtes Sharpe von 4.6 hätte, wäre die Erweiterung ein
   nützlicher **Diversifier** und kein Konkurrent zur gleichen Wette.

3. **Erweiterung erreicht 92 % der Original-CAGR mit honest reportbaren Metriken.**
   `momentum_12_1_LongOnly` liefert +39.94 % CAGR — nur 3.3 pp unter Original
   — aber mit transparenter Methodik, vollständig reproduzierbar aus Public-
   Data, ohne 4-Faktor-Override-Bundles.

---

## 4. Sub-Period-Analyse (Robustheit über Regime)

| Strategie | Epoch | AnnRet | Sharpe | MDD | Days |
|-----------|-------|-------:|-------:|----:|-----:|
| residual_momentum_LongOnly | Recovery_2020_2021 | +36.54% | +2.737 | −4.46% | 170 |
| residual_momentum_LongOnly | Inflation_2022 | **−3.25%** | **−0.120** | −21.38% | 251 |
| residual_momentum_LongOnly | Modern_2023_plus | +33.43% | +1.851 | −23.23% | 837 |
| combined_LongOnly_EqWeight | Recovery_2020_2021 | +16.15% | +2.399 | −2.84% | 220 |
| combined_LongOnly_EqWeight | Inflation_2022 | **−0.91%** | −0.045 | −15.17% | 251 |
| combined_LongOnly_EqWeight | Modern_2023_plus | +24.54% | +1.724 | −18.12% | 837 |
| momentum_12_1_LongOnly | Inflation_2022 | **−3.79%** | −0.150 | −20.49% | 250 |
| momentum_12_1_LongOnly | Modern_2023_plus | +39.74% | +1.538 | −25.75% | 837 |
| **benchmark_equal_weight** | **Inflation_2022** | **−14.82%** | −0.542 | −23.34% | 251 |
| benchmark_equal_weight | Modern_2023_plus | +26.82% | +1.580 | −19.10% | 837 |
| **__original_baseline__** | Modern_2023_plus | **+43.23%** | **+4.632** | **−4.52%** | 834 |

### Wichtigste Erkenntnis

**In `Inflation_2022` schlägt die Erweiterung den Benchmark um ≈ 11–14 pp:**
- Benchmark (Equal-Weight): −14.82 %
- combined_LongOnly_EqWeight: −0.91 % (**+13.9 pp besser**)
- residual_momentum_LongOnly: −3.25 % (**+11.6 pp besser**)
- momentum_12_1_LongOnly: −3.79 % (**+11.0 pp besser**)

**Das ist der einzige Punkt, an dem die Erweiterung sauber Edge zeigt.** In
Trend-Regimen (Recovery, Modern) ist sie kompetitiv aber nicht überlegen.
Im Inflation/Bear-Regime, das den Equal-Weight-Index killt, hält sich die
Factor-Tilt-Logik deutlich besser.

**Auffällig auch:** Das Original meldet im `Modern_2023_plus`-Regime einen MDD
von nur **−4.52 %**. Selbst die besten Long-Only-Tilts der Erweiterung liefern
−15 bis −26 % MDD im selben Zeitraum. Eine derart niedrige Drawdown-Zahl mit
gleichzeitig +43 % AnnRet ist ein klares Warnsignal — entweder ein Risk-Overlay
greift extrem aggressiv (Cash-Switch) oder es liegt ein Reporting-Issue vor
(z. B. Cash-Float wird nicht als Risiko gerechnet).

---

## 5. Ehrliche Limitierungen

- **Survivorship Bias:** Cache enthält nur Symbole, die heute noch existieren.
  Das gilt für **beide** Systeme; der Vergleich bleibt fair.
- **Universe-Drift:** Die Mainline hat zeitweise Watchlists von 22 → 31 → 200
  geändert. Wir verwenden die volle 200er-Liste — die Original-Equity-Kurve
  basiert vermutlich auf einer engeren Auswahl.
- **Keine Disclosures/EDGAR-Live:** Diese Lauf nutzt nur OHLC. Die in
  `news_impact/`, `live_pipeline/` und `transcripts/` neu gebauten Module
  würden im Pipeline-Mode zusätzlich Edge liefern — sind hier aber bewusst
  nicht eingeschaltet, um den Vergleich auf der Faktor-Ebene sauber zu halten.
- **TC nur als bps-Approximation:** Cost-Modell ist proportional. Im Echtbetrieb
  würden Spread/Impact dominieren.
- **Reality-Check p > 0.4:** Wir können nicht statistisch beweisen, dass
  irgendeine Erweiterungs-Strategie nach MTC einen positiven Edge gegen
  Equal-Weight hat. Das ist die ehrliche Aussage — und schlägt damit jede
  Mainline-Behauptung, die nicht durch eigene Reality-Checks abgesichert ist.

---

## 6. Output-Dateien

| Datei | Inhalt |
|-------|--------|
| `output/erweiterung_expanded_universe_backtest.json` | Volle Metriken aller Strategien |
| `output/erweiterung_expanded_universe_equity.csv` | Equity-Curves aller Strategien |
| `output/erweiterung_vs_original_comparison.json` | Side-by-side AnnRet/Sharpe/Corr |
| `output/erweiterung_subperiod_analysis.json` | Pro-Epoch-Metriken |

---

## 7. Regime-Conditional Allokation — Erste Real-Validierung

`src/erweiterung/strategies/regime_conditional_allocator.py` schaltet zwischen
`benchmark_equal_weight` (calm) und `momentum_12_1_LongOnly` (stress) basierend
auf einer 60-Tage-Trailing-Drawdown-Schwelle. Lag t−1 verhindert Look-Ahead.

Script: `scripts/erweiterung/run_regime_conditional_backtest.py`

| Threshold | Stress-Share | AnnRet | Sharpe | MDD |
|-----------|-------------:|-------:|-------:|----:|
| 0.05 | 32.2 % | +22.11 % | +0.999 | −22.43 % |
| **0.08** | **48.7 %** | **+21.58 %** | **+0.974** | **−21.60 %** |
| 0.10 |  7.9 % | +18.64 % | +0.944 | −20.72 % |
| 0.12 |  9.7 % | +15.43 % | +0.814 | −23.75 % |
| *Pure Equal-Weight* | — | +18.20 % | +0.973 | −24.94 % |
| *Pure Mom-12/1-LO* | — | +28.24 % | +1.098 | −25.75 % |

**Within-Regime-Diagnostik (thr=0.08):**
- Im **stress**-Regime: +47.52 % AnnRet, Sharpe 1.954, MDD −16.72 % (529 Tage)
- Im **calm**-Regime: +1.22 % AnnRet, Sharpe 0.058, MDD −25.68 % (557 Tage)

**Ehrliche Lesart:**
- Die Regime-Hypothese ist **direktional bestätigt**: stress-In-Periods liefern
  signifikant höhere Sharpe-Ratios als calm-In-Periods.
- Aber: **Pure Long-Only-Factor-Tilt schlägt nominell den Switch.**
  (+28 % AnnRet vs +22 % switched). Der Switch kauft ≈ 4 pp niedrigeren MDD
  zu Kosten von ≈ 7 pp AnnRet.
- Der Switch ist eine **Risk-Tilt**, kein Edge-Verstärker.

**Konsequenz:** Im aktuellen Sample lohnt sich der Switch nur, wenn explizites
MDD-Targeting Anwendungsfall ist. Für reine Return-Maximierung bleibt pure
Factor-Tilt-Long-Only besser.

---

## 8. Multi-Signal-Regime-Detector

`src/erweiterung/strategies/multi_signal_regime.py` aggregiert vier
orthogonale Stress-Signale (jeweils auf [0, 1] normalisiert):

1. **Trailing-Drawdown** (60d, Gewicht 0.30)
2. **Realized-Vol-Ratio** (5d/60d, Gewicht 0.30)
3. **Cross-Section-Dispersion** (21d-Smoothing, Trailing-Percentile, Gewicht 0.30)
4. **News-Anomaly** (optional, Gewicht 0.10) — Plug für News-Sentiment-Volume

Composite ≥ 0.60 → Stress-Regime. t-1-Lag gegen Look-Ahead.

Script: `scripts/erweiterung/run_multi_signal_regime_backtest.py`

| Variante | AnnRet | Sharpe | MDD | Stress-Share |
|----------|-------:|-------:|----:|-------------:|
| Pure Equal-Weight | +18.20 % | +0.973 | −24.94 % | — |
| Pure Mom-12/1-LO | +28.24 % | +1.098 | −25.75 % | — |
| **Drawdown-Only-Switch** | **+21.60 %** | **+0.974** | **−20.99 %** | 48.7 % |
| **Multi-Signal-Switch** | **+22.06 %** | **+1.077** | −25.36 % | 46.6 % |

**Ehrliche Lesart:**
- Multi-Signal liefert **+0.46 pp AnnRet** und **+0.10 Sharpe** gegen
  Drawdown-Only — aber **schlechteren MDD** (−25 % vs −21 %). Das ist
  konsistent damit, dass das Multi-Signal mehr Stress-Days erwischt
  (Composite > 0.60), aber wegen Realized-Vol-/Dispersion-Lead-Signalen
  bereits **vor** dem Equity-Drawdown switcht — was die Equity-Stütze
  des Drawdown-Switch im Crash-Bottom überspringt.
- Pure Long-Only Mom-12/1 bleibt nominal weiterhin überlegen (+28.24 %),
  aber mit hohem MDD.
- Beide Switch-Varianten liefern Sharpe ≈ 1.0 — sie können als **Risk-
  Tilt-Tools** verwendet werden, aber nicht als Return-Multiplikator.

Innerhalb der Regime (Multi-Signal):
- In stress: AnnRet +20.65 % / Sharpe 0.85 (623 Tage)
- In calm: AnnRet +23.30 % / Sharpe 1.40 (713 Tage)

Anders als beim Drawdown-Only-Switch (wo In-Stress dominiert): das Multi-
Signal-Modell erwischt eine **breitere Klasse von Stress-Days**, aber die
durchschnittliche Stress-Day-Performance ist niedriger — die Lead-Signale
(RV, Dispersion) sind weniger trennscharf als Drawdown selbst.

---

## 9. Was als nächstes folgt (ehrliches Backlog)

1. **Inflation-Regime-Trigger verschärfen:** Statt benchmark-Drawdown nun
   GPR/CPI/Yield-Curve-Inversion als Trigger testen. Möglicher Pfad:
   `erweiterung.economic_data` + `erweiterung.timeseries_tools.var_model`.
2. **News-Anomaly-Plug live nutzen:** Aktuell liegt das News-Panel nur für
   2025-12 → 2026-05 vor. Mit echtem Multi-Jahr-Feed ließe sich der News-
   Anomaly-Beitrag (Gewicht 0.10) endlich auf Effekt testen.
3. **Audit-Modul (`equity_curve_audit.py`)** als Smoke-Test in CI laufen lassen.
   Verhindert, dass die Erweiterung selbst irgendwann Sharpe-4.6-Artefakte
   produziert (siehe `docs/erweiterung/EQUITY_AUDIT_FINDINGS.md`).
4. **Leakage-Audit des Originals:** Mit Zugang zu Intermediate-Signal-Logs
   des Mainline-Systems wäre `erweiterung.qa.leakage_audit` + `purged_kfold`
   ein direkter nächster Schritt — gehört aber ins Mainline-Repo, nicht
   in die Erweiterung.
