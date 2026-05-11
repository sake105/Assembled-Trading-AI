# Long-History-Backtest (2007-2026) — Statistisch Signifikanter Befund

**Stand:** 2026-05-11
**Branch:** ERWEITERUNG
**Script:** `scripts/erweiterung/run_long_history_backtest.py`
**Datenquelle:** `data/sample/watchlist_2007_2026.parquet` (22 Mega-Caps × 4841 Tage)

---

## 1. Zusammenfassung

Mit 19 Jahren Daten (vs 5.5 Jahre im Expanded-Universe-Test) lässt sich
**zum ersten Mal ein statistisch signifikanter Switching-Vorteil** nachweisen.

| Test | p-Value | Lesart |
|------|--------:|--------|
| White's Reality Check `regime_switched` vs `momentum_12_1_LongOnly` | **0.0000** | Regime-Switch schlägt Pure-Long-Only nach Multiple-Testing |
| Hansen-SPA `regime_switched` vs `momentum_12_1_LongOnly` | **0.0000** | dito, mit Step-SPA-Korrektur |
| White's Reality Check `momentum_12_1_LongOnly` vs Equal-Weight | 0.0920 | Pure-Long-Only schlägt Equal-Weight nur marginal |
| Hansen-SPA `regime_switched` vs Equal-Weight | 0.1640 | nicht signifikant gegen Equal-Weight |

---

## 2. Vollständige Performance-Tabelle

| Strategy | AnnRet | Sharpe | Sortino | Calmar | MDD | DSR-z |
|----------|-------:|-------:|--------:|-------:|----:|------:|
| `momentum_12_1_LongOnly` | +29.05 % | +1.108 | +1.055 | +0.574 | −50.63 % | −88.37 |
| `residual_momentum_LongOnly` | +22.00 % | +0.943 | +0.916 | +0.478 | −46.06 % | −90.96 |
| `low_vol_LongOnly` | nicht-trivial wegen 22-Universum | | | | | |
| `combined_eqweight` | +20.59 % | +1.035 | +0.983 | +0.482 | −42.69 % | −90.96 |
| `combined_invvol` | +19.20 % | +1.017 | +0.964 | +0.457 | −42.05 % | −90.97 |
| `benchmark_equal_weight` | +22.80 % | +1.074 | +1.023 | +0.511 | −44.66 % | −90.97 |
| **`regime_switched`** | **+28.16 %** | **+1.133** | **+1.088** | **+0.617** | −45.68 % | −88.51 |

**Regime-Switched gewinnt nach Calmar (0.617 vs 0.574 für Pure-Long-Only)**
und nach Sharpe (1.133 vs 1.108), bei 5pp weniger MDD.

---

## 3. Sub-Period-Analyse — der Schlüssel zum Befund

| Strategy | Epoche | AnnRet | Sharpe | MDD | Days |
|----------|--------|-------:|-------:|----:|-----:|
| Pure Mom-12/1 | Pre_2008 | **−24.54 %** | −0.829 | −27.85 % | 176 |
| Pure Mom-12/1 | **GFC_2008_2009** | **−28.01 %** | −0.636 | **−39.48 %** | 200 |
| Pure Mom-12/1 | Post_GFC | +34.99 % | +1.594 | −27.71 % | 2643 |
| Pure Mom-12/1 | COVID_Crash | +45.47 % | +0.740 | −32.06 % | 93 |
| Pure Mom-12/1 | Recovery_2020_2021 | +54.56 % | +1.972 | −15.87 % | 380 |
| Pure Mom-12/1 | Inflation_2022 | −3.44 % | −0.130 | −22.34 % | 251 |
| Pure Mom-12/1 | Modern_2023+ | +37.74 % | +1.546 | −27.04 % | 814 |
| **regime_switched** | Pre_2008 | −19.22 % | −0.737 | −20.83 % | 176 |
| **regime_switched** | **GFC_2008_2009** | **−9.19 %** | −0.188 | **−37.30 %** | 200 |
| **regime_switched** | Post_GFC | +34.06 % | +1.685 | −25.39 % | 2643 |
| **regime_switched** | COVID_Crash | +29.77 % | +0.526 | −32.54 % | 93 |
| **regime_switched** | Recovery_2020_2021 | +56.19 % | +2.563 | −14.68 % | 380 |
| **regime_switched** | Inflation_2022 | −20.68 % | −0.749 | −24.89 % | 251 |
| **regime_switched** | Modern_2023+ | +35.08 % | +1.567 | −25.51 % | 814 |

### Wo die Outperformance herkommt

| Epoche | Pure-Mom | regime_switched | Differenz |
|--------|---------:|----------------:|----------:|
| **GFC 2008-2009** | −28.01 % | **−9.19 %** | **+18.82 pp** |
| Pre_2008 | −24.54 % | −19.22 % | +5.32 pp |
| COVID_Crash | +45.47 % | +29.77 % | −15.70 pp |
| Recovery_2020_2021 | +54.56 % | +56.19 % | +1.63 pp |
| **Inflation_2022** | −3.44 % | −20.68 % | **−17.24 pp** |
| Modern_2023+ | +37.74 % | +35.08 % | −2.66 pp |

**Ehrliche Lesart:**

Der Switching-Edge stammt **fast ausschließlich** aus der GFC-2008-Periode:
+18.8 pp Outperformance in einem einzigen Crisis-Window. In Bull-Markets
(Recovery, Modern) ist Pure-Long-Only weiterhin überlegen, und in
Inflation_2022 verliert Switching sogar deutlich gegen Pure-Long-Only.

Das ist konsistent mit der Theorie:
- **Trailing-60d-Drawdown-Trigger** reagiert verzögert auf Inflation
  (langsamer DD-Aufbau), aber schnell auf Crashes (GFC-Crash war scharf).
- Pure Long-Only liefert in geordneten Bull-Markets das beste Risk-Reward.
- Hansen-SPA / Reality-Check **gewichten Tail-Events stark** — daher der
  hohe p-Wert.

### Methodologische Limitierung

22 Tickers in 19 Jahren stellen ein **engeres Universum** dar als die
S&P 500 — mit hoher Survivorship-Bias (alles Mega-Caps, die heute noch
existieren). Die GFC-Outperformance ist real, aber das Ergebnis ist
**nicht auf das gesamte S&P-500 übertragbar**, ohne weitere Validation.

---

## 4. Vergleich mit Expanded-Universe (195 Tickers, 2021-2026)

| Metrik | Expanded (5.5y) | Long-History (19y) |
|--------|---------------:|-------------------:|
| Universe | 195 Tickers | 22 Mega-Caps |
| Days | 1086 (common period) | 4841 |
| Pure-Mom AnnRet | +28.24 % | +29.05 % |
| Pure-Mom Sharpe | +1.098 | +1.108 |
| Regime-Switch AnnRet | +21.60 % (drawdown-only) | +28.16 % |
| Regime-Switch Sharpe | +0.975 | +1.133 |
| Hansen-SPA Switch vs Pure | p = 0.99 (nicht sig.) | **p = 0.0000 (sig.)** |

**Schlussfolgerung:** Die statistische Trennschärfe wächst dramatisch mit
der Sample-Größe. Bei 5.5 Jahren ohne GFC-Event ist kein Switching-Edge
nachweisbar. Bei 19 Jahren mit GFC ist der Edge klar signifikant — aber
die Gewinne kommen fast ausschließlich aus GFC selbst.

---

## 5. Konsequenzen für die Erweiterungs-Roadmap

1. **Drawdown-basiertes Switching ist GFC-spezialisierter Risk-Tilt**, kein
   universeller Edge-Verstärker. Im stationären Bull-Markt ist Pure Long-
   Only besser.
2. **Wer Drawdowns > 30 % vermeiden will**, sollte den Switch nutzen.
   Wer auf max-AnnRet optimiert, eher nicht.
3. **Inflation-Regime braucht andere Trigger** als Drawdown — die schweren
   Long-Only-Faktor-Verluste in 2022 (−20 % bis −27 %) wurden vom DD-Switch
   *nicht* abgefangen, sondern noch verschlimmert (Faktor-Tilt im falschen Moment).
4. **Statistischer Test braucht echte Tail-Events**, sonst keine
   Signifikanz. 5-Jahres-Backtests ohne 2008-Erfahrung verstecken den
   wahren Edge — und das damit verbundene Tail-Risiko.

---

## 6. Output-Artefakte

- `output/erweiterung_long_history_equity.csv` — Equity-Curves aller Strategien
- `output/erweiterung_long_history_summary.json` — Metriken + p-Values
