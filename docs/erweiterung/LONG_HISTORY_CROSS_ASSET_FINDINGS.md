# Cross-Asset 19-Jahres-Test — 5y-Befund WIDERLEGT

**Stand:** 2026-05-11
**Branch:** ERWEITERUNG
**Daten:** `data/cache/yfinance_long/` (11 ETFs, 2007-01-03 → 2026-05-07, ~4867 Tage)
**Script:** `scripts/erweiterung/run_cross_asset_long_history.py`

---

## 1. Setup

ETF-Universum gefetcht via `scripts/erweiterung/fetch_long_history_etfs.py`:
SPY, QQQ, IWM, EFA, EEM, AGG, TLT, HYG, GLD, SLV, DBC

**Mean off-diagonal Korrelation: 0.39** (vs 0.377 im 5y-Test — stabil).

---

## 2. Performance 2007-2026 (19y)

| Strategy | AnnRet | Sharpe | Sortino | Calmar | MDD |
|----------|-------:|-------:|--------:|-------:|----:|
| EW_All_11 | +5.71 % | +0.474 | +0.491 | +0.166 | −34.39 % |
| 60/40 Classic | +6.91 % | +0.658 | +0.685 | +0.205 | −33.69 % |
| Risk_Parity | +4.62 % | +0.510 | +0.534 | +0.174 | −26.61 % |
| HRP_Static | +3.99 % | +0.485 | +0.483 | +0.175 | −22.78 % |
| VolTarget_EW | +5.00 % | +0.516 | +0.476 | +0.165 | −30.35 % |
| XAsset_Mom_Top5 | +8.21 % | +0.633 | +0.535 | +0.359 | −22.89 % |
| Hybrid_VT_Mom | +8.49 % | +0.739 | +0.681 | +0.348 | −24.42 % |

### Calmar-Bootstrap vs 60/40 Classic (19y)

| Challenger | p(>0) |
|------------|------:|
| XAsset_Mom_Top5 | 0.664 |
| Hybrid_VT_Mom | 0.539 |
| HRP_Static | 0.520 |
| Risk_Parity | 0.384 |
| EW_All_11 | 0.344 |
| VolTarget_EW | 0.232 |

**Kein Cross-Asset-Strategie ist über 19y signifikant besser als 60/40.**

---

## 3. Vergleich 5y vs 19y (wichtigster Befund)

| Strategy | 5y AnnRet | 19y AnnRet | 5y Sharpe | 19y Sharpe | 5y Calmar | 19y Calmar |
|----------|----------:|-----------:|----------:|-----------:|----------:|-----------:|
| 60/40 Classic | +9.02 % | +6.91 % | +0.828 | +0.658 | +0.439 | +0.205 |
| Hybrid_VT_Mom | **+12.40 %** | **+8.49 %** | **+1.012** | **+0.739** | **+0.711** | **+0.348** |
| XAsset_Mom_Top5 | +15.24 % | +8.21 % | +1.032 | +0.633 | +0.800 | +0.359 |
| VolTarget_EW | +9.45 % | +5.00 % | +0.879 | +0.516 | +0.522 | +0.165 |
| Calmar-p vs 60/40 (Hybrid) | 0.760 | **0.539** | | | | |

**Konsequenz:**
- Sharpe-Ratios des 5y-Tests waren **30-50 % höher** als 19y
- Calmar-Werte halbieren oder mehr im 19y-Sample
- Hybrid_VT_Mom verliert von 0.760 auf 0.539 Calmar-p

**Die 5y-Werte (2021-2026) waren ein Bull-Market-Artefakt**, das sich nicht
auf die volle Historie inkl. GFC, COVID und Inflation übertragen lässt.

---

## 4. Sub-Period-Analyse (was wo gewonnen/verloren wurde)

| Strategy | GFC_2008 | COVID_Crash | Inflation_2022 | Modern_2023+ |
|----------|---------:|------------:|---------------:|-------------:|
| 60/40 Classic | −15.36 % | −5.01 % | −15.68 % | +15.53 % |
| VolTarget_EW | −3.12 % | **−36.35 %** | −11.59 % | +18.03 % |
| XAsset_Mom_Top5 | −0.22 % | +15.63 % | −12.54 % | **+30.58 %** |
| **Hybrid_VT_Mom** | −1.32 % | −13.99 % | −12.01 % | +24.25 % |

**Wichtige Beobachtungen:**

1. **VolTarget_EW versagt im COVID-Crash** (-36.35 %, MDD −22.96 %). Vol-Targeting
   reagiert auf realized-Vol-Trail — im schnellen COVID-Crash war die Trailing-Vol
   noch niedrig, → volle Exposure, dann brutaler Loss.

2. **XAsset_Mom_Top5 ist beste Bull-Strategie** (Modern_2023+ +30.58 %), aber
   verliert in COVID-Aftermath durch Mom-Reversal.

3. **60/40 Classic schwach in GFC** (−15.36 %), aber konsistent im Mittel.

4. **Hybrid_VT_Mom ist ausgewogen aber unspektakulär** — keine Periode mit
   katastrophalem Verlust, aber auch kein massiver Gewinn.

---

## 5. Methodische Lehre

Diese Findings unterstreichen einen kritischen Punkt:

- **Backtests auf 5 Jahre sind unzureichend.** Bull-Market-Bias verzerrt
  jede Sharpe-/Calmar-Statistik nach oben.
- **Echte Tail-Events (GFC 2008, COVID 2020)** sind notwendig, um die
  Robustheit von Allokations-Strategien zu validieren.
- **Vol-Targeting hat Schwächen in schnellen Crashes** — Trailing-Vol ist
  lag-behaftet.

Das **Master_70_30 ist auf 5y-Daten gebaut** — der Walk-Forward zeigte
bereits p=0.808 vs 60/40 OOS. Mit einem 19y-Long-History-Master-Test
würde p vermutlich weiter fallen.

---

## 6. Konsequenzen für die Erweiterungs-Roadmap

1. **Cross-Asset-Strategien haben über 19y keinen statistisch signifikanten
   Edge gegen 60/40 Classic.** Hybrid_VT_Mom marginal nominell besser, aber
   p=0.539 ist weit unter Signifikanz.

2. **60/40 Classic ist über 19y eine schwer-zu-schlagende Baseline.** Sharpe
   0.658, Calmar 0.205, MDD −33.69 %.

3. **Master_70_30 muss auf 19y-Daten re-evaluiert werden.** Die 5y-In-Sample-
   p=0.97 ist vermutlich nicht reproduzierbar im 19y-Sample (analog zum
   Regime-Switching-Fall, In-Sample 0.0000 → OOS 0.99).

4. **Statistische Signifikanz braucht echte Multi-Decade-Daten** — was vor
   2021 fehlte, ist jetzt verfügbar (yfinance_long).

---

## 7. Output-Artefakte

- `data/cache/yfinance_long/{ETF}.parquet` — 11 ETFs, 2007-2026
- `output/erweiterung_cross_asset_long_history_equity.csv`
- `output/erweiterung_cross_asset_long_history_summary.json`
