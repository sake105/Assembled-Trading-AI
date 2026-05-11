# Walk-Forward Out-of-Sample Validierung — Ehrlicher Reality-Check

**Stand:** 2026-05-11
**Branch:** ERWEITERUNG
**Module:** `src/erweiterung/robustness/walk_forward.py`
**Script:** `scripts/erweiterung/run_walk_forward_validation.py`

---

## 1. Zweck

Der vorherige Long-History-Test (`docs/erweiterung/LONG_HISTORY_FINDINGS.md`)
hatte einen verdächtig sauberen Hansen-SPA-p=0.0000 für `regime_switched`.
**Bevor wir das als bewiesen melden**, muss der Regime-Switch streng
out-of-sample getestet werden — sonst wäre das ein typischer In-Sample-Bias.

Walk-Forward-Methodik (Lopez de Prado 2018, §11):

1. Rolling 5-y Train-Window
2. Threshold-Optimierung auf **Calmar-Ratio** im Training-Window
3. Apply den gewählten Threshold auf das nächste, **unsichtbare** 1-y Test-Window
4. Sammele alle Test-Returns zu einer konsistenten OOS-Curve
5. Vergleiche OOS-Curve mit Pure-Long-Only und Fixed-Threshold

**13 Walk-Forward-Windows** über 2013-2026 (= 3276 OOS-Tage).

---

## 2. OOS-Performance — Walk-Forward-Switch vs Pure-Long-Only

| Strategy (OOS-Period 2013-2026) | AnnRet | Sharpe | Sortino | Calmar | MDD |
|----------------------------------|-------:|-------:|--------:|-------:|----:|
| Pure Equal-Weight | +26.61 % | +1.334 | +1.249 | +0.849 | −31.35 % |
| **Pure Mom-12/1 LO** | **+37.90 %** | +1.405 | +1.329 | **+1.182** | −32.06 % |
| Fixed-thr=0.08 Switch | +29.57 % | +1.364 | +1.296 | +0.986 | −29.98 % |
| **Walk-Forward OOS Switch** | +30.72 % | **+1.424** | **+1.357** | +1.061 | **−28.94 %** |

**Beobachtungen:**

- Walk-Forward-Switch hat marginal höchstes **Sharpe** (1.424 vs 1.405) und
  niedrigsten **MDD** (−28.94 % vs −32.06 %).
- Aber **niedrigerer AnnRet** als Pure-Mom (+30.72 % vs +37.90 %).
- **Hansen-SPA p = 0.992**, Reality-Check p = 0.988 vs Pure Mom-12/1 → 
  **kein statistisch signifikanter Edge.**

Vergleich mit dem In-Sample-Befund:

| Test | In-Sample (full 19y) | Walk-Forward OOS (2013-2026) |
|------|---------------------:|-----------------------------:|
| AnnRet vs Pure-Mom | -0.89 pp | -7.18 pp |
| Sharpe vs Pure-Mom | +0.025 | +0.019 |
| Calmar vs Pure-Mom | +0.043 | -0.121 |
| Hansen-SPA p-Value | **0.0000** | **0.9920** |

**Der In-Sample-Edge verschwindet im Out-of-Sample-Test.** Der p-Wert von
0.0000 im 19y-Test war ein In-Sample-Bias: das Switching wurde implizit auf
GFC 2008-2009 gefittet (durch die Threshold=0.08-Wahl in den Initial-Tests),
und im OOS-Period gibt es kein vergleichbares Tail-Event.

---

## 3. Train→Test-Stabilität (Schlüssel-Diagnostik)

**Korrelation Train-Calmar ↔ Test-Calmar: −0.372**

Das ist ein **rotes Warnsignal**:

- Strategien, die **im Trainings-Window gut funktionierten**, performten
  im nächsten Test-Window **schlechter**.
- Wer nach Train-Calmar optimiert, fittet auf Vergangenheits-Idiosynkrasien.
- Echter robuster Edge hätte **positive** Train→Test-Korrelation.

**Anti-prädiktivität deutet auf Overfit-Risk hin.** Der Drawdown-Trigger ist
nicht gleichmäßig genug, um einen stabilen Threshold zu lernen.

---

## 4. Threshold-Verteilung über die Windows

| Window | Train-Start | Test-Start | Best-Threshold | Train-Obj | Test-AnnRet | Test-Sharpe | Test-MDD |
|-------:|:-----------|:-----------|---------------:|----------:|------------:|------------:|---------:|
| 0 | 2008-01 | 2013-01 | 0.10 | 0.349 | +55.32 % | +4.510 | −3.82 % |
| 1 | 2009-01 | 2014-01 | 0.10 | 2.003 | +17.43 % | +1.198 | −8.62 % |
| 2 | 2010-01 | 2015-01 | 0.10 | 1.760 | +28.32 % | +1.590 | −11.46 % |
| 3 | 2011-01 | 2016-01 | 0.10 | 1.767 | +30.47 % | +1.910 | −9.29 % |
| 4 | 2012-01 | 2017-01 | 0.10 | 2.320 | +43.13 % | +4.650 | −4.02 % |
| 5 | 2013-01 | 2018-01 | 0.10 | 2.529 | +17.09 % | +0.713 | −16.02 % |
| 6 | 2014-01 | 2019-01 | 0.10 | 1.681 | +38.41 % | +2.723 | −9.72 % |
| 7 | 2015-01 | 2020-01 | 0.10 | 1.946 | +58.63 % | +1.479 | −28.94 % |
| 8 | 2016-01 | 2021-01 | 0.06 | 1.251 | +30.98 % | +2.055 | −5.76 % |
| 9 | 2017-01 | **2022-01** | 0.06 | 1.285 | **−10.24 %** | **−0.375** | **−26.50 %** |
| 10 | 2018-01 | 2023-01 | 0.06 | 0.874 | +46.24 % | +3.084 | −9.10 % |
| 11 | 2019-01 | 2024-01 | 0.06 | 1.086 | +34.19 % | +2.434 | −7.32 % |
| 12 | 2020-01 | 2025-01 | 0.06 | 1.030 | +25.51 % | +1.051 | −18.26 % |

**Auffälligkeiten:**

- **Threshold-Shift bei Window 8 (2021)**: von 0.10 zu 0.06. Das System
  "lernt" einen sensitiveren Trigger nach COVID-Episode.
- **Window 9 (Inflation 2022): MDD −26.5 %, AnnRet −10.24 %** — das gewählte
  Trigger-Setup hat im Inflation-Regime versagt. Das war auch der Befund
  aus dem Sub-Period-Test in `LONG_HISTORY_FINDINGS.md`.
- Volatile Sharpe-Ratios von 0.71 bis 4.65 zeigen, dass der Test-Window
  Sharpe stark von Marktbedingungen abhängt, nicht von der Train-Window-Wahl.

---

## 5. Ehrliche Schlussfolgerungen

1. **Der In-Sample-p=0.0000 (Hansen-SPA) ist nicht out-of-sample reproduzierbar.**
   Walk-Forward p=0.99 → der Edge ist statistisches Artefakt der GFC-Ereignis-
   Konzentration im 19y-Sample.

2. **Pure Long-Only-Mom-12/1 bleibt der nominal beste Performer in OOS.**
   Die Sharpe-Outperformance des Switching (+0.019) ist innerhalb der
   natürlichen Block-Bootstrap-Variation.

3. **Train→Test-Calmar-Korrelation = −0.372** ist eine ernsthafte Overfit-
   Warnung. Strategien-Selektion nach Train-Performance ist hier kontraproduktiv.

4. **Der einzige robuste Befund**: Switching reduziert **MDD um ≈ 3 pp**
   (−28.94 % vs −32.06 %) bei vergleichbarem Sharpe. Das ist ein
   **Risk-Tilt**, kein Return-Edge.

5. **Wer den Switch nutzt, sollte das wegen MDD-Targeting tun, nicht wegen
   AnnRet-Verbesserung.** Wer nur Sharpe-Pareto-effizient sein will, lässt
   den Switch weg.

---

## 6. Konsequenzen für die Erweiterungs-Roadmap

- **Drawdown-Switch als universeller Edge: WIDERLEGT.**
- **Drawdown-Switch als MDD-Reducer mit −3 pp Effekt: bestätigt** (OOS-konsistent).
- **Inflation-Regime braucht andere Trigger** — Drawdown-Lag versagt 2022.
- **Threshold-Auto-Tuning ist gefährlich** — anti-prädiktive Train-Test-
  Korrelation.
- **Statistisch beweisbarer Switching-Edge braucht entweder:**
  - Drastisch mehr Daten (50y+, mit mehreren Tail-Events)
  - Oder einen Trigger, der **nicht** auf Equity-Drawdown allein basiert
  - Oder ein anderes Asset-Universum (z. B. Bonds, Commodities)

---

## 7. Output-Artefakte

- `output/erweiterung_walk_forward_oos_equity.csv` — OOS-Equity
- `output/erweiterung_walk_forward_windows.csv` — Per-Window-Details
- `output/erweiterung_walk_forward_summary.json` — Aggregat-Metriken
