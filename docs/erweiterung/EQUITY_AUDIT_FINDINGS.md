# Equity-Curve Anomaly Audit — Findings

**Stand:** 2026-05-11
**Branch:** ERWEITERUNG
**Script:** `scripts/erweiterung/run_equity_curve_audit.py`
**Modul:** `src/erweiterung/qa/equity_curve_audit.py`
**Output:** `output/equity_curve_audit.json`

---

## 1. Methodik

Statistische Plausibilitätsprüfung jeder Equity-Curve gegen sechs Heuristiken:

1. Overall Sharpe — extreme Werte (> 3) markieren
2. Autokorrelation Lag-1/Lag-5 — hoch deutet auf NAV-Smoothing
3. Skew/Kurtosis — abnormal symmetrisch oder kurtotisch = synthetisch-verdächtig
4. MDD ↔ Sharpe-Konsistenz — Sharpe > 2 mit |MDD| < 5 % ist suspekt
5. Worst-Day/Vol-Ratio — Tail-Realismus bei langen Zeitreihen
6. Markt-Korrelation gegen SPY — zu niedrig deutet auf orthogonale Konstruktion

Wichtig: Diese Checks **beweisen kein Leakage**. Sie zeigen nur Inkonsistenzen,
die in echtem Equity-Verhalten ungewöhnlich sind.

---

## 2. Top-Befunde

### 2.1 Drei Original-Equity-Files sind exakt identisch

| Name | Sharpe | AC1 | Skew | Kurt | MDD | WD/Vol |
|------|-------:|----:|-----:|-----:|----:|------:|
| `equity_curve_baseline.csv` | **+4.63** | +0.00 | +0.77 | +3.73 | −4.52 % | +2.92 |
| `equity_curve_altdata.csv` | **+4.63** | +0.00 | +0.77 | +3.73 | −4.52 % | +2.92 |
| `equity_curve_test1_aitech_qagate.csv` | **+4.63** | +0.00 | +0.77 | +3.73 | −4.52 % | +2.92 |

Alle drei Files zeigen **bit-genau** dieselben Werte. Das ist konsistent mit
dem dokumentierten Memory-Eintrag (`session-2026-05-03-altdata-ab-test.md`):
*"A/B 2023-2026 AI-Tech: IDENTICAL (CAGR 43.01% / Sharpe 3.898 / MDD -4.52%
baseline=altdata) — altdata columns reach panel but bundle yaml has only 4
price factors, none use altdata"*.

**Implikation:** Die "altdata"- und "qagate"-Varianten waren effektiv No-Ops.
Wer den Original-Backtest als "validated mit alt-data" verkauft, basiert auf
einer Equity, in der die Alt-Data-Spalten **nicht angewendet wurden**.

### 2.2 Sharpe 4.63 mit MDD −4.52 % → mathematisch verdächtig

**Flags ausgelöst:** `SUSPICIOUS_SHARPE_4.63`, `MDD_TOO_LOW_FOR_SHARPE`

Über ≈ 836 Trading-Tage (2023-01-03 → 2026-05-01):
- CAGR ≈ 43 %
- Vol ≈ 9.3 % (impliziert durch Sharpe = 4.63 / AnnRet)
- Worst-Day = 2.92 × Vol ≈ 1.7 % Daily Loss
- Maximum-Drawdown −4.52 %

Reale Portfolios mit Vol ≈ 9 % über 3+ Jahre haben üblicherweise:
- Worst-Day ≥ 4-6 × Vol (Fat Tails)
- MDD ≥ 8-15 %

**Konsistente Erklärungen:**
- aktiver Risk-Overlay / Kill-Switch, der bei Drawdown ≤ 4 % Cash zieht
- Sizing-Cap, das in Drawdown-Phasen automatisch reduziert (z. B. DD-Damper)
- in-sample Optimierung der Risk-Layer

Ohne Zugang zu den Intermediate-Signal-Logs lässt sich nicht entscheiden,
welche dieser Erklärungen zutrifft. Aber die Kombination Sharpe 4.63 / MDD 4.5 %
über 3+ Jahre ist außerhalb des für Long-Only-Faktor-Tilts typischen Bereichs.

### 2.3 T2-200-Sym-Backtest ist deutlich realistischer

| Name | Sharpe | MDD | Kurt | WD/Vol |
|------|-------:|----:|-----:|-------:|
| `equity_curve_t2_nolev_2025_26.csv` | +0.77 | −30.36 % | +15.94 | +4.11 |

Sharpe 0.77, MDD -30 %, Kurtosis 16 → **typische Stresstest-Charakteristik
einer ehrlichen Long-Only-Equity** im 2025-26-Drawdown-Sample. Keine Flags.

Das passt zum Memory-Eintrag *"T2 200-Sym 2025-26 no-leverage CAGR 27.68% /
Sharpe 1.086 / MDD -29.47%"* — also Real-Test, nicht in-sample-Optimierung.

### 2.4 T3-200-Sym-Backtest 2023-24 ebenfalls plausibel

| Name | Sharpe | MDD | Skew | WD/Vol |
|------|-------:|----:|-----:|-------:|
| `equity_curve_t3_2023_24.csv` | +0.54 | −16.53 % | +0.55 | +3.06 |

Sharpe 0.54, MDD -16.5 % — Long-Only-Faktor-Tilt-typisch. Keine Flags.

### 2.5 Erweiterungs-Equities

Alle 13 Erweiterungs-Strategien zeigen **plausible** Profile:
- Sharpe-Range: −0.68 bis +1.30 (typisch für Faktor-Strategien)
- MDD: −15 % bis −82 % (Long-Short-Strategien haben die hohen)
- Worst-Day/Vol: 4-8 (realistic fat-tails)
- Kurtosis 1-10 (gesund)

Eine einzige Flag: `MARKET_CORR_TOO_LOW_-0.05` bei `residual_momentum_LS` —
das ist **erwünscht**, da diese Strategie explizit sektor-neutral konstruiert
ist. Kein Bug.

---

## 3. Zusammenfassung

| Kategorie | Befund |
|-----------|--------|
| Triplikate | 3 Original-Files identisch — Altdata-/QAgate-Varianten effektiv No-Ops |
| Sharpe-Inflation | Original-Baseline Sharpe 4.63 / MDD 4.5 % über 836 Tage — statistisch außerhalb des normalen Bereichs für Long-Only |
| Reale Original-Equities | T2/T3-200-Sym-Backtests sehen ehrlich aus (Sharpe < 1, MDD > 15 %) |
| Erweiterungs-Equities | Alle 13 Profile zeigen normale Tail-/Skew-/MDD-Charakteristik |

**Empfehlung an das Mainline-Projekt:**
1. Klären, warum `baseline`, `altdata` und `test1_aitech_qagate` identische
   Equity-Files produzieren — falls Altdata/QAgate aktiviert sein sollte,
   liegt ein Wiring-Bug vor.
2. Den Sharpe-4.63/MDD-4.5 %-Backtest auf Risk-Overlay-Effekte hin durchsuchen
   (DD-Damper, Cash-Switch, Sizing-Cap im PIT). Wenn der Risk-Overlay
   in-sample auf das Backtest-Window gefittet ist, ist Sharpe optimistisch.
3. Den realen T2-Backtest (Sharpe ≈ 1.08, MDD −29.47 %) **als Headline-
   Number** verwenden, nicht den Baseline-Sharpe 4.63. T2 wurde explizit
   "kein Leverage / 200-Symbol / Out-of-Sample 2025-26" gefahren — das ist
   die ehrlichere Aussage.

---

## 4. Ausblick

Das Audit-Modul wird auch in der Erweiterung selbst eingesetzt: jede neue
Strategie durchläuft als Smoke-Test diese sechs Heuristiken bevor sie in
Backtest-Reports auftaucht. Verhindert, dass die Erweiterung dieselben
Anomalien produziert, die das Original aktuell zeigt.
