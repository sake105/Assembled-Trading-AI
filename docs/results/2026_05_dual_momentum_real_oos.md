# dual_momentum — Echter OOS Walk-Forward (Alpaca, 2026-05)

**Erstellt:** 2026-05-29  
**Branch:** main  
**Zweck:** GO_LIVE Kandidat A — Dual Momentum (Antonacci-Variante).

---

## Strategie

Vier-Asset-Universum: SPY (US-Equity), VEU (Ex-US-Equity), BIL (T-Bills), AGG (Bonds).

```
ret_12m(X) = close_today / close_12M_ago − 1  [kausal, Kalender-Monate]
Outperformer = argmax(ret_12m(SPY), ret_12m(VEU))
Absolute-Filter: wenn ret_12m(Outperformer) > ret_12m(BIL):
    halte Outperformer  (voll investiert, weight=1.0)
Sonst: halte AGG  (Safe-Asset)
Re-Balance: letzter Handelstag jeden Monats
```

## Datenquelle

- **Anbieter:** Alpaca Markets — split-adjustiert, `StockHistoricalDataClient`
- **Symbole:** SPY, VEU, BIL, AGG
- **Tatsächliche Zeitspanne:** 2016-01-04 → 2025-12-30
  (VEU-Inception: 2007-03-02; BIL-Inception: 2007-05-25; limitierender Faktor)

## Walk-Forward-Konfiguration

- Modus: Rolling
- Train/Test/Step: 252/252/252 Handelstage (~1 Jahr)
- Warmup-Buffer: 312 Bars vor Testbeginn (12M Lookback + Buffer)
- Transaktionskosten: 10.0 bps Commission + 0.25+0.5 Spread/Impact = 10.75 bps je Positions-Switch (100 % Turnover)
- Startkapital: 100,000 USD
- Gewichte: monatliches Rebalancing auf Zielgewicht 1.0 (eine Position zur Zeit)

---

## Ergebnisse pro Fold

| Fold | Test-Periode | CAGR | Sharpe | MaxDD | Calmar | Trades | SPY CAGR | SPY Sharpe | SPY MaxDD | 60/40 B&H CAGR | 60/40 Reb. CAGR | 60/40 Reb. Sharpe |
|------|-------------|------|--------|-------|--------|--------|----------|------------|-----------|---------------|----------------|-----------------|
| 1 | 2016-09-12–2017-05-22 | 16.3% | 2.11 | -3.0% | 5.45 | 0 | 15.0% | 1.75 | -4.0% | 7.7% | 7.6% | 1.53 |
| 2 | 2017-05-22–2018-01-29 | 32.8% | 3.88 | -2.7% | 12.32 | 3 | 29.8% | 4.08 | -2.1% | 17.0% | 16.5% | 4.02 |
| 3 | 2018-01-29–2018-10-08 | -4.8% | -0.34 | -9.0% | -0.54 | 1 | 1.6% | 0.18 | -8.7% | -1.1% | -0.9% | -0.06 |
| 4 | 2018-10-08–2019-06-17 | -13.6% | -0.81 | -18.6% | -0.73 | 2 | 0.7% | 0.13 | -18.5% | 4.0% | 4.4% | 0.46 |
| 5 | 2019-06-17–2020-02-24 | 23.0% | 1.79 | -5.9% | 3.88 | 0 | 22.9% | 1.78 | -5.9% | 16.0% | 16.1% | 2.21 |
| 6 | 2020-02-24–2020-11-02 | -21.7% | -0.54 | -30.9% | -0.70 | 2 | 1.9% | 0.24 | -28.9% | 2.1% | 3.8% | 0.28 |
| 7 | 2020-11-02–2021-07-12 | 45.1% | 2.83 | -4.1% | 11.03 | 2 | 50.0% | 3.12 | -4.1% | 28.4% | 27.0% | 2.91 |
| 8 | 2021-07-12–2022-03-21 | 3.0% | 0.26 | -12.9% | 0.23 | 0 | 2.5% | 0.23 | -12.9% | -2.3% | -2.1% | -0.16 |
| 9 | 2022-03-21–2022-11-28 | -16.6% | -1.42 | -18.8% | -0.88 | 1 | -13.4% | -0.44 | -22.7% | -12.8% | -12.3% | -0.70 |
| 10 | 2022-11-28–2023-08-07 | 11.0% | 1.28 | -4.4% | 2.48 | 1 | 19.4% | 1.27 | -7.6% | 10.8% | 10.8% | 1.10 |
| 11 | 2023-08-07–2024-04-15 | 19.7% | 1.55 | -9.0% | 2.19 | 2 | 20.0% | 1.60 | -9.0% | 11.4% | 11.3% | 1.31 |
| 12 | 2024-04-15–2024-12-23 | 23.4% | 1.68 | -8.4% | 2.78 | 0 | 25.7% | 1.84 | -8.4% | 16.2% | 16.1% | 1.82 |
| 13 | 2024-12-23–2025-09-01 | 8.1% | 0.46 | -19.0% | 0.43 | 2 | 12.7% | 0.65 | -19.0% | 9.3% | 10.0% | 0.76 |

_Erfolgreiche Folds: 13/13_

---

## Aggregierte OOS-Metriken

| Metrik | dual_momentum | SPY B&H | 60/40 B&H | 60/40 Rebalanced |
|--------|---------------|---------|-----------|-----------------|
| Ø CAGR | 9.7% | 14.5% | 8.2% | 8.3% |
| Ø Sharpe | 0.98 | 1.26 | 1.18 | 1.19 |
| Ø MaxDD | -11.3% | -11.7% | -8.0% | -7.9% |
| Ø Calmar | 2.92 | — | — | — |
| Win-Rate (CAGR > 0) | 69.2% | — | — | — |
| Folds, die SPY CAGR schlagen | 30.8% | — | — | — |
| Folds, die SPY Sharpe schlagen | 30.8% | — | — | — |
| Folds, die 60/40-Reb. Sharpe schlagen | 30.8% | — | — | — |

---

## Drawdown-Analyse

- Ø MaxDD dual_momentum: **-11.3%**
- Ø MaxDD SPY: **-11.7%**
- MaxDD-Verhältnis overlay/SPY: **0.97x** (3.2% Verbesserung)

---

## Quervergleich mit vol_target_overlay

_Achtung: anderer Testzeitraum (VEU/BIL-Inception 2007 vs IEF-Inception 2003)._
_Ergebnisse aus separatem Lauf (Alpaca, 2026-05-28, 12/13 Folds, 2016–2025)._

| Metrik | dual_momentum (dieser Lauf) | vol_target_overlay (Referenz) |
|--------|-----------------------------|-------------------------------|
| Ø CAGR | 9.7% | 8.8% |
| Ø Sharpe | 0.98 | 0.88 |
| Ø MaxDD | -11.3% | -8.4% |
| MaxDD-Ratio vs SPY | 0.97x | 0.68x |

---

## Bewertung

**Kein Kriterium erfüllt.** Ø CAGR 9.7% vs SPY 14.5%, Ø Sharpe 0.98 vs SPY 1.26 und 60/40 Reb. 1.19. Dual Momentum liefert auf diesem Sample keinen messbaren Mehrwert gegenüber einfachen passiven Benchmarks.

### Einschränkungen

- SPY, VEU, BIL, AGG ohne Dividenden-Reinvestition (Alpaca bar close ≈ Kursrendite).
  VEU-Dividendenrendite ~3 %, AGG-Coupon ~3–4 % p.a. fehlen — IEF-/AGG-Returns unterschätzt.
- BIL als T-Bill-Proxy: Kursrendite nahezu 0 (korrekte Hurdle-Proxy-Eigenschaft).
- Kosten: 10.75 bps je Positions-Switch (monatliches Rebalancing = ~12 Switches/Jahr);
  sehr niedrige Transaktionskosten — Schätzung eher günstig.
- Walk-Forward deckt nur Alpaca-Verfügbarkeit (VEU/BIL ab 2007-05);
  enthält GFC 2008–09 und COVID 2020 — breitere Krisenabdeckung als vol_target (ab 2016).
- Parameter (lookback=12M, BIL-Hurdle) sind Antonacci-Standard, nicht optimiert.
- Quervergleich mit vol_target nicht direkt, da anderer Datenzeitraum und andere Assets.

---

_Dieses Dokument ist ein automatisch erzeugtes Artefakt aus_ `scripts/_oos_wf_dual_momentum.py`. _Nicht manuell editieren._
