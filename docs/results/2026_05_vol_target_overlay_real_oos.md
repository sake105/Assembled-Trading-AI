# vol_target_overlay — Echter OOS Walk-Forward (Alpaca, 2026-05)

**Erstellt:** 2026-05-28  
**Branch:** main  
**Zweck:** GO_LIVE Kandidat B — Strategie-Integration Vol-Target Risk-Overlay.

---

## Strategie

Zwei-Asset-Overlay: SPY (Risiko) + IEF (Defensiv, Barclays 7–10Y Treasuries).

```
realized_vol  = std(daily_ret[-20:]) × √252   [annualisiert, kausal]
raw_weight_spy = min(1.0, 0.12 / realized_vol)
Trend-Filter:  wenn SPY close < 200-Tage-SMA → weight_spy ×= 0.5
weight_ief    = 1 − weight_spy  (immer voll investiert)
```

## Datenquelle

- **Anbieter:** Alpaca Markets — split-adjustiert, `StockHistoricalDataClient`
- **Symbole:** SPY + IEF
- **Tatsächliche Zeitspanne:** 2016-01-04 → 2025-12-30
  (IEF-Inception: 2002-07-30; Anfrage ab 2003-01-01)

## Walk-Forward-Konfiguration

- Modus: Rolling
- Train/Test/Step: 252/252/252 Handelstage (~1 Jahr)
- Warmup-Buffer: 230 Bars vor Testbeginn (SMA + Vol-Lookback initialisiert)
- Transaktionskosten: 10.0 bps Commission + 0.25+0.5 Spread/Impact = 10.75 bps je Turnover
- Startkapital: 100,000 USD
- Gewichte: tägliche Rebalanzierung auf Zielgewichte

---

## Ergebnisse pro Fold

| Fold | Test-Periode | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | SPY MaxDD | 60/40 CAGR | 60/40 Sharpe |
|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|------------|
| 1 | 2016-09-12–2017-05-22 | N/A | N/A | N/A | N/A | 15.0% | 1.75 | -4.0% | 6.8% | 1.37 |
> Fold 1 FAILED: No signals after warmup — insufficient data
| 2 | 2017-05-22–2018-01-29 | 30.8% | 4.18 | -2.1% | 14.90 | 29.8% | 4.08 | -2.1% | 15.8% | 3.89 |
| 3 | 2018-01-29–2018-10-08 | -5.8% | -0.51 | -11.7% | -0.50 | 1.6% | 0.18 | -8.7% | -1.1% | -0.10 |
| 4 | 2018-10-08–2019-06-17 | -3.6% | -0.34 | -9.5% | -0.38 | 0.7% | 0.13 | -18.5% | 6.1% | 0.68 |
| 5 | 2019-06-17–2020-02-24 | 18.0% | 1.66 | -6.6% | 2.73 | 22.9% | 1.78 | -5.9% | 16.3% | 2.39 |
| 6 | 2020-02-24–2020-11-02 | -1.5% | -0.05 | -8.8% | -0.17 | 1.9% | 0.24 | -28.9% | 3.6% | 0.28 |
| 7 | 2020-11-02–2021-07-12 | 33.8% | 2.65 | -4.0% | 8.52 | 50.0% | 3.12 | -4.1% | 27.3% | 2.77 |
| 8 | 2021-07-12–2022-03-21 | -6.0% | -0.49 | -11.2% | -0.54 | 2.5% | 0.23 | -12.9% | -2.1% | -0.18 |
| 9 | 2022-03-21–2022-11-28 | -15.3% | -1.39 | -15.1% | -1.01 | -13.4% | -0.44 | -22.7% | -13.6% | -0.80 |
| 10 | 2022-11-28–2023-08-07 | 13.1% | 1.18 | -5.9% | 2.23 | 19.4% | 1.27 | -7.6% | 10.2% | 1.05 |
| 11 | 2023-08-07–2024-04-15 | 18.2% | 1.53 | -7.7% | 2.35 | 20.0% | 1.60 | -9.0% | 10.3% | 1.18 |
| 12 | 2024-04-15–2024-12-23 | 20.3% | 1.67 | -6.9% | 2.94 | 25.7% | 1.84 | -8.4% | 15.8% | 1.77 |
| 13 | 2024-12-23–2025-09-01 | 4.1% | 0.42 | -10.8% | 0.37 | 12.7% | 0.65 | -19.0% | 10.2% | 0.82 |

_Erfolgreiche Folds: 12/13_

---

## Aggregierte OOS-Metriken

| Metrik | vol_target_overlay | SPY B&H | 60/40 B&H |
|--------|--------------------|---------|-----------|
| Ø CAGR | 8.8% | 14.5% | 8.2% |
| Ø Sharpe | 0.88 | 1.22 | 1.15 |
| Ø MaxDD | -8.4% | -12.3% | -8.1% |
| Ø Calmar | 2.62 | — | — |
| Win-Rate (CAGR > 0) | 58.3% | — | — |
| Folds, die SPY CAGR schlagen | 8.3% | — | — |
| Folds, die SPY Sharpe schlagen | 8.3% | — | — |

---

## Drawdown-Analyse (primäres Overlay-Ziel)

- Ø MaxDD overlay: **-8.4%**
- Ø MaxDD SPY: **-12.3%**
- MaxDD-Verhältnis overlay/SPY: **0.68x** (32.1% Verbesserung)

---

## Bewertung

**MaxDD-Kriterium erfüllt, Sharpe-Kriterium nicht erfüllt.** MaxDD-Ratio 0.68x (≥ 30% Reduktion), aber Ø Sharpe 0.88 < SPY Ø 1.22 + 0.2. Ø CAGR 8.8%, Ø Calmar 2.62. Teilerfolg — das Rendite/Risiko-Ziel wird nicht vollständig erreicht.

### Einschränkungen

- SPY und IEF ohne Dividenden-Reinvestition (Alpaca bar close ≈ Kursrendite, nicht Totalrendite).
  Bei IEF ist die Coupon-Rendite (~2–4% p.a.) nicht enthalten — unterschätzt IEF-Return.
- Kosten: tägliche Rebalanzierung führt zu mehr Turnover als monatliches Rebalancing;
  Kosten sind damit eher konservativ (worst case).
- Keine Transaktionssteuer, kein Spread-Impact über 0.75 bps hinaus.
- Walk-Forward deckt nur Alpaca-Verfügbarkeit; Backfill ab IEF-Inception 2002-07 wäre
  idealer (enthält 2002, 2008, 2020) — hier ab 2003 nach Warmup.

---

_Dieses Dokument ist ein automatisch erzeugtes Artefakt aus_ `scripts/_oos_wf_vol_target_overlay.py`. _Nicht manuell editieren._