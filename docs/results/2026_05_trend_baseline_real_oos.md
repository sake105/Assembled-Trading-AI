# trend_baseline — Echter OOS Walk-Forward (Alpaca, 2026-05)

**Erstellt:** 2026-05-27  
**Branch:** main @ a7e01689  
**Zweck:** Artefakt 2 aus GO_LIVE_CHECKLIST A3/B1 — echter OOS-Nachweis auf realen Kursdaten.

---

## Datenquelle

- **Anbieter:** Alpaca Markets (Free Tier) — `StockHistoricalDataClient`, split-adjustiert
- **Angefordertes Universum:** 195 Symbole (watchlist.txt, US-only, ohne '.')
- **Symbole mit Alpaca-Daten:** 194
- **Tatsächliche Zeitspanne:** 2017-08-24 → 2025-12-30
  (Anfrage: 2018-01-01 → 2025-12-31; Alpaca liefert je nach Symbol ab ~2015–2016)
- **SPY:** Als Buy-and-Hold-Benchmark, gleicher Anbieter

## Walk-Forward-Konfiguration

- Modus: Rolling
- Train-Fenster: 252 Handelstage (~1 Jahr)
- Test-Fenster: 252 Handelstage (~1 Jahr)
- Schrittweite: 252 Handelstage (jährliche Verschiebung)
- MA-Warmup-Buffer: 90 Bars vor Testbeginn (MA initialisiert)
- ma_fast=20, ma_slow=60 (wie PaperPilot paper_runner.py)
- Commission: 10.0 bps (wie April-2026-Report)
- Spread-Weight: 0.25, Impact-Weight: 0.5
- Startkapital: 100,000 USD

---

## Ergebnisse pro Fold

| Fold | Train | Test | CAGR | Sharpe | MaxDD | SPY-CAGR | SPY-Sharpe | Schlägt SPY? |
|------|-------|------|------|--------|-------|----------|------------|-------------|
| 1 | 2018-01-01–2018-09-10 | 2018-09-10–2019-05-20 | -31.3% | -1.68 | -24.6% | -1.1% | 0.02 | Nein |
| 2 | 2018-09-10–2019-05-20 | 2019-05-20–2020-01-27 | 1.4% | 0.13 | -14.3% | 23.8% | 1.85 | Nein |
| 3 | 2019-05-20–2020-01-27 | 2020-01-27–2020-10-05 | -6.3% | 0.04 | -29.3% | 4.6% | 0.31 | Nein |
| 4 | 2020-01-27–2020-10-05 | 2020-10-05–2021-06-14 | 27.2% | 1.05 | -10.3% | 38.2% | 2.31 | Nein |
| 5 | 2020-10-05–2021-06-14 | 2021-06-14–2022-02-21 | -27.4% | -1.50 | -21.2% | 3.1% | 0.28 | Nein |
| 6 | 2021-06-14–2022-02-21 | 2022-02-21–2022-10-31 | -33.6% | -0.88 | -43.4% | -13.4% | -0.44 | Nein |
| 7 | 2022-02-21–2022-10-31 | 2022-10-31–2023-07-10 | 11.2% | 0.44 | -20.0% | 20.6% | 1.18 | Nein |
| 8 | 2022-10-31–2023-07-10 | 2023-07-10–2024-03-18 | 4.4% | 0.26 | -13.4% | 23.9% | 1.90 | Nein |
| 9 | 2023-07-10–2024-03-18 | 2024-03-18–2024-11-25 | 21.4% | 0.98 | -11.0% | 24.0% | 1.75 | Nein |
| 10 | 2024-03-18–2024-11-25 | 2024-11-25–2025-08-04 | -28.0% | -0.64 | -34.2% | 6.1% | 0.37 | Nein |

_Erfolgreiche Folds: 10/10_

---

## Aggregierte OOS-Metriken

| Metrik | trend_baseline | SPY Buy-and-Hold |
|--------|---------------|-----------------|
| Ø CAGR | -6.1% | 13.0% |
| Ø Sharpe | -0.18 | 0.95 |
| Ø MaxDD | -22.2% | — |
| Win-Rate (CAGR > 0) | 50.0% | — |
| Folds, die SPY schlagen | 0.0% | — |

---

## Bewertung

trend_baseline schlägt SPY nur in 0.0% der Folds (Ø CAGR -6.1% vs. SPY 13.0%). Sharpe Ø -0.18. Das Ergebnis ist **negativ** — trend_baseline liefert im OOS-Vergleich keinen robusten Alpha gegenüber einer passiven SPY-Position. Die Strategie profitiert vom Bullmarkt-Bias in bestimmten Perioden, versagt aber in Seitwärts- oder Bear-Phasen. Der PaperPilot-Betrieb sollte dies als Risikofaktor werten.

### Einschränkungen dieses Reports

- Alpaca Free Tier: keine adjustierten Daten für delisted Symbole → Survivorship-Bias möglich.
- walk_forward.py `make_engine_backtest_fn` wurde durch eine custom backtest_fn ersetzt,   die MA-Warmup-Bars vor dem Testzeitraum prepended.
- Die Transaktionskosten (10 bps) entsprechen dem April-2026-Report, aber kein   marktimpact-adjustierter Kostensatz.
- SPY-Vergleich: kein Dividenden-Reinvest im SPY-Buy-and-Hold (Alpaca bar close ≠ total return).

---

_Dieses Dokument ist ein automatisch erzeugtes Artefakt aus_ `scripts/_oos_wf_trend_baseline.py`. _Nicht manuell editieren._