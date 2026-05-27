# multifactor_v2 — Echter OOS Walk-Forward (Alpaca, 2026-05)

**Erstellt:** 2026-05-27  
**Branch:** main @ cc563605  
**Zweck:** GO_LIVE_CHECKLIST Paket 3b — echter OOS-Nachweis auf realen Kursdaten.

---

## Datenquelle

- **Anbieter:** Alpaca Markets (Free Tier) — `StockHistoricalDataClient`, split-adjustiert
- **Angefordertes Universum:** 195 Symbole (watchlist.txt, US-only, ohne '.')
- **Symbole mit Alpaca-Daten:** 194
- **Tatsächliche Zeitspanne:** 2016-11-28 → 2025-12-30
  (Anfrage: 2018-01-01 → 2025-12-31)
- **SPY:** Als Buy-and-Hold-Benchmark, gleicher Anbieter

## Walk-Forward-Konfiguration

- Modus: Rolling
- Train-Fenster: 252 Handelstage (~1 Jahr)
- Test-Fenster: 252 Handelstage (~1 Jahr)
- Schrittweite: 252 Handelstage (jährliche Verschiebung)
- Warmup-Buffer: 250 Bars (MA-200 initialisiert)
- Rebalancierung: Monatlich (erster Handelstag jedes Monats im Testzeitraum)
- Commission: 10.0 bps
- Spread-Weight: 0.25, Impact-Weight: 0.5
- Startkapital: 100,000 USD

**Faktor-Verfügbarkeit in diesem Test:**
- Aktiv (aus OHLCV berechnet): EMA-Spread, MA200-Position, RSI, OBV-Trend, Breadth,
  Bollinger %B, Stochastic, ADX, MACD-Histogramm, Volatilitäts-Regime (soweit TA-Spalten passen)
- Degradiert auf 0.0 (kein Altdata): Earnings-Surprise, Insider, News-Sentiment,
  Makro-Faktoren, Intermarkt, VIX/Put-Call-Options, Congress, GPR, Buyback

---

## Ergebnisse pro Fold

| Fold | Train | Test | CAGR | Sharpe | MaxDD | SPY-CAGR | SPY-Sharpe | Schlägt SPY? |
|------|-------|------|------|--------|-------|----------|------------|-------------|
| 1 | 2018-01-01–2018-09-10 | 2018-09-10–2019-05-20 | 4.9% | 0.21 | -20.1% | -1.1% | 0.02 | Ja |
| 2 | 2018-09-10–2019-05-20 | 2019-05-20–2020-01-27 | 40.0% | 1.38 | -9.1% | 23.8% | 1.85 | Ja |
| 3 | 2019-05-20–2020-01-27 | 2020-01-27–2020-10-05 | 42.4% | 0.73 | -27.4% | 4.6% | 0.31 | Ja |
| 4 | 2020-01-27–2020-10-05 | 2020-10-05–2021-06-14 | 4.4% | 0.19 | -32.8% | 38.2% | 2.31 | Nein |
| 5 | 2020-10-05–2021-06-14 | 2021-06-14–2022-02-21 | -26.1% | -0.68 | -26.2% | 3.1% | 0.28 | Nein |
| 6 | 2021-06-14–2022-02-21 | 2022-02-21–2022-10-31 | -8.6% | -0.12 | -25.4% | -13.4% | -0.44 | Ja |
| 7 | 2022-02-21–2022-10-31 | 2022-10-31–2023-07-10 | 41.9% | 1.14 | -9.0% | 20.6% | 1.18 | Ja |
| 8 | 2022-10-31–2023-07-10 | 2023-07-10–2024-03-18 | 1.0% | 0.10 | -21.5% | 23.9% | 1.90 | Nein |
| 9 | 2023-07-10–2024-03-18 | 2024-03-18–2024-11-25 | -0.8% | 0.07 | -29.7% | 24.0% | 1.75 | Nein |
| 10 | 2024-03-18–2024-11-25 | 2024-11-25–2025-08-04 | 29.9% | 0.57 | -28.5% | 6.1% | 0.37 | Ja |

_Erfolgreiche Folds: 10/10_

---

## Aggregierte OOS-Metriken

| Metrik | multifactor_v2 | SPY Buy-and-Hold |
|--------|---------------|-----------------|
| Ø CAGR | 12.9% | 13.0% |
| Ø Sharpe | 0.36 | 0.95 |
| Ø MaxDD | -23.0% | — |
| Win-Rate (CAGR > 0) | 70.0% | — |
| Folds, die SPY schlagen | 60.0% | — |

---

## Bewertung

multifactor_v2 schlägt SPY in 60.0% der Folds (Ø CAGR 12.9% vs. SPY 13.0%). Sharpe Ø 0.36. Das Ergebnis ist **gemischt**. Wichtige Einschränkung: 19 von 34 Faktoren degradierten auf 0.0 (kein Altdata), sodass dieser Test nur den TA-Subset von mfv2 misst.

### Einschränkungen dieses Reports

- **Haupteinschränkung:** 19/34 Faktoren = 0.0 (kein Altdata aus Alpaca). Dieser Test   misst nur den TA-Subset (EMA-Spread, OBV, RSI, Bollinger, ADX, MACD, Breadth).
- Monatliche Rebalancierung (≠ tägliche Rebalancierung im PaperPilot).
- compute_signals gibt tail(1) zurück → muss pro Monats-Rebalancing-Datum separat aufgerufen werden.
- Alpaca Free Tier: Survivorship-Bias möglich (delisted Symbole fehlen).
- SPY-Vergleich: kein Dividenden-Reinvest.

---

_Dieses Dokument ist ein automatisch erzeugtes Artefakt aus_ `scripts/_oos_wf_mfv2.py`. _Nicht manuell editieren._