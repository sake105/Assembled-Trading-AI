# multifactor_long_short — Echter OOS Walk-Forward (Alpaca, 2026-05)

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
- Warmup-Buffer: 300 Bars (trailing_momentum_12m initialisiert)
- Rebalancierung: Monatlich (erster Handelstag jedes Monats im Testzeitraum)
- **Factor Bundle:** `macro_world_etfs_core_bundle.yaml` (factor_set=core+vol_liquidity)
- **Aktive Faktoren:** trailing_momentum_12m_excl_1m (30%), trend_strength_200 (25%),
  trend_strength_50 (20%), realized_volatility_20 (15%, negativ), trailing_returns_12m (10%)
- **Getestet:** Long-only (TOP 20% quantile). Short-Seite (BOTTOM 20%) nicht in Backtest.
- Commission: 10.0 bps
- Spread-Weight: 0.25, Impact-Weight: 0.5
- Startkapital: 100,000 USD

---

## Ergebnisse pro Fold

| Fold | Train | Test | CAGR | Sharpe | MaxDD | SPY-CAGR | SPY-Sharpe | Schlägt SPY? |
|------|-------|------|------|--------|-------|----------|------------|-------------|
| 1 | 2018-01-01–2018-09-10 | 2018-09-10–2019-05-20 | -22.0% | -1.01 | -17.0% | -1.1% | 0.02 | Nein |
| 2 | 2018-09-10–2019-05-20 | 2019-05-20–2020-01-27 | -10.0% | -0.59 | -11.3% | 23.8% | 1.85 | Nein |
| 3 | 2019-05-20–2020-01-27 | 2020-01-27–2020-10-05 | -32.5% | -0.90 | -32.3% | 4.6% | 0.31 | Nein |
| 4 | 2020-01-27–2020-10-05 | 2020-10-05–2021-06-14 | -23.2% | -0.88 | -17.3% | 38.2% | 2.31 | Nein |
| 5 | 2020-10-05–2021-06-14 | 2021-06-14–2022-02-21 | -29.6% | -1.57 | -23.1% | 3.1% | 0.28 | Nein |
| 6 | 2021-06-14–2022-02-21 | 2022-02-21–2022-10-31 | -30.8% | -1.28 | -30.1% | -13.4% | -0.44 | Nein |
| 7 | 2022-02-21–2022-10-31 | 2022-10-31–2023-07-10 | -21.6% | -0.84 | -23.7% | 20.6% | 1.18 | Nein |
| 8 | 2022-10-31–2023-07-10 | 2023-07-10–2024-03-18 | -0.7% | 0.02 | -22.6% | 23.9% | 1.90 | Nein |
| 9 | 2023-07-10–2024-03-18 | 2024-03-18–2024-11-25 | -24.6% | -1.04 | -23.7% | 24.0% | 1.75 | Nein |
| 10 | 2024-03-18–2024-11-25 | 2024-11-25–2025-08-04 | -0.1% | 0.09 | -22.4% | 6.1% | 0.37 | Nein |

_Erfolgreiche Folds: 10/10_

---

## Aggregierte OOS-Metriken

| Metrik | multifactor_long_short | SPY Buy-and-Hold |
|--------|----------------------|-----------------|
| Ø CAGR | -19.5% | 13.0% |
| Ø Sharpe | -0.80 | 0.95 |
| Ø MaxDD | -22.3% | — |
| Win-Rate (CAGR > 0) | 0.0% | — |
| Folds, die SPY schlagen | 0.0% | — |

---

## Bewertung

multifactor_long_short (Long-only) schlägt SPY nur in 0.0% der Folds (Ø CAGR -19.5% vs. SPY 13.0%). Sharpe Ø -0.80. Das Ergebnis ist **negativ**. Das Momentum-Ranking des macro_world_etfs_core_bundle liefert im Long-only-Modus keinen robusten Mehrwert gegenüber SPY. Die Short-Seite wurde nicht getestet; das Long-Short-Gesamtergebnis kann abweichen.

### Einschränkungen dieses Reports

- **Long-only:** SHORT-Seite (BOTTOM-20%-Quantile) nicht in Backtest einbezogen.
  Ein vollständiger Long-Short-Backtest würde einen dedizierten Short-Selling-fähigen Engine benötigen.
- **Bundle:** macro_world_etfs_core_bundle (OHLCV-only). Andere Bundles (ai_tech, alternative_risk_premia)
  könnten andere Ergebnisse liefern, erfordern aber ggf. Altdata.
- Monatliche Rebalancierung (≠ höhere Frequenz bei aktivem Betrieb). Dieser Report wurde mit
  per-Symbol-Monatsanker erzeugt (erster verfügbarer Bar des Monats je Symbol). Skript-Fix
  (2026-05-27): einheitlicher Kalenderanker pro Monat über alle Symbole. Praktischer Effekt gering
  (Alpaca-Symbole teilen denselben Handelstag), Richtungsaussage bleibt unverändert.
- Alpaca Free Tier: Survivorship-Bias möglich (delisted Symbole fehlen).
- SPY-Vergleich: kein Dividenden-Reinvest.

---

_Dieses Dokument ist ein automatisch erzeugtes Artefakt aus_ `scripts/_oos_wf_mfv_long_short.py`. _Nicht manuell editieren._