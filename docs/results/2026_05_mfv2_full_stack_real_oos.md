# multifactor_v2 Full-Stack — Echter OOS Walk-Forward (Alpaca, 2026-05)

> ⚠️ **SUPERSEDED (2026-06-04):** Diese Zahlen wurden VOR dem Look-Ahead-Fix
> `fd8a192c` (Anti-Pattern E-038) berechnet. Die mfv2-Macro-Faktoren
> (`macro_growth_momentum` w=0.03, `macro_inflation_surprise` w=0.02) liefen über
> einen Macro-Loader mit Forward-Shift-Look-Ahead, der zu alte Beobachtungen ins
> Lookback-Fenster zog; jedes Fold-Signal hängt daran, also sind Per-Fold- und
> Aggregat-Zeilen unzuverlässig. **GO_LIVE_CHECKLIST ist NICHT betroffen** (keine
> Macro-Abhängigkeit). Re-Run (`scripts/_oos_wf_mfv2_full.py`) ist aufgesetzt, aber
> bewusst NICHT ausgeführt: der ursprüngliche Alpaca-Preis-Cache existiert nicht mehr
> (kein byte-reproduzierbares Before/After) und die mfv2-Produktionsgewichtung ist
> derzeit ~0. Re-Run erst bei erneuter mfv2-Produktionsbetrachtung. Siehe KNOWN_ISSUES.md.
>
> **UPDATE (2026-06-06) — Re-Run ausgeführt, Zweifel ausgeräumt.** Der Re-Run lief jetzt
> (frische Alpaca-Daten, da der Original-Cache fehlt → Code+Daten kombiniert, offengelegt).
> Ergebnis: Ø CAGR **10.7% unverändert**, Ø Sharpe **0.36→0.37**, Sharpe-Δ vs TA-only
> **+0.00→+0.01** — der Look-Ahead-Fix (E-038) ist **immateriell**. Die obigen Zahlen waren in
> der **Schlussfolgerung nie unzuverlässig**: mfv2 Full-Stack hat mit oder ohne Macro-Look-Ahead
> keinen risk-adjustierten Edge gegenüber SPY (0.37 vs 0.95); die Produktionsgewichtung bleibt ~0.
> Vollständiger Vergleich: `docs/results/2026_06_oos_edge_rerun_postsweep.md`.

**Erstellt:** 2026-05-28  
**Branch:** main  
**Zweck:** GO_LIVE_CHECKLIST Paket 3c.2 — mfv2 mit vollem verfügbarem Faktor-Stack.

---

## Datenquelle

- **Anbieter:** Alpaca Markets (Free Tier) + yfinance (ETF-Panel)
- **Angefordertes Universum:** 195 Symbole (watchlist.txt)
- **Symbole mit Alpaca-Daten:** 194
- **Tatsächliche Zeitspanne:** 2016-11-28 → 2025-12-30
- **SPY:** Buy-and-Hold-Benchmark

## Walk-Forward-Konfiguration

- Modus: Rolling
- Train-Fenster: 252 Handelstage (~1 Jahr)
- Test-Fenster: 252 Handelstage (~1 Jahr)
- Schrittweite: 252 Handelstage
- Warmup-Buffer: 250 Bars
- Rebalancierung: Monatlich (einheitlicher Kalender-Anker)
- Commission: 10.0 bps, Spread-Weight: 0.25, Impact-Weight: 0.5
- Startkapital: 100,000 USD

**Aktive Faktoren in diesem Test:**
- 9 von ~35 Faktoren aktiv (basierend auf Factor-Audit 2023-06-30)
- Neu aktiviert (ACTIVE im Factor-Audit 2023-06-30): macro_growth_momentum_z,
  macro_inflation_surprise_z, intermarket_credit_spread, intermarket_yield_curve,
  geo_risk_composite
- Weiterhin 0.0: earnings_surprise_z (altdata_loader gap), sector_rotation_bias
  (security_meta/ETF-Wiring-Gap), news_sentiment (Daten erst ab 2025-12),
  insider (all unknown), options/VIX (live-only), congress/buyback (keine Daten)

---

## Ergebnisse pro Fold

| Fold | Train | Test | CAGR | Sharpe | MaxDD | SPY-CAGR | SPY-Sharpe | Schlägt SPY? |
|------|-------|------|------|--------|-------|----------|------------|-------------|
| 1 | 2018-01-01–2018-09-10 | 2018-09-10–2019-05-20 | 4.7% | 0.23 | -16.1% | -1.1% | 0.02 | Ja |
| 2 | 2018-09-10–2019-05-20 | 2019-05-20–2020-01-27 | 30.9% | 1.37 | -7.3% | 23.8% | 1.85 | Ja |
| 3 | 2019-05-20–2020-01-27 | 2020-01-27–2020-10-05 | 35.9% | 0.76 | -22.3% | 4.6% | 0.31 | Ja |
| 4 | 2020-01-27–2020-10-05 | 2020-10-05–2021-06-14 | 4.4% | 0.19 | -27.1% | 38.2% | 2.31 | Nein |
| 5 | 2020-10-05–2021-06-14 | 2021-06-14–2022-02-21 | -21.2% | -0.69 | -21.4% | 3.1% | 0.28 | Nein |
| 6 | 2021-06-14–2022-02-21 | 2022-02-21–2022-10-31 | -6.4% | -0.12 | -20.3% | -13.4% | -0.44 | Ja |
| 7 | 2022-02-21–2022-10-31 | 2022-10-31–2023-07-10 | 33.9% | 1.17 | -7.1% | 20.6% | 1.18 | Ja |
| 8 | 2022-10-31–2023-07-10 | 2023-07-10–2024-03-18 | 0.9% | 0.09 | -17.4% | 23.9% | 1.90 | Nein |
| 9 | 2023-07-10–2024-03-18 | 2024-03-18–2024-11-25 | -0.2% | 0.06 | -24.3% | 24.0% | 1.75 | Nein |
| 10 | 2024-03-18–2024-11-25 | 2024-11-25–2025-08-04 | 24.4% | 0.56 | -23.2% | 6.1% | 0.37 | Ja |

_Erfolgreiche Folds: 10/10_

---

## Aggregierte OOS-Metriken

| Metrik | mfv2 Full-Stack | mfv2 TA-only (3b) | SPY Buy-and-Hold |
|--------|----------------|------------------|-----------------|
| Ø CAGR | 10.7% | +12.9% | 13.0% |
| Ø Sharpe | 0.36 | 0.36 | 0.95 |
| Ø MaxDD | -18.6% | -23.0% | — |
| Win-Rate (CAGR > 0) | 70.0% | 70% | — |
| Folds vs SPY | 60.0% | 60% | — |

---

## Bewertung

mfv2 Full-Stack schlägt SPY in 60.0% der Folds (Ø CAGR 10.7% vs. SPY 13.0%). Sharpe Δ gegenüber TA-only: +0.00. Das Ergebnis ist **gemischt**.

### Vergleich zu TA-only Baseline (Paket 3b)

- TA-only Baseline (Paket 3b): Ø CAGR +12.9%, Ø Sharpe 0.36, 6/10 Folds vs SPY
- Full-Stack (Paket 3c.2): Messung ob aktivierte Faktoren Verbesserung bringen.
- Sharpe-Delta: +0.00 — unverändert durch Altdata-Stack.

### Einschränkungen

- News-Sentiment (F21/F22) = 0.0 in allen historischen Folds (Daten erst ab 2025-12-22).
- Insider-Faktoren (F20/F32) = 0.0 (insider_trading.parquet enthält nur 'unknown' Trades).
- VIX/Options-Faktoren (F28/F29/F35) = 0.0 (live CBOE-Fetch, kein historisches Parquet).
- Congress/Buyback (F30/F34) = 0.0 (keine Datendateien vorhanden).
- Alpaca Free Tier: Survivorship-Bias möglich.
- SPY-Benchmark ohne Dividenden-Reinvest.

---

_Quelldokumente:_ `docs/results/2026_05_mfv2_factor_activation_log.md`  
_Skript:_ `scripts/_oos_wf_mfv2_full.py`  
_Nicht manuell editieren._