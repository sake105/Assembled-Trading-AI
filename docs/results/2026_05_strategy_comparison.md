# Strategievergleich — OOS Walk-Forward (Alpaca, 2026-05)

**Erstellt:** 2026-05-27  
**Branch:** main @ cc563605  
**Zweck:** Paket 3b — Vergleich aller drei aktiven Strategien auf gleicher Datenbasis.

Alle drei Strategien wurden mit identischer Methodik getestet:
- Alpaca Tageskurse, 194 Symbole (watchlist.txt), 2018–2025
- 10-Fold Rolling Walk-Forward: 252 Tage Train / 252 Tage Test / 252 Tage Schritt
- Commission 10 bps, spread_w=0.25, impact_w=0.5
- Startkapital 100.000 USD
- SPY Buy-and-Hold als Benchmark (identische Kostenbasis)

---

## Ergebnisübersicht

| Strategie | Ø CAGR | Ø Sharpe | Ø MaxDD | Win-Rate (CAGR>0) | Folds vs SPY | Einschränkung |
|-----------|--------|----------|---------|-------------------|--------------|---------------|
| **SPY Buy-and-Hold** | +13.0% | 0.95 | — | — | — | Benchmark |
| **multifactor_v2** | +12.9% | 0.36 | -23.0% | 70% | 6/10 (60%) | TA-only: 19/34 Faktoren = 0.0 |
| **trend_baseline** | -6.1% | -0.18 | -22.2% | 50% | 0/10 (0%) | Vollständig getestet |
| **multifactor_long_short** | -19.5% | -0.80 | -22.3% | 0% | 0/10 (0%) | Long-only, macro_world_etfs_core_bundle |

---

## Fold-Detail: multifactor_v2

| Fold | Test-Zeitraum | CAGR | Sharpe | MaxDD | SPY-CAGR | Schlägt SPY? |
|------|--------------|------|--------|-------|----------|-------------|
| 1 | 2018-09-10–2019-05-20 | +4.9% | 0.21 | -20.1% | -1.1% | Ja |
| 2 | 2019-05-20–2020-01-27 | +40.0% | 1.38 | -9.1% | +23.8% | Ja |
| 3 | 2020-01-27–2020-10-05 | +42.4% | 0.73 | -27.4% | +4.6% | Ja |
| 4 | 2020-10-05–2021-06-14 | +4.4% | 0.19 | -32.8% | +38.2% | Nein |
| 5 | 2021-06-14–2022-02-21 | -26.1% | -0.68 | -26.2% | +3.1% | Nein |
| 6 | 2022-02-21–2022-10-31 | -8.6% | -0.12 | -25.4% | -13.4% | Ja |
| 7 | 2022-10-31–2023-07-10 | +41.9% | 1.14 | -9.0% | +20.6% | Ja |
| 8 | 2023-07-10–2024-03-18 | +1.0% | 0.10 | -21.5% | +23.9% | Nein |
| 9 | 2024-03-18–2024-11-25 | -0.8% | 0.07 | -29.7% | +24.0% | Nein |
| 10 | 2024-11-25–2025-08-04 | +29.9% | 0.57 | -28.5% | +6.1% | Ja |

## Fold-Detail: trend_baseline

| Fold | Test-Zeitraum | CAGR | Sharpe | MaxDD | SPY-CAGR | Schlägt SPY? |
|------|--------------|------|--------|-------|----------|-------------|
| 1 | 2018-09-10–2019-05-20 | -31.3% | -1.68 | -24.6% | -1.1% | Nein |
| 2 | 2019-05-20–2020-01-27 | +1.4% | 0.13 | -14.3% | +23.8% | Nein |
| 3 | 2020-01-27–2020-10-05 | -6.3% | 0.04 | -29.3% | +4.6% | Nein |
| 4 | 2020-10-05–2021-06-14 | +27.2% | 1.05 | -10.3% | +38.2% | Nein |
| 5 | 2021-06-14–2022-02-21 | -27.4% | -1.50 | -21.2% | +3.1% | Nein |
| 6 | 2022-02-21–2022-10-31 | -33.6% | -0.88 | -43.4% | -13.4% | Nein |
| 7 | 2022-10-31–2023-07-10 | +11.2% | 0.44 | -20.0% | +20.6% | Nein |
| 8 | 2023-07-10–2024-03-18 | +4.4% | 0.26 | -13.4% | +23.9% | Nein |
| 9 | 2024-03-18–2024-11-25 | +21.4% | 0.98 | -11.0% | +24.0% | Nein |
| 10 | 2024-11-25–2025-08-04 | -28.0% | -0.64 | -34.2% | +6.1% | Nein |

## Fold-Detail: multifactor_long_short

| Fold | Test-Zeitraum | CAGR | Sharpe | MaxDD | SPY-CAGR | Schlägt SPY? |
|------|--------------|------|--------|-------|----------|-------------|
| 1 | 2018-09-10–2019-05-20 | -22.0% | -1.01 | -17.0% | -1.1% | Nein |
| 2 | 2019-05-20–2020-01-27 | -10.0% | -0.59 | -11.3% | +23.8% | Nein |
| 3 | 2020-01-27–2020-10-05 | -32.5% | -0.90 | -32.3% | +4.6% | Nein |
| 4 | 2020-10-05–2021-06-14 | -23.2% | -0.88 | -17.3% | +38.2% | Nein |
| 5 | 2021-06-14–2022-02-21 | -29.6% | -1.57 | -23.1% | +3.1% | Nein |
| 6 | 2022-02-21–2022-10-31 | -30.8% | -1.28 | -30.1% | -13.4% | Nein |
| 7 | 2022-10-31–2023-07-10 | -21.6% | -0.84 | -23.7% | +20.6% | Nein |
| 8 | 2023-07-10–2024-03-18 | -0.7% | 0.02 | -22.6% | +23.9% | Nein |
| 9 | 2024-03-18–2024-11-25 | -24.6% | -1.04 | -23.7% | +24.0% | Nein |
| 10 | 2024-11-25–2025-08-04 | -0.1% | 0.09 | -22.4% | +6.1% | Nein |

---

## Schlussfolgerung

**Keine der drei Strategien zeigt einen robusten, reproduzierbaren OOS-Edge gegenüber SPY Buy-and-Hold.**

Das beste Einzelergebnis liefert **multifactor_v2** mit Ø CAGR +12.9% (≈ SPY) und 6/10 gewonnenen Folds — jedoch mit drei entscheidenden Einschränkungen: (1) Der Test ist ein degradierter TA-only-Test: 19 der 34 Faktoren (Earnings, Insider, News, Makro, GPR, VIX, Options, Congress, Buyback) degradierten auf 0.0, weil Alpaca keine Fundamentaldaten liefert. Das echte mfv2 mit vollem Altdata-Stack ist damit OOS **nicht bewertet**. (2) Der Sharpe (0.36 vs. SPY 0.95) zeigt deutlich höhere Volatilität der Renditen. (3) Die positiven Folds (2, 3, 7, 10) kommen alle aus Momentum-Phasen; die negativen (5, 8, 9) aus Seitwärts-/Bear-Phasen — klassisches Momentum-Risikoprofil ohne Downside-Schutz.

**trend_baseline** verliert alle 10 Folds gegen SPY (Ø CAGR -6.1%) und ist vollständig getestet. Kein statistischer Edge nachweisbar.

**multifactor_long_short** produziert das schlechteste Ergebnis (Ø CAGR -19.5%, 0/10 Folds). Das ist jedoch kein fairer Test: Die Strategie ist für Long-Short ausgelegt; der Long-only-Subset des Momentum-Rankings addiert negative Selektion, weil die besten Momentum-Titel oft bereits teuer bewertet sind. Ohne Short-Engine kein sinnvoller Vergleich.

**Konkrete Empfehlung:** Keine der Strategien ist in diesem Testzustand für Go-Live geeignet. Wenn eine Strategie priorisiert werden soll, bietet multifactor_v2 das beste Ausgangsprofil — aber nur mit vollem Altdata-Stack (News, Makro, GPR) und einem dedizierten OOS-Lauf auf vollständiger Datenbasis. trend_baseline benötigt vor Go-Live eine Kostensenkung, ein engeres Universum oder eine Exposure-Steuerung, die Bear-Phasen abfedert.

---

## Einschränkungen dieses Vergleichs

- **multifactor_v2:** TA-only (19/34 Faktoren inaktiv) — kein vollständiger Test.
- **multifactor_long_short:** Long-only (Short-Seite nicht getestet) — kein vollständiger Test.
- **trend_baseline:** Vollständig getestet, aber ohne Altdata-Faktoren (by design).
- Alle Tests: Alpaca Free Tier (Survivorship-Bias), kein Corporate-Actions-Adjust, kein Dividenden-Reinvest im SPY-Benchmark.
- Monatliche Rebalancierung bei mfv2 und mfv_long_short (tägliche Rebalancierung im PaperPilot).
- 10 Folds × 1 Jahr sind statistisch begrenzt — keine Signifikanzaussage möglich.

---

_Quelldokumente:_
- `docs/results/2026_05_trend_baseline_real_oos.md`
- `docs/results/2026_05_multifactor_v2_real_oos.md`
- `docs/results/2026_05_multifactor_long_short_real_oos.md`

_Skripte: `scripts/_oos_wf_trend_baseline.py`, `scripts/_oos_wf_mfv2.py`, `scripts/_oos_wf_mfv_long_short.py`_
