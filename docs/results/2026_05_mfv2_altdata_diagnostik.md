# multifactor_v2 — Altdata-Diagnostik (2026-05)

**Erstellt:** 2026-05-27  
**Branch:** main @ bda5826f  
**Zweck:** Paket 3c.1 — Reiner Analyse-Auftrag. Kein Code geändert.  
**Basis:** `src/assembled_core/strategies/multifactor_v2.py`, lokale Dateiprüfung, OOS-Ergebnis `2026_05_multifactor_v2_real_oos.md`

---

## Schritt 1 — Faktor-Inventar (34 Faktoren)

### Legende
- **OOS-Status:** AKTIV = im letzten Walk-Forward tatsächlich berechnet (Alpaca OHLCV-Basis). STILL = auf 0.0 degradiert.  
- **ZEROED** = Policy-Entscheidung, unabhängig von Datenlage (weight=0.0 in DEFAULT_V2_WEIGHTS).

---

### Gruppe A: TA / Trend (Faktoren 1–4)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 1 | trend_ema_spread | `multifactor_v1._compute_ema_spread()` — close | TA/Trend | **AKTIV** |
| 2 | trend_ma200_position | `multifactor_v1._ma_position()` — close, sma_200 (add_all_features) | TA/Trend | **AKTIV** |
| 3 | trend_adx_strength | `multifactor_v1._safe_col()` — ta_adx_v1 (add_all_features) | TA/Trend | **AKTIV** |
| 4 | trend_macd_hist | `multifactor_v1._safe_col()` — ta_macd_hist_v1 (add_all_features) | TA/Trend | **AKTIV** |

### Gruppe B: TA / Momentum (Faktoren 5–7)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 5 | mom_rsi_centered | `multifactor_v1._rsi_score()` — ta_rsi_v1 (add_all_features) | TA/Momentum | **AKTIV** |
| 6 | mom_volume_weighted | `multifactor_v1._safe_col()` — ta_vol_weighted_mom_20d_v1 (add_all_features) | TA/Momentum | **AKTIV** |
| 7 | mom_obv_trend | `multifactor_v1._obv_trend()` — close, volume | TA/Momentum | **AKTIV** |

### Gruppe C: TA / Mean-Reversion (Faktoren 8–9, 16–17)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 8 | mr_bollinger_pctb | `multifactor_v1._bollinger_score()` — ta_bollinger_pctb_v1 | TA/MR | **AKTIV** |
| 9 | mr_stoch_oversold | `multifactor_v1._stochastic_score()` — ta_stoch_pctk_v1 | TA/MR | **AKTIV** |
| 16 | mr_zscore_reversal_3d | `features/mean_reversion_factors.compute_mean_reversion_factors()` — close (3-Tages-Return-Zscore) | TA/MR | **AKTIV** |
| 17 | mr_rsi_extreme_uptrend | `features/mean_reversion_factors.compute_mean_reversion_factors()` — ta_rsi_v1, RSI<30 im Uptrend | TA/MR | **AKTIV** |

### Gruppe D: TA / Volume (Faktoren 10–11)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 10 | vol_abnormal | `multifactor_v1._abnormal_volume_score()` — ta_volume_ratio_v1 | TA/Volume | **AKTIV** |
| 11 | vol_tick_imbalance | `multifactor_v1._safe_col()` — tick_imbalance_20d | TA/Volume | **STILL** — Spalte nicht in add_all_features |

### Gruppe E: TA / Volatilität (Faktoren 12–13)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 12 | vola_regime_score | `multifactor_v1._volatility_regime_score()` — hml_volume_20d, daily returns | TA/Volatilität | **AKTIV** |
| 13 | vola_vov_penalty | `multifactor_v1._vov_penalty()` — close, ta_atr_v1 | TA/Volatilität | **AKTIV** |

### Gruppe F: TA / Breadth (Faktoren 14–15)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 14 | breadth_above_ma | `multifactor_v1._compute_breadth_score()` — symbol, close (% über MA50 per Datum, cross-sectional) | TA/Breadth | **AKTIV** |
| 15 | breadth_ad_line | `multifactor_v2._compute_breadth_ad_slope()` — symbol, close (5-Tages-A/D-Slope) | TA/Breadth | **AKTIV** |

### Gruppe G: Sektor (Faktor 18)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 18 | sector_rotation_bias | `features/sector_rotation.compute_sector_scores()` — symbol, close; GICS-Zuordnung via `configs/security_meta.csv` (1 KB, 195 Symbole) | Sektor | **STILL** — security_meta.csv im OOS-Skript nicht geladen |

### Gruppe H: Earnings / Insider (Faktoren 19–22)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 19 | earnings_surprise_z | `data/altdata/finnhub_events.load_earnings_history()` — Finnhub EPS-Daten → `output/events_earnings.parquet` | Earnings | **STILL** — Parquet nicht ins OOS-Skript geladen |
| 20 | insider_activity_score | `data/altdata/finnhub_events.load_insider_filings()` — Finnhub Insider-Filings → `output/insider_trading.parquet` | Insider | **ZEROED** (Policy) — 59.506 Zeilen, 100% "unknown" classification |
| 21 | insider_cluster_score | `features/insider_cluster` — SEC EDGAR Form-4-Filings (≥3 Insider in 30 Tagen) | Insider | **STILL** — kein lokales Form-4-Parquet |
| 22 | pead_sue_score | `data/altdata/finnhub_events.load_earnings_history()` via `batch_sue()` — Finnhub | Earnings/PEAD | **STILL** — Parquet nicht ins OOS-Skript geladen |

### Gruppe I: News / Makro (Faktoren 23–26)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 23 | news_sentiment_7d | `data/altdata/finnhub_news_macro.load_news_sentiment()` → `output/news_sentiment_fused.parquet` | News | **STILL** — Datei nur ab 2025-12-22 (kein historisches Coverage 2018–2025) |
| 24 | news_volume_spike | Gleiche Quelle wie F23 | News | **STILL** — gleiche Coverage-Lücke |
| 25 | macro_growth_momentum | `data/altdata/finnhub_news_macro.load_macro_indicators()` → `output/macro.parquet` | Makro | **STILL** — macro.parquet nicht ins OOS-Skript geladen |
| 26 | macro_inflation_surprise | Gleiche Quelle wie F25 — Spalten: cpi_yoy, industrial_prod | Makro | **STILL** — gleiche Ursache |

### Gruppe J: Intermarkt (Faktoren 27–29)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 27 | intermarket_bond_equity | `features/intermarket_factors.build_intermarket_factors()` — TLT vs. SPY Preis-Momentum-Divergenz | Intermarkt | **STILL** — Intermarkt-Spalten nicht vorberechnet |
| 28 | intermarket_credit_spread | Gleiche Funktion — HYG vs. TLT Credit-Spread-Änderung 5-Tages | Intermarkt | **STILL** — gleiche Ursache |
| 29 | intermarket_yield_curve | Gleiche Funktion — Yield-Kurven-Slope (10Y − 2Y Treasury) aus `output/macro.parquet` Spalte yield_curve_spread | Intermarkt | **STILL** — macro.parquet nicht geladen |

### Gruppe K: Options / VIX (Faktoren 30–31)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 30 | options_put_call_extreme | `signals/options_derived_signals.build_options_regime_factors()` — CBOE Put-Call Ratio (CBOESource) | Options | **STILL** — kein lokales PCR-Parquet |
| 31 | vix_regime_score | Gleiche Funktion — VIX (252-Tages-Zscore) via CBOESource oder `output/macro.parquet` Spalte vix | VIX | **STILL** — VIX-Quelle nicht ins OOS-Skript geladen |

### Gruppe L: Congress (Faktor 32)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 32 | congress_activity | `features/congress_features.py` + `events/disclosures/fetch_house_ptr.py` — House PTR-Meldungen (öffentlich) | Congress | **ZEROED** (Policy) — keine lokalen Datendateien; Module existieren |

### Gruppe M: Geo-Risk / GPR (Faktor 33)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 33 | geo_risk_composite | `data/gpr` — Caldara-Iacoviello GPR-Index (monatlich, öffentlich) → `output/macro_gpr.parquet` | GPR | **STILL** — macro_gpr.parquet nicht ins OOS-Skript geladen |

### Gruppe N: Buyback (Faktor 34)

| # | Name | Datenquelle | Kategorie | OOS-Status |
|---|------|-------------|-----------|------------|
| 34 | buyback_drift_score | `signals/buyback_drift.buyback_signal_score()` — SEC 8-K-Filings (öffentlich) | Buyback | **STILL** — kein lokales Buyback-Parquet; Live-Fetch im Backtest deaktiviert |

---

## Schritt 2 — Datenquellen-Check (stille Faktoren)

### F11 — vol_tick_imbalance

| | |
|---|---|
| **Modul** | `src/assembled_core/strategies/multifactor_v1.py` — `_safe_col("tick_imbalance_20d")` |
| **Lokale Daten** | Keine — Spalte wird von `add_all_features()` nicht erzeugt |
| **API-Key** | Keiner — aus OHLCV berechenbar |
| **Limitierung** | `tick_imbalance_20d` ist eine intraday-abgeleitete Feature-Spalte. Die tägliche OHLCV-Pipeline berechnet sie nicht. |

### F18 — sector_rotation_bias

| | |
|---|---|
| **Modul** | `src/assembled_core/features/sector_rotation.py` |
| **Lokale Daten** | `configs/security_meta.csv` — 1 KB, GICS-Sektorzuordnung für 195 Symbole |
| **API-Key** | Keiner |
| **Limitierung** | File existiert. Im OOS-Skript nicht geladen (kein `strategy_cfg`-Parameter übergeben). |

### F19 + F22 — earnings_surprise_z, pead_sue_score

| | |
|---|---|
| **Modul** | `src/assembled_core/data/altdata/finnhub_events.py` |
| **Lokale Daten** | `output/events_earnings.parquet` — 51 KB, 1.087 Zeilen, 2020-01-29 bis 2026-08-05, zuletzt geändert 2026-05-03 |
| **API-Key** | `FINNHUB_API_KEY` (Free Tier: 60 req/min) |
| **Limitierung** | Historische Lücke 2018–2020 (2 Jahre der OOS-Periode nicht abgedeckt). Finnhub Free Tier erlaubt keinen unbegrenzten Historien-Abruf. |

### F21 — insider_cluster_score

| | |
|---|---|
| **Modul** | `src/assembled_core/features/` (insider_cluster-Modul) |
| **Lokale Daten** | Kein Form-4-Parquet. `output/insider_trading.parquet` (583 KB, 59.506 Zeilen, 2018–2026) existiert, ist aber 100% "unknown" und für diesen Faktor unbrauchbar. |
| **API-Key** | Keiner — SEC EDGAR ist öffentlich |
| **Limitierung** | Das EDGAR Form-4-Parquet wurde nie lokal aufgebaut. Live-Fetch im Backtest-Modus deaktiviert (PIT-Sicherheitsguard). |

### F23 + F24 — news_sentiment_7d, news_volume_spike

| | |
|---|---|
| **Modul** | `src/assembled_core/data/altdata/finnhub_news_macro.py` |
| **Lokale Daten** | `output/news_sentiment_fused.parquet` — 19 KB, 1.714 Zeilen, **nur 2025-12-22 bis 2026-05-21**, zuletzt geändert 2026-05-27 |
| **API-Key** | `FINNHUB_API_KEY`, `NEWSAPI_KEY` (100 req/Tag Free Tier), `POLYGON_API_KEY` (5 req/min) |
| **Limitierung** | Historische Abdeckung beginnt 2025-12. Kein rückwirkender Datenabruf für 2018–2025 möglich — Finnhub News Free Tier erlaubt maximal ~90 Tage Historie. Die gesamte OOS-Periode (2018–2025) bleibt für diese Faktoren unbewertet. |

### F25 + F26 — macro_growth_momentum, macro_inflation_surprise

| | |
|---|---|
| **Modul** | `src/assembled_core/data/altdata/finnhub_news_macro.py` (FRED-Loader) |
| **Lokale Daten** | `output/macro.parquet` — 438 KB, 20.533 Zeilen, **1919-01-01 bis 2026-05-18**, zuletzt geändert 2026-05-19. Enthält: `cpi_yoy`, `industrial_prod`, `m2_money_supply`, `usd_index`, `wti_crude_oil`, `fed_funds_rate`, `unemployment_rate`, `initial_claims`. |
| **API-Key** | `FRED_API_KEY` (Free, 120 req/min) |
| **Limitierung** | Keine — Daten decken die gesamte OOS-Periode ab. Im OOS-Skript nicht geladen. |

### F27 + F28 — intermarket_bond_equity, intermarket_credit_spread

| | |
|---|---|
| **Modul** | `src/assembled_core/features/intermarket_factors.py` |
| **Lokale Daten** | TLT und HYG als Symbole in `output/oos_alpaca_prices_cache.parquet` (7.949 KB). Alpaca-Preise für diese ETFs ab 2018 vorhanden. |
| **API-Key** | Keiner — ETF-Preisdaten aus existierendem Alpaca-Cache |
| **Limitierung** | `build_intermarket_factors()` erwartet TLT/SPY/HYG als Spalten in einem vorberechneten Intermarkt-Panel. Ob TLT und HYG im 194-Symbol-Panel enthalten sind, muss beim Wiring geprüft werden. |

### F29 — intermarket_yield_curve

| | |
|---|---|
| **Modul** | `src/assembled_core/features/intermarket_factors.py` |
| **Lokale Daten** | `output/macro.parquet` — Spalte `yield_curve_spread` (treasury_10y − treasury_2y), vollständig ab 1919 |
| **API-Key** | Keiner (Daten in macro.parquet via FRED) |
| **Limitierung** | Keine — nur Wiring fehlt. |

### F30 — options_put_call_extreme

| | |
|---|---|
| **Modul** | `src/assembled_core/signals/options_derived_signals.py` — `CBOESource` |
| **Lokale Daten** | Kein PCR-Parquet vorhanden. VIX/VIX3M-Preise existieren in `data/raw/equities_eod/yfinance/VIX.parquet` (28 KB) und `VIX3M.parquet` (28 KB), aber **kein Put-Call Ratio** |
| **API-Key** | Keiner — CBOE PCR ist öffentlich downloadbar (cboe.com) |
| **Limitierung** | CBOE bietet historische PCR-Daten nur manuell als CSV-Download. Kein Live-API. Historisches Parquet wurde nie aufgebaut. |

### F31 — vix_regime_score

| | |
|---|---|
| **Modul** | `src/assembled_core/signals/options_derived_signals.py` — `CBOESource` oder macro.parquet |
| **Lokale Daten** | `output/macro.parquet` Spalte `vix` (ab 1919-01-01). Zusätzlich: `data/raw/equities_eod/yfinance/VIX.parquet` (28 KB). |
| **API-Key** | Keiner |
| **Limitierung** | Keine — zwei lokale Quellen vorhanden. Nur Wiring fehlt. |

### F32 — congress_activity

| | |
|---|---|
| **Modul** | `src/assembled_core/features/congress_features.py` (8 KB), `src/assembled_core/events/disclosures/fetch_house_ptr.py` (21 KB) |
| **Lokale Daten** | **Keine Parquet-Dateien.** mypy-Cache-Artefakte (`fetch_house_ptr.data.json`) sind keine Nutzdaten. |
| **API-Key** | Keiner — House PTR-Meldungen sind auf disclosures.house.gov öffentlich |
| **Limitierung** | Modul implementiert, historische Daten nie heruntergeladen. PTR-Meldungen ab 2012 verfügbar, aber kein automatischer Fetcher für Massenabruf. ZEROED by policy (weight=0.0). |

### F33 — geo_risk_composite

| | |
|---|---|
| **Modul** | `src/assembled_core/data/gpr/` (GPR-Feeder) |
| **Lokale Daten** | `output/macro_gpr.parquet` — 38 KB, 1.516 Zeilen monatlich, Vollabdeckung bis 2026-05, zuletzt geändert 2026-05-23. Spalten: `gpr_index`, `gpr_threats`, `gpr_acts`, `gpr_historical`. |
| **API-Key** | Keiner — Caldara-Iacoviello GPR-Index ist öffentlich (matteoiacoviello.com, monatliche Excel-Datei) |
| **Limitierung** | Daten vorhanden. Monatliche Frequenz (kein Tages-GPR). Im OOS-Skript nicht geladen. |

### F34 — buyback_drift_score

| | |
|---|---|
| **Modul** | `src/assembled_core/signals/buyback_drift.py` (5 KB) |
| **Lokale Daten** | **Kein Buyback-Parquet.** Archiv-Kopie: `archive/observability_graveyard_2026q2/features/buyback_features.py` — ehemaliges Modul. |
| **API-Key** | Keiner — SEC 8-K-Filings öffentlich (EDGAR EFTS-API, kostenlos) |
| **Limitierung** | Kein historischer Cache aufgebaut. Live-Fetch im Backtest-Modus durch PIT-Guard deaktiviert. Faktor gewichtet mit 0.0 in Crisis-Regime. |

---

## Schritt 3 — Aktivierungsaufwand

| # | Name | Aktivierungsaufwand | Begründung |
|---|------|--------------------|-|
| 11 | vol_tick_imbalance | **DATEN FEHLEN** | Spalte tick_imbalance_20d nicht in add_all_features — separate intraday-Berechnung nötig |
| 18 | sector_rotation_bias | **SOFORT AKTIVIERBAR** | configs/security_meta.csv vorhanden (1 KB); nur Wiring ins OOS-Skript fehlt |
| 19 | earnings_surprise_z | **API-KEY NÖTIG** | Lokales Parquet vorhanden (2020–2026), aber Lücke 2018–2020; FINNHUB_API_KEY (Free: 60 req/min) für Backfill |
| 21 | insider_cluster_score | **DATEN FEHLEN** | Kein Form-4-Parquet aufgebaut; EDGAR öffentlich, aber kein Massenabruf-Fetcher aktiv |
| 22 | pead_sue_score | **API-KEY NÖTIG** | Gleiche Datenbasis wie F19; FINNHUB_API_KEY für historischen Abruf |
| 23 | news_sentiment_7d | **DATEN FEHLEN** | Parquet nur ab 2025-12; Finnhub Free Tier erlaubt kein Backfill 2018–2025 |
| 24 | news_volume_spike | **DATEN FEHLEN** | Gleiche Coverage-Lücke wie F23 |
| 25 | macro_growth_momentum | **SOFORT AKTIVIERBAR** | output/macro.parquet vorhanden (1919–2026-05); nur Wiring ins OOS-Skript fehlt |
| 26 | macro_inflation_surprise | **SOFORT AKTIVIERBAR** | Gleiche Quelle wie F25; cpi_yoy-Spalte in macro.parquet vorhanden |
| 27 | intermarket_bond_equity | **SOFORT AKTIVIERBAR** | TLT/SPY in oos_alpaca_prices_cache.parquet vorhanden; Wiring der Intermarkt-Spalten nötig |
| 28 | intermarket_credit_spread | **SOFORT AKTIVIERBAR** | HYG in oos_alpaca_prices_cache.parquet vorhanden (als Watchlist-Symbol) |
| 29 | intermarket_yield_curve | **SOFORT AKTIVIERBAR** | yield_curve_spread-Spalte in macro.parquet (treasury_10y − treasury_2y) vorhanden |
| 30 | options_put_call_extreme | **DATEN FEHLEN** | Kein PCR-Parquet; CBOE CSV-Download manuell, kein automatischer Fetcher |
| 31 | vix_regime_score | **SOFORT AKTIVIERBAR** | VIX-Daten in macro.parquet (Spalte vix, 1919–2026) und in data/raw/yfinance/VIX.parquet vorhanden |
| 32 | congress_activity | **DATEN FEHLEN** | ZEROED by policy; Modul vorhanden, kein Daten-Fetcher für Massenabruf aktiv |
| 33 | geo_risk_composite | **SOFORT AKTIVIERBAR** | macro_gpr.parquet vorhanden (38 KB, bis 2026-05); nur Wiring fehlt |
| 34 | buyback_drift_score | **DATEN FEHLEN** | Kein Buyback-Parquet; SEC 8-K öffentlich, aber Fetcher im Backtest-Modus deaktiviert |

---

## Schritt 4 — Übersichtstabelle

| Faktor | Kategorie | OOS AKTIV? | Aktivierungsaufwand | Kosten |
|--------|-----------|-----------|--------------------|-|
| 1 trend_ema_spread | TA/Trend | ✓ AKTIV | — | — |
| 2 trend_ma200_position | TA/Trend | ✓ AKTIV | — | — |
| 3 trend_adx_strength | TA/Trend | ✓ AKTIV | — | — |
| 4 trend_macd_hist | TA/Trend | ✓ AKTIV | — | — |
| 5 mom_rsi_centered | TA/Momentum | ✓ AKTIV | — | — |
| 6 mom_volume_weighted | TA/Momentum | ✓ AKTIV | — | — |
| 7 mom_obv_trend | TA/Momentum | ✓ AKTIV | — | — |
| 8 mr_bollinger_pctb | TA/MR | ✓ AKTIV | — | — |
| 9 mr_stoch_oversold | TA/MR | ✓ AKTIV | — | — |
| 10 vol_abnormal | TA/Volume | ✓ AKTIV | — | — |
| 11 vol_tick_imbalance | TA/Volume | ✗ STILL | DATEN FEHLEN | — |
| 12 vola_regime_score | TA/Volatilität | ✓ AKTIV | — | — |
| 13 vola_vov_penalty | TA/Volatilität | ✓ AKTIV | — | — |
| 14 breadth_above_ma | TA/Breadth | ✓ AKTIV | — | — |
| 15 breadth_ad_line | TA/Breadth | ✓ AKTIV | — | — |
| 16 mr_zscore_reversal_3d | TA/MR | ✓ AKTIV | — | — |
| 17 mr_rsi_extreme_uptrend | TA/MR | ✓ AKTIV | — | — |
| 18 sector_rotation_bias | Sektor | ✗ STILL | SOFORT AKTIVIERBAR | kostenlos |
| 19 earnings_surprise_z | Earnings | ✗ STILL | API-KEY NÖTIG | kostenlos (Finnhub Free) |
| 20 insider_activity_score | Insider | ZEROED | ZEROED (Policy) | — |
| 21 insider_cluster_score | Insider | ✗ STILL | DATEN FEHLEN | kostenlos (SEC EDGAR) |
| 22 pead_sue_score | Earnings | ✗ STILL | API-KEY NÖTIG | kostenlos (Finnhub Free) |
| 23 news_sentiment_7d | News | ✗ STILL | DATEN FEHLEN | kostenlos (Finnhub/NewsAPI Free) |
| 24 news_volume_spike | News | ✗ STILL | DATEN FEHLEN | kostenlos (gleiche Quellen) |
| 25 macro_growth_momentum | Makro | ✗ STILL | SOFORT AKTIVIERBAR | kostenlos |
| 26 macro_inflation_surprise | Makro | ✗ STILL | SOFORT AKTIVIERBAR | kostenlos |
| 27 intermarket_bond_equity | Intermarkt | ✗ STILL | SOFORT AKTIVIERBAR | kostenlos |
| 28 intermarket_credit_spread | Intermarkt | ✗ STILL | SOFORT AKTIVIERBAR | kostenlos |
| 29 intermarket_yield_curve | Intermarkt | ✗ STILL | SOFORT AKTIVIERBAR | kostenlos |
| 30 options_put_call_extreme | Options | ✗ STILL | DATEN FEHLEN | kostenlos (CBOE public) |
| 31 vix_regime_score | VIX | ✗ STILL | SOFORT AKTIVIERBAR | kostenlos |
| 32 congress_activity | Congress | ZEROED | DATEN FEHLEN | kostenlos (public) |
| 33 geo_risk_composite | GPR | ✗ STILL | SOFORT AKTIVIERBAR | kostenlos |
| 34 buyback_drift_score | Buyback | ✗ STILL | DATEN FEHLEN | kostenlos (SEC EDGAR) |

---

## Schlussfolgerung

Von 19 stillen Faktoren (OOS 2018–2025, Alpaca OHLCV-Basis) sind **8 sofort aktivierbar** (F18, F25, F26, F27, F28, F29, F31, F33 — lokale Daten vorhanden, nur Wiring ins Backtest-Skript fehlt), **2 benötigen einen kostenlosen API-Key** (F19, F22 — Finnhub Free Tier, partielles Parquet vorhanden), **7 haben Datenlücken** (F11, F21, F23, F24, F30, F32, F34 — Module vorhanden, kein historisches Parquet aufgebaut), und **keiner erfordert bezahlte Daten** (2 davon sind zusätzlich durch Policy auf weight=0.0 gesetzt: F20, F32).

---

_Quelldokumente: `docs/results/2026_05_multifactor_v2_real_oos.md`, `src/assembled_core/strategies/multifactor_v2.py`_  
_Lokale Dateichecks: `output/`, `data/`, `configs/`, `src/assembled_core/data/altdata/`_  
_Reiner Analyse-Auftrag — kein Code geändert._

---

## 2026-06-01 Forensik-Closure — `sector_rotation_bias` + `earnings_surprise_z`

**Auftrag:** Forensische Einzelfall-Prüfung der beiden Faktoren F18 (`sector_rotation_bias`) und
F19 (`earnings_surprise_z`) unter der harten Randbedingung **ausschließlich kostenlose Datenfeeds**.
Für jeden Faktor: behebbarer Bug oder Free-Feed-Decke? Bringt eine Behebung etwas? Wenn ja → fixen.
Wenn nein → ehrlich vermerken, dass es „auf dem kostenlosen Feed nichts wird", und benennen, ob ein
akzeptabler Decken-Zustand erreicht ist oder ob Entwicklungspfade verbleiben.

Diese Closure ersetzt die obigen Zeilen für F18/F19 in der Schlussfolgerung **nicht**, sondern
präzisiert sie nach forensischer Tiefenprüfung. Die ältere Tabellen-Einstufung „F18 SOFORT
AKTIVIERBAR" war zu optimistisch (siehe Tier A: Code-Aktivierung ≠ Produktions-Beitrag).

Code-/Config-Stand dieser Closure: `multifactor_v2.py`, `configs/factor_weights_by_regime.json`,
`scripts/train_regime_weights.py`, zugehörige Tests editiert. **Nur lokal geprüft (51 gezielte Tests
+ Smoke), CI unbestätigt. Kein Commit zum Zeitpunkt dieser Doku.**

### Tier A — `sector_rotation_bias`: Bug behoben, aber Produktions-Beitrag bleibt ~0 bis Re-Fit

**Root Causes (beide gefixt):**
- **RC#1 — fehlende Sektor-ETFs im Panel:** Die 8 SPDR-Sektor-ETFs (XLK/XLF/XLE/XLV/XLI/XLU/XLP/XLY)
  fehlten im stock-only Universe → `len < 3` → Faktor lieferte konstant `0.0`. Behoben durch
  PIT-gegateten Free-Store-Fallback (`_sector_prices_from_store(as_of)`): die ETF-Closes werden aus
  einem lokalen Offline-Store injiziert, **streng `as_of`-gesliced** und mit Live-Staleness-Guard
  (`SECTOR_STORE_STALE_DAYS = 7`) gegen Look-Ahead/veraltete Daten geschützt.
- **RC#2 — String-Mapping-Bruch:** `SECTOR_NAMES` („Technology"/„Healthcare") matchte nie die
  `security_meta.csv`-Sektorlabels („Information Technology"/„Health Care") für 2/8 Sektoren. Behoben
  durch `_SECTOR_NAME_CANON` + `_canon_sector()`.

**Was das BEDEUTET (ehrlich):**
- Der Code ist jetzt **feuerfähig** — Smoke-Test zeigt 6/8 Sektoren mit ±1.0-Tilt statt durchgehend 0.
- **ABER:** Die operativen Produktionsgewichte in `configs/factor_weights_by_regime.json` weisen
  `sector_rotation_bias` ~0 zu (bull/bear/crisis = 0.0, sideways ≈ 0.0048), **weil der Faktor tot
  war, als die Gewichte gefittet wurden**. Solange kein **Gewichts-Re-Fit** auf dem korrigierten
  Panel läuft, bleibt der reale Produktions-Beitrag praktisch 0 — egal dass der Code jetzt feuert.
- **ZUSÄTZLICH:** Der Offline-Sektor-Store wird nicht täglich aktualisiert. Im **Live-Modus** greift
  daher der Staleness-Guard und liefert neutral (0.0); nur **Backtests mit in-range `as_of`** üben
  den Faktor tatsächlich aus.
- **Free-Feed-Decke? NEIN.** Sektor-ETF-Preise (alle 8 + SPY) sind kostenlos und liegen vor. Der
  Limiter ist **nicht** der Feed, sondern (1) ein ausstehender Re-Fit und (2) die Live-Frische des
  Offline-Stores. Beide sind lösbar (siehe Follow-ups) — ohne Bezahl-Daten.

### Tier B — `earnings_surprise_z`: Wiring-Bug + ECHTE Free-Feed-Decke → genullt

**Root Causes:**
- **Wiring-Bug:** `altdata_loader.load_earnings_history` strippt `eps_actual`/`eps_estimate`, die
  `earnings_insider_wrapper._validate_columns` zwingend braucht → `ValueError` → still `0.0`
  (Silent-Degradation, E-025-Muster).
- **Free-Feed-Coverage-Wall (der eigentliche Show-Stopper):** EPS-Schätzungen sind auf dem freien
  Feed nur für **~44 Mega-Caps** gecacht → degenerierte Cross-Section. Selbst mit gefixtem Wiring
  bliebe der Faktor über das investierbare Universum quasi-konstant → kein verwertbares Cross-
  Sectional-Signal.

**Entscheidung & Wirkung:**
- `earnings_surprise_z` auf **weight = 0.0** gesetzt — in `DEFAULT_V2_WEIGHTS` (multifactor_v2.py)
  **und** in allen Regimes von `configs/factor_weights_by_regime.json` (crisis war bereits 0.0).
- **Runtime-neutral, doppelt inert:** Der Dead-Factor-Filter droppt den All-Zero-Faktor ohnehin am
  Scoring; zusätzlich trägt `weight = 0.0` exakt 0.0 zum Composite bei und addiert 0.0 zur
  `total_weight` (Renorm-sicher über BEIDE Konsumpfade). Das Nullen ändert kein Laufzeitverhalten —
  es entfernt eine **irreführende 21.6%-Bull-Gewichtung** und eine **Reaktivierungs-Landmine**.
- **Reaktivierungs-Schutz:** `scripts/train_regime_weights.py` force-zeroed `earnings_surprise_z`
  (+ `insider_activity_score`, `congress_activity`) jetzt **nach** dem IC-Fit
  (`INTENTIONALLY_ZEROED_FACTORS`), damit ein Retrain den Faktor nicht still wieder reaktiviert.
- **Free-Feed-Decke? JA, real.** Auf dem kostenlosen Feed „wird das nichts": die ~44-Mega-Cap-
  Abdeckung ist ein Feed-Ceiling, kein Code-Bug. Entwicklungspfad existiert **nur** mit einem
  **bezahlten EPS-Estimate-Feed** (siehe Follow-up iii).

### Tier C — Empirischer Anker (kein frisch gemessener Edge)

Aus **Paket 3c.2** (10-Fold OOS, mfv2 mit voll aktivierbarer Altdata): Ø Sharpe **0.36 == TA-only
0.36**, beide **unter SPY**. Weder F18 noch F19 dreht das SPY-Verdikt. Die in dieser Session
genannten Fire-/Smoke-Zahlen sind **Capability-Nachweise (Code feuert), kein neu gemessener Edge** —
ein echter Edge-Nachweis für F18 erfordert erst den Re-Fit + eine frische OOS-Messung mit Delta
gegen die TA-only-Baseline.

### Offene Entwicklungspfade (als separate Follow-ups ausgelagert)

1. **Regime-Gewichts-Re-Fit für `sector_rotation_bias`** auf dem korrigierten Panel via
   `scripts/train_regime_weights.py`; OOS-Delta gegen TA-only **messen, bevor** der Faktor scharf
   gestellt wird. (Limiter Tier A, lösbar ohne Bezahl-Daten.)
2. **Live-Frische des Offline-Sektor-Stores:** Sektor-ETF- + SPY-Closes in den täglichen EOD-Ingest
   aufnehmen, damit der Live-Pfad innerhalb `SECTOR_STORE_STALE_DAYS` bleibt. (Limiter Tier A.)
3. **`earnings_surprise_z`-Coverage über ~44 Mega-Caps hinaus:** nur mit **bezahltem** EPS-Estimate-
   Feed lösbar. Reaktivierung (Entfernen aus `INTENTIONALLY_ZEROED_FACTORS` + Re-Fit) **nur** falls
   ein solcher Feed beschafft wird. (Free-Feed-Decke Tier B — akzeptierter Decken-Zustand bis dahin.)
