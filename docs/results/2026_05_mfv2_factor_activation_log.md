# multifactor_v2 — Faktor-Aktivierungs-Log (Paket 3c.2)

**Erstellt:** 2026-05-28  
**Audit-Datum:** 2023-06-30  
**Branch:** main  
**Zweck:** Paket 3c.2 — Verifikation der Faktor-Aktivierung vor Walk-Forward.

---

## Zusammenfassung

- **ACTIVE:** 9 Faktoren (% non-zero > 0)
- **ZERO (erwartet):** 15 Faktoren (kein Datenfehler — Daten nicht verfügbar oder live-only)
- **STOPP:** 0 Faktoren (sollten aktiv sein, sind es aber nicht)

**STOPP-Check: BESTANDEN.** Alle erwarteten Faktoren aktiv.

---

## Faktor-Detail

| Faktor | Kategorie | Mean Value | % Non-Zero | Status |
|--------|-----------|-----------|-----------|--------|
| earnings_surprise_z | Altdata/Earnings | 0.0000 | 0% | — ZERO |
| insider_activity_score | Altdata/Insider | 0.0000 | 0% | — ZERO |
| news_sentiment_7d_z | Altdata/News | 0.0000 | 0% | — ZERO |
| news_volume_spike_z | Altdata/News | 0.0000 | 0% | — ZERO |
| macro_growth_momentum_z | Altdata/Makro | -2.0686 | 100% | ✅ ACTIVE |
| macro_inflation_surprise_z | Altdata/Makro | 1.4649 | 100% | ✅ ACTIVE |
| intermarket_bond_equity | Altdata/Intermarkt | 0.0000 | 0% | — ZERO |
| intermarket_credit_spread | Altdata/Intermarkt | 0.0144 | 100% | ✅ ACTIVE |
| intermarket_yield_curve | Altdata/Intermarkt | -1.0200 | 100% | ✅ ACTIVE |
| options_put_call_extreme | Altdata/Options | 0.0000 | 0% | — ZERO |
| vix_regime_score | Altdata/Options | 0.0000 | 0% | — ZERO |
| geo_risk_composite | Altdata/GPR | -2.2106 | 100% | ✅ ACTIVE |
| sector_rotation_bias | Altdata/Sektor | 0.0000 | 0% | — ZERO |
| trend_ema_spread | TA/Trend | 0.0000 | 0% | — ZERO |
| trend_ma200_position | TA/Trend | 155.0487 | 100% | ✅ ACTIVE |
| trend_adx_strength | TA/Trend | 23.7135 | 100% | ✅ ACTIVE |
| trend_macd_hist | TA/Trend | -0.0375 | 100% | ✅ ACTIVE |
| mom_rsi_centered | TA/Momentum | 0.0000 | 0% | — ZERO |
| mom_volume_weighted | TA/Momentum | 0.0000 | 0% | — ZERO |
| mom_obv_trend | TA/Momentum | 0.0000 | 0% | — ZERO |
| mr_bollinger_pctb | TA/MR | 0.0000 | 0% | — ZERO |
| mr_stoch_oversold | TA/MR | 58.7708 | 100% | ✅ ACTIVE |
| vol_abnormal | TA/Volume | 0.0000 | 0% | — ZERO |
| vola_regime_score | TA/Volatility | 0.0000 | 0% | — ZERO |

---

## Aktivierungs-Methodologie

- **gpr_index**: aus `output/macro_gpr.parquet` (Caldara-Iacoviello, monatlich → tägl. ffill)
- **yield_curve_slope**: aus `output/macro.parquet` Spalte `yield_curve_spread` (monatlich → tägl. ffill)
- **bond_equity_divergence_flag**: TLT 20d-Return > 0 UND SPY 20d-Return < 0 (aus ETF-Panel via yfinance)
- **credit_spread_change_5d**: HYG/IEF-Ratio 5-Tage-pct_change (aus ETF-Panel via yfinance)
- **sector_rotation_bias**: Sector-ETF-Rows (XLK/XLF/XLE/…) im Price-Panel + configs/security_meta.csv
- **earnings_surprise_z**: auto via altdata_loader aus `output/events_earnings.parquet`
- **macro_growth_momentum_z / macro_inflation_surprise_z**: auto via altdata_loader aus `output/macro.parquet`

_Dieses Dokument ist ein automatisch erzeugtes Artefakt aus_ `scripts/_oos_wf_mfv2_full.py`.