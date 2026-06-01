# sector_rotation_bias — OOS Walk-Forward Falsification (the actual edge test)

Run date (UTC): 2026-06-01  
Data: `output/aggregates/daily.parquet` — the SAME offline store the live factor reads  
Universe: 8 SPDR sector ETFs ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY'] + SPY (benchmark/RS factor)  
History: 2113 bars 2018-01-02 → 2026-05-29 (Alpaca/master-panel era; NOT full SPDR history)  
Signal: production `compute_sector_scores` (3m·0.50 + 6m·0.30 + 20d-RS·0.20), top-3 long / bottom-2 short — unchanged  
WF: 252/252/252 (train/test/step), month-end rebalance, 2-bar data→fill gap (1-bar signal-ref + 1-bar exec shift)  
Frictions: 5 bps/leg turnover, 30 bps/yr short borrow (liquid ETFs)  
DSR multiple-testing deflation: n_trials = 3 (one fixed signal config, not parameter-searched)  

**Context / honesty:** `sector_rotation_bias` was a *dead* factor in production (stale offline store → 7-day staleness guard neutralised it to 0.0). Commit 433c2c03 fixed the freshness so it can now compute on live data. That unlocked CAPABILITY only. This harness is the falsification test that decides whether the factor deserves a non-zero regime weight. Survivorship bias is N/A (the 8 SPDR sector ETFs + SPY did not delist over 2018-2026); the factor is tested at the ETF level precisely to isolate the pure ranking signal and avoid the survivorship + security_meta mapping noise a stock-level test would carry. Because the factor value is constant within a sector, a stock-level L/S is mathematically this sector-ETF L/S weighted by universe composition. History starts 2018 (the live store's depth); deeper SPDR history (→1998) is NOT what the live factor sees. The production composite is denom-weighted, so an early score is finite from the 20d-RS term alone (an RS-tilted partial composite) before the 3m/6m terms exist; the 130-bar warm-up skips that window so every evaluated bar is dominated by the full 3m+6m terms, and all 8 ETFs + SPY share the 2018-01-02 start (no staggered-inception leakage). Prices are RAW close (not dividend/total-return adjusted), so absolute CAGRs understate total return; the relative SPY-beat verdict is unaffected because every book AND the SPY benchmark use the same raw close (common-mode omission). CI: not run; local one-shot. No production module touched.

## Verdict (auto-generated)

- **sector_ls** [REJECTED] (Sector-rotation L/S — long top-3 / short bottom-2 (the factor ranking, dollar-neutral)): pooled-OOS Sharpe +0.10 vs SPY +0.91; IR vs SPY -0.54 (t=-1.42); DSR-prob 0.28 (pass5%=False); beta -0.19; vol-matched ann.ret +2.0%.
- **sector_lo** [REJECTED] (Sector-rotation long-only tilt — long top-3 equal-weight (the mfv2 use-case)): pooled-OOS Sharpe +0.92 vs SPY +0.91; IR vs SPY +0.02 (t=+0.06); DSR-prob 0.94 (pass5%=False); beta +0.88; vol-matched ann.ret +18.2%.
- **eq_sector** [REJECTED] (Equal-weight 8 sectors (baseline — no rotation)): pooled-OOS Sharpe +0.85 vs SPY +0.91; IR vs SPY -0.34 (t=-0.89); DSR-prob 0.92 (pass5%=False); beta +0.91; vol-matched ann.ret +16.9%.

**ALL sector-rotation books REJECTED** — none clears SPY's pooled-OOS Sharpe with a multiple-testing-deflated (DSR) AND statistically significant (IR t>1.96) edge over 2018-2026. Fixing the factor's data freshness (commit 433c2c03) unlocked capability but did NOT reveal an edge. The production regime weight for `sector_rotation_bias` therefore stays ~0; a non-zero weight is not justified on this evidence.

## OOS-Edge table (pooled out-of-sample)

_sector_ls beta ≈ 0 confirms the dollar-neutral book isolates the ranking's alpha. IR vs SPY = annualised mean excess-over-SPY / its vol; IR t = IR·√years (|t|>1.96 ≈ 5% significant). DSR-prob is deflated for n_trials (Bailey-López de Prado); DSR✓ = passes 5%. PSR>SPY = prob true Sharpe exceeds SPY's. VolMatchRet = annual return if levered to SPY's vol — the honest 'beats SPY CAGR?' figure for a market-neutral book._

| Strategy | AnnSharpe | Sharpe t | CAGR | MaxDD | Beta | IR vs SPY | IR t | DSR-prob | DSR✓ | PSR>SPY | Trn/yr | FoldWin | VolMatchRet |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sector_ls | +0.10 | +0.26 | +0.1% | -37.8% | -0.19 | -0.54 | -1.42 | 0.28 | N | 0.02 | 16 | 2/7 | +2.0% |
| sector_lo | +0.92 | +2.43 | +17.6% | -31.4% | +0.88 | +0.02 | +0.06 | 0.94 | N | 0.51 | 7 | 2/7 | +18.2% |
| eq_sector | +0.85 | +2.26 | +15.3% | -37.0% | +0.91 | -0.34 | -0.89 | 0.92 | N | 0.44 | 0 | 3/7 | +16.9% |
| **SPY (bench)** | +0.91 | +2.41 | +17.3% | -33.7% | +1.00 | — | — | 0.94 | N | — | 0 | — | +17.3% |

## Sector-rotation L/S — long top-3 / short bottom-2 (the factor ranking, dollar-neutral)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +1.5% | +0.20 | -7.7% | -0.10 | 2.00 | +32.3% | +2.30 | 17 |
| 2 | 2020-01-03–2020-12-31 | +20.6% | +0.80 | -27.6% | -0.22 | 2.00 | +17.2% | +0.64 | 11 |
| 3 | 2021-01-04–2021-12-31 | +4.7% | +0.35 | -16.7% | +0.31 | 2.00 | +28.7% | +2.01 | 15 |
| 4 | 2022-01-03–2023-01-03 | +6.8% | +0.39 | -17.6% | -0.55 | 2.00 | -18.5% | -0.73 | 16 |
| 5 | 2023-01-04–2024-01-04 | -11.5% | -0.73 | -16.3% | +0.17 | 2.00 | +24.6% | +1.75 | 15 |
| 6 | 2024-01-05–2025-01-06 | -0.6% | +0.02 | -16.9% | +0.11 | 2.00 | +29.0% | +2.09 | 21 |
| 7 | 2025-01-07–2026-01-08 | -16.5% | -1.00 | -21.1% | -0.07 | 2.00 | +17.2% | +0.91 | 16 |
| **Ø (7/7)** | — | **+0.7%** | **+0.00** | **-17.7%** | **-0.05** | **2.00** | +18.6% | +1.28 | 16 |

## Sector-rotation long-only tilt — long top-3 equal-weight (the mfv2 use-case)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +26.5% | +1.99 | -8.2% | +0.87 | 1.00 | +32.3% | +2.30 | 11 |
| 2 | 2020-01-03–2020-12-31 | +25.9% | +0.84 | -31.4% | +1.01 | 1.00 | +17.2% | +0.64 | 3 |
| 3 | 2021-01-04–2021-12-31 | +30.6% | +1.57 | -8.6% | +1.01 | 1.00 | +28.7% | +2.01 | 6 |
| 4 | 2022-01-03–2023-01-03 | +3.4% | +0.26 | -14.4% | +0.64 | 1.00 | -18.5% | -0.73 | 7 |
| 5 | 2023-01-04–2024-01-04 | +10.4% | +0.77 | -11.3% | +0.97 | 1.00 | +24.6% | +1.75 | 7 |
| 6 | 2024-01-05–2025-01-06 | +24.7% | +1.89 | -5.3% | +0.80 | 1.00 | +29.0% | +2.09 | 8 |
| 7 | 2025-01-07–2026-01-08 | +5.1% | +0.38 | -19.3% | +0.81 | 1.00 | +17.2% | +0.91 | 7 |
| **Ø (7/7)** | — | **+18.1%** | **+1.10** | **-14.1%** | **+0.87** | **1.00** | +18.6% | +1.28 | 7 |

## Equal-weight 8 sectors (baseline — no rotation)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +29.1% | +2.29 | -6.1% | +0.90 | 1.00 | +32.3% | +2.30 | 0 |
| 2 | 2020-01-03–2020-12-31 | +8.4% | +0.41 | -37.0% | +1.05 | 1.00 | +17.2% | +0.64 | 0 |
| 3 | 2021-01-04–2021-12-31 | +29.8% | +2.18 | -4.4% | +0.89 | 1.00 | +28.7% | +2.01 | 0 |
| 4 | 2022-01-03–2023-01-03 | -4.9% | -0.14 | -16.5% | +0.84 | 1.00 | -18.5% | -0.73 | 0 |
| 5 | 2023-01-04–2024-01-04 | +13.9% | +1.16 | -9.9% | +0.86 | 1.00 | +24.6% | +1.75 | 0 |
| 6 | 2024-01-05–2025-01-06 | +19.7% | +1.79 | -5.9% | +0.73 | 1.00 | +29.0% | +2.09 | 0 |
| 7 | 2025-01-07–2026-01-08 | +15.2% | +0.98 | -14.8% | +0.76 | 1.00 | +17.2% | +0.91 | 0 |
| **Ø (7/7)** | — | **+15.9%** | **+1.24** | **-13.5%** | **+0.86** | **1.00** | +18.6% | +1.28 | 0 |

---
_Script: `scripts/_oos_wf_sector_rotation.py` (read-only research harness; imports the live `compute_sector_scores` unchanged; no production file modified)._  
_Signal: `src/assembled_core/signals/sector_rotation.py`; factor wiring: `src/assembled_core/strategies/multifactor_v2.py::_compute_sector_rotation_bias`._  
_Edge helpers: `src/assembled_core/qa/deflated_sharpe.py`, `src/assembled_core/qa/metrics.py`._  
_Freshness fix that motivated this test: commit 433c2c03 (`scripts/ops/refresh_sector_etf_cache.py` + `daily_paper_trading.bat` Step 1b)._  