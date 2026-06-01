# sector_rotation_bias — Full-History OOS Robustness (1998–2026, total-return + raw)

Run date (UTC): 2026-06-01  
**Status: HISTORICAL-ROBUSTNESS EXTENSION — NOT the production verdict.** The live factor reads `output/aggregates/daily.parquet` (Alpaca era, 2018+); the binding falsification is `docs/results/2026_06_sector_rotation_oos.md`. This study sources a DIFFERENT feed (yfinance) and DEEPER history purely to test whether that REJECTED verdict is an artifact of the short 2018+ window or of raw (dividend-omitted) close.  

> **CORRECTION (2026-06-01) — price-type label.** A later feed-divergence check (`docs/results/2026_06_sector_fullhist_feed_divergence.md`) found that the live `daily.parquet` `close` is **total-return (split+dividend) adjusted**, not raw: the live close matches yfinance **Adj Close** to ~0.00 bps median across all 9 symbols. So the `adj` (total-return) mode below — not `raw` — is the true live-methodology match, and the live store's adjusted prices agree with this study's yfinance Adj Close almost exactly. This changes wording only: **all books stay REJECTED in both modes** (an adjusted/total-return book still fails to beat SPY on a deflated, significant basis). The generating script's prose has since been corrected (the `adj` mode is now labeled the live-methodology match and `raw` a conservative different-basis cross-check); the line-4 Status note and the doc body below predate that regeneration and are superseded by this correction.

Data: yfinance — full Select-Sector-SPDR history, raw Close + Adj Close  
Universe: 8 SPDR sector ETFs ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY'] + SPY (benchmark/RS factor)  
History: 7145 bars 1998-01-02 → 2026-05-29 (~28y; vs the live store's ~8y 2018+)  
Signal: production `compute_sector_scores` (3m·0.50 + 6m·0.30 + 20d-RS·0.20), top-3 long / bottom-2 short — unchanged  
WF: 252/252/252 (train/test/step), month-end rebalance, 2-bar data→fill gap — IMPORTED unchanged from the live harness  
Frictions: 5 bps/leg turnover, 30 bps/yr short borrow — IMPORTED unchanged  
DSR deflation: n_trials = 3 per price mode (one fixed signal config, 3 portfolio constructions). The 2 price modes are a robustness cross-check, not independent parameter searches; read a marginal pass conservatively.  

## Honesty / caveats (binding)

- **Two price modes.** `adj` = Adj Close (total-return) — fixes the absolute-CAGR understatement the live verdict flagged (auditor follow-up #2). `raw` = raw Close — PIT-clean and methodology-matched to the live verdict for apples-to-apples.
- **Adjusted-close is PIT-clean for THIS signal.** Score and L/S returns are purely ratio-based; yfinance's today-anchored normalization constant cancels in every ratio, and a ratio adj(t)/adj(t−k) depends only on corporate actions with ex-dates inside (t−k, t] — all in the past relative to t. The only residual is possible *retroactive* yfinance adjustment revisions, a data-QUALITY caveat (free feed), not a structural look-ahead.
- **Feed provenance.** yfinance ≠ the production Alpaca store; back-history splits/adjustments and exact closes can differ. This is why the study is kept SEPARATE from the live verdict rather than overriding it.
- **Survivorship N/A.** All 8 ranked ETFs are original Select Sector SPDRs live since Dec-1998; SPY since 1993. None delisted; no staggered inception in the ranked set.
- **CI:** not run; local one-shot. No production module touched. WF/friction/edge methodology imported unchanged from `scripts/_oos_wf_sector_rotation.py`.

## Verdict — total-return (Adj Close)

- **sector_ls** [REJECTED] (Sector-rotation L/S — long top-3 / short bottom-2 (the factor ranking, dollar-neutral)): pooled-OOS Sharpe -0.01 vs SPY +0.52; IR vs SPY -0.33 (t=-1.73); DSR-prob 0.18 (pass5%=False); beta -0.30; vol-matched ann.ret -0.3%.
- **sector_lo** [REJECTED] (Sector-rotation long-only tilt — long top-3 equal-weight (the mfv2 use-case)): pooled-OOS Sharpe +0.65 vs SPY +0.52; IR vs SPY +0.16 (t=+0.85); DSR-prob 0.99 (pass5%=True); beta +0.80; vol-matched ann.ret +12.5%.
- **eq_sector** [REJECTED] (Equal-weight 8 sectors (baseline — no rotation)): pooled-OOS Sharpe +0.58 vs SPY +0.52; IR vs SPY +0.08 (t=+0.44); DSR-prob 0.98 (pass5%=True); beta +0.90; vol-matched ann.ret +11.2%.

### OOS-Edge table — total-return (Adj Close)

| Strategy | AnnSharpe | Sharpe t | CAGR | MaxDD | Beta | IR vs SPY | IR t | DSR-prob | DSR✓ | PSR>SPY | Trn/yr | FoldWin | VolMatchRet |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sector_ls | -0.01 | -0.08 | -2.0% | -54.3% | -0.30 | -0.33 | -1.73 | 0.18 | N | 0.00 | 17 | 8/27 | -0.3% |
| sector_lo | +0.65 | +3.35 | +10.5% | -40.6% | +0.80 | +0.16 | +0.85 | 0.99 | Y | 0.75 | 7 | 11/27 | +12.5% |
| eq_sector | +0.58 | +3.01 | +9.2% | -52.2% | +0.90 | +0.08 | +0.44 | 0.98 | Y | 0.63 | 0 | 19/27 | +11.2% |
| **SPY (bench)** | +0.52 | +2.68 | +8.4% | -55.2% | +1.00 | — | — | 0.97 | Y | — | 0 | — | +8.4% |

## Verdict — raw Close (matches live methodology)

- **sector_ls** [REJECTED] (Sector-rotation L/S — long top-3 / short bottom-2 (the factor ranking, dollar-neutral)): pooled-OOS Sharpe +0.02 vs SPY +0.42; IR vs SPY -0.26 (t=-1.33); DSR-prob 0.23 (pass5%=False); beta -0.28; vol-matched ann.ret +0.4%.
- **sector_lo** [REJECTED] (Sector-rotation long-only tilt — long top-3 equal-weight (the mfv2 use-case)): pooled-OOS Sharpe +0.52 vs SPY +0.42; IR vs SPY +0.12 (t=+0.65); DSR-prob 0.97 (pass5%=True); beta +0.81; vol-matched ann.ret +10.2%.
- **eq_sector** [REJECTED] (Equal-weight 8 sectors (baseline — no rotation)): pooled-OOS Sharpe +0.47 vs SPY +0.42; IR vs SPY +0.04 (t=+0.22); DSR-prob 0.94 (pass5%=False); beta +0.90; vol-matched ann.ret +9.1%.

### OOS-Edge table — raw Close (matches live methodology)

| Strategy | AnnSharpe | Sharpe t | CAGR | MaxDD | Beta | IR vs SPY | IR t | DSR-prob | DSR✓ | PSR>SPY | Trn/yr | FoldWin | VolMatchRet |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sector_ls | +0.02 | +0.11 | -1.4% | -52.9% | -0.28 | -0.26 | -1.33 | 0.23 | N | 0.02 | 17 | 9/27 | +0.4% |
| sector_lo | +0.52 | +2.73 | +8.1% | -42.2% | +0.81 | +0.12 | +0.65 | 0.97 | Y | 0.70 | 7 | 12/27 | +10.2% |
| eq_sector | +0.47 | +2.43 | +7.1% | -53.6% | +0.90 | +0.04 | +0.22 | 0.94 | N | 0.59 | 0 | 16/27 | +9.1% |
| **SPY (bench)** | +0.42 | +2.21 | +6.6% | -56.5% | +1.00 | — | — | 0.91 | N | — | 0 | — | +6.6% |

**ALL books REJECTED in BOTH price modes** — neither deeper history (back to the 1998 SPDR inception, ~3.5x the live window) NOR total-return (adjusted) prices reveal a multiple-testing-deflated, statistically-significant edge over SPY. The live-store 2018+ rejection (docs/results/2026_06_sector_rotation_oos.md) is therefore NOT an artifact of the short window or the raw-close dividend omission. The production regime weight for `sector_rotation_bias` stays ~0.

## Per-fold detail — total-return (Adj Close)

### Sector-rotation L/S — long top-3 / short bottom-2 (the factor ranking, dollar-neutral)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 1999-01-04–1999-12-31 | +8.7% | +0.56 | -15.0% | -0.07 | 1.85 | +20.4% | +1.12 | 21 |
| 2 | 2000-01-03–2000-12-29 | +1.0% | +0.18 | -26.6% | -0.26 | 2.00 | -9.7% | -0.31 | 17 |
| 3 | 2001-01-02–2002-01-07 | -19.4% | -0.66 | -23.2% | -0.58 | 2.00 | -9.8% | -0.36 | 17 |
| 4 | 2002-01-08–2003-01-07 | +2.8% | +0.24 | -21.3% | -0.14 | 2.00 | -19.3% | -0.67 | 14 |
| 5 | 2003-01-08–2004-01-07 | -9.3% | -0.73 | -13.7% | +0.05 | 2.00 | +23.8% | +1.40 | 19 |
| 6 | 2004-01-08–2005-01-06 | -0.5% | +0.01 | -9.1% | -0.20 | 2.00 | +7.0% | +0.66 | 15 |
| 7 | 2005-01-07–2006-01-06 | +0.9% | +0.14 | -9.6% | +0.04 | 2.00 | +10.2% | +0.99 | 19 |
| 8 | 2006-01-09–2007-01-09 | -16.6% | -1.65 | -17.5% | +0.02 | 2.00 | +11.9% | +1.19 | 20 |
| 9 | 2007-01-10–2008-01-09 | +29.7% | +2.16 | -8.8% | -0.21 | 2.00 | +1.3% | +0.16 | 12 |
| 10 | 2008-01-10–2009-01-08 | -11.1% | -0.08 | -23.1% | -0.74 | 2.00 | -33.6% | -0.78 | 15 |
| 11 | 2009-01-09–2010-01-08 | -8.9% | -0.12 | -36.2% | -0.59 | 2.00 | +28.8% | +1.09 | 17 |
| 12 | 2010-01-11–2011-01-07 | -8.3% | -0.78 | -11.0% | +0.10 | 2.00 | +13.1% | +0.78 | 22 |
| 13 | 2011-01-10–2012-01-09 | +9.3% | +0.63 | -14.3% | -0.46 | 2.00 | +2.8% | +0.24 | 14 |
| 14 | 2012-01-10–2013-01-10 | -8.0% | -0.87 | -13.0% | -0.02 | 2.00 | +17.4% | +1.31 | 17 |
| 15 | 2013-01-11–2014-01-10 | +3.8% | +0.49 | -10.6% | +0.16 | 2.00 | +27.7% | +2.32 | 12 |
| 16 | 2014-01-13–2015-01-12 | -3.7% | -0.33 | -12.3% | -0.02 | 2.00 | +12.2% | +1.05 | 19 |
| 17 | 2015-01-13–2016-01-12 | -1.0% | -0.01 | -12.7% | -0.07 | 2.00 | -2.5% | -0.08 | 18 |
| 18 | 2016-01-13–2017-01-11 | -10.1% | -0.77 | -18.7% | -0.36 | 2.00 | +19.8% | +1.49 | 21 |
| 19 | 2017-01-12–2018-01-11 | +3.5% | +0.40 | -8.3% | +0.44 | 2.00 | +23.9% | +3.20 | 17 |
| 20 | 2018-01-12–2019-01-14 | -6.1% | -0.34 | -12.5% | +0.13 | 2.00 | -5.0% | -0.21 | 15 |
| 21 | 2019-01-15–2020-01-14 | +6.0% | +0.62 | -6.4% | -0.04 | 2.00 | +29.6% | +2.25 | 17 |
| 22 | 2020-01-15–2021-01-13 | +29.8% | +1.05 | -27.6% | -0.22 | 2.00 | +18.2% | +0.67 | 12 |
| 23 | 2021-01-14–2022-01-12 | -14.3% | -0.79 | -25.0% | +0.32 | 2.00 | +25.7% | +1.83 | 17 |
| 24 | 2022-01-13–2023-01-13 | +18.5% | +0.82 | -17.6% | -0.56 | 2.00 | -14.0% | -0.50 | 16 |
| 25 | 2023-01-17–2024-01-17 | -6.6% | -0.38 | -13.0% | +0.22 | 2.00 | +20.3% | +1.51 | 15 |
| 26 | 2024-01-18–2025-01-17 | -7.1% | -0.46 | -16.9% | +0.15 | 2.00 | +28.2% | +2.00 | 21 |
| 27 | 2025-01-21–2026-01-21 | -18.1% | -1.13 | -21.1% | -0.08 | 2.00 | +16.0% | +0.86 | 16 |
| **Ø (27/27)** | — | **-1.3%** | **-0.07** | **-16.5%** | **-0.11** | **1.99** | +9.8% | +0.86 | 17 |

### Sector-rotation long-only tilt — long top-3 equal-weight (the mfv2 use-case)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 1999-01-04–1999-12-31 | +21.4% | +1.21 | -9.8% | +0.72 | 0.92 | +20.4% | +1.12 | 10 |
| 2 | 2000-01-03–2000-12-29 | +7.0% | +0.44 | -10.8% | +0.56 | 1.00 | -9.7% | -0.31 | 7 |
| 3 | 2001-01-02–2002-01-07 | -11.4% | -0.62 | -18.0% | +0.62 | 1.00 | -9.8% | -0.36 | 9 |
| 4 | 2002-01-08–2003-01-07 | -16.3% | -0.55 | -28.6% | +0.90 | 1.00 | -19.3% | -0.67 | 8 |
| 5 | 2003-01-08–2004-01-07 | +20.0% | +1.20 | -13.0% | +0.92 | 1.00 | +23.8% | +1.40 | 9 |
| 6 | 2004-01-08–2005-01-06 | +13.4% | +1.17 | -5.4% | +0.78 | 1.00 | +7.0% | +0.66 | 6 |
| 7 | 2005-01-07–2006-01-06 | +19.0% | +1.39 | -10.4% | +1.03 | 1.00 | +10.2% | +0.99 | 7 |
| 8 | 2006-01-09–2007-01-09 | +9.7% | +0.91 | -10.2% | +0.91 | 1.00 | +11.9% | +1.19 | 9 |
| 9 | 2007-01-10–2008-01-09 | +16.2% | +0.99 | -10.1% | +0.95 | 1.00 | +1.3% | +0.16 | 5 |
| 10 | 2008-01-10–2009-01-08 | -26.5% | -0.93 | -34.7% | +0.61 | 1.00 | -33.6% | -0.78 | 6 |
| 11 | 2009-01-09–2010-01-08 | +33.7% | +1.33 | -18.2% | +0.81 | 1.00 | +28.8% | +1.09 | 6 |
| 12 | 2010-01-11–2011-01-07 | +14.0% | +0.77 | -18.0% | +1.06 | 1.00 | +13.1% | +0.78 | 7 |
| 13 | 2011-01-10–2012-01-09 | +8.6% | +0.53 | -15.8% | +0.78 | 1.00 | +2.8% | +0.24 | 6 |
| 14 | 2012-01-10–2013-01-10 | +11.3% | +0.92 | -10.4% | +0.91 | 1.00 | +17.4% | +1.31 | 9 |
| 15 | 2013-01-11–2014-01-10 | +25.2% | +1.95 | -7.7% | +1.04 | 1.00 | +27.7% | +2.32 | 6 |
| 16 | 2014-01-13–2015-01-12 | +8.3% | +0.74 | -7.7% | +0.94 | 1.00 | +12.2% | +1.05 | 9 |
| 17 | 2015-01-13–2016-01-12 | -7.0% | -0.40 | -14.1% | +0.93 | 1.00 | -2.5% | -0.08 | 8 |
| 18 | 2016-01-13–2017-01-11 | +18.1% | +1.42 | -4.4% | +0.82 | 1.00 | +19.8% | +1.49 | 7 |
| 19 | 2017-01-12–2018-01-11 | +21.5% | +2.48 | -3.7% | +1.05 | 1.00 | +23.9% | +3.20 | 8 |
| 20 | 2018-01-12–2019-01-14 | -0.5% | +0.06 | -15.3% | +0.91 | 1.00 | -5.0% | -0.21 | 3 |
| 21 | 2019-01-15–2020-01-14 | +25.7% | +1.98 | -8.2% | +0.91 | 1.00 | +29.6% | +2.25 | 11 |
| 22 | 2020-01-15–2021-01-13 | +33.9% | +1.01 | -31.4% | +1.01 | 1.00 | +18.2% | +0.67 | 3 |
| 23 | 2021-01-14–2022-01-12 | +19.0% | +1.07 | -8.6% | +1.01 | 1.00 | +25.7% | +1.83 | 7 |
| 24 | 2022-01-13–2023-01-13 | +11.5% | +0.64 | -14.4% | +0.64 | 1.00 | -14.0% | -0.50 | 7 |
| 25 | 2023-01-17–2024-01-17 | +5.6% | +0.46 | -11.3% | +0.98 | 1.00 | +20.3% | +1.51 | 7 |
| 26 | 2024-01-18–2025-01-17 | +24.9% | +1.84 | -6.0% | +0.82 | 1.00 | +28.2% | +2.00 | 8 |
| 27 | 2025-01-21–2026-01-21 | +2.6% | +0.23 | -19.3% | +0.80 | 1.00 | +16.0% | +0.86 | 7 |
| **Ø (27/27)** | — | **+11.5%** | **+0.82** | **-13.5%** | **+0.87** | **1.00** | +9.8% | +0.86 | 7 |

### Equal-weight 8 sectors (baseline — no rotation)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 1999-01-04–1999-12-31 | +13.2% | +0.89 | -11.5% | +0.73 | 0.92 | +20.4% | +1.12 | 1 |
| 2 | 2000-01-03–2000-12-29 | +4.1% | +0.31 | -12.6% | +0.66 | 1.00 | -9.7% | -0.31 | 0 |
| 3 | 2001-01-02–2002-01-07 | -6.5% | -0.26 | -25.0% | +0.82 | 1.00 | -9.8% | -0.36 | 0 |
| 4 | 2002-01-08–2003-01-07 | -17.2% | -0.63 | -32.3% | +0.92 | 1.00 | -19.3% | -0.67 | 0 |
| 5 | 2003-01-08–2004-01-07 | +23.7% | +1.51 | -12.2% | +0.89 | 1.00 | +23.8% | +1.40 | 0 |
| 6 | 2004-01-08–2005-01-06 | +10.6% | +1.03 | -5.9% | +0.88 | 1.00 | +7.0% | +0.66 | 0 |
| 7 | 2005-01-07–2006-01-06 | +13.8% | +1.29 | -6.2% | +0.98 | 1.00 | +10.2% | +0.99 | 0 |
| 8 | 2006-01-09–2007-01-09 | +12.0% | +1.25 | -6.7% | +0.92 | 1.00 | +11.9% | +1.19 | 0 |
| 9 | 2007-01-10–2008-01-09 | +5.2% | +0.41 | -8.6% | +0.92 | 1.00 | +1.3% | +0.16 | 0 |
| 10 | 2008-01-10–2009-01-08 | -30.4% | -0.76 | -44.1% | +0.91 | 1.00 | -33.6% | -0.78 | 0 |
| 11 | 2009-01-09–2010-01-08 | +29.2% | +1.10 | -24.4% | +1.01 | 1.00 | +28.8% | +1.09 | 0 |
| 12 | 2010-01-11–2011-01-07 | +13.6% | +0.82 | -14.6% | +0.98 | 1.00 | +13.1% | +0.78 | 0 |
| 13 | 2011-01-10–2012-01-09 | +5.5% | +0.35 | -17.2% | +0.97 | 1.00 | +2.8% | +0.24 | 0 |
| 14 | 2012-01-10–2013-01-10 | +16.5% | +1.34 | -8.4% | +0.93 | 1.00 | +17.4% | +1.31 | 0 |
| 15 | 2013-01-11–2014-01-10 | +27.2% | +2.31 | -5.8% | +0.98 | 1.00 | +27.7% | +2.32 | 0 |
| 16 | 2014-01-13–2015-01-12 | +12.6% | +1.12 | -6.7% | +0.96 | 1.00 | +12.2% | +1.05 | 0 |
| 17 | 2015-01-13–2016-01-12 | -3.5% | -0.16 | -11.6% | +0.96 | 1.00 | -2.5% | -0.08 | 0 |
| 18 | 2016-01-13–2017-01-11 | +21.0% | +1.63 | -5.0% | +0.95 | 1.00 | +19.8% | +1.49 | 0 |
| 19 | 2017-01-12–2018-01-11 | +20.5% | +3.12 | -2.5% | +0.86 | 1.00 | +23.9% | +3.20 | 0 |
| 20 | 2018-01-12–2019-01-14 | -5.1% | -0.26 | -18.1% | +0.88 | 1.00 | -5.0% | -0.21 | 0 |
| 21 | 2019-01-15–2020-01-14 | +26.1% | +2.20 | -6.1% | +0.90 | 1.00 | +29.6% | +2.25 | 0 |
| 22 | 2020-01-15–2021-01-13 | +11.5% | +0.48 | -37.0% | +1.05 | 1.00 | +18.2% | +0.67 | 0 |
| 23 | 2021-01-14–2022-01-12 | +27.0% | +2.03 | -4.4% | +0.88 | 1.00 | +25.7% | +1.83 | 0 |
| 24 | 2022-01-13–2023-01-13 | -2.2% | -0.00 | -16.5% | +0.84 | 1.00 | -14.0% | -0.50 | 0 |
| 25 | 2023-01-17–2024-01-17 | +8.6% | +0.76 | -9.9% | +0.85 | 1.00 | +20.3% | +1.51 | 0 |
| 26 | 2024-01-18–2025-01-17 | +23.6% | +2.08 | -5.9% | +0.73 | 1.00 | +28.2% | +2.00 | 0 |
| 27 | 2025-01-21–2026-01-21 | +14.0% | +0.91 | -14.8% | +0.76 | 1.00 | +16.0% | +0.86 | 0 |
| **Ø (27/27)** | — | **+10.2%** | **+0.92** | **-13.9%** | **+0.89** | **1.00** | +9.8% | +0.86 | 0 |

## Per-fold detail — raw Close (sector_ls, methodology-matched)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 1999-01-04–1999-12-31 | +9.1% | +0.58 | -15.2% | -0.05 | 1.85 | +19.1% | +1.06 | 21 |
| 2 | 2000-01-03–2000-12-29 | +0.7% | +0.17 | -26.8% | -0.26 | 2.00 | -10.7% | -0.35 | 17 |
| 3 | 2001-01-02–2002-01-07 | -22.1% | -0.74 | -26.2% | -0.52 | 2.00 | -11.0% | -0.42 | 18 |
| 4 | 2002-01-08–2003-01-07 | +4.4% | +0.31 | -20.9% | -0.13 | 2.00 | -20.6% | -0.73 | 14 |
| 5 | 2003-01-08–2004-01-07 | -10.4% | -0.89 | -12.7% | +0.11 | 2.00 | +21.8% | +1.30 | 21 |
| 6 | 2004-01-08–2005-01-06 | -3.9% | -0.33 | -8.6% | -0.19 | 2.00 | +5.0% | +0.49 | 15 |
| 7 | 2005-01-07–2006-01-06 | +3.0% | +0.33 | -8.8% | +0.04 | 2.00 | +8.3% | +0.82 | 17 |
| 8 | 2006-01-09–2007-01-09 | -17.4% | -1.76 | -18.2% | +0.04 | 2.00 | +9.8% | +0.99 | 20 |
| 9 | 2007-01-10–2008-01-09 | +29.8% | +2.14 | -8.5% | -0.17 | 2.00 | -0.5% | +0.05 | 13 |
| 10 | 2008-01-10–2009-01-08 | -12.0% | -0.11 | -22.9% | -0.73 | 2.00 | -35.1% | -0.84 | 15 |
| 11 | 2009-01-09–2010-01-08 | -5.6% | -0.01 | -34.9% | -0.59 | 2.00 | +25.8% | +1.00 | 16 |
| 12 | 2010-01-11–2011-01-07 | -1.9% | -0.14 | -9.2% | +0.15 | 2.00 | +11.0% | +0.67 | 18 |
| 13 | 2011-01-10–2012-01-09 | +6.8% | +0.49 | -14.3% | -0.45 | 2.00 | +0.7% | +0.15 | 16 |
| 14 | 2012-01-10–2013-01-10 | -5.7% | -0.60 | -11.4% | +0.03 | 2.00 | +14.9% | +1.14 | 18 |
| 15 | 2013-01-11–2014-01-10 | +1.3% | +0.21 | -9.6% | +0.20 | 2.00 | +25.2% | +2.12 | 13 |
| 16 | 2014-01-13–2015-01-12 | -3.8% | -0.34 | -12.2% | +0.05 | 2.00 | +10.1% | +0.88 | 19 |
| 17 | 2015-01-13–2016-01-12 | -1.7% | -0.07 | -11.7% | -0.07 | 2.00 | -4.4% | -0.21 | 19 |
| 18 | 2016-01-13–2017-01-11 | -4.8% | -0.31 | -17.8% | -0.34 | 2.00 | +17.3% | +1.32 | 18 |
| 19 | 2017-01-12–2018-01-11 | +4.3% | +0.47 | -8.5% | +0.58 | 2.00 | +21.6% | +2.91 | 18 |
| 20 | 2018-01-12–2019-01-14 | -5.6% | -0.30 | -11.9% | +0.13 | 2.00 | -6.8% | -0.31 | 15 |
| 21 | 2019-01-15–2020-01-14 | +8.1% | +0.81 | -6.4% | -0.04 | 2.00 | +27.2% | +2.08 | 17 |
| 22 | 2020-01-15–2021-01-13 | +34.7% | +1.15 | -26.5% | -0.25 | 2.00 | +16.0% | +0.61 | 13 |
| 23 | 2021-01-14–2022-01-12 | -14.6% | -0.81 | -25.4% | +0.32 | 2.00 | +24.0% | +1.71 | 18 |
| 24 | 2022-01-13–2023-01-13 | +10.6% | +0.57 | -18.1% | -0.49 | 2.00 | -15.4% | -0.57 | 19 |
| 25 | 2023-01-17–2024-01-17 | -2.3% | -0.08 | -11.1% | +0.17 | 2.00 | +18.5% | +1.38 | 14 |
| 26 | 2024-01-18–2025-01-17 | -4.3% | -0.26 | -14.7% | +0.18 | 2.00 | +26.5% | +1.90 | 21 |
| 27 | 2025-01-21–2026-01-21 | -14.3% | -0.86 | -21.0% | -0.08 | 2.00 | +14.7% | +0.80 | 17 |
| **Ø (27/27)** | — | **-0.6%** | **-0.01** | **-16.1%** | **-0.09** | **1.99** | +7.9% | +0.74 | 17 |

---
_Script: `scripts/_oos_wf_sector_rotation_fullhist.py` (read-only research harness; imports the live `compute_sector_scores` AND the committed WF/edge methodology from `scripts/_oos_wf_sector_rotation.py` unchanged; no production file modified)._  
_Live verdict (binding): `docs/results/2026_06_sector_rotation_oos.md`._  
_Signal: `src/assembled_core/signals/sector_rotation.py`._  
_Data cache (gitignored): `output/research/sector_fullhist_yf.parquet` (yfinance)._  