# Long/Short + Levered Strategies — OOS Walk-Forward + Edge Suite

Run date (UTC): 2026-05-31  
Data: local offline cache via `load_eod_prices(None)` — survivors only  
Universe: 75 tradeable symbols (data ≤ 2018-01-31, ≥ 500 bars; SPY = market factor + benchmark)  
WF: 252/252/252 (train/test/step), monthly rebalance, top/bottom quintile  
Frictions: 10.75 bps/leg turnover, 50 bps/yr short borrow, 100 bps/yr financing on long notional > 1.0, BAB gross cap 3.0, 1-bar exec lag  
DSR multiple-testing deflation: n_trials = 16  
Pooled-OOS bars: 1512 (per strategy)  

**Honesty note:** Survivorship-only cache. Bias DIRECTION is strategy-dependent for L/S: short legs of mom_ls/bab_ls/lowvol_ls cannot short delisted losers → short leg UNDERSTATED → those results are a CONSERVATIVE lower bound. reversal_ls/_lo long the recent losers that survived (recovered) → OPTIMISTIC upper bound. QMJ not tested (needs fundamentals, absent). dual_momentum (owned) not driven (needs VEU/BIL, absent). The repo's LIVE-owned strategies (trend_baseline, multifactor_v2, news_alpha, crisis_alpha) are NOT in this table — they run on different universes/harnesses and were OOS-evaluated in prior sessions (e.g. trend_baseline 10-fold OOS Ø CAGR -6.1% vs SPY +13%); the 6 long-only rows here are factor-concept re-runs, not those production strategies. Leverage/borrow/financing are modelled with flat assumptions (no rate term structure). CI: not run; local one-shot.

## Verdict (auto-generated)

- **bab_ls** [REJECTED] (Betting-Against-Beta L/S, beta-targeted (Frazzini-Pedersen 2014)): pooled-OOS Sharpe +0.47 vs SPY +0.91; IR vs SPY -0.17 (t=-0.42); DSR-prob 0.26 (pass5%=False); beta +0.44; vol-matched ann.ret +9.2%.
- **mom_ls** [REJECTED] (Cross-sectional Momentum L/S, 12-1 WML (Jegadeesh-Titman 1993)): pooled-OOS Sharpe +0.78 vs SPY +0.91; IR vs SPY +0.14 (t=+0.35); DSR-prob 0.54 (pass5%=False); beta -0.05; vol-matched ann.ret +15.5%.
- **resmom_ls** [REJECTED] (Residual Momentum L/S (Blitz-Huij-Martens 2011)): pooled-OOS Sharpe -0.35 vs SPY +0.91; IR vs SPY -0.90 (t=-2.20); DSR-prob 0.00 (pass5%=False); beta -0.00; vol-matched ann.ret -7.0%.
- **reversal_ls** [REJECTED] (1-Month Reversal L/S (Jegadeesh 1990)): pooled-OOS Sharpe -0.55 vs SPY +0.91; IR vs SPY -1.12 (t=-2.75); DSR-prob 0.00 (pass5%=False); beta +0.30; vol-matched ann.ret -10.9%.
- **lowvol_ls** [REJECTED] (Low-Volatility L/S): pooled-OOS Sharpe -0.72 vs SPY +0.91; IR vs SPY -0.89 (t=-2.18); DSR-prob 0.00 (pass5%=False); beta -0.95; vol-matched ann.ret -14.3%.

**ALL 5 NEW L/S strategies REJECTED** — none clears SPY's pooled-OOS Sharpe with a multiple-testing-deflated (DSR) AND statistically significant (IR t>1.96) edge. This holds even though survivorship bias is CONSERVATIVE for the short-the-junk books (mom_ls/bab_ls/lowvol_ls). Consistent with the decay literature (BAB micro-cap/profitability artifact; reversal arbitraged; WML crash-prone). No prospect on this universe under realistic frictions.

## Consolidated OOS-Edge table (pooled out-of-sample, all new + prior candidates)

_Beta ≈ 0 confirms market-neutrality of the L/S books. IR vs SPY = annualised mean excess-over-SPY / its vol; IR t = IR·√years (|t|>1.96 ≈ 5% significant). DSR-prob is deflated for n_trials (Bailey-López de Prado); DSR✓ = passes 5%. PSR>SPY = prob true Sharpe exceeds SPY's. VolMatchRet = annual return if levered to SPY's vol, net of financing — the honest 'beats SPY CAGR?' figure for a market-neutral book._

| Strategy | AnnSharpe | Sharpe t | CAGR | MaxDD | Beta | IR vs SPY | IR t | DSR-prob | DSR✓ | PSR>SPY | Trn/yr | FoldWin | VolMatchRet |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bab_ls | +0.47 | +1.14 | +9.5% | -45.2% | +0.44 | -0.17 | -0.42 | 0.26 | N | 0.14 | 8 | 1/6 | +9.2% |
| mom_ls | +0.78 | +1.91 | +20.6% | -36.1% | -0.05 | +0.14 | +0.35 | 0.54 | N | 0.38 | 11 | 3/6 | +15.5% |
| resmom_ls | -0.35 | -0.86 | -7.9% | -54.5% | -0.00 | -0.90 | -2.20 | 0.00 | N | 0.00 | 26 | 1/6 | -7.0% |
| reversal_ls | -0.55 | -1.34 | -16.0% | -66.3% | +0.30 | -1.12 | -2.75 | 0.00 | N | 0.00 | 33 | 1/6 | -10.9% |
| lowvol_ls | -0.72 | -1.76 | -27.1% | -84.9% | -0.95 | -0.89 | -2.18 | 0.00 | N | 0.00 | 4 | 1/6 | -14.3% |
| mom_lo | +1.36 | +3.33 | +42.9% | -39.8% | +1.09 | +1.09 | +2.67 | 0.93 | N | 0.86 | 5 | 3/6 | +26.9% |
| high52w_lo | +1.10 | +2.70 | +16.4% | -21.3% | +0.57 | -0.13 | -0.33 | 0.81 | N | 0.68 | 12 | 2/6 | +21.5% |
| reversal_lo | +0.64 | +1.56 | +15.6% | -37.8% | +1.16 | +0.05 | +0.12 | 0.41 | N | 0.26 | 16 | 3/6 | +12.7% |
| lowbeta_lo | +1.01 | +2.47 | +11.8% | -21.6% | +0.40 | -0.41 | -1.01 | 0.74 | N | 0.59 | 2 | 3/6 | +19.3% |
| resmom_lo | +0.80 | +1.97 | +16.8% | -33.4% | +0.94 | +0.00 | +0.01 | 0.57 | N | 0.40 | 13 | 3/6 | +15.9% |
| eq_weight | +1.01 | +2.48 | +20.1% | -30.5% | +0.93 | +0.29 | +0.72 | 0.75 | N | 0.60 | 1 | 3/6 | +20.1% |
| **SPY (bench)** | +0.91 | +2.22 | +17.4% | -33.7% | +1.00 | — | — | 0.66 | N | — | 0 | — | +17.4% |

## Betting-Against-Beta L/S, beta-targeted (Frazzini-Pedersen 2014)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +54.0% | +2.33 | -10.6% | +0.02 | 2.76 | +35.7% | +2.54 | 7 |
| 2 | 2020-01-03–2020-12-31 | -29.6% | -0.54 | -44.1% | +0.82 | 2.76 | +18.2% | +0.67 | 8 |
| 3 | 2021-01-04–2021-12-31 | +7.7% | +0.46 | -17.7% | +0.02 | 2.77 | +30.6% | +2.13 | 8 |
| 4 | 2022-01-03–2023-01-03 | +18.6% | +0.73 | -24.7% | +0.26 | 2.76 | -19.1% | -0.75 | 7 |
| 5 | 2023-01-04–2024-01-04 | +22.0% | +1.00 | -15.9% | -0.10 | 2.77 | +23.7% | +1.69 | 8 |
| 6 | 2024-01-05–2025-01-06 | +2.2% | +0.21 | -27.6% | -0.13 | 2.79 | +29.0% | +2.08 | 8 |
| **Ø (6/6)** | — | **+12.5%** | **+0.70** | **-23.4%** | **+0.15** | **2.77** | +19.7% | +1.40 | 8 |

## Cross-sectional Momentum L/S, 12-1 WML (Jegadeesh-Titman 1993)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +16.6% | +0.94 | -16.8% | +0.11 | 1.84 | +35.7% | +2.54 | 12 |
| 2 | 2020-01-03–2020-12-31 | +65.1% | +1.68 | -17.4% | +0.08 | 1.84 | +18.2% | +0.67 | 9 |
| 3 | 2021-01-04–2021-12-31 | -12.1% | -0.30 | -35.7% | +0.93 | 1.85 | +30.6% | +2.13 | 11 |
| 4 | 2022-01-03–2023-01-03 | +34.0% | +0.99 | -23.7% | -0.77 | 1.84 | -19.1% | -0.75 | 11 |
| 5 | 2023-01-04–2024-01-04 | +64.1% | +2.14 | -9.4% | -0.35 | 1.85 | +23.7% | +1.69 | 13 |
| 6 | 2024-01-05–2025-01-06 | -17.3% | -0.41 | -33.5% | +0.78 | 1.86 | +29.0% | +2.08 | 10 |
| **Ø (6/6)** | — | **+25.1%** | **+0.84** | **-22.7%** | **+0.13** | **1.85** | +19.7% | +1.40 | 11 |

## Residual Momentum L/S (Blitz-Huij-Martens 2011)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +15.3% | +1.28 | -11.3% | -0.32 | 1.84 | +35.7% | +2.54 | 26 |
| 2 | 2020-01-03–2020-12-31 | +9.2% | +0.54 | -17.0% | +0.02 | 1.84 | +18.2% | +0.67 | 24 |
| 3 | 2021-01-04–2021-12-31 | -6.8% | -0.42 | -15.5% | +0.07 | 1.85 | +30.6% | +2.13 | 23 |
| 4 | 2022-01-03–2023-01-03 | -12.9% | -0.53 | -19.7% | +0.01 | 1.84 | -19.1% | -0.75 | 28 |
| 5 | 2023-01-04–2024-01-04 | +4.4% | +0.38 | -10.2% | +0.08 | 1.85 | +23.7% | +1.69 | 25 |
| 6 | 2024-01-05–2025-01-06 | -42.8% | -2.06 | -42.8% | -0.06 | 1.86 | +29.0% | +2.08 | 27 |
| **Ø (6/6)** | — | **-5.6%** | **-0.14** | **-19.4%** | **-0.04** | **1.85** | +19.7% | +1.40 | 26 |

## 1-Month Reversal L/S (Jegadeesh 1990)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | -11.8% | -0.71 | -19.3% | -0.15 | 1.84 | +35.7% | +2.54 | 34 |
| 2 | 2020-01-03–2020-12-31 | -36.7% | -1.42 | -36.7% | +0.37 | 1.84 | +18.2% | +0.67 | 30 |
| 3 | 2021-01-04–2021-12-31 | +16.7% | +0.83 | -10.7% | -0.14 | 1.85 | +30.6% | +2.13 | 35 |
| 4 | 2022-01-03–2023-01-03 | -6.4% | -0.03 | -27.2% | +0.57 | 1.84 | -19.1% | -0.75 | 34 |
| 5 | 2023-01-04–2024-01-04 | -4.3% | -0.09 | -19.2% | -0.01 | 1.85 | +23.7% | +1.69 | 33 |
| 6 | 2024-01-05–2025-01-06 | -40.0% | -1.65 | -41.7% | +0.14 | 1.86 | +29.0% | +2.08 | 32 |
| **Ø (6/6)** | — | **-13.7%** | **-0.51** | **-25.8%** | **+0.13** | **1.85** | +19.7% | +1.40 | 33 |

## Low-Volatility L/S

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | -40.7% | -2.30 | -40.7% | -0.88 | 1.84 | +35.7% | +2.54 | 4 |
| 2 | 2020-01-03–2020-12-31 | -56.3% | -2.36 | -65.6% | -0.61 | 1.84 | +18.2% | +0.67 | 5 |
| 3 | 2021-01-04–2021-12-31 | -10.5% | -0.19 | -34.5% | -1.09 | 1.85 | +30.6% | +2.13 | 4 |
| 4 | 2022-01-03–2023-01-03 | +4.1% | +0.33 | -39.5% | -1.39 | 1.84 | -19.1% | -0.75 | 3 |
| 5 | 2023-01-04–2024-01-04 | +0.2% | +0.16 | -27.3% | -1.40 | 1.85 | +23.7% | +1.69 | 3 |
| 6 | 2024-01-05–2025-01-06 | -37.7% | -1.05 | -47.7% | -1.23 | 1.86 | +29.0% | +2.08 | 4 |
| **Ø (6/6)** | — | **-23.5%** | **-0.90** | **-42.5%** | **-1.10** | **1.85** | +19.7% | +1.40 | 4 |

## Total-Return Momentum 12-1, long-only (control)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +62.5% | +2.39 | -10.4% | +0.98 | 0.92 | +35.7% | +2.54 | 5 |
| 2 | 2020-01-03–2020-12-31 | +118.4% | +1.93 | -39.8% | +1.12 | 0.92 | +18.2% | +0.67 | 4 |
| 3 | 2021-01-04–2021-12-31 | +12.1% | +0.52 | -29.8% | +1.46 | 0.92 | +30.6% | +2.13 | 5 |
| 4 | 2022-01-03–2023-01-03 | +9.1% | +0.49 | -16.1% | +0.81 | 0.92 | -19.1% | -0.75 | 6 |
| 5 | 2023-01-04–2024-01-04 | +48.4% | +2.17 | -10.3% | +1.03 | 0.92 | +23.7% | +1.69 | 7 |
| 6 | 2024-01-05–2025-01-06 | +32.1% | +1.14 | -20.1% | +1.72 | 0.93 | +29.0% | +2.08 | 5 |
| **Ø (6/6)** | — | **+47.1%** | **+1.44** | **-21.1%** | **+1.19** | **0.92** | +19.7% | +1.40 | 5 |

## 52-Week-High Momentum, long-only (George-Hwang 2004)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +20.1% | +1.79 | -4.0% | +0.60 | 0.92 | +35.7% | +2.54 | 12 |
| 2 | 2020-01-03–2020-12-31 | +35.9% | +1.45 | -21.3% | +0.59 | 0.92 | +18.2% | +0.67 | 13 |
| 3 | 2021-01-04–2021-12-31 | +27.5% | +2.12 | -4.4% | +0.71 | 0.92 | +30.6% | +2.13 | 12 |
| 4 | 2022-01-03–2023-01-03 | +1.9% | +0.19 | -17.0% | +0.47 | 0.92 | -19.1% | -0.75 | 7 |
| 5 | 2023-01-04–2024-01-04 | +1.8% | +0.21 | -11.7% | +0.63 | 0.92 | +23.7% | +1.69 | 13 |
| 6 | 2024-01-05–2025-01-06 | +15.4% | +1.23 | -6.1% | +0.66 | 0.93 | +29.0% | +2.08 | 13 |
| **Ø (6/6)** | — | **+17.1%** | **+1.16** | **-10.7%** | **+0.61** | **0.92** | +19.7% | +1.40 | 12 |

## 1-Month Reversal, long-only (Jegadeesh 1990)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +44.5% | +2.69 | -5.8% | +0.74 | 0.92 | +35.7% | +2.54 | 18 |
| 2 | 2020-01-03–2020-12-31 | +24.6% | +0.76 | -37.8% | +1.10 | 0.92 | +18.2% | +0.67 | 16 |
| 3 | 2021-01-04–2021-12-31 | +33.7% | +1.58 | -9.9% | +0.94 | 0.92 | +30.6% | +2.13 | 17 |
| 4 | 2022-01-03–2023-01-03 | -16.1% | -0.18 | -32.9% | +1.48 | 0.92 | -19.1% | -0.75 | 16 |
| 5 | 2023-01-04–2024-01-04 | +3.4% | +0.26 | -23.1% | +1.08 | 0.92 | +23.7% | +1.69 | 16 |
| 6 | 2024-01-05–2025-01-06 | +14.2% | +0.64 | -17.2% | +1.21 | 0.93 | +29.0% | +2.08 | 15 |
| **Ø (6/6)** | — | **+17.4%** | **+0.96** | **-21.1%** | **+1.09** | **0.92** | +19.7% | +1.40 | 16 |

## Low-Beta tilt, long-only (BAB no-leverage subset)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +25.6% | +3.11 | -4.1% | +0.17 | 0.92 | +35.7% | +2.54 | 2 |
| 2 | 2020-01-03–2020-12-31 | +16.3% | +0.81 | -21.6% | +0.55 | 0.92 | +18.2% | +0.67 | 2 |
| 3 | 2021-01-04–2021-12-31 | +10.2% | +1.24 | -8.1% | +0.36 | 0.92 | +30.6% | +2.13 | 3 |
| 4 | 2022-01-03–2023-01-03 | +5.1% | +0.48 | -11.6% | +0.31 | 0.92 | -19.1% | -0.75 | 2 |
| 5 | 2023-01-04–2024-01-04 | +10.1% | +1.28 | -4.9% | +0.23 | 0.92 | +23.7% | +1.69 | 3 |
| 6 | 2024-01-05–2025-01-06 | +5.1% | +0.73 | -8.1% | +0.12 | 0.93 | +29.0% | +2.08 | 2 |
| **Ø (6/6)** | — | **+12.1%** | **+1.28** | **-9.7%** | **+0.29** | **0.92** | +19.7% | +1.40 | 2 |

## Residual Momentum, long-only (Blitz-Huij-Martens 2011)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +33.8% | +2.62 | -5.0% | +0.57 | 0.92 | +35.7% | +2.54 | 12 |
| 2 | 2020-01-03–2020-12-31 | +55.3% | +1.40 | -33.4% | +0.98 | 0.92 | +18.2% | +0.67 | 12 |
| 3 | 2021-01-04–2021-12-31 | +20.2% | +1.15 | -8.8% | +0.97 | 0.92 | +30.6% | +2.13 | 12 |
| 4 | 2022-01-03–2023-01-03 | -15.5% | -0.47 | -25.1% | +0.96 | 0.92 | -19.1% | -0.75 | 14 |
| 5 | 2023-01-04–2024-01-04 | +10.3% | +0.69 | -10.6% | +0.96 | 0.92 | +23.7% | +1.69 | 12 |
| 6 | 2024-01-05–2025-01-06 | +9.0% | +0.60 | -15.4% | +0.91 | 0.93 | +29.0% | +2.08 | 14 |
| **Ø (6/6)** | — | **+18.8%** | **+1.00** | **-16.4%** | **+0.89** | **0.92** | +19.7% | +1.40 | 13 |

## Equal-Weight universe (baseline)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +33.9% | +2.57 | -5.5% | +0.78 | 0.92 | +35.7% | +2.54 | 1 |
| 2 | 2020-01-03–2020-12-31 | +43.9% | +1.30 | -30.5% | +0.94 | 0.92 | +18.2% | +0.67 | 1 |
| 3 | 2021-01-04–2021-12-31 | +24.3% | +1.66 | -7.1% | +0.89 | 0.92 | +30.6% | +2.13 | 1 |
| 4 | 2022-01-03–2023-01-03 | -10.9% | -0.32 | -22.9% | +1.01 | 0.92 | -19.1% | -0.75 | 1 |
| 5 | 2023-01-04–2024-01-04 | +12.1% | +0.91 | -13.0% | +0.91 | 0.92 | +23.7% | +1.69 | 1 |
| 6 | 2024-01-05–2025-01-06 | +25.3% | +1.66 | -9.4% | +0.90 | 0.93 | +29.0% | +2.08 | 1 |
| **Ø (6/6)** | — | **+21.4%** | **+1.30** | **-14.7%** | **+0.90** | **0.92** | +19.7% | +1.40 | 1 |

---
_Script: `scripts/_oos_wf_leverage_short.py` (read-only research harness, no production changes)._  
_References: Frazzini & Pedersen (2014) JFE 111(1); Novy-Marx & Velikov (2022) 'Betting Against Betting Against Beta' JFE; Jegadeesh & Titman (1993) J.Finance 48(1); Daniel & Moskowitz (2016) 'Momentum Crashes' JFE 122(2); Blitz, Huij & Martens (2011) J.Emp.Finance 18(3); Jegadeesh (1990) J.Finance 45(3); Asness, Frazzini & Pedersen (2019) 'Quality Minus Junk' Rev.Acc.Studies; McLean & Pontiff (2016) J.Finance 71(1)._  
_Edge helpers: `src/assembled_core/qa/deflated_sharpe.py`, `src/assembled_core/qa/metrics.py`._  