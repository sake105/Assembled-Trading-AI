# Pipeline-Realistic OOS Walk-Forward — 11 Strategies through `run_trading_cycle`

Run date (UTC): 2026-05-31  
Data: local offline cache via `load_eod_prices(None)` — survivors only  
Universe: 75 tradeable symbols (data ≤ 2018-01-31, ≥ 500 bars; SPY = market factor + benchmark, never traded)  
WF: 252/252/252 (train/test/step), monthly rebalance, top/bottom quintile  
Execution: LITERAL `run_portfolio_backtest(cycle_fn=make_cycle_fn(...))` → `run_trading_cycle` → feature enrichment → `generate_orders_from_targets` (signed notional → integer shares) → `simulate_with_costs`  
Config: `enable_risk_controls=False` (matches `run_backtest_strategy.py` backtest path), `include_costs=True` (real cost model)  
DSR multiple-testing deflation: n_trials = 16  
Pooled-OOS bars: 1512 (per strategy)  

**What 'pipeline-realistic' means here (honest scope):** the selection logic is IDENTICAL to the prior vectorized study `_oos_wf_leverage_short.py`; the difference is that orders are now generated and filled through the REAL production cycle. The literal pipeline ADDS a real cost model (commission+spread+impact), real signed-notional→integer-share order generation, and feature enrichment (HMM/behavioral/macro/rv) — the last is IMMATERIAL because these signals read only `close`. It does NOT model the prior study's explicit short-borrow (50 bps/yr) or margin financing (100 bps/yr), so on the carry side it is LESS conservative; net cost direction vs the prior study is therefore ambiguous and both are shown.

**Live-mode caveat (separate layer):** with `enable_risk_controls=True` (paper/live, NOT the backtest path), a smoke test confirmed the cycle preserves signed shorts (first-rebalance BUY notional ≈ SELL notional, ratio 0.99) but de-levers gross ~5x to the default `risk_limits.max_gross_exposure`=1.20 cap. In live mode every book below would therefore be further de-levered, pushing absolute returns DOWN and leaving Sharpe ≈ unchanged (cash drag is ~vol-neutral). That cannot rescue a rejected strategy.

**Honesty note:** Survivorship-only cache. Bias DIRECTION is strategy-dependent for L/S: short legs of mom_ls/bab_ls/lowvol_ls cannot short delisted losers → short leg UNDERSTATED → CONSERVATIVE lower bound. reversal_ls/_lo LONG recovered losers → OPTIMISTIC upper bound. The repo's LIVE-owned strategies (trend_baseline, multifactor_v2, news_alpha, crisis_alpha) are evaluated separately (see companion section / prior sessions) on their own universes. CI: not run; local one-shot.

## Verdict (auto-generated)

- **bab_ls** [REJECTED] (Betting-Against-Beta L/S, beta-targeted (Frazzini-Pedersen 2014)): pooled-OOS Sharpe -0.06 vs SPY +0.91; IR vs SPY -0.76 (t=-1.86); DSR-prob 0.03 (pass5%=False); beta +0.02; vol-matched ann.ret -1.4%.
- **mom_ls** [REJECTED] (Cross-sectional Momentum L/S, 12-1 WML (Jegadeesh-Titman 1993)): pooled-OOS Sharpe +0.80 vs SPY +0.91; IR vs SPY +0.01 (t=+0.01); DSR-prob 0.56 (pass5%=False); beta +0.02; vol-matched ann.ret +15.8%.
- **resmom_ls** [REJECTED] (Residual Momentum L/S (Blitz-Huij-Martens 2011)): pooled-OOS Sharpe -0.10 vs SPY +0.91; IR vs SPY -0.68 (t=-1.67); DSR-prob 0.02 (pass5%=False); beta -0.02; vol-matched ann.ret -2.0%.
- **reversal_ls** [REJECTED] (1-Month Reversal L/S (Jegadeesh 1990)): pooled-OOS Sharpe -0.42 vs SPY +0.91; IR vs SPY -0.96 (t=-2.35); DSR-prob 0.00 (pass5%=False); beta +0.16; vol-matched ann.ret -8.4%.
- **lowvol_ls** [REJECTED] (Low-Volatility L/S): pooled-OOS Sharpe -0.68 vs SPY +0.91; IR vs SPY -0.91 (t=-2.23); DSR-prob 0.00 (pass5%=False); beta -0.63; vol-matched ann.ret -13.5%.
- **mom_lo** [REJECTED] (Total-Return Momentum 12-1, long-only (control)): pooled-OOS Sharpe +1.38 vs SPY +0.91; IR vs SPY +0.58 (t=+1.42); DSR-prob 0.94 (pass5%=False); beta +0.66; vol-matched ann.ret +27.4%.
- **high52w_lo** [REJECTED] (52-Week-High Momentum, long-only (George-Hwang 2004)): pooled-OOS Sharpe +1.16 vs SPY +0.91; IR vs SPY -0.37 (t=-0.91); DSR-prob 0.84 (pass5%=False); beta +0.36; vol-matched ann.ret +22.1%.
- **reversal_lo** [REJECTED] (1-Month Reversal, long-only (Jegadeesh 1990)): pooled-OOS Sharpe +0.73 vs SPY +0.91; IR vs SPY -0.19 (t=-0.47); DSR-prob 0.50 (pass5%=False); beta +0.73; vol-matched ann.ret +14.5%.
- **lowbeta_lo** [REJECTED] (Low-Beta tilt, long-only (BAB no-leverage subset)): pooled-OOS Sharpe +1.21 vs SPY +0.91; IR vs SPY -0.52 (t=-1.28); DSR-prob 0.87 (pass5%=False); beta +0.22; vol-matched ann.ret +22.3%.
- **resmom_lo** [REJECTED] (Residual Momentum, long-only (Blitz-Huij-Martens 2011)): pooled-OOS Sharpe +0.94 vs SPY +0.91; IR vs SPY -0.28 (t=-0.69); DSR-prob 0.69 (pass5%=False); beta +0.59; vol-matched ann.ret +18.3%.
- **eq_weight** [REJECTED] (Equal-Weight universe (baseline)): pooled-OOS Sharpe +1.07 vs SPY +0.91; IR vs SPY -0.30 (t=-0.73); DSR-prob 0.79 (pass5%=False); beta +0.61; vol-matched ann.ret +20.8%.

**ALL 11 strategies REJECTED through the LITERAL pipeline** — none clears SPY's pooled-OOS Sharpe with a DSR-deflated AND significant (IR t>1.96) edge, even executed through the real production cycle (feature enrichment → order generation → cost simulation). Pipeline-realism CONFIRMS the prior vectorized-harness rejections rather than overturning them: real costs + share-rounding only reduce returns, enrichment is immaterial to pure-price signals, and with risk-controls OFF there is no overlay that could improve the risk-adjusted outcome.

## Consolidated OOS-Edge table (pooled out-of-sample, LITERAL pipeline)

_Beta ≈ 0 confirms market-neutrality of the L/S books. IR vs SPY = annualised mean excess-over-SPY / its vol; IR t = IR·√years (|t|>1.96 ≈ 5% significant). DSR-prob is deflated for n_trials (Bailey-López de Prado); DSR✓ = passes 5%. PSR>SPY = prob true Sharpe exceeds SPY's. VolMatchRet = annual return if levered to SPY's vol, net of financing — the honest 'beats SPY CAGR?' figure for a market-neutral book. NOTE: this table's SPY bench is POOLED-daily over all 1512 OOS bars (AnnSharpe +0.91 / CAGR +17.4%); the per-strategy fold tables below show SPY as the mean-of-6-folds (Sharpe +1.40 / CAGR +19.7%). Two different estimators by construction, not an inconsistency._

| Strategy | AnnSharpe | Sharpe t | CAGR | MaxDD | Beta | IR vs SPY | IR t | DSR-prob | DSR✓ | PSR>SPY | Trn/yr | FoldWin | VolMatchRet |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bab_ls | -0.06 | -0.14 | -2.1% | -49.1% | +0.02 | -0.76 | -1.86 | 0.03 | N | 0.01 | 8.4 | 1/6 | -1.4% |
| mom_ls | +0.80 | +1.95 | +16.8% | -33.2% | +0.02 | +0.01 | +0.01 | 0.56 | N | 0.40 | 12.0 | 3/6 | +15.8% |
| resmom_ls | -0.10 | -0.24 | -4.4% | -46.2% | -0.02 | -0.68 | -1.67 | 0.02 | N | 0.01 | 22.7 | 2/6 | -2.0% |
| reversal_ls | -0.42 | -1.03 | -12.7% | -61.1% | +0.16 | -0.96 | -2.35 | 0.00 | N | 0.00 | 28.0 | 1/6 | -8.4% |
| lowvol_ls | -0.68 | -1.67 | -23.0% | -79.1% | -0.63 | -0.91 | -2.23 | 0.00 | N | 0.00 | 6.4 | 1/6 | -13.5% |
| mom_lo | +1.38 | +3.39 | +29.3% | -27.7% | +0.66 | +0.58 | +1.42 | 0.94 | N | 0.87 | 6.3 | 3/6 | +27.4% |
| high52w_lo | +1.16 | +2.83 | +12.6% | -14.3% | +0.36 | -0.37 | -0.91 | 0.84 | N | 0.72 | 11.1 | 2/6 | +22.1% |
| reversal_lo | +0.73 | +1.79 | +13.8% | -27.9% | +0.73 | -0.19 | -0.47 | 0.50 | N | 0.33 | 14.8 | 3/6 | +14.5% |
| lowbeta_lo | +1.21 | +2.95 | +9.4% | -12.1% | +0.22 | -0.52 | -1.28 | 0.87 | N | 0.76 | 3.4 | 3/6 | +22.3% |
| resmom_lo | +0.94 | +2.30 | +14.1% | -22.2% | +0.59 | -0.28 | -0.69 | 0.69 | N | 0.53 | 11.6 | 2/6 | +18.3% |
| eq_weight | +1.07 | +2.62 | +15.0% | -21.1% | +0.61 | -0.30 | -0.73 | 0.79 | N | 0.65 | 2.6 | 2/6 | +20.8% |
| **SPY (bench)** | +0.91 | +2.22 | +17.4% | -33.7% | +1.00 | — | — | 0.66 | N | — | 0 | — | +17.4% |

## Betting-Against-Beta L/S, beta-targeted (Frazzini-Pedersen 2014)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +3.8% | +0.44 | -11.0% | -0.22 | 2.43 | +35.7% | +2.54 | 7.3 |
| 2 | 2020-01-03–2020-12-31 | -38.3% | -1.72 | -38.5% | +0.23 | 1.97 | +18.2% | +0.67 | 9.5 |
| 3 | 2021-01-04–2021-12-31 | +9.5% | +0.69 | -13.8% | -0.25 | 2.35 | +30.6% | +2.13 | 9.7 |
| 4 | 2022-01-03–2023-01-03 | +13.9% | +1.18 | -8.6% | -0.07 | 1.88 | -19.1% | -0.75 | 6.9 |
| 5 | 2023-01-04–2024-01-04 | +14.8% | +1.03 | -14.1% | -0.25 | 2.40 | +23.7% | +1.69 | 8.1 |
| 6 | 2024-01-05–2025-01-06 | -3.8% | -0.29 | -13.4% | -0.23 | 2.28 | +29.0% | +2.08 | 8.9 |
| **Ø (6/6)** | — | **-0.0%** | **+0.22** | **-16.6%** | **-0.13** | **2.22** | +19.7% | +1.40 | 8.4 |

## Cross-sectional Momentum L/S, 12-1 WML (Jegadeesh-Titman 1993)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +19.9% | +1.28 | -12.7% | +0.14 | 1.81 | +35.7% | +2.54 | 13.4 |
| 2 | 2020-01-03–2020-12-31 | +38.1% | +1.62 | -10.9% | +0.04 | 1.46 | +18.2% | +0.67 | 10.7 |
| 3 | 2021-01-04–2021-12-31 | -7.9% | -0.19 | -33.2% | +0.81 | 1.62 | +30.6% | +2.13 | 12.3 |
| 4 | 2022-01-03–2023-01-03 | +30.9% | +1.50 | -11.9% | -0.37 | 1.31 | -19.1% | -0.75 | 9.8 |
| 5 | 2023-01-04–2024-01-04 | +47.6% | +2.29 | -6.6% | -0.20 | 1.73 | +23.7% | +1.69 | 14.0 |
| 6 | 2024-01-05–2025-01-06 | -13.7% | -0.28 | -32.3% | +0.59 | 1.58 | +29.0% | +2.08 | 12.0 |
| **Ø (6/6)** | — | **+19.1%** | **+1.04** | **-17.9%** | **+0.17** | **1.58** | +19.7% | +1.40 | 12.0 |

## Residual Momentum L/S (Blitz-Huij-Martens 2011)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +18.1% | +1.65 | -9.5% | -0.29 | 1.81 | +35.7% | +2.54 | 25.2 |
| 2 | 2020-01-03–2020-12-31 | +15.3% | +0.92 | -14.6% | -0.02 | 1.46 | +18.2% | +0.67 | 20.3 |
| 3 | 2021-01-04–2021-12-31 | -4.1% | -0.30 | -10.5% | +0.03 | 1.62 | +30.6% | +2.13 | 23.1 |
| 4 | 2022-01-03–2023-01-03 | -2.1% | -0.10 | -11.3% | -0.02 | 1.31 | -19.1% | -0.75 | 19.7 |
| 5 | 2023-01-04–2024-01-04 | +5.8% | +0.52 | -8.8% | +0.09 | 1.73 | +23.7% | +1.69 | 23.9 |
| 6 | 2024-01-05–2025-01-06 | -43.7% | -1.08 | -44.1% | +0.03 | 1.58 | +29.0% | +2.08 | 24.3 |
| **Ø (6/6)** | — | **-1.8%** | **+0.27** | **-16.5%** | **-0.03** | **1.58** | +19.7% | +1.40 | 22.7 |

## 1-Month Reversal L/S (Jegadeesh 1990)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | -10.2% | -0.59 | -18.0% | -0.20 | 1.75 | +35.7% | +2.54 | 30.4 |
| 2 | 2020-01-03–2020-12-31 | -22.3% | -1.11 | -24.0% | +0.16 | 1.46 | +18.2% | +0.67 | 24.7 |
| 3 | 2021-01-04–2021-12-31 | +20.0% | +1.24 | -7.2% | -0.07 | 1.57 | +30.6% | +2.13 | 30.9 |
| 4 | 2022-01-03–2023-01-03 | -5.5% | -0.14 | -20.1% | +0.35 | 1.31 | -19.1% | -0.75 | 23.4 |
| 5 | 2023-01-04–2024-01-04 | +2.9% | +0.25 | -14.1% | +0.03 | 1.68 | +23.7% | +1.69 | 30.3 |
| 6 | 2024-01-05–2025-01-06 | -45.6% | -1.15 | -53.2% | +0.17 | 1.58 | +29.0% | +2.08 | 28.0 |
| **Ø (6/6)** | — | **-10.1%** | **-0.25** | **-22.8%** | **+0.07** | **1.56** | +19.7% | +1.40 | 28.0 |

## Low-Volatility L/S

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | -43.2% | -1.85 | -43.2% | -1.17 | 1.81 | +35.7% | +2.54 | 6.6 |
| 2 | 2020-01-03–2020-12-31 | -57.0% | -2.66 | -64.0% | -0.37 | 1.46 | +18.2% | +0.67 | 7.1 |
| 3 | 2021-01-04–2021-12-31 | +7.8% | +0.42 | -27.1% | -0.85 | 1.62 | +30.6% | +2.13 | 6.7 |
| 4 | 2022-01-03–2023-01-03 | +19.7% | +0.79 | -21.2% | -0.73 | 1.31 | -19.1% | -0.75 | 5.1 |
| 5 | 2023-01-04–2024-01-04 | -0.6% | +0.11 | -27.9% | -1.12 | 1.73 | +23.7% | +1.69 | 5.5 |
| 6 | 2024-01-05–2025-01-06 | -33.4% | -0.68 | -45.3% | -0.81 | 1.58 | +29.0% | +2.08 | 7.1 |
| **Ø (6/6)** | — | **-17.8%** | **-0.64** | **-38.1%** | **-0.84** | **1.58** | +19.7% | +1.40 | 6.4 |

## Total-Return Momentum 12-1, long-only (control)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +45.2% | +2.35 | -7.1% | +0.77 | 0.90 | +35.7% | +2.54 | 6.5 |
| 2 | 2020-01-03–2020-12-31 | +57.2% | +1.93 | -24.8% | +0.54 | 0.77 | +18.2% | +0.67 | 6.2 |
| 3 | 2021-01-04–2021-12-31 | +6.9% | +0.39 | -27.7% | +1.15 | 0.83 | +30.6% | +2.13 | 6.1 |
| 4 | 2022-01-03–2023-01-03 | +6.6% | +0.47 | -11.6% | +0.54 | 0.67 | -19.1% | -0.75 | 5.2 |
| 5 | 2023-01-04–2024-01-04 | +40.6% | +2.34 | -7.0% | +0.78 | 0.88 | +23.7% | +1.69 | 8.0 |
| 6 | 2024-01-05–2025-01-06 | +27.7% | +1.34 | -13.1% | +1.15 | 0.81 | +29.0% | +2.08 | 6.0 |
| **Ø (6/6)** | — | **+30.7%** | **+1.47** | **-15.2%** | **+0.82** | **0.81** | +19.7% | +1.40 | 6.3 |

## 52-Week-High Momentum, long-only (George-Hwang 2004)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +16.0% | +1.61 | -3.7% | +0.54 | 0.90 | +35.7% | +2.54 | 12.0 |
| 2 | 2020-01-03–2020-12-31 | +20.6% | +1.35 | -14.3% | +0.30 | 0.77 | +18.2% | +0.67 | 11.6 |
| 3 | 2021-01-04–2021-12-31 | +20.6% | +2.06 | -3.9% | +0.56 | 0.83 | +30.6% | +2.13 | 12.1 |
| 4 | 2022-01-03–2023-01-03 | +2.3% | +0.26 | -12.0% | +0.31 | 0.67 | -19.1% | -0.75 | 5.7 |
| 5 | 2023-01-04–2024-01-04 | +3.1% | +0.37 | -8.9% | +0.50 | 0.88 | +23.7% | +1.69 | 12.9 |
| 6 | 2024-01-05–2025-01-06 | +14.6% | +1.46 | -4.4% | +0.48 | 0.81 | +29.0% | +2.08 | 12.0 |
| **Ø (6/6)** | — | **+12.9%** | **+1.18** | **-7.9%** | **+0.45** | **0.81** | +19.7% | +1.40 | 11.1 |

## 1-Month Reversal, long-only (Jegadeesh 1990)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +32.9% | +2.58 | -5.0% | +0.60 | 0.90 | +35.7% | +2.54 | 16.8 |
| 2 | 2020-01-03–2020-12-31 | +24.8% | +1.14 | -23.3% | +0.56 | 0.77 | +18.2% | +0.67 | 13.8 |
| 3 | 2021-01-04–2021-12-31 | +24.1% | +1.50 | -8.6% | +0.73 | 0.83 | +30.6% | +2.13 | 16.7 |
| 4 | 2022-01-03–2023-01-03 | -12.3% | -0.25 | -27.9% | +1.05 | 0.67 | -19.1% | -0.75 | 11.9 |
| 5 | 2023-01-04–2024-01-04 | +8.1% | +0.52 | -16.4% | +0.83 | 0.88 | +23.7% | +1.69 | 15.7 |
| 6 | 2024-01-05–2025-01-06 | +11.3% | +0.63 | -14.5% | +0.84 | 0.81 | +29.0% | +2.08 | 14.0 |
| **Ø (6/6)** | — | **+14.8%** | **+1.02** | **-16.0%** | **+0.77** | **0.81** | +19.7% | +1.40 | 14.8 |

## Low-Beta tilt, long-only (BAB no-leverage subset)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +22.0% | +3.10 | -3.2% | +0.15 | 0.90 | +35.7% | +2.54 | 3.2 |
| 2 | 2020-01-03–2020-12-31 | +10.5% | +0.92 | -12.1% | +0.25 | 0.77 | +18.2% | +0.67 | 4.1 |
| 3 | 2021-01-04–2021-12-31 | +8.5% | +1.31 | -6.2% | +0.28 | 0.83 | +30.6% | +2.13 | 3.6 |
| 4 | 2022-01-03–2023-01-03 | +3.1% | +0.43 | -8.6% | +0.20 | 0.67 | -19.1% | -0.75 | 2.8 |
| 5 | 2023-01-04–2024-01-04 | +9.1% | +1.34 | -3.7% | +0.19 | 0.88 | +23.7% | +1.69 | 3.5 |
| 6 | 2024-01-05–2025-01-06 | +4.1% | +0.77 | -5.8% | +0.09 | 0.81 | +29.0% | +2.08 | 3.6 |
| **Ø (6/6)** | — | **+9.5%** | **+1.31** | **-6.6%** | **+0.19** | **0.81** | +19.7% | +1.40 | 3.4 |

## Residual Momentum, long-only (Blitz-Huij-Martens 2011)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +26.6% | +2.51 | -3.5% | +0.48 | 0.90 | +35.7% | +2.54 | 12.1 |
| 2 | 2020-01-03–2020-12-31 | +36.8% | +1.61 | -22.2% | +0.51 | 0.77 | +18.2% | +0.67 | 11.0 |
| 3 | 2021-01-04–2021-12-31 | +13.7% | +1.02 | -7.9% | +0.76 | 0.83 | +30.6% | +2.13 | 11.6 |
| 4 | 2022-01-03–2023-01-03 | -8.8% | -0.38 | -19.8% | +0.67 | 0.67 | -19.1% | -0.75 | 10.2 |
| 5 | 2023-01-04–2024-01-04 | +10.3% | +0.79 | -8.5% | +0.79 | 0.88 | +23.7% | +1.69 | 11.9 |
| 6 | 2024-01-05–2025-01-06 | +11.5% | +0.96 | -10.2% | +0.64 | 0.81 | +29.0% | +2.08 | 12.7 |
| **Ø (6/6)** | — | **+15.0%** | **+1.08** | **-12.0%** | **+0.64** | **0.81** | +19.7% | +1.40 | 11.6 |

## Equal-Weight universe (baseline)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |
|------|-------------|------|--------|-------|------|-------|----------|------------|--------|
| 1 | 2019-01-03–2020-01-02 | +25.1% | +2.32 | -4.6% | +0.65 | 0.90 | +35.7% | +2.54 | 2.3 |
| 2 | 2020-01-03–2020-12-31 | +28.2% | +1.40 | -21.1% | +0.53 | 0.77 | +18.2% | +0.67 | 3.7 |
| 3 | 2021-01-04–2021-12-31 | +16.4% | +1.48 | -5.5% | +0.70 | 0.83 | +30.6% | +2.13 | 2.5 |
| 4 | 2022-01-03–2023-01-03 | -8.7% | -0.41 | -18.9% | +0.70 | 0.67 | -19.1% | -0.75 | 2.2 |
| 5 | 2023-01-04–2024-01-04 | +12.4% | +1.08 | -9.9% | +0.75 | 0.88 | +23.7% | +1.69 | 2.1 |
| 6 | 2024-01-05–2025-01-06 | +20.5% | +1.70 | -6.7% | +0.64 | 0.81 | +29.0% | +2.08 | 2.7 |
| **Ø (6/6)** | — | **+15.7%** | **+1.26** | **-11.1%** | **+0.66** | **0.81** | +19.7% | +1.40 | 2.6 |

# Part B — Live-Owned Registered Strategies

**Scope split.** Part A (above) pushes the 11 *research* concepts through the literal
`run_trading_cycle`. Part B does the apples-to-apples counterpart for the strategies the repo
already registers and runs in paper/live: `low_max_lottery`, `trend_baseline`, `multifactor_v2`,
`dual_momentum`, `etf_pairs_meanrev`. Goal: does the same pipeline-realism that confirmed every
research rejection rescue any *owned* strategy above SPY on a risk-adjusted basis? It does not.

Two evidence tiers are kept distinct and **not** mixed:
- **Literal (this harness):** driven through the same `run_portfolio_backtest` / `make_cycle_fn`
  path, same 75-symbol universe, same 6 folds (2019–2024), same SPY pooled bench
  (AnnSharpe **+0.91** / CAGR **+17.4%**). Directly comparable to Part A.
- **Cited (prior runs):** numbers read verbatim from earlier result docs that used a *different*
  universe/period/fold count. Useful as corroboration, **not** as same-axis comparison. Each cite
  names its source doc.

## B.1 low_max_lottery — LITERAL (directly comparable to Part A)

Production "lottery-aversion" concept (long the lowest 20% by trailing 20-day max daily return)
driven through the literal pipeline this session (`lowmax_lo` mode, real cost model, risk-controls
OFF — same config as Part A).

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe |
|------|-------------|------|--------|-------|------|-------|----------|------------|
| 1 | 2019-01-03–2020-01-02 | +15.4% | +2.47 | -4.0% | +0.38 | 0.88 | +35.7% | +2.54 |
| 2 | 2020-01-03–2020-12-31 | +11.0% | +1.00 | -14.1% | +0.28 | 0.79 | +18.2% | +0.67 |
| 3 | 2021-01-04–2021-12-31 | +8.0% | +1.28 | -4.3% | +0.34 | 0.85 | +30.6% | +2.13 |
| 4 | 2022-01-03–2023-01-03 | +0.4% | +0.09 | -10.1% | +0.33 | 0.68 | -19.1% | -0.75 |
| 5 | 2023-01-04–2024-01-04 | +3.9% | +0.56 | -7.2% | +0.41 | 0.88 | +23.7% | +1.69 |
| 6 | 2024-01-05–2025-01-06 | +5.1% | +0.98 | -5.2% | +0.21 | 0.82 | +29.0% | +2.08 |
| **Ø (6/6)** | — | **+7.3%** | **+1.06** | **-7.5%** | **+0.33** | **0.82** | +19.7% | +1.39 |

**Pooled edge:** AnnSharpe **0.92** vs SPY 0.91 · CAGR **+7.2%** vs SPY +17.4% · IR vs SPY **-0.74
(t=-1.81)** · DSR-prob **0.67 (pass=False)**. → **REJECTED.** Matches SPY's *Sharpe* but at <half the
return and a strongly negative information ratio; the low beta (0.33) shows the defensive tilt is
real but it buys risk reduction, not excess return. Note: a prior *vectorized* standalone run
(`docs/results/2026_05_low_max_lottery_real_oos.md`) reported Ø CAGR +9.8% / Sharpe +1.06 against a
SPY +19.7% bench — the literal pipeline's real commission+spread+impact+share-rounding pulled CAGR
down to +7.2%, i.e. realism *deepened* the rejection, consistent with Part A's structural finding.

## B.2 trend_baseline — CITED (different universe/period — corroboration only)

From `docs/results/2026_05_trend_baseline_real_oos.md` (10-fold WF, 194 symbols, 2018–2025 Alpaca):
Ø CAGR **-6.1%** / Sharpe **-0.18** / MaxDD **-22.2%** vs SPY +13.0% / +0.95; **0/10 folds beat SPY.**
Negative verdict. This is the registered pilot strategy; its own OOS already fails SPY on every
axis, so pipeline-realism has nothing to rescue.

## B.3 multifactor_v2 — CITED (different universe/period — corroboration only)

From `docs/results/2026_05_mfv2_full_stack_real_oos.md` (10-fold, full activatable Altdata stack):
Ø CAGR **+10.7%** / Sharpe **+0.36** / MaxDD **-18.6%** vs SPY +13.0% / +0.95; TA-only baseline
+12.9% / +0.36 → **Sharpe Δ vs TA-only = +0.00** (the alt-data stack adds no risk-adjusted edge);
60% of folds beat SPY on CAGR but none on Sharpe. "Gemischt"/below-SPY on a risk-adjusted basis.

## B.3b dual_momentum — DRIVEN through the literal pipeline on a sibling Alpaca universe (REJECTED)

The offline `load_eod_prices` cache lacks VEU & BIL, so dual_momentum cannot run on Part B's
75-symbol universe. Rather than exclude it, a sibling harness sources the required 4-asset menu
(SPY / VEU / BIL / AGG) from Alpaca split-adjusted daily bars and drives the **registered**
`dual_momentum.compute_signals` through the **identical** literal machinery (real `run_trading_cycle`,
cost model, order-gen, enrichment, risk-controls OFF, monthly rebalance, 252/252/252 WF). This is a
DIFFERENT data source and DIFFERENT folds (7 folds 2017-08…2024-08, Alpaca starts earlier than the
offline cache), so it is comparable to Part B on the **execution axis only**, with its own SPY series.

Result: pooled-OOS AnnSharpe **+0.65** vs SPY +0.72 · CAGR **+7.6%** vs SPY +12.7% · IR vs SPY -0.49
(t=-1.29, not significant) · DSR-prob 0.47 (pass=False) · beta +0.52 · MaxDD -28.0% vs SPY -34.2%.
**REJECTED** — the absolute-momentum trend filter trims drawdowns modestly but drags absolute return
below buy-and-hold SPY in this bull-dominated sample, and the risk-adjusted edge is not significant.
Directionally consistent with the standalone vectorized study (`docs/results/2026_05_dual_momentum_real_oos.md`:
Ø CAGR 9.7% / Sharpe 0.98 vs SPY 14.5% / 1.26). Full doc: `docs/results/2026_05_dual_momentum_literal_oos.md`.

**Methodology note (material) — fill-model cash buffer.** dual_momentum deploys **98%** of capital
into the single held asset, not 100%. This is REQUIRED: the literal fill model's non-negative-cash
gate (`execution/fill_model.apply_cash_gate`) rejects any BUY whose `notional + cost` would drive
cash below ~0, so a 100%-notional single-asset order is structurally un-fillable — the position never
establishes and the equity path is a phantom (verified: at weight 1.0 fold-4 realized -8.2% with the
establishing BUY rejected, while SPY did +30.6%; at 0.98 the same fold realizes +20.7% with zero
rejects). **Implication for Part A/B:** any research book gross-100% invested loses its
last-alphabetical position to the same gate each rebalance — diluted across many names, but a known
small drag; single-asset dual_momentum merely exposes it in full.

## B.3c etf_pairs_meanrev — DRIVEN through the literal pipeline on its native pair universe (REJECTED)

etf_pairs_meanrev is structurally incompatible with the cross-sectional top/bottom-quintile harness
(it trades *cointegrated relative-value SHORT pairs*, not a rank), so it was previously listed as a
by-design exclusion. Rather than leave it excluded, a sibling harness sources the 6 default pairs
(12 ETFs: SPY/IVV, GDX/GDXJ, XLE/VDE, EWA/EWC, XLF/KBE, XLK/VGT) from Alpaca split-adjusted daily
bars and drives the **registered** `etf_pairs_meanrev` signal through the **identical** literal
machinery (real `run_trading_cycle`, cost model, order-gen, enrichment, risk-controls OFF, 252/252/252
WF) — but with two design-faithful differences from Part B: **DAILY** rebalance (it is a daily Z-score
strategy; monthly would miss most entries/exits) and **FULL long-short** (each active pair emits a
LONG + a SHORT leg at ±1/k → gross ≈ 200%, NET ≈ 0, market-neutral). 8 folds 2017-01…2025-01 on its
own Alpaca SPY series — comparable to Part B on the **execution axis only**.

Result: pooled-OOS AnnSharpe **-0.06** vs SPY +0.76 · CAGR **-0.6%** vs SPY +12.9% · IR vs SPY -0.77
(t=-2.18, **significantly negative**) · DSR-prob 0.02 (pass=False) · beta **+0.06** (market-neutral) ·
MaxDD -13.7% vs SPY -34.2% · vol-matched ann.ret -2.7%. **REJECTED** — a beta≈0 book structurally
trails a bull-market SPY on absolute CAGR (expected, not the interesting signal), but the risk-adjusted
line is *also* a miss: the net-of-cost spread alpha is negative-to-flat and IR-t is significantly
NEGATIVE. The book does show genuine crisis-period diversification (fold 6 = 2022: Sharpe +2.14 while
SPY did -0.78; fold 5 = 2021: +2.10), yet the 2017-18 and 2023 folds drag the pool below zero.
Costs bite hard: commission+spread+impact on every leg change at gross 200% is a heavy tax on a thin
relative-value edge, and NO short-borrow fee is even modelled (optimistic for the short side). Full
doc: `docs/results/2026_05_etf_pairs_literal_oos.md`.

**Short-side sanity (material).** All 8 folds recorded **0 rejected trades** at avg gross ≈ 1.6: the
fill model's non-negative-cash gate (which forced the 98% buffer for single-asset dual_momentum in
B.3b) does NOT bind here because the short SELL legs credit cash *before* the long BUY legs fill on
the same bar — so the market-neutral book self-funds its longs and needs no cash buffer. This is the
mirror-image confirmation of the B.3b cash-gate finding.

## B.5 Part B verdict

Pipeline-realism does **not** rescue any owned strategy above SPY on a risk-adjusted basis:
`low_max_lottery` matches SPY's Sharpe but loses ~10pp CAGR and has IR-t -1.81 (literal);
`trend_baseline` is negative on every axis (cited); `multifactor_v2` adds zero Sharpe over its own
TA-only baseline and sits below SPY risk-adjusted (cited); `dual_momentum` driven on its own Alpaca
4-asset menu is REJECTED (AnnSharpe 0.65 vs 0.72, CAGR 7.6% vs 12.7%, IR-t -1.29); `etf_pairs_meanrev`
driven on its native 6-pair Alpaca universe (daily, full long-short, market-neutral) is REJECTED
(AnnSharpe -0.06 vs 0.76, CAGR -0.6% vs 12.9%, IR-t -2.18, beta +0.06). Combined with
Part A (all 11 research
concepts REJECTED), the consolidated finding holds: **on this data, no strategy — new or owned —
clears SPY on a deflated/risk-adjusted basis.** Every Part B owned strategy has now been *driven
through the real cycle* (none remain as by-design exclusions): the cross-sectional names via the
matrix harness, dual_momentum and etf_pairs_meanrev via their native sibling harnesses. `mom_lo` (Part A) remains the single concept that
beats SPY on *absolute* CAGR (+29.3%) but fails the deflated-Sharpe and IR-t significance bars.

**Caveats (binding):** survivorship bias in the 75-symbol universe is OPTIMISTIC for long-only
books (B.1 and the long-only Part A modes) — the cross-section excludes delisted names, so realized
long-only returns are upward-biased; the true edge vs SPY is therefore *no better* than shown.
Costs model commission+spread+impact but **not** short-borrow/financing (immaterial for the
long-only B.1). All numbers are local; **CI-unverified.** Cited B.2/B.3 figures are on a different
universe/period and are corroboration, not same-axis comparison.

---
_Script: `scripts/_oos_wf_pipeline_realistic.py` (research harness; executes the real `run_trading_cycle`, reads the live `policy.yaml` read-only, EDITS no production module, and forces crisis-overlay state to dry-run via `ASSEMBLED_NO_CRISIS_OVERLAY=1` so it mutates NO production state. Risk-review note: earlier exploratory runs this session DID persist time-traveled crisis state to `output/ops/crisis_alpha_state.json`; that file was deleted (re-inits to the WATCH fail-safe default) and the env-isolation added — re-running `lowmax_lo` with isolation reproduced identical pooled numbers, confirming the state write was an inert side-effect, not a result input.)._  
_Selection/edge logic reused verbatim from `scripts/_oos_wf_leverage_short.py` (DRY)._  
_Pipeline entry: `src/assembled_core/qa/backtest_engine.py` (`run_portfolio_backtest` / `make_cycle_fn`)._  
_Edge helpers: `src/assembled_core/qa/deflated_sharpe.py`, `src/assembled_core/qa/metrics.py`._  