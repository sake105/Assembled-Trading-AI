# New-Factor Sweep — OOS Walk-Forward Backtest (3 NEW long-only signals)

Run date (UTC): 2026-05-31  
Data: local offline cache via `load_eod_prices(None)` — survivors only  
Universe: 75 tradeable symbols (data ≤ 2018-01-31, ≥ 500 bars; SPY excluded)  
Signals (each top-quintile, monthly, long-only, equal-weight):  
- **high52w** — close / trailing-252d-high (George-Hwang 2004)  
- **reversal_1m** — −(last 21-bar return), buy losers (Jegadeesh 1990)  
- **low_beta** — long lowest-beta quintile, rolling 252d OLS vs SPY (Frazzini-Pedersen 2014, long-only NO-leverage tilt only)  
WF: 252/252/252 (train/test/step)  
Costs: 10.75 bps per leg, 1-bar execution lag  

**Honesty note:** offline cache is survivors-only (no delisted names) → all of
these signals are INFLATED to some degree (the worst names that delisted are
absent; this especially flatters reversal, which buys losers). Treat any
outperformance as an OPTIMISTIC upper bound. The low_beta mode is the long-only
NO-leverage subset of Betting-Against-Beta, structurally weaker than the levered
long/short original. CI status: not run in CI; local one-shot only.

## Verdict (auto-generated)

- **high52w** [REJECTED]: Ø CAGR +17.1% vs SPY +19.7% (beats 2/6); Ø Sharpe +1.16 vs SPY +1.40 (beats 2/6); Ø MaxDD -10.7%.
- **reversal_1m** [REJECTED]: Ø CAGR +17.4% vs SPY +19.7% (beats 4/6); Ø Sharpe +0.96 vs SPY +1.40 (beats 3/6); Ø MaxDD -21.1%.
- **low_beta** [REJECTED]: Ø CAGR +12.1% vs SPY +19.7% (beats 1/6); Ø Sharpe +1.28 vs SPY +1.40 (beats 3/6); Ø MaxDD -9.7%.

**ALL THREE REJECTED as irrelevant** — none beats SPY risk-adjusted or absolute even on the survivorship-INFLATED offline universe. On a survivorship-clean universe they would be weaker still. No prospect; do not pursue.

## 52-Week-High Momentum (George-Hwang 2004)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | SPY MaxDD | Corr(SPY) | Trades/yr |
|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|-----------|
| 1 | 2019-01-03–2020-01-02 | +20.1% | +1.79 | -4.0% | +5.09 | +35.7% | +2.54 | -6.6% | +0.71 | 12 |
| 2 | 2020-01-03–2020-12-31 | +35.9% | +1.45 | -21.3% | +1.68 | +18.2% | +0.67 | -33.7% | +0.84 | 12 |
| 3 | 2021-01-04–2021-12-31 | +27.5% | +2.12 | -4.4% | +6.30 | +30.6% | +2.13 | -5.1% | +0.78 | 12 |
| 4 | 2022-01-03–2023-01-03 | +1.9% | +0.19 | -17.0% | +0.11 | -19.1% | -0.75 | -24.5% | +0.70 | 12 |
| 5 | 2023-01-04–2024-01-04 | +1.8% | +0.21 | -11.7% | +0.15 | +23.7% | +1.69 | -10.0% | +0.73 | 12 |
| 6 | 2024-01-05–2025-01-06 | +15.4% | +1.23 | -6.1% | +2.51 | +29.0% | +2.08 | -8.4% | +0.67 | 12 |
| **Ø (6/6)** | — | **+17.1%** | **+1.16** | **-10.7%** | **+2.64** | +19.7% | +1.40 | -14.7% | +0.74 | 12 |

## 1-Month Reversal (Jegadeesh 1990)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | SPY MaxDD | Corr(SPY) | Trades/yr |
|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|-----------|
| 1 | 2019-01-03–2020-01-02 | +44.5% | +2.69 | -5.8% | +7.65 | +35.7% | +2.54 | -6.6% | +0.66 | 12 |
| 2 | 2020-01-03–2020-12-31 | +24.6% | +0.76 | -37.8% | +0.65 | +18.2% | +0.67 | -33.7% | +0.93 | 12 |
| 3 | 2021-01-04–2021-12-31 | +33.7% | +1.58 | -9.9% | +3.40 | +30.6% | +2.13 | -5.1% | +0.62 | 12 |
| 4 | 2022-01-03–2023-01-03 | -16.1% | -0.18 | -32.9% | -0.49 | -19.1% | -0.75 | -24.5% | +0.81 | 12 |
| 5 | 2023-01-04–2024-01-04 | +3.4% | +0.26 | -23.1% | +0.15 | +23.7% | +1.69 | -10.0% | +0.63 | 12 |
| 6 | 2024-01-05–2025-01-06 | +14.2% | +0.64 | -17.2% | +0.83 | +29.0% | +2.08 | -8.4% | +0.58 | 12 |
| **Ø (6/6)** | — | **+17.4%** | **+0.96** | **-21.1%** | **+2.03** | +19.7% | +1.40 | -14.7% | +0.71 | 12 |

## Low-Beta Tilt (Frazzini-Pedersen 2014, long-only no-lev)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | SPY MaxDD | Corr(SPY) | Trades/yr |
|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|-----------|
| 1 | 2019-01-03–2020-01-02 | +25.6% | +3.11 | -4.1% | +6.22 | +35.7% | +2.54 | -6.6% | +0.28 | 9 |
| 2 | 2020-01-03–2020-12-31 | +16.3% | +0.81 | -21.6% | +0.75 | +18.2% | +0.67 | -33.7% | +0.85 | 6 |
| 3 | 2021-01-04–2021-12-31 | +10.2% | +1.24 | -8.1% | +1.26 | +30.6% | +2.13 | -5.1% | +0.57 | 8 |
| 4 | 2022-01-03–2023-01-03 | +5.1% | +0.48 | -11.6% | +0.44 | -19.1% | -0.75 | -24.5% | +0.64 | 7 |
| 5 | 2023-01-04–2024-01-04 | +10.1% | +1.28 | -4.9% | +2.08 | +23.7% | +1.69 | -10.0% | +0.39 | 10 |
| 6 | 2024-01-05–2025-01-06 | +5.1% | +0.73 | -8.1% | +0.63 | +29.0% | +2.08 | -8.4% | +0.21 | 10 |
| **Ø (6/6)** | — | **+12.1%** | **+1.28** | **-9.7%** | **+1.90** | +19.7% | +1.40 | -14.7% | +0.49 | 8 |

## Equal-Weight universe (baseline)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | SPY MaxDD | Corr(SPY) | Trades/yr |
|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|-----------|
| 1 | 2019-01-03–2020-01-02 | +33.9% | +2.57 | -5.5% | +6.11 | +35.7% | +2.54 | -6.6% | +0.83 | 1 |
| 2 | 2020-01-03–2020-12-31 | +43.9% | +1.30 | -30.5% | +1.44 | +18.2% | +0.67 | -33.7% | +0.97 | 1 |
| 3 | 2021-01-04–2021-12-31 | +24.3% | +1.66 | -7.1% | +3.44 | +30.6% | +2.13 | -5.1% | +0.84 | 1 |
| 4 | 2022-01-03–2023-01-03 | -10.9% | -0.32 | -22.9% | -0.48 | -19.1% | -0.75 | -24.5% | +0.94 | 1 |
| 5 | 2023-01-04–2024-01-04 | +12.1% | +0.91 | -13.0% | +0.93 | +23.7% | +1.69 | -10.0% | +0.87 | 1 |
| 6 | 2024-01-05–2025-01-06 | +25.3% | +1.66 | -9.4% | +2.68 | +29.0% | +2.08 | -8.4% | +0.80 | 1 |
| **Ø (6/6)** | — | **+21.4%** | **+1.30** | **-14.7%** | **+2.35** | +19.7% | +1.40 | -14.7% | +0.88 | 1 |

## Attribution (Ø across OK folds)

| Mode | Ø CAGR | Ø Sharpe | Ø MaxDD | Ø Calmar |
|------|--------|----------|---------|----------|
| high52w | +17.1% | +1.16 | -10.7% | +2.64 |
| reversal_1m | +17.4% | +0.96 | -21.1% | +2.03 |
| low_beta | +12.1% | +1.28 | -9.7% | +1.90 |
| eq_weight | +21.4% | +1.30 | -14.7% | +2.35 |
| **SPY (bench)** | +19.7% | +1.40 | -14.7% | — |

---
_Script: `scripts/_oos_wf_new_factors_sweep.py` (read-only research harness, no production changes)_  
_References: George & Hwang (2004) J. Finance 59(5); Jegadeesh (1990) J. Finance 45(3); Frazzini & Pedersen (2014) 'Betting Against Beta', J. Financial Economics 111(1)._  