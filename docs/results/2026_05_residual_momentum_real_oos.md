# Residual Momentum — OOS Walk-Forward Backtest (NEW strategy)

Run date (UTC): 2026-05-31  
Data: local offline cache via `load_eod_prices(None)` — survivors only  
Universe: 75 tradeable symbols (data ≤ 2018-01-31, ≥ 500 bars; SPY excluded)  
Strategy: single-factor (market=SPY) residual momentum, top quintile, monthly, long-only, equal-weight  
Market model: rolling OLS r_i = α_i + β_i·r_SPY + e_i over 252 bars  
Formation: Σ residual over last 126 bars skipping 21, standardised by residual vol  
WF: 252/252/252 (train/test/step)  
Costs: 10.75 bps per leg, 1-bar execution lag  

**Honesty note:** offline cache is survivors-only (no delisted names) → momentum-type
signals are INFLATED (biggest losers that delisted are absent). Treat any outperformance
as an OPTIMISTIC upper bound. CI status: not run in CI; local one-shot only.

## Verdict (auto-generated)

- Ø CAGR (residual_mom): +18.8% vs SPY +19.7% → beats SPY CAGR in 2/6 folds
- Ø Sharpe (residual_mom): +1.00 vs SPY +1.40 → beats SPY Sharpe in 3/6 folds
- Ø MaxDD: -16.4% | Ø Calmar: +1.94
- Control — total-return momentum Ø Sharpe +1.22 / Ø CAGR +27.8%: residualisation does NOT raise Sharpe (+1.00 vs +1.22).

**REJECTED as irrelevant** — does NOT beat SPY risk-adjusted or absolute even on the survivorship-INFLATED offline universe. On a survivorship-clean universe it would be weaker still. No prospect; do not pursue.

## Residual Momentum — THE STRATEGY

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | SPY MaxDD | Corr(SPY) | Trades/yr |
|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|-----------|
| 1 | 2019-01-03–2020-01-02 | +33.8% | +2.62 | -5.0% | +6.77 | +35.7% | +2.54 | -6.6% | +0.63 | 12 |
| 2 | 2020-01-03–2020-12-31 | +55.3% | +1.40 | -33.4% | +1.66 | +18.2% | +0.67 | -33.7% | +0.90 | 12 |
| 3 | 2021-01-04–2021-12-31 | +20.2% | +1.15 | -8.8% | +2.28 | +30.6% | +2.13 | -5.1% | +0.73 | 12 |
| 4 | 2022-01-03–2023-01-03 | -15.5% | -0.47 | -25.1% | -0.62 | -19.1% | -0.75 | -24.5% | +0.84 | 12 |
| 5 | 2023-01-04–2024-01-04 | +10.3% | +0.69 | -10.6% | +0.96 | +23.7% | +1.69 | -10.0% | +0.78 | 12 |
| 6 | 2024-01-05–2025-01-06 | +9.0% | +0.60 | -15.4% | +0.58 | +29.0% | +2.08 | -8.4% | +0.68 | 12 |
| **Ø (6/6)** | — | **+18.8%** | **+1.00** | **-16.4%** | **+1.94** | +19.7% | +1.40 | -14.7% | +0.76 | 12 |

## Total-Return Momentum (control)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | SPY MaxDD | Corr(SPY) | Trades/yr |
|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|-----------|
| 1 | 2019-01-03–2020-01-02 | +35.9% | +2.32 | -8.5% | +4.24 | +35.7% | +2.54 | -6.6% | +0.50 | 12 |
| 2 | 2020-01-03–2020-12-31 | +68.6% | +1.58 | -31.4% | +2.19 | +18.2% | +0.67 | -33.7% | +0.85 | 12 |
| 3 | 2021-01-04–2021-12-31 | +12.1% | +0.62 | -17.2% | +0.71 | +30.6% | +2.13 | -5.1% | +0.71 | 12 |
| 4 | 2022-01-03–2023-01-03 | +7.0% | +0.43 | -13.2% | +0.53 | -19.1% | -0.75 | -24.5% | +0.81 | 12 |
| 5 | 2023-01-04–2024-01-04 | +31.2% | +1.64 | -10.2% | +3.06 | +23.7% | +1.69 | -10.0% | +0.74 | 12 |
| 6 | 2024-01-05–2025-01-06 | +11.8% | +0.71 | -14.9% | +0.79 | +29.0% | +2.08 | -8.4% | +0.76 | 12 |
| **Ø (6/6)** | — | **+27.8%** | **+1.22** | **-15.9%** | **+1.92** | +19.7% | +1.40 | -14.7% | +0.73 | 12 |

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
| residual_mom | +18.8% | +1.00 | -16.4% | +1.94 |
| total_mom | +27.8% | +1.22 | -15.9% | +1.92 |
| eq_weight | +21.4% | +1.30 | -14.7% | +2.35 |
| **SPY (bench)** | +19.7% | +1.40 | -14.7% | — |

---
_Script: `scripts/_oos_wf_residual_momentum.py` (read-only research harness, no production changes)_  
_Reference: Blitz, Huij & Martens (2011) 'Residual Momentum', J. Empirical Finance 18(3), 506-521._  
_Repo helpers (unused by any strategy): `src/assembled_core/features/residual_momentum.py`._