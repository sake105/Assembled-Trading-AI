# Cross-Sectional Low-Vol + Momentum — OOS Walk-Forward Backtest (NEW strategy)

Run date: 2026-05-31  
Data: local offline cache via `load_eod_prices(None)` — survivors only  
Universe: 75 tradeable symbols (data ≤ 2018-01-31, ≥ 500 bars; SPY excluded)  
Strategy: top-quintile by 0.5·lowvol_rank + 0.5·momentum_rank, monthly, long-only, equal-weight  
Low-vol: std of last 60 daily returns  
Momentum: price[t-21] / price[t-147] − 1 (126d formation, skip 21d)  
WF: 252/252/252 (train/test/step)  
Costs: 10.75 bps per leg (one-sided), 1-bar execution lag  

**Honesty note:** offline cache is survivors-only (no delisted names). This *inflates*
the momentum sleeve (the biggest losers that delisted are absent) and the whole universe.
Treat any outperformance here as an OPTIMISTIC upper bound, not a production claim.
CI status: not run in CI; local one-shot only.

## Verdict (auto-generated)

- Ø CAGR (combo): +17.5% vs SPY +19.7% → beats SPY CAGR in 2/6 folds
- Ø Sharpe (combo): +1.30 vs SPY +1.40 → beats SPY Sharpe in 2/6 folds
- Ø MaxDD (combo): -10.3% | Ø Calmar: +2.55

**REJECTED as irrelevant** — does NOT beat SPY risk-adjusted or absolute even on the survivorship-INFLATED offline universe. On a survivorship-clean universe it would be weaker still. No prospect; do not pursue.

## Combo (Low-Vol + Momentum) — THE STRATEGY

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | SPY MaxDD | Corr(SPY) | Trades/yr |
|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|-----------|
| 1 | 2019-01-03–2020-01-02 | +22.0% | +2.16 | -4.4% | +4.99 | +35.7% | +2.54 | -6.6% | +0.69 | 12 |
| 2 | 2020-01-03–2020-12-31 | +32.9% | +1.21 | -25.3% | +1.30 | +18.2% | +0.67 | -33.7% | +0.90 | 11 |
| 3 | 2021-01-04–2021-12-31 | +28.4% | +2.00 | -6.1% | +4.67 | +30.6% | +2.13 | -5.1% | +0.89 | 12 |
| 4 | 2022-01-03–2023-01-03 | -4.4% | -0.22 | -13.3% | -0.33 | -19.1% | -0.75 | -24.5% | +0.78 | 12 |
| 5 | 2023-01-04–2024-01-04 | +8.9% | +0.91 | -7.9% | +1.13 | +23.7% | +1.69 | -10.0% | +0.84 | 12 |
| 6 | 2024-01-05–2025-01-06 | +17.3% | +1.77 | -4.9% | +3.56 | +29.0% | +2.08 | -8.4% | +0.79 | 12 |
| **Ø (6/6)** | — | **+17.5%** | **+1.30** | **-10.3%** | **+2.55** | +19.7% | +1.40 | -14.7% | +0.82 | 12 |

## Low-Vol sleeve alone

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | SPY MaxDD | Corr(SPY) | Trades/yr |
|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|-----------|
| 1 | 2019-01-03–2020-01-02 | +20.4% | +2.89 | -2.6% | +7.87 | +35.7% | +2.54 | -6.6% | +0.72 | 12 |
| 2 | 2020-01-03–2020-12-31 | +16.7% | +0.89 | -21.2% | +0.79 | +18.2% | +0.67 | -33.7% | +0.93 | 11 |
| 3 | 2021-01-04–2021-12-31 | +13.4% | +1.67 | -4.8% | +2.77 | +30.6% | +2.13 | -5.1% | +0.80 | 12 |
| 4 | 2022-01-03–2023-01-03 | -3.1% | -0.19 | -14.8% | -0.21 | -19.1% | -0.75 | -24.5% | +0.84 | 12 |
| 5 | 2023-01-04–2024-01-04 | +4.4% | +0.58 | -7.2% | +0.62 | +23.7% | +1.69 | -10.0% | +0.76 | 11 |
| 6 | 2024-01-05–2025-01-06 | +12.1% | +1.66 | -5.2% | +2.33 | +29.0% | +2.08 | -8.4% | +0.69 | 11 |
| **Ø (6/6)** | — | **+10.7%** | **+1.25** | **-9.3%** | **+2.36** | +19.7% | +1.40 | -14.7% | +0.79 | 12 |

## Momentum sleeve alone

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | SPY MaxDD | Corr(SPY) | Trades/yr |
|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|-----------|
| 1 | 2019-01-03–2020-01-02 | +65.8% | +2.46 | -11.1% | +5.91 | +35.7% | +2.54 | -6.6% | +0.69 | 12 |
| 2 | 2020-01-03–2020-12-31 | +116.0% | +1.95 | -36.4% | +3.19 | +18.2% | +0.67 | -33.7% | +0.82 | 12 |
| 3 | 2021-01-04–2021-12-31 | +22.4% | +0.82 | -22.5% | +1.00 | +30.6% | +2.13 | -5.1% | +0.71 | 12 |
| 4 | 2022-01-03–2023-01-03 | -17.1% | -0.61 | -24.7% | -0.69 | -19.1% | -0.75 | -24.5% | +0.81 | 12 |
| 5 | 2023-01-04–2024-01-04 | +40.9% | +1.67 | -13.4% | +3.06 | +23.7% | +1.69 | -10.0% | +0.78 | 12 |
| 6 | 2024-01-05–2025-01-06 | +49.3% | +1.67 | -20.6% | +2.40 | +29.0% | +2.08 | -8.4% | +0.80 | 12 |
| **Ø (6/6)** | — | **+46.2%** | **+1.33** | **-21.4%** | **+2.48** | +19.7% | +1.40 | -14.7% | +0.77 | 12 |

## Equal-Weight universe (baseline)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | SPY MaxDD | Corr(SPY) | Trades/yr |
|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|-----------|
| 1 | 2019-01-03–2020-01-02 | +46.0% | +3.00 | -5.5% | +8.31 | +35.7% | +2.54 | -6.6% | +0.94 | 0 |
| 2 | 2020-01-03–2020-12-31 | +46.4% | +1.34 | -30.5% | +1.52 | +18.2% | +0.67 | -33.7% | +0.98 | 0 |
| 3 | 2021-01-04–2021-12-31 | +26.9% | +1.70 | -7.1% | +3.80 | +30.6% | +2.13 | -5.1% | +0.90 | 0 |
| 4 | 2022-01-03–2023-01-03 | -16.7% | -0.55 | -25.5% | -0.65 | -19.1% | -0.75 | -24.5% | +0.96 | 0 |
| 5 | 2023-01-04–2024-01-04 | +21.5% | +1.41 | -13.0% | +1.65 | +23.7% | +1.69 | -10.0% | +0.93 | 0 |
| 6 | 2024-01-05–2025-01-06 | +29.3% | +1.84 | -9.4% | +3.10 | +29.0% | +2.08 | -8.4% | +0.82 | 0 |
| **Ø (6/6)** | — | **+25.6%** | **+1.46** | **-15.2%** | **+2.96** | +19.7% | +1.40 | -14.7% | +0.92 | 0 |

## Attribution (Ø across OK folds)

| Mode | Ø CAGR | Ø Sharpe | Ø MaxDD | Ø Calmar |
|------|--------|----------|---------|----------|
| combo | +17.5% | +1.30 | -10.3% | +2.55 |
| lowvol | +10.7% | +1.25 | -9.3% | +2.36 |
| momentum | +46.2% | +1.33 | -21.4% | +2.48 |
| eq_weight | +25.6% | +1.46 | -15.2% | +2.96 |
| **SPY (bench)** | +19.7% | +1.40 | -14.7% | — |

---
_Script: `scripts/_oos_wf_lowvol_momentum.py` (read-only research harness, no production changes)_  
_Low-vol anomaly: Baker/Bradley/Wurgler (2011); Frazzini/Pedersen 'Betting Against Beta' (2014)._  
_Momentum: Jegadeesh/Titman (1993); 12-1 / 6-1 formation. Combination = 'defensive momentum'._