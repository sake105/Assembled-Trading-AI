# Low-MAX / Lottery-Avoidance Strategy — OOS Walk-Forward Backtest

Run date: 2026-05-29  
Data: Alpaca daily bars (local cache) — 2018-01-02 → 2025-12-31  
Universe: 75 tradeable symbols (data from ≤ 2018-01-31, ≥ 500 bars; SPY excluded)  
Strategy: Low-MAX quintile selection — monthly rebalancing, long-only, equal-weight within bucket  
MAX definition: max daily return over last 20 trading days  
Quintile: bottom/top 20% by MAX score  
WF: 252-bar train / 252-bar test / 252-bar step  
Costs: 10.75 bps per leg (one-sided), 1-bar execution lag  

Reference: Bali, Cakici & Whitelaw (2011) 'Maxing Out: Stocks as Lotteries and the
Cross-Section of Expected Returns'. Journal of Financial Economics 99(2), 427-446.

**Implementation note:** This OOS script is a standalone research replication.
It does NOT call `generate_low_max_signals_from_prices` from the production module.
Three parameters differ from the production strategy:

| Parameter | WF Script | Production Strategy |
|-----------|-----------|---------------------|
| MAX_LOOKBACK | 20 bars | 21 bars (default) |
| Rebalance anchor | Month-end (`freq="ME"`) | First trading day of month |
| Quintile method | `quantile(0.20)` + `<=` threshold | `pd.qcut(..., duplicates='drop')` |

Results below validate the MAX anomaly research hypothesis on the available universe;
they cannot be directly attributed to the production `low_max_lottery` module.

## Walk-Forward Results — Low-MAX (Bottom Quintile)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | 60/40 CAGR | Corr(SPY) | Trades/yr | AvgHold | WinRate |
|------|-------------|------|--------|-------|--------|----------|------------ |-----------|-----------|-----------|---------|---------|
| 1 | 2019-01-03–2020-01-02 | +20.8% | +2.67 | -5.1% | +4.12 | +35.7% | +2.54 | +27.0% | +0.81 | 12 | 252d | 59.1% |
| 2 | 2020-01-03–2020-12-31 | +16.2% | +0.83 | -21.3% | +0.76 | +18.2% | +0.67 | +20.2% | +0.94 | 12 | 252d | 58.3% |
| 3 | 2021-01-04–2021-12-31 | +11.2% | +1.41 | -5.0% | +2.25 | +30.6% | +2.13 | +15.8% | +0.82 | 12 | 252d | 54.0% |
| 4 | 2022-01-03–2023-01-03 | +0.6% | +0.11 | -13.1% | +0.04 | -19.1% | -0.75 | -22.0% | +0.84 | 12 | 252d | 50.0% |
| 5 | 2023-01-04–2024-01-04 | +2.6% | +0.34 | -9.6% | +0.27 | +23.7% | +1.69 | +13.2% | +0.75 | 12 | 252d | 49.6% |
| 6 | 2024-01-05–2025-01-06 | +7.7% | +1.02 | -6.3% | +1.21 | +29.0% | +2.08 | +14.1% | +0.65 | 12 | 252d | 52.4% |
| **Ø (6/6)** | — | **+9.8%** | **+1.06** | **-10.1%** | — | +19.7% | +1.40 | +11.4% | +0.80 | 12 | 252d | 53.9% |

## Walk-Forward Results — Equal-Weight Universe

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | 60/40 CAGR | Corr(SPY) | Trades/yr | AvgHold | WinRate |
|------|-------------|------|--------|-------|--------|----------|------------ |-----------|-----------|-----------|---------|---------|
| 1 | 2019-01-03–2020-01-02 | +46.0% | +3.00 | -5.5% | +8.31 | +35.7% | +2.54 | +27.0% | +0.94 | 0 | 252d | 61.5% |
| 2 | 2020-01-03–2020-12-31 | +46.4% | +1.34 | -30.5% | +1.52 | +18.2% | +0.67 | +20.2% | +0.98 | 0 | 252d | 60.3% |
| 3 | 2021-01-04–2021-12-31 | +26.9% | +1.70 | -7.1% | +3.80 | +30.6% | +2.13 | +15.8% | +0.90 | 0 | 252d | 55.2% |
| 4 | 2022-01-03–2023-01-03 | -16.7% | -0.55 | -25.5% | -0.65 | -19.1% | -0.75 | -22.0% | +0.96 | 0 | 252d | 47.6% |
| 5 | 2023-01-04–2024-01-04 | +21.5% | +1.41 | -13.0% | +1.65 | +23.7% | +1.69 | +13.2% | +0.93 | 0 | 252d | 53.2% |
| 6 | 2024-01-05–2025-01-06 | +29.3% | +1.84 | -9.4% | +3.10 | +29.0% | +2.08 | +14.1% | +0.82 | 0 | 252d | 54.0% |
| **Ø (6/6)** | — | **+25.6%** | **+1.46** | **-15.2%** | — | +19.7% | +1.40 | +11.4% | +0.92 | 0 | 252d | 55.3% |

## Walk-Forward Results — High-MAX (Top Quintile)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | 60/40 CAGR | Corr(SPY) | Trades/yr | AvgHold | WinRate |
|------|-------------|------|--------|-------|--------|----------|------------ |-----------|-----------|-----------|---------|---------|
| 1 | 2019-01-03–2020-01-02 | +97.4% | +3.14 | -8.3% | +11.72 | +35.7% | +2.54 | +27.0% | +0.72 | 12 | 252d | 59.9% |
| 2 | 2020-01-03–2020-12-31 | +134.4% | +2.13 | -37.8% | +3.56 | +18.2% | +0.67 | +20.2% | +0.88 | 12 | 252d | 63.1% |
| 3 | 2021-01-04–2021-12-31 | +21.0% | +0.75 | -25.2% | +0.83 | +30.6% | +2.13 | +15.8% | +0.62 | 12 | 252d | 57.5% |
| 4 | 2022-01-03–2023-01-03 | -39.7% | -0.75 | -45.9% | -0.86 | -19.1% | -0.75 | -22.0% | +0.83 | 12 | 252d | 46.8% |
| 5 | 2023-01-04–2024-01-04 | +19.8% | +0.75 | -26.2% | +0.75 | +23.7% | +1.69 | +13.2% | +0.78 | 12 | 252d | 51.6% |
| 6 | 2024-01-05–2025-01-06 | +52.9% | +1.28 | -35.1% | +1.51 | +29.0% | +2.08 | +14.1% | +0.53 | 12 | 252d | 53.6% |
| **Ø (6/6)** | — | **+47.6%** | **+1.22** | **-29.8%** | — | +19.7% | +1.40 | +11.4% | +0.73 | 12 | 252d | 55.4% |

## MAX-Spread (CAGR Low-MAX minus CAGR High-MAX)

| Fold | Test Period | CAGR Low-MAX | CAGR High-MAX | Spread |
|------|-------------|--------------|---------------|--------|
| 1 | 2019-01-03–2020-01-02 | +20.8% | +97.4% | -76.6% |
| 2 | 2020-01-03–2020-12-31 | +16.2% | +134.4% | -118.3% |
| 3 | 2021-01-04–2021-12-31 | +11.2% | +21.0% | -9.9% |
| 4 | 2022-01-03–2023-01-03 | +0.6% | -39.7% | +40.3% |
| 5 | 2023-01-04–2024-01-04 | +2.6% | +19.8% | -17.2% |
| 6 | 2024-01-05–2025-01-06 | +7.7% | +52.9% | -45.3% |
| **Ø** | — | **+9.8%** | **+47.6%** | **-37.8%** |

## Assessment

### 1. Does Low-MAX beat Equal-Weight risk-adjusted?

- Ø CAGR Low-MAX vs Equal-Weight: -15.7% (NO)
- Ø Sharpe Low-MAX vs Equal-Weight: -0.39 (NO)

### 2. Is the MAX-Spread positive (Lottery effect present)?

- Ø MAX-Spread (Low-MAX CAGR − High-MAX CAGR): -37.8%  
- **NO — Lottery effect absent or reversed**  
  (Positive spread confirms high-MAX stocks underperform low-MAX in-sample.)

### 3. Important Caveats

**Universe bias (large/mid-cap dampening):**  
The academic MAX effect is strongest in small- and microcap stocks (Bali et al. 2011).  
This universe (Alpaca local cache, mostly large/mid-cap, liquid names) systematically  
dampens the effect. A positive result here is noteworthy; a null result is ambiguous.  

**Survivorship bias:**  
The local Alpaca cache only contains currently-available (surviving) symbols.  
High-MAX stocks that blew up (the biggest lottery losers) are missing.  
This compresses the observable spread downward and makes high-MAX appear less bad  
than the full universe. A null result may understate the real effect.  

**No short leg:**  
The academic result is long-short. This backtest is long-only (bottom quintile only).  
Long-only capture of the factor is weaker and more market-beta-driven.  

### 4. GO_LIVE_CHECKLIST B-tier Criterion Check (Research Script — See Implementation Note)

| Criterion | Threshold | Achieved | Pass? |
|-----------|-----------|----------|-------|
| Ø CAGR > 5% | 5% | +9.8% | ✓ |
| Ø Sharpe > 0.5 | 0.5 | +1.06 | ✓ |
| MaxDD > -30% | -30% | -10.1% | ✓ |
| Beat SPY Sharpe | N/A | +1.06 vs +1.40 | ✗ |

---
_Script: `scripts/_oos_wf_low_max_lottery.py`_  
_Strategy: `src/assembled_core/strategies/low_max_lottery.py`_  
_Tests: `tests/test_low_max_lottery_pit_safety.py`_  
_Reference: Bali, Cakici & Whitelaw (2011), J. Financial Economics 99(2)_