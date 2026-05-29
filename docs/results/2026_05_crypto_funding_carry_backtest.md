# Crypto Funding-Rate-Carry — Backtest Study

Run date: 2026-05-29

> **Disclaimer:** This is a pure backtest research study.
> No exchange account, no real-money execution, no EU/MiCA compliance assessment.
> Live implementation would require: exchange account setup, retail-access
> review under MiCA/BaFin rules, and a separate live risk model.
> Counterparty/exchange risk is **NOT modelled** here (see §5).

---

## 1. Data Source & Universe

| Field | Detail |
|-------|--------|
| Source | Binance FAPI public endpoint — no auth required |
| Endpoints | `/fapi/v1/fundingRate`, `/fapi/v1/klines?interval=8h` |
| Funding interval | 8 hours (Binance standard for USDT-M perps) |
| BTC (BTCUSDT) | 2019-09-10 → 2026-05-29 — 7,359 funding intervals |
| ETH (ETHUSDT) | 2019-11-27 → 2026-05-29 — 7,125 funding intervals |
| Cache | `data/crypto_funding_cache/` (Parquet) |

**Note on data quality:** Binance historical funding rates are generally complete
back to contract launch. A small number of intervals may show exactly 0.01%
(the exchange default floor) rather than a market-driven rate.

---

## 2. Strategy Rules

**Position:** Long Spot + Short Perp (delta-neutral) on the same notional X.

| Rule | Detail |
|------|--------|
| Entry | APR > 8% AND last 6 funding intervals all positive |
| Exit | APR < 5% OR funding becomes negative |
| Fees | 0.04% taker × 2 legs open + 2 legs close = 0.16% roundtrip |
| Execution lag | Signals based on interval-end rates; costs apply on transition |
| Notional (display) | USD 10,000 per asset |

**Negative funding is NOT filtered out.** When funding turns negative during an
active position, the short-perp leg PAYS funding (PnL negative that interval).
The exit rule closes the position only at the START of the next evaluation.

---

## 3. Performance Summary

### BTC (BTCUSDT)

| Metric | Value |
|--------|-------|
| Sample period | 2019-09-10 → 2026-05-29 (6.7 years) |
| Funding intervals | 7,359 total |
| % time positive funding | 85.4% |
| % time negative funding | 14.6% |
| Mean 8 h funding rate | +0.0108% |
| Median 8 h funding rate | +0.0100% |
| Implied mean APR (full sample) | +11.8% |
| % time in position | 57.0% |
| **Net APR after fees** | **+4.5%** |
| Total PnL (USD, $10,000 notional) | $+3,031 |
| Sharpe (all periods, flat=0) | +4.40 |
| Sharpe (active-only) | +5.86 ⚠ see caveat §4 |
| Max drawdown (USD) | $-1,389 |
| Worst single-interval PnL | $-8.0 |
| Return skewness | -0.19 |
| Excess kurtosis | +3.36 |

### ETH (ETHUSDT)

| Metric | Value |
|--------|-------|
| Sample period | 2019-11-27 → 2026-05-29 (6.5 years) |
| Funding intervals | 7,125 total |
| % time positive funding | 86.5% |
| % time negative funding | 13.5% |
| Mean 8 h funding rate | +0.0130% |
| Median 8 h funding rate | +0.0100% |
| Implied mean APR (full sample) | +14.3% |
| % time in position | 59.8% |
| **Net APR after fees** | **+6.7%** |
| Total PnL (USD, $10,000 notional) | $+4,378 |
| Sharpe (all periods, flat=0) | +5.64 |
| Sharpe (active-only) | +7.37 ⚠ see caveat §4 |
| Max drawdown (USD) | $-1,611 |
| Worst single-interval PnL | $-8.0 |
| Return skewness | +0.69 |
| Excess kurtosis | +6.11 |

### BTC — Year-by-Year Net PnL

| Year | Mean FR (8h) | % Time Positive | In-Position | Net PnL (USD) | Net APR |
|------|-------------|-----------------|-------------|---------------|---------|
| 2019 | +0.0068% | 81.7% | 64% | $+97 | +3.1% |
| 2020 | +0.0157% | 85.7% | 73% | $+1,542 | +15.4% |
| 2021 | +0.0280% | 92.7% | 83% | $+2,759 | +27.6% |
| 2022 | +0.0038% | 77.9% | 35% | $-755 | -7.6% |
| 2023 | +0.0072% | 89.9% | 55% | $-95 | -1.0% |
| 2024 | +0.0109% | 91.6% | 72% | $+545 | +5.4% |
| 2025 | +0.0047% | 87.1% | 40% | $-846 | -8.5% |
| 2026 | +0.0007% | 58.2% | 11% | $-216 | -5.3% |

### ETH — Year-by-Year Net PnL

| Year | Mean FR (8h) | % Time Positive | In-Position | Net PnL (USD) | Net APR |
|------|-------------|-----------------|-------------|---------------|---------|
| 2019 | +0.0081% | 95.2% | 85% | $-30 | -3.1% |
| 2020 | +0.0250% | 97.4% | 90% | $+2,465 | +24.6% |
| 2021 | +0.0343% | 95.9% | 87% | $+3,483 | +34.8% |
| 2022 | +0.0007% | 65.8% | 23% | $-651 | -6.5% |
| 2023 | +0.0075% | 90.9% | 57% | $-325 | -3.2% |
| 2024 | +0.0118% | 95.8% | 82% | $+462 | +4.6% |
| 2025 | +0.0045% | 83.8% | 36% | $-760 | -7.6% |
| 2026 | +0.0003% | 57.5% | 12% | $-267 | -6.6% |

---

## 4. Risk Analysis

### 4.1 Negative Funding Periods

**BTC**

- % of all intervals with negative funding: 14.6%
- Minimum 8 h rate recorded: -0.3000%  (annualized: -328.5%)
- Longest consecutive negative-funding streak: **24 intervals** = **192 hours** (8 days)

**ETH**

- % of all intervals with negative funding: 13.5%
- Minimum 8 h rate recorded: -0.3563%  (annualized: -390.2%)
- Longest consecutive negative-funding streak: **25 intervals** = **200 hours** (8 days)

### 4.2 Carry Sharpe — ⚠ Critical Caveat

> **The carry Sharpe ratio is structurally misleading.**
>
> Funding-rate carry is a classic 'picking up pennies in front of a steamroller'
> strategy. The return distribution looks like this:
>
> - **Normal regime** (most of the time): small positive returns every 8 hours.
>   Low variance → high apparent Sharpe.
> - **Tail regime** (rare): funding turns sharply negative during crypto market
>   dislocations (e.g. 2022 LUNA/3AC collapse, FTX implosion), or margin is called
>   on the perp leg before the spot can be liquidated.
>
> The Sharpe ratio computed from 8 h intervals does NOT capture this tail risk,
> because the tail events are exactly as rare as they are severe. The true
> risk-adjusted return is substantially lower than the Sharpe implies.
>
> **Do NOT use this Sharpe to compare against equity strategies.**
>
> Skewness: -0.19 / +0.69 (BTC/ETH)
> Excess kurtosis: +3.36 / +6.11
> Negative skew and fat tails confirm the steamroller profile.

### 4.3 Liquidation / Margin Risk (Short Perp Leg)

The short-perp position is subject to margin calls if price spikes sharply
upward before the opposing long-spot can be sold. Even if the net position
is delta-neutral on paper, exchanges liquidate the perp leg independently.

**BTC — largest single 8 h intrabar spike (open→high):**

- Worst upward spike: **+36.2%** on 2021-07-26
- At 5× isolated margin (liquidation at ≥20% adverse move):
  **2 intervals** in the sample exceeded the liquidation threshold
  (0.03% of all intervals).
- A liquidation would wipe the entire perp margin even though the spot leg gains.
  Net loss ≈ 1/(leverage) = **20% of total notional** (margin forfeited,
  spot gain partially offsetting — but timing and slippage make recovery uncertain).

**ETH — largest single 8 h intrabar spike (open→high):**

- Worst upward spike: **+202.0%** on 2020-03-13
- At 5× isolated margin (liquidation at ≥20% adverse move):
  **2 intervals** in the sample exceeded the liquidation threshold
  (0.03% of all intervals).
- A liquidation would wipe the entire perp margin even though the spot leg gains.
  Net loss ≈ 1/(leverage) = **20% of total notional** (margin forfeited,
  spot gain partially offsetting — but timing and slippage make recovery uncertain).

### 4.4 Counterparty / Exchange Risk (NOT MODELLED)

> **This risk cannot be quantified via backtesting.**
>
> Historical carry metrics assume the exchange remains solvent and accessible.
> Real-world evidence (FTX November 2022, Celsius, Voyager) shows that exchange
> failure can result in total loss of both legs of the position with no recovery.
>
> Additional unmodelled risks:
> - API downtime / inability to close positions during market stress
> - Regulatory freeze / asset seizure (especially EU/MiCA retail restrictions)
> - Smart-contract / oracle manipulation on the funding rate settlement
> - Forced ADL (auto-deleveraging) by the exchange during extreme volatility

### 4.5 99th-Percentile (VaR) per Interval

| Asset | VaR 99% (single 8 h interval, $10,000 notional) |
|-------|------|
| BTC | $-8.00 |
| ETH | $-8.00 |

---

## 5. Honest Assessment

### Does a carry edge remain after fees?

- **BTC** net APR after fees: **+4.5%**  
  Gross implied APR from mean rate: +11.8%  
  Fee drag (roundtrip × turnover): approx. +0.16% per trade  
  Edge remaining: **YES (marginal)**

- **ETH** net APR after fees: **+6.7%**  
  Gross implied APR from mean rate: +14.3%  
  Fee drag (roundtrip × turnover): approx. +0.16% per trade  
  Edge remaining: **YES (marginal)**

### Does the carry justify the tail and counterparty risk?

The carry is **real and persistent** — Bitcoin and Ethereum perpetual funding has
been consistently positive over most of the sample, reflecting leveraged-long
demand from retail speculators paying the funding rate to hold perp longs.

However:

1. **Negative-funding periods are common** (≥10–20% of intervals) and can
   sustain for days to weeks during bear markets. A position cannot always
   be closed instantly if an exchange has withdrawal queues or halts.

2. **The Sharpe is structurally overstated.** The 'risk' in the denominator
   does not include the probability of exchange insolvency or a 30–40% funding
   rate collapse (e.g. the LUNA-2022 or FTX-2022 events).

3. **Margin liquidation risk is non-trivial.** Even at conservative 2–3× leverage,
   extreme intrabar moves can trigger liquidation before the spot leg can be sold.

4. **Institutional vs. retail:** This strategy is routinely run by crypto market
   makers and hedge funds with direct exchange APIs, insurance funds, and legal
   recourse. A retail investor operates at a structural disadvantage.

**Conclusion:** The carry edge exists and is persistent, but the unmodelled tail
and counterparty risks make it inappropriate to evaluate this strategy on its
backtest Sharpe alone. It may be suitable as one small component of a
well-diversified portfolio for investors with crypto-native infrastructure
and appropriate risk controls — but **not as a standalone strategy**.

---

_Script: `scripts/crypto_funding_carry_backtest.py`_  
_Data: Binance FAPI (public, no auth)_  
_Cache: `data/crypto_funding_cache/`_  
_This is a research-only backtest. No live execution components._