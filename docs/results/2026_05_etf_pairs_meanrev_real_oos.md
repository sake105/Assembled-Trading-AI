# ETF-Pairs Cointegration Mean-Reversion — OOS Walk-Forward Backtest

Run date: 2026-05-29  
Data: Alpaca daily bars (local cache) — 2018-01-02 → 2026-05-18  
Strategy: `etf_pairs_meanrev` — rolling 252-bar Engle-Granger cointegration,
OLS hedge ratio, Z-score (60d), entry |Z|>2.0, exit |Z|<0.5, stop |Z|>3.5  
WF: 252-bar train / 252-bar test / 252-bar step

## Pairs used (local-data substitutes for original 6)

| Requested | Substitute | Rationale |
|-----------|------------|-----------|
| SPY/IVV   | SPY/IWM    | large-cap / small-cap US equity |
| GDX/GDXJ  | GLD/SLV    | gold / silver (precious metals) |
| XLK/VGT   | XLK/QQQ    | technology ETFs |
| EWA/EWC   | TLT/XLF    | rates / financials |
| XLF/KBE   | XLV/XLY    | healthcare / consumer discretionary |
| XLE/VDE   | XLE/XLI    | energy / industrials |

The original symbols (IVV, GDX, GDXJ, VDE, EWA, EWC, KBE, VGT) are not
present in the local Alpaca price cache.  Pairs are not the same and results
should not be compared directly to the original specification.

## Walk-Forward Results — Full (Long-Short)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | 60/40 CAGR | Corr(SPY) | Trades/yr | AvgHold | WinRate |
|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|-----------|---------|---------|
| 1 | 2019-01-03–2020-01-02 | -0.9% | -0.29 | -2.9% | -0.32 | +35.7% | +2.54 | +27.0% | +0.05 | 6 | 5d | 37.5% |
| 2 | 2020-01-03–2020-12-31 | +0.0% | — | +0.0% | — | +18.2% | +0.67 | +20.2% | — | 0 | — | — |
| 3 | 2021-01-04–2021-12-31 | +2.2% | +0.38 | -4.5% | +0.48 | +30.6% | +2.13 | +15.8% | +0.00 | 13 | 4d | 32.0% |
| 4 | 2022-01-03–2023-01-03 | -1.8% | -1.03 | -1.8% | -1.00 | -19.1% | -0.75 | -22.0% | -0.10 | 2 | 2d | 50.0% |
| 5 | 2023-01-04–2024-01-04 | -1.3% | -1.49 | -1.3% | -1.00 | +23.7% | +1.69 | +13.2% | -0.00 | 2 | 6d | 33.3% |
| 6 | 2024-01-05–2025-01-06 | -0.1% | -0.04 | -1.6% | -0.03 | +29.0% | +2.08 | +14.1% | -0.07 | 3 | 10d | 47.4% |
| **Ø (6/6)** | — | **-0.3%** | **-0.49** | **-2.0%** | — | +19.7% | +1.40 | +11.4% | -0.02 | 4 | 5d | 40.0% |

## Walk-Forward Results — Long-Only

| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | WinRate |
|------|-------------|------|--------|-------|--------|----------|------------|---------|
| 1 | 2019-01-03–2020-01-02 | +5.8% | +1.83 | -1.3% | +4.49 | +35.7% | +2.54 | 50.0% |
| 2 | 2020-01-03–2020-12-31 | +0.0% | — | +0.0% | — | +18.2% | +0.67 | — |
| 3 | 2021-01-04–2021-12-31 | +5.2% | +0.61 | -7.4% | +0.70 | +30.6% | +2.13 | 36.0% |
| 4 | 2022-01-03–2023-01-03 | -0.5% | -1.28 | -0.5% | -1.00 | -19.1% | -0.75 | 0.0% |
| 5 | 2023-01-04–2024-01-04 | +6.1% | +2.01 | -0.1% | +56.39 | +23.7% | +1.69 | 83.3% |
| 6 | 2024-01-05–2025-01-06 | +2.0% | +0.40 | -3.8% | +0.52 | +29.0% | +2.08 | 52.6% |
| **Ø (6/6)** | — | **+3.1%** | **+0.71** | **-2.2%** | — | +19.7% | +1.40 | 44.4% |

## Assessment

**Data note:** Substitute pairs used — results not directly comparable to original
6-pair specification.  Original symbols missing from local Alpaca cache.

**Full mode** Ø CAGR vs SPY: -20.0% | Ø Sharpe vs SPY: -1.89 | Ø SPY correlation: -0.02  

**Criterion check (GO_LIVE_CHECKLIST B-tier):**

| Criterion | Threshold | Achieved | Pass? |
|-----------|-----------|----------|-------|
| Ø CAGR > 5% | 5% | -0.3% | ✗ |
| Ø Sharpe > 0.5 | 0.5 | -0.49 | ✗ |
| MaxDD > -30% | -30% | -2.0% | ✓ |
| Beat SPY Sharpe | N/A | -0.49 vs +1.40 | — |

**Verdict:** Informational only — substitute pairs, not original spec.
Original pairs (IVV, GDXJ, VDE, EWA/EWC, KBE, VGT) require Alpaca data download.
Consider fetching original pairs for definitive Kandidat D assessment.

---
_Script: `scripts/_oos_wf_etf_pairs_meanrev.py`_  
_Strategy: `src/assembled_core/strategies/etf_pairs_meanrev.py`_  
_Tests: `tests/test_etf_pairs_meanrev_pit_safety.py`_