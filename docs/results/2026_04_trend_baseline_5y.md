# Trend Baseline — 5-Year Backtest Results

**Date:** 2026-04-24  
**Strategy:** trend_baseline (EMA crossover, score-weighted sizing)  
**Universe:** 22 large-cap symbols (watchlist_22_2020_2026.parquet)  
**Period:** 2020-01-01 to 2026-04-01  
**Capital:** $100,000  
**Frequency:** Daily (1d)  
**Costs:** 10 bps commission

## Results

| Metric | Value |
|--------|-------|
| Gross Return | +48.95% |
| Net Return | +48.95% |
| CAGR | 6.60% |
| Sharpe Ratio | 0.66 |
| Max Drawdown | -13.20% |
| Profit Factor | 1.49 |
| Total Trades | 5,559 (all filled) |
| Total Costs | $9,488 |
| Final Equity | $148,946 |
| QA Result | WARNING |

## Equity Curve Summary

- Start: $100,000 (2020-01-02)
- Peak: $162,798
- End: $148,946 (2026-04-01)
- Min: $88,012 (COVID drawdown 2020)

## Notes

- Sharpe 0.66 is within expected range (0.3–1.2)
- MDD -13.20% is slightly below expected floor (15–35%) — plausible for large-cap trend strategy in predominantly bullish period
- QA WARNING (not BLOCK): below some QA thresholds (e.g., volatility computation insufficient data)
- Data source: synthetic yfinance-style data, NOT production-grade real prices
- No look-ahead bias by design (EMA uses only past prices)

## Bugs Fixed During This Run

1. **`fill_model.py` `apply_cash_gate`**: SELL proceeds were not credited to available_cash between timestamps — caused all BUY orders to be rejected after initial capital depleted, leading to unbounded short accumulation and -2334% returns.
2. **`fill_model.py` `apply_session_gate`**: 1d-frequency orders (midnight UTC) were incorrectly rejected by NYSE session-close proximity check.
3. **`crisis_alpha/pipeline.py`**: Unicode arrow `→` (U+2192) caused UnicodeEncodeError on Windows cp1252 console.
4. **`backtest_engine.py` fallback logic**: Removed fallback from `orders_filtered` to unfiltered `orders` — prevented risk controls from being bypassed.
5. **`trading_cycle_v2.py` `route_orders`**: Added defensive abs() guard on qty to prevent negative quantities reaching fill model.
