# Point-in-Time Safety and Latency for Alt-Data Factors

## Overview

**Point-in-Time (PIT) Safety** ensures that backtest strategies only use information that was actually available at each point in time, preventing look-ahead bias. This is critical for realistic backtesting and production trading.

**Key Concepts:**

- **event_date**: When the underlying real-world event actually happened (e.g., trade execution, earnings call, news publication).
- **disclosure_date**: When the event becomes observable to the strategy (e.g., filing publication, data vendor delivery, batch ingestion).

**The Core Rule:**

> For backtest date T, only events with `disclosure_date <= T` may contribute to features or factors used on T.

**Why This Matters:**

Without PIT safety, strategies can appear profitable in backtests but fail in production because they "see" information that wasn't available at the time. Common look-ahead pitfalls:

- Using insider trades before they are filed (Form 4 typically filed T+2 after trade)
- Using earnings surprises before the announcement is published
- Using news sentiment before the article is available in the feed
- Using congressional trades before the periodic transaction report (PTR) is published

---

## Per Data Source: Latency Models

| Data Source | event_date | disclosure_date | Typical Latency | Notes |
|-------------|------------|-----------------|-----------------|-------|
| **Insider Trades** | Trade execution date | Form 4 filing date | T+2 business days | SEC requires filing within 2 business days. We use filing date as disclosure_date. |
| **Congress Trades** | Transaction date | PTR publication date | Days to weeks | Periodic Transaction Reports (PTR) published periodically, often with significant delay. |
| **Earnings** | Earnings call/announcement time | Publication timestamp | Usually same day | Announcement timestamp is treated as disclosure_date. |
| **News Sentiment** | Article publication time | Feed delivery timestamp | Minutes to hours | Daily aggregated panels use date as disclosure_date (end-of-day snapshot). |
| **Shipping** | Vessel event (arrival/departure) | Daily snapshot consolidation | End-of-day | Congestion/flows available as of daily snapshot date. |

**Default Behavior:**

If `disclosure_date` is not explicitly provided in the event data:
- For earnings/insider: `disclosure_date = event_date` (conservative default)
- For news: `disclosure_date = daily_timestamp` (end-of-day aggregation)
- For shipping: `disclosure_date = snapshot_date`

**Upstream Enhancement:**

When upstream data sources (e.g., Finnhub, SEC filings) provide explicit filing/publication dates, these are used as `disclosure_date`. Otherwise, conservative defaults ensure no look-ahead bias.

---

## Rules for Feature Builds

### Implementation Pattern

All Alt-Data feature builders follow this pattern:

1. **Normalize timestamps to UTC**
2. **Ensure event_date and disclosure_date exist** (derive from timestamp if missing)
3. **Filter by disclosure_date** (if `as_of` parameter is provided):
   ```python
   if as_of is not None:
       events = events[events["disclosure_date"] <= as_of.normalize()]
   ```
4. **Perform aggregations/joins** only on the filtered subset

### Feature Builder Functions

All Alt-Data feature builders now accept an optional `as_of` parameter:

- `build_earnings_surprise_factors(events_earnings, prices, ..., as_of=None)`
- `build_insider_activity_factors(events_insider, prices, ..., as_of=None)`
- `build_news_sentiment_factors(news_sentiment_daily, prices, ..., as_of=None)`

**Behavior:**

- If `as_of=None` (default): All events are used (backward compatible, but assumes `disclosure_date = event_date`).
- If `as_of` is provided: Only events with `disclosure_date <= as_of` are included in factor computation.

**Module References:**

- Earnings/Insider: `src/assembled_core/features/altdata_earnings_insider_factors.py`
- News/Macro: `src/assembled_core/features/altdata_news_macro_factors.py`
- Data Loading: `src/assembled_core/data/altdata/finnhub_events.py`, `finnhub_news_macro.py`

---

## Examples

### Example 1: Insider Trade with Delayed Disclosure

**Scenario:**
- Event: Insider buys 1000 shares of AAPL on Monday, 2024-01-10
- Disclosure: Form 4 filed on Wednesday, 2024-01-15 (T+3 due to holiday)
- Backtest dates: 2024-01-11, 2024-01-14, 2024-01-15, 2024-01-16

**Event Data:**
```
event_date: 2024-01-10
disclosure_date: 2024-01-15
```

**Factor Behavior:**

- **2024-01-11** (as_of=2024-01-11): Event filtered out (`disclosure_date > as_of`)
  - `insider_net_notional_60d` = 0.0 or NaN
  - No buy signal from insider activity

- **2024-01-14** (as_of=2024-01-14): Event still filtered out
  - `insider_net_notional_60d` = 0.0 or NaN

- **2024-01-15** (as_of=2024-01-15): Event now visible (`disclosure_date <= as_of`)
  - `insider_net_notional_60d` > 0 (reflects $150k buy)
  - `insider_buy_count_60d` = 1
  - Signal can now react to insider activity

- **2024-01-16** (as_of=2024-01-16): Event remains visible (forward-filled)
  - Factors continue to reflect the insider buy

**Key Point:** The strategy cannot trade on insider information before it is publicly disclosed.

### Example 2: Earnings Surprise with Immediate Disclosure

**Scenario:**
- Event: Earnings announcement on 2024-01-12 at 4:00 PM ET
- Disclosure: Same timestamp (announcement is disclosure)
- Backtest dates: 2024-01-11, 2024-01-12, 2024-01-13

**Event Data:**
```
event_date: 2024-01-12
disclosure_date: 2024-01-12
```

**Factor Behavior:**

- **2024-01-11** (as_of=2024-01-11): Event filtered out
  - `earnings_eps_surprise_last` = NaN

- **2024-01-12** (as_of=2024-01-12): Event visible
  - `earnings_eps_surprise_last` = 8.7% (positive surprise)
  - Signal can react to earnings on announcement day

- **2024-01-13** (as_of=2024-01-13): Event remains visible (forward-filled)
  - `earnings_eps_surprise_last` = 8.7%

**Key Point:** For earnings, disclosure is typically immediate, but the PIT rule still ensures we don't use future announcements.

---

## Integration in Backtests

### Backtest Loop Pattern

In a backtest, factors should be computed per day with `as_of=current_date`:

```python
for current_date in trading_dates:
    # Compute factors with as_of=current_date
    factors = build_earnings_surprise_factors(
        events_earnings=events,
        prices=prices,
        as_of=current_date,  # Only events disclosed by current_date
    )
    
    # Generate signals from factors
    signals = signal_fn(factors)
    
    # Execute trades based on signals
    # ...
```

**Important:** The backtest engine (`run_portfolio_backtest`) does not automatically pass `as_of` to feature builders. If you need PIT-safe factors in backtests, you must:

1. Pre-filter events by `disclosure_date` before building factors, OR
2. Modify your feature computation to pass `as_of=current_date` per timestamp

**Future Enhancement:** The backtest engine may automatically pass `as_of=current_timestamp` to feature builders in future versions.

### EOD Pipeline

For EOD (End-of-Day) runs, `as_of` should be set to the current EOD date:

```python
from datetime import date
from pandas import Timestamp

eod_date = date.today()  # or from config
as_of = Timestamp(eod_date, tz="UTC")

factors = build_earnings_surprise_factors(
    events_earnings=events,
    prices=prices,
    as_of=as_of,
)
```

This ensures that only events disclosed by EOD are used in live trading decisions.

---

## Testing & Guarantees

### Test Suite

Comprehensive PIT tests are available in `tests/test_point_in_time_altdata.py`:

- **Test Coverage:**
  - Earnings factors respect disclosure_date
  - Insider factors respect disclosure_date
  - News sentiment factors respect disclosure_date
  - No look-ahead bias (comparison with/without delayed events)
  - Mini backtest scenario (factors appear only after disclosure)

**Example Test:**
```python
def test_earnings_factors_respect_disclosure_date(...):
    # Event A: disclosure_date=2024-01-15
    # as_of=2024-01-11: Event A filtered out
    # as_of=2024-01-15: Event A visible
```

**Run Tests:**
```bash
pytest tests/test_point_in_time_altdata.py -xvs
```

### Guarantees

**The test suite guarantees:**

1. **No Future Data:** Features computed with `as_of=T` contain no information from events with `disclosure_date > T`.
2. **Deterministic Behavior:** Same `as_of` + same events = identical factors.
3. **Backward Compatibility:** If `as_of=None`, behavior matches pre-B2 implementation (assumes `disclosure_date = event_date`).

**Verification:**

- All tests in `test_point_in_time_altdata.py` pass.
- Look-ahead bias tests verify that factors are identical before disclosure.
- Mini backtest tests verify that factors appear only after disclosure_date.

---

## Best Practices

### For Factor Development

1. **Always use `as_of` in backtests:** Pass `as_of=current_date` when computing factors per timestamp.
2. **Verify disclosure dates:** Check that upstream data sources provide accurate `disclosure_date` (or use conservative defaults).
3. **Test with delayed disclosure:** Use synthetic events with `disclosure_date > event_date` to verify PIT behavior.

### For Data Ingestion

1. **Preserve filing/publication dates:** When loading events from APIs or files, store both `event_date` and `disclosure_date`.
2. **Use conservative defaults:** If `disclosure_date` is unknown, set it equal to `event_date` (no look-ahead, but may miss some latency).
3. **Document latency assumptions:** If you introduce fixed lags (e.g., "T+2 for insider trades"), document them clearly.

### For Backtest Analysis

1. **Check factor timelines:** Verify that factors change only after `disclosure_date` (not `event_date`).
2. **Compare with/without PIT:** Run backtests with `as_of=None` vs. `as_of=current_date` to quantify the impact of latency.
3. **Review test results:** Run `pytest tests/test_point_in_time_altdata.py` to ensure PIT safety.

---

## References

- **Design Document:** [Point-in-Time and Latency B2 Design](POINT_IN_TIME_AND_LATENCY_B2_DESIGN.md)
- **Test Suite:** `tests/test_point_in_time_altdata.py`
- **Feature Modules:**
  - `src/assembled_core/features/altdata_earnings_insider_factors.py`
  - `src/assembled_core/features/altdata_news_macro_factors.py`
- **Data Loading:**
  - `src/assembled_core/data/altdata/finnhub_events.py`
  - `src/assembled_core/data/altdata/finnhub_news_macro.py`
- **Factor Store (P2):** Factor panels stored via `factor_store` are point-in-time safe (see [Factor Store P2 Design](FACTOR_STORE_P2_DESIGN.md))

---

## FAQ

**Q: Do I need to pass `as_of` in every feature builder call?**  
A: Only if you need PIT safety. Default behavior (`as_of=None`) assumes `disclosure_date = event_date` (backward compatible).

**Q: What if my event data doesn't have `disclosure_date`?**  
A: The feature builders automatically derive `disclosure_date = event_date` if missing. This is conservative (no look-ahead) but may not model real latency.

**Q: How do I verify PIT safety in my backtests?**  
A: Run `pytest tests/test_point_in_time_altdata.py` and check that factors appear only after `disclosure_date`. Compare backtest results with/without `as_of` to quantify latency impact.

**Q: Can I use PIT-safe factors with the Factor Store (P2)?**  
A: Yes. Factor panels stored via `factor_store` preserve `disclosure_date` and can be loaded with point-in-time filtering (see [Factor Store P2 Design](FACTOR_STORE_P2_DESIGN.md)).

