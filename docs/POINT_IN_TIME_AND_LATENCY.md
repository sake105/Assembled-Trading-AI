POINT-IN-TIME SAFETY AND LATENCY FOR ALT-DATA (B2)
==================================================

This document describes the point-in-time (PIT) safety rules and latency
handling for alt-data sources in the Assembled Trading AI backend. It
ensures that backtests and live trading only use information that was
actually available at the time of decision-making, preventing look-ahead bias.

1. Core Concepts
----------------

1.1 event_date
~~~~~~~~~~~~~~

The `event_date` is when the underlying real-world event actually happens.

Examples:
- The actual trade date of an insider purchase
- The date when a member of Congress executes a trade
- The time when an earnings call takes place
- The time when a ship departs or arrives
- The instant when a news item is first written

This timestamp is often recorded in raw event feeds as `event_time`,
`trade_date`, `event_timestamp`, or similar.

1.2 disclosure_date
~~~~~~~~~~~~~~~~~~

The `disclosure_date` is when the event becomes observable to the model or
strategy. This is when the information can first influence features,
signals, and decisions.

Examples:
- For insider trades: The filing date when the Form 4 is published by
  the regulator (typically T+2 relative to trade date)
- For Congress trades: The date when a periodic transaction report (PTR)
  is published (often 10-30 days after the actual trade)
- For earnings: The timestamp when the earnings announcement is published
- For news: The timestamp when the news article is available via the
  data vendor's feed to the backend
- For shipping: The timestamp when a congestion signal is computed and
  stored (e.g., daily snapshot after all port events for the day are
  ingested)

1.3 Latency
~~~~~~~~~~~

Latency is the delay between `event_date` and `disclosure_date`. In many
real-world feeds:

- `disclosure_date` is explicitly provided (e.g., `disclosure_date` column)
- Or derived as: `disclosure_date = event_date + reporting_lag`
- Or the ingestion time when the record first appears in our local snapshot

2. Core Rule: PIT Safety
-------------------------

For backtest date T, only events with `disclosure_date <= T` may be used
in feature computation.

This rule ensures that we never use information that was not yet known at
that date, preventing look-ahead bias due to reporting lags or ingestion
delays.

Boundary semantics (strict <=):
- `disclosure_date < as_of`: INCLUDED (past disclosure)
- `disclosure_date == as_of`: INCLUDED (inclusive boundary)
- `disclosure_date > as_of`: EXCLUDED (future disclosure, strict)

3. Typical Latencies by Source
-------------------------------

3.1 Insider Trades (Form 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Event: Trade execution date
- Disclosure: Form 4 filing date (typically T+2 business days)
- Default latency: 2 days
- Implementation: `add_insider_features(..., disclosure_latency_days=2)`

3.2 Congress Trades (PTR)
~~~~~~~~~~~~~~~~~~~~~~~~~

- Event: Transaction date
- Disclosure: Periodic Transaction Report publication date
- Default latency: 10-30 days (conservative: 10 days)
- Implementation: `add_congress_features(..., disclosure_latency_days=10)`

3.3 Earnings Announcements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Event: Earnings call/announcement time
- Disclosure: Publication timestamp (often same day, but may have delays)
- Default latency: 0 days (same day)
- Implementation: `build_earnings_surprise_factors(..., as_of=...)`

3.4 News Sentiment
~~~~~~~~~~~~~~~~~

- Event: News article publication time
- Disclosure: Feed ingestion timestamp (may have intraday delays)
- Default latency: 0 days (same day, end-of-day snapshot)
- Implementation: `build_news_sentiment_factors(..., as_of=...)`

3.5 Shipping Data
~~~~~~~~~~~~~~~~~

- Event: Vessel arrival/departure time
- Disclosure: Daily snapshot consolidation timestamp
- Default latency: 1 day (end-of-day snapshot)
- Implementation: Shipping features use daily timestamps as disclosure_date

4. Implementation Pattern
---------------------------

All event feature builders follow this pattern:

4.1 Step 1: Schema Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use `ensure_event_schema()` to validate required columns:

```python
from src.assembled_core.data.latency import ensure_event_schema

events = ensure_event_schema(
    events_df,
    required_cols=["timestamp", "symbol"],
    strict=False  # Creates missing columns with defaults
)
```

4.2 Step 2: Derive disclosure_date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If `disclosure_date` is missing, derive it using `apply_source_latency()`:

```python
from src.assembled_core.data.latency import apply_source_latency

if "disclosure_date" not in events.columns:
    events = apply_source_latency(
        events,
        days=disclosure_latency_days,  # e.g., 2 for insider, 10 for congress
        event_date_col="event_date",
        timestamp_col="timestamp"
    )
```

4.3 Step 3: PIT-Safe Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Filter events by `disclosure_date <= as_of`:

```python
from src.assembled_core.data.latency import filter_events_as_of

if as_of is not None:
    events = filter_events_as_of(
        events,
        as_of,
        disclosure_col="disclosure_date"
    )
```

4.4 Step 4: Feature Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aggregate events into features (only disclosed events are used):

```python
# For each price timestamp, filter by disclosure_date <= price_time
for price_time in prices["timestamp"]:
    row_events = events[
        events["disclosure_date"] <= price_time.normalize()
    ]
    # Compute features from row_events
```

5. Reference Implementation
----------------------------

See `src/assembled_core/features/event_features.py` for a minimal,
complete reference implementation:

- `build_event_feature_panel()` demonstrates the full B2 pattern
- Shows how to integrate latency helpers
- Provides a template for new event feature builders

6. Existing Implementations
----------------------------

6.1 Insider Features
~~~~~~~~~~~~~~~~~~~~

- Module: `src/assembled_core/features/insider_features.py`
- Function: `add_insider_features(prices, events, as_of=None, disclosure_latency_days=2)`
- Default latency: 2 days (Form 4 filing delay)

6.2 Congress Features
~~~~~~~~~~~~~~~~~~~~

- Module: `src/assembled_core/features/congress_features.py`
- Function: `add_congress_features(prices, events, as_of=None, disclosure_latency_days=10)`
- Default latency: 10 days (PTR publication delay)

6.3 Earnings/Insider Factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Module: `src/assembled_core/features/altdata_earnings_insider_factors.py`
- Functions: `build_earnings_surprise_factors()`, `build_insider_activity_factors()`
- Already implements PIT-safe filtering with `as_of` parameter

6.4 News/Macro Factors
~~~~~~~~~~~~~~~~~~~~~~

- Module: `src/assembled_core/features/altdata_news_macro_factors.py`
- Functions: `build_news_sentiment_factors()`, `build_macro_regime_factors()`
- Already implements PIT-safe filtering with `as_of` parameter

7. How to Add a New Event Source
---------------------------------

7.1 Step 1: Understand the Latency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Research the typical delay between event and disclosure:
- What is the event? (e.g., trade execution, announcement)
- When is it disclosed? (e.g., filing date, publication date)
- What is the typical latency? (e.g., 2 days, 10 days, same day)

7.2 Step 2: Create Feature Builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new module in `src/assembled_core/features/` following the pattern:

```python
from src.assembled_core.data.latency import (
    apply_source_latency,
    ensure_event_schema,
    filter_events_as_of,
)

def add_your_event_features(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    disclosure_latency_days: int = YOUR_DEFAULT_LATENCY,
) -> pd.DataFrame:
    # Step 1: Schema validation
    events = ensure_event_schema(events, required_cols=["timestamp", "symbol"], strict=False)
    
    # Step 2: Derive disclosure_date if missing
    if "disclosure_date" not in events.columns:
        events = apply_source_latency(
            events,
            days=disclosure_latency_days,
            event_date_col="event_date",
            timestamp_col="timestamp",
        )
    
    # Step 3: PIT-safe filtering
    if as_of is not None:
        events = filter_events_as_of(events, as_of, disclosure_col="disclosure_date")
    
    # Step 4: Feature aggregation (per symbol, per price timestamp)
    # ... your feature computation logic ...
    
    return result
```

7.3 Step 3: Write Tests
~~~~~~~~~~~~~~~~~~~~~~~

Create tests in `tests/test_your_event_features_pit.py`:

```python
def test_your_event_features_pit_safe():
    # Create events with disclosure_date > as_of
    # Verify features are "blind" (no leakage)
    # Verify features become visible after disclosure_date
```

See `tests/test_event_features_pit.py` for examples.

7.4 Step 4: Document Latency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add your source to section 3 (Typical Latencies by Source) in this document.

8. Testing PIT Safety
----------------------

8.1 Test Files
~~~~~~~~~~~~~~

- `tests/test_latency_point_in_time.py`: Core PIT filtering tests
- `tests/test_event_features_pit.py`: Feature builder PIT tests
- `tests/test_data_latency_helpers.py`: Latency helper unit tests

8.2 Test Pattern
~~~~~~~~~~~~~~~~

All PIT tests follow this pattern:

1. Create events with `event_date` and `disclosure_date`
2. Filter with `as_of` before disclosure: verify event is excluded
3. Filter with `as_of` on/after disclosure: verify event is included
4. Verify no value leakage (features are "blind" before disclosure)

Example:

```python
def test_pit_safety():
    events = pd.DataFrame({
        "event_date": [pd.Timestamp("2025-01-05", tz="UTC")],
        "disclosure_date": [pd.Timestamp("2025-01-08", tz="UTC")],
        "symbol": ["AAPL"],
        "value": [100.0],
    })
    
    # Before disclosure: must be excluded
    filtered_before = filter_events_as_of(
        events, pd.Timestamp("2025-01-07", tz="UTC"), disclosure_col="disclosure_date"
    )
    assert len(filtered_before) == 0  # Feature is blind
    
    # After disclosure: must be included
    filtered_after = filter_events_as_of(
        events, pd.Timestamp("2025-01-08", tz="UTC"), disclosure_col="disclosure_date"
    )
    assert len(filtered_after) == 1  # Event is visible
```

9. Integration with Factor Store (P2)
--------------------------------------

The Factor Store (P2) also implements PIT-safe filtering:

- `load_factors(..., as_of=...)` filters factors by timestamp <= as_of
- Pre-computed factor panels are marked with effective `as_of` date
- See `docs/FACTOR_STORE_P2_DESIGN.md` for details

When using Factor Store with event-based factors:

1. Build factors with `as_of` parameter (PIT-safe)
2. Store factors with metadata (effective `as_of` date)
3. Load factors with `as_of` parameter (ensures no future data)

10. Integration in Backtests and EOD
-------------------------------------

10.1 Backtest Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

In portfolio backtests, the engine loops over time (e.g., daily timestamps).
For each step with date T:

- Any feature-building function that depends on alt-data must enforce:
  `disclosure_date <= T` at the data access boundary
- Practical patterns:
  - Pre-build daily panels of alt-data factors that are already PIT-safe
  - Or, when building features on-the-fly, pass `as_of=T` to feature builders

10.2 EOD Integration
~~~~~~~~~~~~~~~~~~~

For EOD jobs (e.g., `scripts/run_daily.py`), the notion of `as_of` is:

- EOD date `T_eod` (e.g., "today" in UTC)

Feature-building during EOD should:

- Use `as_of = T_eod` consistently for all alt-data domains
- Ensure that any alt-data snapshots written to disk are marked with the
  effective `as_of` date in metadata

11. Related Documentation
--------------------------

- `docs/POINT_IN_TIME_AND_LATENCY_B2_DESIGN.md`: Detailed design document
- `docs/FACTOR_STORE_P2_DESIGN.md`: Factor Store PIT integration
- `src/assembled_core/data/latency.py`: Implementation of latency helpers
- `src/assembled_core/features/event_features.py`: Reference implementation

12. Summary
-----------

Key takeaways:

1. Always distinguish `event_date` (when it happened) from `disclosure_date`
   (when it becomes observable)

2. Core rule: Only use events with `disclosure_date <= as_of` in features

3. Use latency helpers (`filter_events_as_of`, `apply_source_latency`) for
   consistent PIT safety

4. Test PIT safety explicitly: verify features are "blind" before disclosure

5. Document typical latencies for each source (helps with defaults)

6. Follow the reference implementation pattern for new event sources
