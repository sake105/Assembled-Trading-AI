# Alt-Data Event Contract

## Purpose

This document defines the standardized contract for alt-data events (insider trades,
shipping data, news events, etc.) used throughout the trading system. The contract
ensures consistent schema, validation, and PIT-safe filtering across all alt-data sources.

## Schema

### Required Columns

- **symbol** (str): Symbol/ticker identifier
- **event_date** (datetime, UTC): When the event actually happened
- **disclosure_date** (datetime, UTC): When the event becomes publicly observable

### Optional Columns

- **effective_date** (datetime, UTC): When the event becomes usable for features
  - Default: `disclosure_date` (if missing, automatically set to disclosure_date)
- **event_type** (str): Type of event (e.g., "BUY", "SELL", "SHIPMENT", "NEWS")
- **source** (str): Data source identifier (e.g., "SEC_FORM4", "SHIPPING_API")
- **value** (float/str): Event value for aggregation (e.g., transaction amount, shipment volume)

## Date Semantics

### event_date

The date when the event actually occurred. For example:
- Insider trade: Date of the transaction
- Shipping event: Date of shipment departure/arrival
- News event: Date of publication

### disclosure_date

The date when the event becomes publicly observable. This is critical for PIT-safe
feature building. For example:
- Insider trade: Filing date (Form 4 typically T+2 after transaction)
- Shipping event: Date when data is published/available
- News event: Publication date

**PIT Rule**: Features must only use events where `disclosure_date <= as_of`.
This prevents look-ahead bias in backtests.

### effective_date

The date when the event becomes usable for feature computation. Defaults to
`disclosure_date` if not provided. This allows modeling additional delays beyond
disclosure (e.g., data processing time, API latency).

**Constraint**: `effective_date >= disclosure_date >= event_date`

## Late-Arrival Semantics

Events can have `event_date` in the past but `disclosure_date` in the future.
This models late-arriving data (e.g., delayed filings, retroactive updates).

**Example**:
- `event_date = 2024-01-15` (transaction happened)
- `disclosure_date = 2024-01-17` (filing published T+2)
- `effective_date = 2024-01-17` (default)

For a backtest at `as_of = 2024-01-16`, this event is NOT available (disclosure_date
is in the future). For `as_of = 2024-01-17`, the event IS available.

## Normalization

The `normalize_alt_events()` function performs:

1. **String trimming**: `symbol`, `event_type`, `source` are trimmed of whitespace
2. **UTC normalization**: All timestamps converted to UTC (naive -> UTC, tz-aware -> UTC)
3. **Date normalization**: Timestamps normalized to date (end-of-day)
4. **Effective date fallback**: Missing `effective_date` set to `disclosure_date`
5. **Constraint validation**: Ensures `effective_date >= disclosure_date >= event_date`
6. **Deduplication**: Removes duplicates on `(symbol, event_date, disclosure_date, effective_date)`
   - Deterministic: keeps first occurrence
7. **Deterministic sorting**: Sorted by `symbol`, `event_date`, `disclosure_date`, `effective_date`

## PIT Filtering

The `filter_events_pit()` function implements PIT-safe filtering:

- **Rule**: `disclosure_date <= as_of`
- **Semantics**: Strict "less than or equal" (inclusive boundary)
- **Purpose**: Prevents look-ahead bias in backtests

**Example**:
```python
events = filter_events_pit(events, as_of=pd.Timestamp("2024-01-17", tz="UTC"))
# Returns only events with disclosure_date <= 2024-01-17
```

## UTC Policy

All timestamps must be UTC-aware. The normalization function:
- Converts naive timestamps to UTC (assumes UTC)
- Converts tz-aware timestamps to UTC
- Normalizes to date (end-of-day) for consistent comparison

## Deduplication Policy

Duplicates are removed deterministically:
- **Subset**: `(symbol, event_date, disclosure_date, effective_date)`
- **Keep**: First occurrence (deterministic)
- **Rationale**: Same event (same dates) should appear only once

## Usage Example

```python
from src.assembled_core.data.altdata import normalize_alt_events, filter_events_pit
import pandas as pd

# Raw events (may have missing columns, naive timestamps)
raw_events = pd.DataFrame({
    "symbol": ["AAPL", "AAPL"],
    "event_date": ["2024-01-15", "2024-01-15"],
    "disclosure_date": ["2024-01-17", "2024-01-17"],
    "event_type": ["BUY", "BUY"],
    "value": [1000.0, 1000.0],
})

# Normalize to contract
events = normalize_alt_events(raw_events)
# - effective_date added (default = disclosure_date)
# - Timestamps normalized to UTC
# - Duplicates removed (if any)
# - Sorted deterministically

# PIT-safe filtering
as_of = pd.Timestamp("2024-01-16", tz="UTC")
filtered = filter_events_pit(events, as_of)
# Returns empty (disclosure_date 2024-01-17 > as_of 2024-01-16)

as_of = pd.Timestamp("2024-01-17", tz="UTC")
filtered = filter_events_pit(events, as_of)
# Returns events (disclosure_date 2024-01-17 <= as_of 2024-01-17)
```

## Public Disclosures Only

**Policy**: All alt-data events must represent public disclosures only. No MNPI (Material Non-Public Information) is allowed.

### Requirements

1. **Insider/Congress Trades**: Only public filings/disclosures with appropriate lags
   - Form 4 filings: T+2 after transaction (disclosure_date = transaction_date + 2 days)
   - Congress trades: Public disclosure dates only
   - No "instant knowledge" or pre-filing data

2. **No MNPI**: Events must be publicly observable at `disclosure_date`
   - No private information
   - No leaked data
   - No "insider" knowledge

3. **PIT Enforcement**: `disclosure_date` is mandatory
   - Without `disclosure_date`, PIT filtering is impossible
   - Fail-fast: Missing `disclosure_date` raises `ValueError`

### Validation

The `normalize_alt_events()` function enforces this policy:

- **`is_public` field**: If present and `False`, raises `ValueError` (fail-fast)
- **Missing `disclosure_date`**: Raises `ValueError` (PIT not definable)

**Example**:
```python
# Valid: Public disclosure
events = pd.DataFrame({
    "symbol": ["AAPL"],
    "event_date": pd.to_datetime(["2024-01-15"], utc=True),
    "disclosure_date": pd.to_datetime(["2024-01-17"], utc=True),  # T+2 filing
    "is_public": [True],  # Optional, but if present must be True
})

# Invalid: Non-public data
events = pd.DataFrame({
    "symbol": ["AAPL"],
    "event_date": pd.to_datetime(["2024-01-15"], utc=True),
    "disclosure_date": pd.to_datetime(["2024-01-17"], utc=True),
    "is_public": [False],  # Raises ValueError
})

# Invalid: Missing disclosure_date
events = pd.DataFrame({
    "symbol": ["AAPL"],
    "event_date": pd.to_datetime(["2024-01-15"], utc=True),
    # disclosure_date missing -> Raises ValueError
})
```

### Rationale

- **Regulatory Compliance**: Ensures no use of MNPI in trading strategies
- **PIT-Safety**: `disclosure_date` is essential for point-in-time filtering
- **Transparency**: Clear policy prevents accidental use of non-public data

## Integration with Feature Builders

All alt-data feature builders should:
1. Normalize events using `normalize_alt_events()`
2. Filter by PIT using `filter_events_pit(events, as_of)`
3. Aggregate events into features (respecting `effective_date` if needed)

This ensures consistent, PIT-safe feature computation across all alt-data sources.

## Leakage Test Suite

The leakage test suite (`src/assembled_core/qa/leakage_tests/altdata_leakage.py`) provides
helpers to detect look-ahead bias in alt-data features.

### Helper Function

**`assert_feature_zero_before_disclosure()`**:
- Validates that features are zero before `disclosure_date`
- Validates that features are non-zero after `disclosure_date`
- Provides clear error messages identifying which symbol/date violates PIT-safety

### Test Scenarios

1. **Future Event Inserted**: Event with disclosure far in future -> feature remains 0 before disclosure
2. **Late Arrival**: `event_date` old, `disclosure_date` late -> 0 before, >0 after
3. **Multiple Symbols**: Only affected symbol rises, others remain 0

### Usage

```python
from src.assembled_core.qa.leakage_tests import assert_feature_zero_before_disclosure
from src.assembled_core.features.event_features import add_disclosure_count_feature

def feature_fn(prices, events, as_of):
    return add_disclosure_count_feature(prices, events, window_days=30, as_of=as_of)

# Validate PIT-safety
assert_feature_zero_before_disclosure(
    prices,
    events,
    feature_fn,
    as_of_before=pd.Timestamp("2024-01-14", tz="UTC"),
    as_of_after=pd.Timestamp("2024-01-15", tz="UTC"),
)
```

### Test Execution

Leakage tests are mandatory and run in CI (no markers, always executed).
They verify that features do not leak future information by using events
that have not yet been disclosed.
