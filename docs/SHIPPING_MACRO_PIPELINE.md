# Shipping/Macro Pipeline Contract (Sprint 11.E3)

## Purpose

This document defines the standardized contract for shipping and macro data releases
with release calendar and availability times. The contract ensures:
- Reproducible ingestion (deterministic deduplication)
- Timestamp sanity (available_ts >= release_ts)
- Point-in-time (PIT) safety for feature building (available_ts <= as_of)
- Release calendar support (release_ts for scheduled releases)

## Schema

### Required Columns

- `series_id` (str): Series identifier (e.g., "SHIPPING_CONTAINER_INDEX", "GDP_US")
- `release_ts` (datetime, UTC): Scheduled release timestamp
- `available_ts` (datetime, UTC): When data becomes available (actual availability time)
- `value` (float): Data value

### Optional Columns

**Shipping:**
- `region` (str): Geographic region
- `source` (str): Data source identifier
- `revision_id` (str): Revision identifier for revised releases
- `metric` (str): Alternative to series_id

**Macro:**
- `country` (str): Country identifier
- `currency` (str): Currency identifier
- `source` (str): Data source identifier
- `revision_id` (str): Revision identifier for revised releases
- `metric` (str): Alternative to series_id

## Timestamp Semantics

### Release Timestamp (release_ts)

The `release_ts` represents the scheduled release time of the data.
This is typically when the data is officially published (e.g., economic calendar).

### Availability Timestamp (available_ts)

The `available_ts` represents when the data actually becomes available in the system.
This may be:
- Equal to `release_ts` (data available immediately)
- Later than `release_ts` (delayed availability, e.g., data provider lag)

### Timestamp Sanity Rules

1. **Valid Availability**: `available_ts >= release_ts`
   - Data cannot be available before it is released
   - Revisions must have `available_ts >= release_ts` of original release

2. **No Future Availability**: `available_ts` must not be in the future relative to ingest time
   (enforced during normalization if `ingest_ts` is provided)

## Point-in-Time (PIT) Filtering

For feature building, only releases with `available_ts <= as_of` should be used.
This prevents look-ahead bias by ensuring only data that was actually available
at the given point in time is used.

**Important**: Use `available_ts` (not `release_ts`) for PIT filtering, as this
represents when the data was actually available.

The `filter_shipping_pit()` and `filter_macro_pit()` functions enforce this rule.

## Deduplication

Deduplication is deterministic and uses the following key:
- Primary: `(series_id, release_ts, revision_id)` if `revision_id` is present
- Fallback: `(series_id, release_ts)` if `revision_id` is missing

The `dedupe_keep` parameter controls which duplicate is kept:
- `"first"`: Keep the first occurrence (earliest `available_ts` or row order)
- `"last"`: Keep the last occurrence (latest `available_ts` or row order)

**Important**: Deduplication must be deterministic (same input -> same output).

## Release Calendar Behavior

The release calendar is represented by `release_ts`. This allows:
- Scheduled releases (known in advance)
- Actual availability tracking (via `available_ts`)
- Revision handling (via `revision_id`)

### Revision Handling

Revisions are handled via `revision_id`:
- Original release: `revision_id` = None or "initial"
- Revised release: `revision_id` = unique identifier (e.g., "rev1", "rev2")

Deduplication uses `(series_id, release_ts, revision_id)` as key, so revisions
are treated as separate releases.

## Usage Examples

### Normalize Shipping Releases

```python
from src.assembled_core.data.shipping import normalize_shipping_releases

shipping_df = pd.DataFrame({
    "series_id": ["SHIPPING_CONTAINER_INDEX"],
    "release_ts": ["2024-01-15 08:00:00"],
    "available_ts": ["2024-01-15 08:30:00"],  # 30 min delay
    "value": [1500.0],
    "region": ["Global"],
})

normalized = normalize_shipping_releases(shipping_df, dedupe_keep="first")
```

### Normalize Macro Releases

```python
from src.assembled_core.data.macro import normalize_macro_releases

macro_df = pd.DataFrame({
    "series_id": ["GDP_US"],
    "release_ts": ["2024-01-15 08:30:00"],
    "available_ts": ["2024-01-15 08:30:00"],
    "value": [2.5],
    "country": ["US"],
})

normalized = normalize_macro_releases(macro_df, dedupe_keep="first")
```

### PIT Filtering

```python
from src.assembled_core.data.shipping import filter_shipping_pit
from src.assembled_core.data.macro import filter_macro_pit

as_of = pd.Timestamp("2024-01-15 09:00:00", tz="UTC")

# Filter shipping data
filtered_shipping = filter_shipping_pit(shipping_df, as_of)

# Filter macro data
filtered_macro = filter_macro_pit(macro_df, as_of)
```

### Macro Features

```python
from src.assembled_core.features.macro_features import add_latest_macro_value

panel_index = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
    "symbol": ["AAPL"] * 10,
})

macro_df = pd.DataFrame({
    "series_id": ["GDP_US"] * 5,
    "release_ts": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
    "available_ts": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
    "value": [2.5, 2.6, 2.7, 2.8, 2.9],
})

as_of = pd.Timestamp("2024-01-10", tz="UTC")

result = add_latest_macro_value(
    panel_index,
    macro_df,
    as_of,
    series_id="GDP_US",
)
```

## Integration with Feature Builders

Shipping/Macro feature builders should:
1. Load releases from storage
2. Apply `filter_shipping_pit()` or `filter_macro_pit()` before feature calculation
3. Use `available_ts` (not `release_ts`) for window calculations
4. Use `merge_asof` on `available_ts` to join latest available values

This ensures PIT-safety and prevents look-ahead bias.

## Error Handling

- **Missing Required Columns**: `ValueError` with list of missing columns
- **Timestamp Sanity Violations**: `ValueError` with description of violations
- **Invalid PIT Filtering**: `ValueError` if `available_ts` column is missing

## Determinism

All operations are deterministic:
- Same input -> same output
- Deduplication is stable (same key -> same result)
- Sorting is stable (same data -> same order)

This ensures reproducible ingestion and feature building.

## Timezone Invariance

All timestamps are normalized to UTC:
- Naive timestamps are assumed to be UTC
- Timezone-aware timestamps are converted to UTC

This ensures consistent behavior regardless of input timezone.
