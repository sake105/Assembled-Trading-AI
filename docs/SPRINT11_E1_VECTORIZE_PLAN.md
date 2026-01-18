# Sprint 11.E1: Vectorization Plan for Event Features

## Current Implementation Summary

### Functions Analyzed

1. **`build_event_feature_panel()`** (lines 20-138)
   - Input: `events_df`, `prices_df`, `as_of`, `lookback_days`, `feature_prefix`
   - Output: `prices_df` with columns: `{prefix}_count_{days}d`, `{prefix}_sum_{days}d`, `{prefix}_mean_{days}d`
   - Semantics: Count/sum/mean of events in disclosure_date window per (symbol, timestamp)

2. **`add_disclosure_count_feature()`** (lines 141-229)
   - Input: `prices`, `events`, `window_days`, `out_col`, `as_of`
   - Output: `prices` with column `{out_col}` (event count)
   - Semantics: Count events in disclosure_date window per (symbol, timestamp)

### Current Algorithm (Pseudo-Code)

```
For each symbol in prices:
    Get events for symbol (already PIT-filtered)
    For each price row (timestamp):
        Filter events: disclosure_date <= price_timestamp
        Filter events: disclosure_date in [price_timestamp - window, price_timestamp]
        Count/sum/mean window_events
        Write to result.loc[idx, feature_col]
```

## Bottlenecks

### Identified Hotspots

1. **Nested Loops (Lines 96-137, 196-227)**
   - Outer loop: `for symbol in result["symbol"].unique()`
   - Inner loop: `for idx in symbol_prices.index`
   - Complexity: O(n_symbols * n_prices_per_symbol)
   - For 100 symbols * 1000 prices = 100,000 iterations

2. **Per-Row DataFrame Filtering (Lines 114-122, 213-224)**
   - `row_events = symbol_events[symbol_events["disclosure_date"] <= price_time_normalized].copy()`
   - `window_events = row_events[(row_events["disclosure_date"] <= ...) & (...)].copy()`
   - Creates new DataFrame for each price row
   - Memory allocation overhead

3. **Per-Row `.loc[]` Assignment (Lines 125-136, 227)**
   - `result.loc[idx, feature_col] = value`
   - Index-based assignment is slow for large DataFrames
   - No vectorization

4. **Redundant PIT Filtering (Lines 114-116, 212-218)**
   - Per-row filtering even when `as_of` is provided (globally filtered)
   - Can be optimized away if `as_of` is set correctly

### Performance Impact

- **Current**: ~O(n_symbols * n_prices * n_events_per_symbol)
- **Expected after vectorization**: ~O(n_symbols * log(n_events) * n_prices) using merge_asof + rolling

## Vectorization Strategy

### Approach: Daily Aggregation + merge_asof + Rolling Window Stats

#### Step 1: Daily Event Aggregation (per symbol, per day)

```
events_daily = events.groupby(["symbol", "disclosure_date"]).agg({
    "value": ["count", "sum", "mean"]  # or just count if no value column
}).reset_index()
```

**Benefits**:
- Reduces event DataFrame size (one row per symbol per disclosure_date)
- Enables efficient merge with prices

#### Step 2: Merge with Prices using merge_asof

```
# For each symbol, merge events to prices using disclosure_date <= price_timestamp
result = pd.merge_asof(
    prices_sorted,
    events_daily_sorted,
    left_on="timestamp",
    right_on="disclosure_date",
    by="symbol",
    direction="backward"  # disclosure_date <= timestamp
)
```

**Benefits**:
- Vectorized merge (no loops)
- Handles PIT filtering automatically (backward merge)

#### Step 3: Rolling Window Statistics

```
# For each symbol, compute rolling window stats
result = result.groupby("symbol").apply(
    lambda g: g.rolling(window=f"{window_days}D", on="timestamp", closed="both")
    .agg({"event_count": "sum", "event_sum": "sum", "event_mean": "mean"})
)
```

**Benefits**:
- Vectorized rolling window computation
- Handles lookback window efficiently

### Alternative: Cross Join + Filter (if merge_asof insufficient)

If merge_asof doesn't handle all edge cases:

```
# Cross join prices with events (per symbol)
cross = prices.merge(events, on="symbol", how="left", suffixes=("_price", "_event"))

# Vectorized filtering
mask = (
    (cross["disclosure_date"] <= cross["timestamp"])
    & (cross["disclosure_date"] > cross["timestamp"] - pd.Timedelta(days=window_days))
)

# Groupby + aggregate
result = cross[mask].groupby(["symbol", "timestamp"]).agg({
    "value": ["count", "sum", "mean"]
})
```

**Trade-off**: More memory (cross join), but fully vectorized

## Non-Negotiable Semantics

### PIT-Safety (MUST PRESERVE)

1. **disclosure_date <= as_of**: Events must be filtered by `disclosure_date <= as_of` (or `price_timestamp` if `as_of` is None)
2. **Window based on disclosure_date**: Lookback window uses `disclosure_date`, NOT `event_date`
3. **Per-row PIT filtering**: If `as_of` is None, each price row must filter events by `disclosure_date <= price_timestamp`

### UTC Policy (MUST PRESERVE)

1. All timestamps must be UTC-aware
2. Normalization to date (end-of-day) for consistent comparison

### Deterministic Sorting (MUST PRESERVE)

1. Output must be sorted by `symbol`, `timestamp` (ascending)
2. Same input -> same output (deterministic)

### Deduplication Policy (MUST PRESERVE)

1. Events are deduplicated in `normalize_alt_events()` (by symbol, event_date, disclosure_date, effective_date)
2. No additional deduplication needed in feature computation

### Output Schema (MUST PRESERVE)

1. **`build_event_feature_panel`**:
   - Columns: `{prefix}_count_{days}d`, `{prefix}_sum_{days}d`, `{prefix}_mean_{days}d`
   - Data types: `count` (int), `sum` (float), `mean` (float/NA)
   - NaN handling: `mean` can be `pd.NA` if no events, `sum` is `0.0`, `count` is `0`

2. **`add_disclosure_count_feature`**:
   - Column: `{out_col}` (int, default 0)

## Compatibility Strategy

### Legacy Path (for Testing)

1. Keep current implementation as `_build_event_feature_panel_legacy()` (private)
2. New vectorized implementation as `build_event_feature_panel()` (public)
3. Add equivalence test: `test_vectorized_equals_legacy()`
   - Compare outputs for same inputs
   - Verify identical feature values (within floating-point tolerance)

### Migration Path

1. **Phase 1**: Implement vectorized version alongside legacy
2. **Phase 2**: Run equivalence tests on sample data
3. **Phase 3**: Switch default to vectorized, keep legacy as fallback
4. **Phase 4**: Remove legacy after validation period

## Implementation Notes

### Edge Cases to Handle

1. **Empty events**: Return prices with zero features (current behavior)
2. **No events for symbol**: Feature = 0 (current behavior)
3. **Events outside window**: Feature = 0 (current behavior)
4. **Multiple events per disclosure_date**: Aggregate correctly (count/sum/mean)

### Testing Requirements

1. **Equivalence tests**: Vectorized == Legacy for same inputs
2. **PIT-safety tests**: Verify disclosure_date filtering still works
3. **Performance tests**: Measure speedup (target: 10x-100x for large datasets)
4. **Edge case tests**: Empty events, no events per symbol, etc.

## File Locations

- **Current Implementation**: `src/assembled_core/features/event_features.py`
- **Tests**: `tests/test_event_features_pit.py`, `tests/test_leakage_altdata_pit.py`
- **Contract**: `src/assembled_core/data/altdata/contract.py`

## Next Steps (Sprint 11.E1)

1. Implement vectorized `build_event_feature_panel()` using merge_asof + rolling
2. Implement vectorized `add_disclosure_count_feature()` using same approach
3. Add equivalence tests comparing vectorized vs legacy
4. Benchmark performance improvement
5. Update documentation if API changes
