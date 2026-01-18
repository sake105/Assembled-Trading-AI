# News Pipeline Contract (Sprint 11.E2)

## Purpose

This document defines the standardized contract for news event ingestion, storage,
and retrieval in the Assembled Trading AI system. The contract ensures:
- Reproducible ingestion (deterministic deduplication)
- Timestamp sanity (no future revisions, no future publish dates)
- Point-in-time (PIT) safety for feature building
- Atomic storage operations

## Schema

### Required Columns

- `publish_ts` (datetime, UTC): Publication timestamp of the news article
- `source` (str): News source identifier (e.g., "reuters", "bloomberg", "yahoo")

### Optional Columns

- `symbol` (str): Single symbol this news relates to
- `symbols` (str): Comma-separated list of symbols (alternative to `symbol`)
- `headline` (str): News headline (used for deduplication if no provider_id)
- `url` (str): URL to the news article
- `provider_id` (str): Provider-specific unique identifier (preferred for deduplication)
- `ingest_ts` (datetime, UTC): Timestamp when news was ingested into the system
- `revised_ts` (datetime, UTC): Timestamp when news was revised (if applicable)
- `sentiment` (str/float): Sentiment score or label
- `raw_url` (str): Original URL before any processing

### Identifier Requirement

At least one of the following must be present and non-empty:
- `headline`
- `url`
- `provider_id`

## Timestamp Sanity Rules

1. **No Future Publish Dates**: `publish_ts` must not be in the future relative to `ingest_ts`
   (if `ingest_ts` is present). This prevents "future news" from being ingested.

2. **Valid Revisions**: `revised_ts >= publish_ts` (if `revised_ts` is present).
   Revisions must occur after the original publication.

3. **No Future Revisions Policy**: News events cannot be backdated. Once published,
   the `publish_ts` cannot be changed. Revisions must have `revised_ts >= publish_ts`.

## Deduplication

Deduplication is deterministic and uses the following key strategy:

1. **Primary Key**: If both `source` and `provider_id` are present and non-NaN:
   - Key = `(source, provider_id)`

2. **Fallback Key**: If `provider_id` is missing:
   - Key = `hash(headline + publish_ts + source)`

The `dedupe_keep` parameter controls which duplicate is kept:
- `"first"`: Keep the first occurrence (earliest `ingest_ts` or row order)
- `"last"`: Keep the last occurrence (latest `ingest_ts` or row order)

**Important**: Deduplication must be deterministic (same input -> same output).

## Point-in-Time (PIT) Filtering

For feature building, only news events with `publish_ts <= as_of` should be used.
This prevents look-ahead bias.

The `filter_news_pit()` function enforces this rule.

## Storage Layout

News events are stored in parquet format with partitioning:

```
{root}/news/{source}/{year}/{month:02d}/news_{source}_{year}_{month:02d}.parquet
```

Example:
```
data/news/reuters/2024/01/news_reuters_2024_01.parquet
data/news/bloomberg/2024/12/news_bloomberg_2024_12.parquet
```

### Storage Operations

- **Append Mode**: Loads existing partition, merges with new data, deduplicates, writes atomically
- **Replace Mode**: Overwrites existing partition (use with caution)

### Atomic Writes

All writes use atomic operations:
1. Write to temporary file: `news_{source}_{year}_{month:02d}.tmp.parquet`
2. Rename to final file: `news_{source}_{year}_{month:02d}.parquet`

This ensures no partial files are left behind on errors.

## Usage Examples

### Normalize News Events

```python
from src.assembled_core.data.news import normalize_news_events

news_df = pd.DataFrame({
    "publish_ts": ["2024-01-15 10:00:00"],
    "source": ["reuters"],
    "headline": ["Apple reports strong earnings"],
    "symbol": ["AAPL"],
})

normalized = normalize_news_events(news_df, dedupe_keep="first")
```

### Store News Events

```python
from src.assembled_core.data.news import store_news_parquet

store_news_parquet(
    news_df,
    root="data",
    source="reuters",
    year=2024,
    month=1,
    mode="append",
    dedupe_keep="first",
)
```

### Load News Events

```python
from src.assembled_core.data.news import load_news_parquet

news_df = load_news_parquet(
    root="data",
    source="reuters",
    year=2024,
    month=1,
)
```

### PIT Filtering

```python
from src.assembled_core.data.news import filter_news_pit

as_of = pd.Timestamp("2024-01-20", tz="UTC")
filtered_news = filter_news_pit(news_df, as_of)
```

## Entity Linking

News events can be linked to symbols using the `link_news_to_symbols()` function.

### Mapping Sources

1. **Existing 'symbol' column**: If news already has a 'symbol' column, it is passed through (trimmed).

2. **Mapping DataFrame**: If news has 'ticker' or 'entity' columns, they can be mapped via a mapping DataFrame:
   ```python
   mapping_df = pd.DataFrame({
       "entity": ["AAPL", "MSFT"],
       "symbol": ["AAPL", "MSFT"],
   })
   ```

3. **Security Master**: If 'ticker' or 'entity' matches a symbol in the security master, it is used directly.

### Missing Policy

The `missing` parameter controls how unmapped entities are handled:
- `"raise"` (default): Raise `ValueError` with list of missing entities
- `"drop"`: Remove rows without valid symbol mapping
- `"keep_unknown"`: Set `symbol="UNKNOWN"` for unmapped entities

### Usage Example

```python
from src.assembled_core.data.news import link_news_to_symbols

news_df = pd.DataFrame({
    "publish_ts": ["2024-01-15 10:00:00"],
    "source": ["reuters"],
    "headline": ["Apple reports earnings"],
    "ticker": ["AAPL"],
})

# Map ticker to symbol
mapping_df = pd.DataFrame({
    "entity": ["AAPL"],
    "symbol": ["AAPL"],
})

linked_news = link_news_to_symbols(
    news_df,
    mapping_df=mapping_df,
    missing="raise",
)
```

## Integration with Feature Builders

News feature builders should:
1. Load news events from partitions
2. Apply `link_news_to_symbols()` if needed (if ticker/entity columns present)
3. Apply `filter_news_pit(news_df, as_of)` before feature calculation
4. Use `publish_ts` (not `ingest_ts` or `revised_ts`) for window calculations

This ensures PIT-safety and prevents look-ahead bias.

## Error Handling

- **Missing Required Columns**: `ValueError` with list of missing columns
- **Timestamp Sanity Violations**: `ValueError` with description of violations
- **Empty DataFrame**: `ValueError` if trying to store empty DataFrame
- **Missing Identifier**: `ValueError` if no identifier column (headline/url/provider_id) is present

## Determinism

All operations are deterministic:
- Same input -> same output
- Deduplication is stable (same key -> same result)
- Sorting is stable (same data -> same order)

This ensures reproducible ingestion and feature building.
