# P2 - Factor Store & Data Layout (Design)

## 1. Overview & Goals

The goal of P2 is to introduce a small, reusable factor store API that:

- Stores factor panels on disk under `data/factors/...` in a structured, partitioned format
- Can reload factor panels quickly for backtests, EOD workflows, and ML experiments
- Enforces point-in-time safety by filtering on `end`/`as_of` dates (no future data)
- Provides a simple, filesystem-based storage layer without external services or live APIs

This factor store serves as the foundation for faster backtest iterations and reproducible research. It separates factor computation (expensive, done once) from factor consumption (frequent, should be fast).

The store is intentionally minimal in P2 - we focus on core storage/retrieval functionality. Advanced features like lazy loading, caching, and incremental updates are deferred to later phases (P3/P4).

---

## 2. Data Contracts

### Price Panel

Required columns:
- `timestamp`: datetime (UTC-aware preferred, but UTC-naive acceptable)
- `symbol`: string
- Price columns: `open`, `high`, `low`, `close`, `volume` (or subset - at minimum `close` is required)

The timestamp can be either:
- Index level (MultiIndex with `timestamp`, `symbol`)
- Regular column named `timestamp`

### Factor Panel

Required columns:
- `timestamp`: datetime (UTC-aware or UTC-naive, must match price panel)
- `symbol`: string
- Factor columns: `factor_*` or other naming conventions (e.g., `momentum_12m`, `rv_20`, `earnings_yield`)

**Note:** Forward return columns (`fwd_return_*d`) are not stored in the factor store itself in P2. They are typically computed on-demand during ML validation or can be added in separate workflows. This keeps the factor store focused on raw factor values.

### Point-in-Time Safety

All load operations enforce point-in-time constraints:
- Data loaded for a given `end` date must not contain any rows with `timestamp > end`
- This prevents look-ahead bias in backtests and research

---

## 3. Factor Store API

The API is implemented in `src/assembled_core/data/factor_store.py` with the following functions:

### `load_price_panel()`

```python
def load_price_panel(
    freq: str,
    universe: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    price_loader: Optional[Callable[..., pd.DataFrame]] = None,
) -> pd.DataFrame
```

Loads a price panel for the given universe and date range. If `price_loader` is provided, it is used to fetch prices (e.g., from existing `load_eod_prices` or `load_intraday_prices` functions). Otherwise, uses the default price loading mechanism.

Returns a DataFrame with required price columns (`timestamp`, `symbol`, `close`, etc.).

### `store_factors()`

```python
def store_factors(
    freq: str,
    group: str,
    df: pd.DataFrame,
    *,
    root: Optional[Path] = None,
) -> Dict[str, Path]
```

Stores a factor panel DataFrame to disk. The `group` parameter categorizes the factor set:
- `"ta"`: Technical analysis / price-based factors
- `"alt_insider"`: Alternative data - insider trading factors
- `"alt_earnings"`: Alternative data - earnings factors
- `"alt_news"`: Alternative data - news sentiment factors
- Other groups as needed

The function partitions data by year:
- Storage path: `data/factors/<freq>/<group>/<year>.parquet`
- Example: `data/factors/1d/ta/2023.parquet`

If the input DataFrame spans multiple years, it is split and one file per year is written.

Returns a dictionary mapping year (as string, e.g., "2023") to the written file path.

### `load_factors()`

```python
def load_factors(
    freq: str,
    universe: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    groups: Sequence[str] = ("ta",),
    root: Optional[Path] = None,
) -> pd.DataFrame
```

Loads factor panels for the given universe and date range. Behavior:

1. Detects all years between `start` and `end` (inclusive)
2. For each `group` in `groups`, loads all existing parquet files matching `data/factors/<freq>/<group>/<year>.parquet`
3. Concatenates all loaded DataFrames
4. Filters by `timestamp` between `start` and `end` (point-in-time: `timestamp <= end`)
5. Filters by `symbol` in `universe`
6. If multiple groups are requested, merges them horizontally on `(timestamp, symbol)`
7. Returns the combined factor panel

Missing files for a group/year combination are silently skipped (allows partial data).

### `list_available_panels()`

```python
def list_available_panels(
    root: Optional[Path] = None,
) -> pd.DataFrame
```

Scans the factor store root directory recursively and detects available panels. Returns a DataFrame with at least the following columns:

- `freq`: Frequency string (e.g., "1d", "5min")
- `group`: Factor group name (e.g., "ta", "alt_insider")
- `year`: Year as integer or string
- `path`: Full path to the parquet file

Optionally, if cheap to compute, can include:
- `n_rows`: Number of rows in the file
- `min_timestamp`: Earliest timestamp in the file
- `max_timestamp`: Latest timestamp in the file

This function is useful for discovering what data is available before attempting to load.

---

## 4. Storage Layout & Conventions

### Root Directory

- Environment variable: `ASSEMBLED_FACTOR_STORE_ROOT`
- Default: `data/factors/` (relative to project root)
- The root can be absolute or relative; all paths within the store are relative to this root

### Directory Structure

```
data/factors/
  <freq>/
    <group>/
      <year>.parquet
```

Examples:
- `data/factors/1d/ta/2023.parquet`
- `data/factors/1d/alt_insider/2023.parquet`
- `data/factors/5min/ta/2024.parquet`

### File Format

- Format: Parquet (Apache Parquet)
- Required columns in every file: `timestamp`, `symbol`
- Additional columns: factor columns (names vary by group)
- All files are self-contained and can be loaded independently

### Conventions

- All timestamps are stored as datetime objects (UTC-aware or UTC-naive, consistent within a file)
- Symbols are stored as strings
- Missing data (NaN) is allowed in factor columns, but `timestamp` and `symbol` must never be NaN
- No external services, no live APIs: everything is local and filesystem-based
- Files are append-only (yearly partitioning means new data for a year overwrites the file, but this is acceptable for P2)

---

## 5. Implementation Plan

### P2.1: Implement Factor Store Module

- Create `src/assembled_core/data/factor_store.py`
- Implement `load_price_panel()`, `store_factors()`, `load_factors()`, `list_available_panels()`
- Handle edge cases: empty DataFrames, missing files, date range filtering, symbol filtering
- Add basic logging for store/load operations

### P2.2: Unit Tests

- Create `tests/test_data_factor_store.py`
- Test roundtrip: store a factor panel, then load it back and verify data integrity
- Test `list_available_panels()` with various directory structures
- Test point-in-time safety: verify that data beyond `end` date is never returned
- Test multiple groups: verify horizontal merge behavior
- Test edge cases: empty universe, date range with no data, missing files

### P2.3: Documentation & Integration Points

- Update relevant workflow documents to mention the factor store
- Add examples of using the factor store API
- Optionally wire factor store into existing backtest workflows (can be minimal integration for P2, full integration in later sprints)

### P2.4: Performance Considerations

- Document current performance characteristics (load time for typical panels)
- Identify bottlenecks for large panels (many years, many symbols, many groups)
- Plan for future optimizations:
  - Lazy loading (only load columns needed)
  - Partitioning strategies (monthly instead of yearly for very large datasets)
  - Caching layer (keep frequently accessed panels in memory)
  - These optimizations will be addressed in P3/P4 based on actual usage patterns

---

## 6. References

- [Performance Profiling P1 Design](PERFORMANCE_PROFILING_P1_DESIGN.md): Performance baseline for measuring factor store impact
- [Advanced Analytics Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md): Overall roadmap context

