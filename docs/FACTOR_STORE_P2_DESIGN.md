# Factor Store (P2) - Design Document

**Last Updated:** 2025-12-23  
**Status:** Design Phase  
**Related:** [Performance Profiling P1](PERFORMANCE_PROFILING_P1_DESIGN.md), [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md)

---

## 1. Overview & Goals

**Goal:** Centralize factor/feature computation and caching to avoid redundant calculations across EOD pipelines, backtests, and ML workflows.

**Scope P2:**
- Define storage structure for computed factors (organized by group, frequency, universe, year)
- Implement `build_or_load_factors()` API that checks cache before computing
- Integrate with existing EOD (`run_daily.py`) and backtest (`run_backtest_strategy.py`) workflows
- Ensure Point-in-Time (PIT) safety: Loader filters by `start_date`, `end_date`, `as_of`
- Store metadata (computation date, hash, parameters) for cache invalidation

**Key Benefits:**
- **Performance:** Avoid recomputing expensive features (e.g., 200-day moving averages) for same data
- **Consistency:** Same features used across EOD, backtests, and ML workflows
- **Reproducibility:** Stored factors are deterministic and can be versioned
- **PIT Safety:** Time-based filtering ensures no look-ahead bias

**Non-Goals:**
- P2 does NOT optimize feature computation itself (that's P3+)
- P2 does NOT handle factor versioning/lineage tracking (future work)
- P2 focuses on TA/Price factors; Alt-Data factors may be handled separately

---

## 2. Storage Structure

### 2.1. Directory Layout

Factors are stored in a hierarchical structure organized by:
1. **Factor Group** (e.g., `core_ta`, `vol_liquidity`, `alt_insider`)
2. **Frequency** (e.g., `1d`, `5min`)
3. **Universe Key** (hash or name derived from universe symbols, sorted)
4. **Year** (partitioning for large datasets)

```
data/factors/
├── core_ta/
│   ├── 1d/
│   │   ├── universe_sp500/
│   │   │   ├── year=2023.parquet
│   │   │   ├── year=2024.parquet
│   │   │   └── _metadata.json
│   │   └── universe_watchlist/
│   │       ├── year=2023.parquet
│   │       └── year=2024.parquet
│   └── 5min/
│       └── universe_watchlist/
│           └── year=2024.parquet
├── vol_liquidity/
│   └── 1d/
│       └── universe_sp500/
│           └── year=2024.parquet
└── alt_insider/
    └── 1d/
        └── universe_watchlist/
            └── year=2024.parquet
```

### 2.2. Universe Key Generation

Universe keys are deterministic hashes derived from sorted symbol lists:

```python
def compute_universe_key(symbols: list[str]) -> str:
    """Generate deterministic universe key from symbol list."""
    sorted_symbols = sorted([s.upper() for s in symbols])
    # Use hash of sorted list (short hash for readability, e.g., first 8 chars of MD5)
    import hashlib
    symbols_str = ",".join(sorted_symbols)
    hash_hex = hashlib.md5(symbols_str.encode()).hexdigest()[:8]
    # For small universes, use readable name; for large, use hash
    if len(sorted_symbols) <= 20:
        return f"universe_{'_'.join(sorted_symbols[:5])}_{hash_hex}"
    else:
        return f"universe_{hash_hex}"
```

**Examples:**
- `["AAPL", "MSFT", "GOOGL"]` → `universe_AAPL_GOOGL_MSFT_a1b2c3d4`
- `["AAPL", "MSFT", ..., "NVDA"]` (200 symbols) → `universe_e5f6g7h8`

### 2.3. File Format: Parquet

Each factor file is a Parquet dataset with:
- **Index:** Multi-index `(timestamp, symbol)` or columns `timestamp`, `date`, `symbol`
- **Columns:**
  - Required: `timestamp` (pd.Timestamp, UTC), `date` (date string YYYY-MM-DD), `symbol` (string)
  - Price columns (if included): `px_open`, `px_high`, `px_low`, `px_close`, `px_volume`
  - Feature columns with prefixes:
    - `ta_*` for technical analysis (e.g., `ta_ma_20`, `ta_rsi_14`, `ta_atr_14`)
    - `vol_*` for volatility factors (e.g., `vol_rv_20`, `vol_vov_20_60`)
    - `liq_*` for liquidity factors (e.g., `liq_turnover`, `liq_volume_zscore`)
    - `alt_*` for alternative data factors (e.g., `alt_insider_net_buy_20d`)
- **Sorting:** Sorted by `timestamp`, then `symbol` (ascending)
- **Partitioning:** By year (enables efficient date filtering)

### 2.4. Metadata File

Each universe directory contains `_metadata.json`:

```json
{
  "factor_group": "core_ta",
  "freq": "1d",
  "universe_key": "universe_sp500",
  "symbols": ["AAPL", "MSFT", ...],
  "computed_at": "2025-12-23T10:00:00Z",
  "computation_params": {
    "ma_windows": [20, 50, 200],
    "atr_window": 14,
    "rsi_window": 14
  },
  "data_hash": "abc123...",
  "factor_columns": ["ta_ma_20", "ta_ma_50", "ta_atr_14", "ta_rsi_14"],
  "date_range": {
    "start": "2023-01-01",
    "end": "2024-12-31"
  }
}
```

**Purpose:**
- Cache invalidation (if params or data hash change, recompute)
- Discovery (list available factor groups/universes)
- Debugging (know what factors are stored and when computed)

---

## 3. API Design

### 3.1. Core Functions

#### `load_price_panel()`

```python
def load_price_panel(
    freq: str,
    universe_symbols: list[str] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load price data for factor computation.
    
    Args:
        freq: Frequency ("1d" or "5min")
        universe_symbols: Optional list of symbols (default: from settings.watchlist_file)
        start_date: Optional start date (inclusive, UTC)
        end_date: Optional end date (inclusive, UTC)
        as_of: Optional point-in-time date (loads data up to and including this date)
    
    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        Sorted by timestamp, then symbol
        Filtered to requested date range (if provided)
    
    Notes:
        - If both start_date/end_date and as_of are provided, as_of takes precedence
        - as_of ensures PIT safety: only data <= as_of is returned
    """
```

**Implementation:** Wrapper around existing `load_eod_prices_for_universe()` with date filtering.

#### `load_factors()`

```python
def load_factors(
    factor_group: str,
    freq: str,
    universe_symbols: list[str] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    as_of: str | pd.Timestamp | None = None,
    factors_root: Path | str | None = None,
) -> pd.DataFrame | None:
    """Load cached factors from factor store.
    
    Args:
        factor_group: Factor group name (e.g., "core_ta", "vol_liquidity")
        freq: Frequency ("1d" or "5min")
        universe_symbols: List of symbols (for universe key generation)
        start_date: Optional start date filter (inclusive, UTC)
        end_date: Optional end date filter (inclusive, UTC)
        as_of: Optional point-in-time date (loads factors up to and including this date)
        factors_root: Optional root directory (default: data/factors/)
    
    Returns:
        DataFrame with factors (columns: timestamp, date, symbol, ta_*, vol_*, etc.)
        or None if factors not found in cache
    
    Notes:
        - PIT Safety: Filters factors by date range (start_date/end_date or as_of)
        - Returns None if cache miss (caller should compute and store)
    """
```

#### `store_factors()`

```python
def store_factors(
    factors_df: pd.DataFrame,
    factor_group: str,
    freq: str,
    universe_symbols: list[str],
    computation_params: dict[str, Any] | None = None,
    data_hash: str | None = None,
    factors_root: Path | str | None = None,
) -> Path:
    """Store computed factors in factor store.
    
    Args:
        factors_df: DataFrame with factors (must have timestamp, symbol columns)
        factor_group: Factor group name (e.g., "core_ta")
        freq: Frequency ("1d" or "5min")
        universe_symbols: List of symbols (for universe key generation)
        computation_params: Optional dict of parameters used for computation (stored in metadata)
        data_hash: Optional hash of source data (for cache invalidation)
        factors_root: Optional root directory (default: data/factors/)
    
    Returns:
        Path to stored factor directory
    
    Notes:
        - Partitions by year automatically (creates year=YYYY.parquet files)
        - Updates _metadata.json with computation info
        - Atomic writes (write to temp, then rename)
    """
```

#### `build_or_load_factors()`

```python
def build_or_load_factors(
    factor_group: str,
    freq: str,
    universe_symbols: list[str] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    as_of: str | pd.Timestamp | None = None,
    computation_params: dict[str, Any] | None = None,
    force_recompute: bool = False,
    builder_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build or load factors from cache.
    
    High-level API that:
    1. Checks cache for existing factors
    2. If cache hit and params match: load and return
    3. If cache miss or force_recompute: load prices, compute factors, store, return
    
    Args:
        factor_group: Factor group name (e.g., "core_ta")
        freq: Frequency ("1d" or "5min")
        universe_symbols: Optional list of symbols (default: from settings)
        start_date: Optional start date (inclusive, UTC)
        end_date: Optional end date (inclusive, UTC)
        as_of: Optional point-in-time date (PIT-safe filtering)
        computation_params: Dict of parameters for computation (for cache key)
        force_recompute: If True, recompute even if cache exists
        builder_fn: Function that takes prices DataFrame, returns factors DataFrame
                   (default: uses built-in builder for factor_group)
    
    Returns:
        DataFrame with factors (filtered to requested date range)
    
    Example:
        # Build core TA factors for watchlist, last 2 years
        factors = build_or_load_factors(
            factor_group="core_ta",
            freq="1d",
            start_date="2023-01-01",
            end_date="2024-12-31",
            computation_params={"ma_windows": [20, 50, 200], "rsi_window": 14}
        )
    """
```

#### `list_available_panels()`

```python
def list_available_panels(
    factor_group: str | None = None,
    freq: str | None = None,
    factors_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List available factor panels in the store.
    
    Args:
        factor_group: Optional filter by factor group
        freq: Optional filter by frequency
        factors_root: Optional root directory (default: data/factors/)
    
    Returns:
        List of dicts with keys: factor_group, freq, universe_key, date_range, computed_at
    """
```

---

## 4. Integration Points

### 4.1. EOD Pipeline (`run_daily.py`)

**Current Flow:**
1. Load prices (`load_eod_prices_for_universe()`)
2. Compute features (`add_all_features()`)
3. Generate signals
4. Generate orders

**New Flow (with Factor Store):**
1. Load prices (if needed for features not in store)
2. **Load or compute factors** (`build_or_load_factors(factor_group="core_ta", ...)`)
3. Merge prices with factors (if factors don't include prices)
4. Generate signals (using factors)
5. Generate orders

**Hook Location:**
- Replace `add_all_features()` call (around line 348) with `build_or_load_factors()`
- Pass `as_of=target_date` for PIT safety

**Example:**
```python
# Step 4: Load or compute factors
from src.assembled_core.factor_store import build_or_load_factors

factors = build_or_load_factors(
    factor_group="core_ta",
    freq="1d",
    universe_symbols=universe_symbols,
    as_of=target_date,  # PIT-safe
    computation_params={"ma_windows": (20, 50, 200), "rsi_window": 14},
)

# Merge with prices (if factors don't include price columns)
prices_with_features = prices.merge(factors, on=["timestamp", "symbol"], how="left")
```

### 4.2. Backtest Runner (`run_backtest_strategy.py`)

**Current Flow:**
- Signal function receives `prices_df` and computes features inline
- For `multifactor_long_short`: factors are computed from prices within signal function

**New Flow (with Factor Store):**
- Option 1: Pre-compute factors before backtest loop
- Option 2: Lazy load within signal function (via `build_or_load_factors()`)

**Hook Location:**
- In `run_backtest_from_args()`, before creating signal function
- Or within signal function (if lazy loading preferred)

**Example:**
```python
# Pre-compute factors for entire backtest date range
factors = build_or_load_factors(
    factor_group="core_ta",
    freq=args.freq,
    universe_symbols=universe_symbols,
    start_date=args.start_date,
    end_date=args.end_date,
    computation_params={"ma_windows": (20, 50, 200)},
)

# Pass factors to signal function
signal_fn = create_trend_baseline_signal_fn(ma_fast=20, ma_slow=50)
signals = signal_fn(prices_df.merge(factors, on=["timestamp", "symbol"], how="left"))
```

---

## 5. Point-in-Time (PIT) Safety

### 5.1. Date Filtering in Loader

The `load_factors()` function must filter factors by date range to ensure PIT safety:

```python
def load_factors(..., as_of: pd.Timestamp | None = None, ...):
    # Load all parquet files for the universe
    factors_dir = factors_root / factor_group / freq / universe_key
    
    # Filter by date range
    if as_of is not None:
        # PIT-safe: only factors <= as_of
        factors_df = factors_df[factors_df["timestamp"] <= as_of]
    elif start_date is not None or end_date is not None:
        # Date range filtering
        if start_date:
            factors_df = factors_df[factors_df["timestamp"] >= start_date]
        if end_date:
            factors_df = factors_df[factors_df["timestamp"] <= end_date]
    
    return factors_df
```

### 5.2. Metadata Validation

When loading factors, validate that stored factors cover the requested date range:

```python
# Check metadata
metadata = load_metadata(factors_dir / "_metadata.json")
requested_start = start_date or factors_df["timestamp"].min()
requested_end = end_date or as_of or factors_df["timestamp"].max()

if metadata["date_range"]["start"] > requested_start:
    # Cache miss: need to compute earlier dates
    return None

if metadata["date_range"]["end"] < requested_end:
    # Partial cache hit: may need to compute additional dates
    # (For P2, we can recompute entire range for simplicity, or implement incremental updates)
    return None  # For simplicity in P2, treat as cache miss
```

---

## 6. Implementation Plan

### Phase 1: Core Storage & Loader (Week 1)
- [ ] Implement directory structure creation
- [ ] Implement `store_factors()` with Parquet partitioning by year
- [ ] Implement `load_factors()` with date filtering
- [ ] Implement metadata JSON read/write

### Phase 2: Builder Integration (Week 1-2)
- [ ] Implement `build_or_load_factors()` with cache check
- [ ] Create builder registry for different factor groups
- [ ] Integrate with `add_all_features()` for "core_ta" group
- [ ] Add unit tests for cache hit/miss scenarios

### Phase 3: EOD Integration (Week 2)
- [ ] Integrate `build_or_load_factors()` into `run_daily.py`
- [ ] Add `--factor-store` flag (default: enabled)
- [ ] Test EOD pipeline with factor caching

### Phase 4: Backtest Integration (Week 2-3)
- [ ] Integrate `build_or_load_factors()` into `run_backtest_strategy.py`
- [ ] Update signal functions to accept pre-computed factors
- [ ] Test backtest workflows with factor caching

### Phase 5: Documentation & Testing (Week 3)
- [ ] Write integration tests (EOD + Backtest workflows)
- [ ] Update documentation (workflow guides)
- [ ] Performance profiling (compare with/without factor store)

---

## 7. Open Questions & Future Work

### P2 Scope Decisions:
1. **Incremental Updates:** Should we support appending new dates to existing factor files? (For P2: No, recompute entire range if dates extend beyond cache)
2. **Factor Versioning:** Should we track factor computation versions? (For P2: No, use params hash for cache key)
3. **Alt-Data Factors:** Should alt-data factors (insider, shipping) be in same store? (For P2: Separate store, but same API)

### Future Enhancements (P3+):
- **Incremental Updates:** Append-only factor files for new dates
- **Factor Lineage:** Track which features depend on which source data
- **Multi-Factor Groups:** Combine multiple factor groups into single panel
- **Factor Validation:** Checksums and validation after load
- **Compression:** Column compression for large factor panels

---

## 8. References

- **Design Docs:**
  - [Performance Profiling P1](PERFORMANCE_PROFILING_P1_DESIGN.md) - Baseline measurements
  - [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) - Factor development roadmap

- **Code References:**
  - `src/assembled_core/features/ta_features.py` - Current feature computation
  - `scripts/run_daily.py` - EOD pipeline (hook point)
  - `scripts/run_backtest_strategy.py` - Backtest runner (hook point)
  - `src/assembled_core/data/prices_ingest.py` - Price loading (similar pattern)

---

## 9. Acceptance Criteria

**P2 is considered "done" when:**

1. ✅ Factor store structure is created (`data/factors/<group>/<freq>/<universe>/year=YYYY.parquet`)
2. ✅ `build_or_load_factors()` API works for "core_ta" group
3. ✅ EOD pipeline (`run_daily.py`) uses factor store (with `--factor-store` flag)
4. ✅ Backtest runner can use factor store (optional, via `--use-factor-store`)
5. ✅ PIT safety: `as_of` filtering works correctly
6. ✅ Unit tests: cache hit/miss scenarios
7. ✅ Integration tests: EOD + Backtest workflows produce same results with/without cache
8. ✅ Performance: Factor store reduces EOD runtime by >50% for repeated runs (measured via P1 profiling)

---

**Status:** Design complete, ready for implementation.
