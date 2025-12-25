# Factor Store (P2) - API & Layout Standard

**Last Updated:** 2025-01-XX  
**Status:** Standard Contract Document  
**Related:** [Factor Store P2 Design](FACTOR_STORE_P2_DESIGN.md), [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md)

---

## 1. Overview

This document defines the **standard contract** (API and layout) for the Factor Store system, enabling caching of computed factors/features to avoid redundant calculations across EOD pipelines, backtests, and ML workflows.

**Key Benefits:**
- **Performance:** Avoid recomputing expensive features (e.g., 200-day moving averages)
- **Consistency:** Same features used across EOD, backtests, and ML workflows
- **Reproducibility:** Stored factors are deterministic and can be versioned
- **PIT Safety:** Time-based filtering ensures no look-ahead bias

---

## 2. Storage Layout

### 2.1. Directory Structure

Factors are stored in a hierarchical structure organized by:
1. **Factor Group** (e.g., `core_ta`, `vol_liquidity`, `alt_insider_...`)
2. **Frequency** (e.g., `1d`, `5min`)
3. **Universe Key** (deterministic hash/name derived from sorted symbol list)
4. **Year** (partitioning for large datasets)

```
data/factors/
├── core_ta/                    # Factor group: core technical analysis
│   ├── 1d/                     # Frequency: daily
│   │   ├── universe_sp500/     # Universe key (deterministic from symbols)
│   │   │   ├── year=2023.parquet
│   │   │   ├── year=2024.parquet
│   │   │   └── _metadata.json
│   │   └── universe_watchlist/
│   │       ├── year=2023.parquet
│   │       └── year=2024.parquet
│   └── 5min/                   # Frequency: 5-minute
│       └── universe_watchlist/
│           └── year=2024.parquet
├── vol_liquidity/              # Factor group: volatility & liquidity
│   └── 1d/
│       └── universe_sp500/
│           └── year=2024.parquet
├── alt_insider_.../            # Factor group: alternative data (insider, shipping, etc.)
│   └── 1d/
│       └── universe_watchlist/
│           └── year=2024.parquet
└── px_.../                     # Factor group prefix: price data (if stored)
    └── 1d/
        └── universe_watchlist/
            └── year=2024.parquet
```

### 2.2. Factor Group Prefixes

Factor groups follow naming conventions:

- **`core_ta`**: Core technical analysis (MA, RSI, ATR, MACD, Bollinger Bands)
- **`vol_liquidity`**: Volatility and liquidity factors (realized vol, volume z-score, turnover)
- **`alt_insider_...`**: Alternative data factors (insider trades, shipping congestion, news sentiment)
- **`px_...`**: Price data panels (if stored separately, e.g., `px_base` for OHLCV)

**Note:** Group prefixes are descriptive and can be extended (e.g., `alt_insider_net_buy_20d`, `alt_shipping_congestion_7d`).

### 2.3. Year Partitioning

Each universe directory contains Parquet files partitioned by year:
- Format: `year=YYYY.parquet` (e.g., `year=2023.parquet`, `year=2024.parquet`)
- Enables efficient date filtering (only load relevant year files)
- Combines multiple year files when loading date ranges

### 2.4. Metadata File

Each universe directory contains `_metadata.json`:

```json
{
  "factor_group": "core_ta",
  "freq": "1d",
  "universe_key": "universe_sp500",
  "computed_at": "2025-01-15T10:00:00Z",
  "date_range": {
    "start": "2023-01-01T00:00:00Z",
    "end": "2024-12-31T23:59:59Z"
  },
  "years": [2023, 2024],
  "factor_columns": ["ta_ma_20", "ta_ma_50", "ta_atr_14", "ta_rsi_14"],
  "schema": {
    "columns": ["timestamp", "date", "symbol", "ta_ma_20", "ta_ma_50", "ta_atr_14", "ta_rsi_14"],
    "dtypes": {
      "timestamp": "datetime64[ns, UTC]",
      "date": "object",
      "symbol": "object",
      "ta_ma_20": "float64",
      "ta_ma_50": "float64"
    }
  },
  "config_hash": "a1b2c3d4"
}
```

---

## 3. Standard API

### 3.1. `load_price_panel()`

Load price data for factor computation.

**Note:** This function is not yet implemented as a separate API. Price data loading is currently handled by:
- `src/assembled_core/data/prices_ingest.py::load_eod_prices_for_universe()`
- `src/assembled_core/pipeline/io.py::load_prices()`

**Future API (planned):**
```python
def load_price_panel(
    freq: str,
    universe: list[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load price data for factor computation.
    
    Args:
        freq: Frequency ("1d" or "5min")
        universe: Optional list of symbols (default: from settings.watchlist_file)
        start: Optional start date (inclusive, UTC)
        end: Optional end date (inclusive, UTC)
    
    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        Sorted by timestamp, then symbol
        Filtered to requested date range (if provided)
    """
```

**Current Status:** Price loading is handled directly by existing functions in the pipeline.

---

### 3.2. `load_factors()`

Load cached factors from factor store.

```python
def load_factors(
    factor_group: str,
    freq: str,
    universe_key: str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    as_of: str | pd.Timestamp | None = None,
    factors_root: Path | None = None,
) -> pd.DataFrame | None:
    """Load cached factors from factor store.
    
    Args:
        factor_group: Factor group name (e.g., "core_ta", "vol_liquidity")
        freq: Frequency ("1d" or "5min")
        universe_key: Universe key (from compute_universe_key)
        start_date: Optional start date filter (inclusive, UTC)
        end_date: Optional end date filter (inclusive, UTC)
        as_of: Optional point-in-time cutoff (inclusive, UTC)
        factors_root: Optional root directory (default: data/factors/)
    
    Returns:
        DataFrame with factors (columns: timestamp, date, symbol, <feature_columns>)
        or None if factors not found in cache
    
    Notes:
        - PIT Safety: Filters factors by date range (start_date/end_date/as_of)
        - Returns None if cache miss (caller should compute and store)
        - To load multiple groups, call multiple times and merge
    """
```

**Current Implementation:** `src/assembled_core/data/factor_store.py::load_factors()`  
**Status:** ✅ API matches standard contract

---

### 3.3. `store_factors()`

Store computed factors in factor store.

```python
def store_factors(
    df: pd.DataFrame,
    factor_group: str,
    freq: str,
    universe_key: str,
    mode: str = "overwrite",
    factors_root: Path | None = None,
    metadata: dict[str, Any] | None = None,
    write_manifest: bool = True,
) -> Path:
    """Store computed factors in factor store.
    
    Args:
        df: DataFrame with factors (must have timestamp, symbol columns)
        factor_group: Factor group name (e.g., "core_ta", "vol_liquidity")
        freq: Frequency ("1d" or "5min")
        universe_key: Universe key (from compute_universe_key)
        mode: Storage mode: "overwrite" (default) or "append"
        factors_root: Optional root directory (default: data/factors/)
        metadata: Optional metadata dict to store in _metadata.json
        write_manifest: If True, write _metadata.json file
    
    Returns:
        Path to stored factor directory
    
    Notes:
        - Partitions by year automatically (creates year=YYYY.parquet files)
        - Updates _metadata.json with computation info
        - Atomic writes (write to temp, then rename)
        - For append mode: merges with existing files, deduplicates by (timestamp, symbol)
    """
```

**Current Implementation:** `src/assembled_core/data/factor_store.py::store_factors()`  
**Status:** ✅ API matches standard contract

---

### 3.4. `list_available_panels()`

List available factor panels in the store.

```python
def list_available_panels(
    factor_group: str | None = None,
    freq: str | None = None,
    factors_root: Path | None = None,
) -> list[dict[str, Any]]:
    """List available factor panels in the store.
    
    Args:
        factor_group: Optional filter by factor group
        freq: Optional filter by frequency
        factors_root: Optional root directory (default: data/factors/)
    
    Returns:
        List of dicts with keys:
        - factor_group, freq, universe_key
        - date_range (start, end)
        - years (list of available years)
        - computed_at, config_hash (if manifest exists)
        - factor_columns (if manifest exists)
    """
```

**Current Implementation:** `src/assembled_core/data/factor_store.py::list_available_panels()`  
**Status:** ✅ Matches standard API

---

## 4. Universe Key Generation

Universe keys are deterministic hashes derived from sorted symbol lists:

```python
def compute_universe_key(symbols: list[str]) -> str:
    """Generate deterministic universe key from symbol list."""
    sorted_symbols = sorted([s.upper() for s in symbols])
    symbols_str = ",".join(sorted_symbols)
    hash_hex = hashlib.md5(symbols_str.encode()).hexdigest()[:8]
    
    # For small universes, include symbol names for readability
    if len(sorted_symbols) <= 20:
        return f"universe_{'_'.join(sorted_symbols[:5])}_{hash_hex}"
    else:
        return f"universe_{hash_hex}"
```

**Examples:**
- `["AAPL", "MSFT", "GOOGL"]` → `universe_AAPL_GOOGL_MSFT_a1b2c3d4`
- `["AAPL", "MSFT", ..., "NVDA"]` (200 symbols) → `universe_e5f6g7h8`

**Implementation:** `src/assembled_core/data/factor_store.py::compute_universe_key()`

---

## 5. File Format: Parquet

Each factor file is a Parquet dataset with:

- **Required Columns:**
  - `timestamp` (pd.Timestamp, UTC-aware)
  - `date` (date string YYYY-MM-DD, redundant but useful for filtering)
  - `symbol` (string)

- **Feature Columns** (with prefixes):
  - `ta_*` for technical analysis (e.g., `ta_ma_20`, `ta_rsi_14`, `ta_atr_14`)
  - `vol_*` for volatility factors (e.g., `vol_rv_20`, `vol_vov_20_60`)
  - `liq_*` for liquidity factors (e.g., `liq_turnover`, `liq_volume_zscore`)
  - `alt_*` for alternative data factors (e.g., `alt_insider_net_buy_20d`)
  - `px_*` for price columns (if included: `px_open`, `px_high`, `px_low`, `px_close`, `px_volume`)

- **Sorting:** Sorted by `timestamp`, then `symbol` (ascending)
- **Partitioning:** By year (`year=YYYY.parquet`)

---

## 6. Point-in-Time (PIT) Safety

All load functions support PIT-safe filtering:

- **`start` / `end`:** Date range filtering (inclusive)
- **`as_of`:** Point-in-time cutoff (only returns data with `timestamp <= as_of`)

**Example:**
```python
# Load factors up to 2024-01-15 (PIT-safe for backtesting)
factors = load_factors(
    freq="1d",
    universe=["AAPL", "MSFT"],
    start="2023-01-01",
    as_of="2024-01-15",  # Only data <= 2024-01-15
    groups=["core_ta"],
)
```

**Implementation:** `src/assembled_core/data/factor_store.py::load_factors()` filters by `start_date`, `end_date`, and `as_of`.

---

## 7. Integration Example

### EOD Pipeline Integration

```python
from src.assembled_core.features.factor_store_integration import build_or_load_factors

# Load or compute factors
factors = build_or_load_factors(
    prices=prices_df,
    factor_group="core_ta",
    freq="1d",
    as_of=target_date,  # PIT-safe
)

# Merge with prices (if factors don't include price columns)
prices_with_features = prices_df.merge(
    factors, on=["timestamp", "symbol"], how="left"
)
```

---

## 8. References

- **Design Document:** [Factor Store P2 Design](FACTOR_STORE_P2_DESIGN.md)
- **Implementation:** `src/assembled_core/data/factor_store.py`
- **Integration:** `src/assembled_core/features/factor_store_integration.py`
- **Factor Labs:** [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md)

---

**Status:** Standard contract defined. API aligns with implementation in `factor_store.py`.

