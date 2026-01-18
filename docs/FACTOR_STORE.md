# Factor Store - Design & API (Sprint 5 / F1)

**Status:** Implementiert (P2)  
**Last Updated:** 2025-01-04  
**ASCII-only:** Yes

---

## Overview

Der **Factor Store** ist ein versioniertes, wiederverwendbares Caching-System fuer berechnete Faktoren und Features. Er speichert technische Indikatoren und andere Features in einer strukturierten, partitionierten Form, um teure Neuberechnungen zu vermeiden und Performance zu verbessern.

### Ziele

- **Versionierung**: Faktoren sind deterministisch und reproduzierbar
- **Wiederverwendbarkeit**: Gleiche Features werden in EOD-Pipelines, Backtests und ML-Workflows verwendet
- **Performance**: Vermeidet wiederholte Berechnungen teurer Features (z.B. 200-Tage Moving Averages)
- **PIT-Safety**: Point-in-Time-Filterung verhindert Look-Ahead-Bias
- **Layering-konform**: Implementierung im data-layer (keine pipeline/qa imports)

---

## Directory Structure

Der Factor Store verwendet eine hierarchische Ordnerstruktur mit Jahres-Partitionierung:

```
data/factors/
├── <factor_group>/              # Factor Group (z.B. "core_ta", "vol_liquidity")
│   ├── <freq>/                  # Frequency ("1d" oder "5min")
│   │   ├── <universe_key>/      # Universe Key (deterministischer Hash/Name)
│   │   │   ├── year=2023.parquet
│   │   │   ├── year=2024.parquet
│   │   │   └── _metadata.json   # Metadaten (Schema, Parameter, etc.)
│   │   └── <universe_key_2>/
│   │       └── year=2024.parquet
│   └── <freq_2>/
│       └── <universe_key>/
│           └── year=2024.parquet
```

### Pfadkonvention

**Standard-Pfad:** `data/factors/<group>/<freq>/<universe>/year=YYYY.parquet`

- **Factor Group**: Kategorie der Faktoren (`core_ta`, `vol_liquidity`, `alt_insider`, etc.)
- **Frequency**: Trading-Frequenz (`1d`, `5min`)
- **Universe Key**: Deterministischer Hash/Name aus der Symbol-Liste (z.B. `universe_watchlist`, `universe_sp500`)
- **Year Partitioning**: Jahres-Partitionierung fuer grosse Datensaetze (`year=2023.parquet`, `year=2024.parquet`)

### Default Location

Der Standard-Pfad ist `data/factors/` im Repository-Root (via `settings.data_dir / "factors"`). Dieser kann ueber `--factor-store-root` oder `factors_root` Parameter ueberschrieben werden.

---

## API Functions

### Core Functions

#### `factor_partition_path(group, freq, universe, year=None, root=None) -> Path`

Berechnet den deterministischen Pfad zu einer Factor-Partition.

**Parameters:**
- `group`: Factor group name (z.B. "core_ta")
- `freq`: Frequency string ("1d" oder "5min")
- `universe`: Universe key (aus `compute_universe_key()`)
- `year`: Optional year fuer partitionierte Datei (z.B. 2024 -> "year=2024.parquet")
- `root`: Optional root directory (default: `get_factor_store_root()`)

**Returns:**
- Path zu Factor-Panel-Datei (wenn `year` gesetzt) oder Verzeichnis (wenn `year=None`)

**Example:**
```python
from src.assembled_core.data.factor_store import factor_partition_path

path = factor_partition_path("core_ta", "1d", "universe_sp500", year=2024)
# Path('data/factors/core_ta/1d/universe_sp500/year=2024.parquet')
```

#### `store_factors_parquet(df, group, freq, universe, mode="replace", root=None, metadata=None) -> Path`

Speichert Faktoren als Parquet-Dateien mit Jahres-Partitionierung.

**Parameters:**
- `df`: DataFrame mit Faktoren (muss `timestamp`, `symbol` Spalten haben)
- `group`: Factor group name (z.B. "core_ta")
- `freq`: Frequency string ("1d" oder "5min")
- `universe`: Universe key (aus `compute_universe_key()`)
- `mode`: Storage mode: "replace" (default) oder "append"
- `root`: Optional root directory (default: `get_factor_store_root()`)
- `metadata`: Optional metadata dict (wird in `_metadata.json` gespeichert)

**Returns:**
- Path zu Factor-Panel-Verzeichnis

**Raises:**
- `ValueError`: Wenn required columns fehlen oder mode ungültig ist
- `OSError`: Wenn atomic write fehlschlägt

**Notes:**
- Partitions data by year automatically (creates `year=YYYY.parquet` files)
- Ensures timestamp column is UTC-aware
- Adds `date` column (YYYY-MM-DD string) for filtering convenience
- Sorts by timestamp, then symbol
- For append mode: merges with existing files, deduplicates by (timestamp, symbol) using `keep="last"`
- Atomic write (temp file + replace) for data integrity

**Example:**
```python
from src.assembled_core.data.factor_store import store_factors_parquet

df_factors = add_all_features(prices)  # DataFrame mit timestamp, symbol, <features>
panel_dir = store_factors_parquet(
    df=df_factors,
    group="core_ta",
    freq="1d",
    universe="universe_watchlist",
    mode="append",
    metadata={"builder_fn": "add_all_features", "ma_windows": [20, 50, 200]}
)
```

#### `load_factors_parquet(group, freq, universe, start=None, end=None, root=None) -> pd.DataFrame | None`

Laedt Faktoren aus dem Factor Store mit Datums-Filterung (PIT-safe).

**Parameters:**
- `group`: Factor group name (z.B. "core_ta")
- `freq`: Frequency string ("1d" oder "5min")
- `universe`: Universe key (aus `compute_universe_key()`)
- `start`: Optional start date filter (inclusive, UTC)
- `end`: Optional end date filter (inclusive, UTC)
- `root`: Optional root directory (default: `get_factor_store_root()`)

**Returns:**
- DataFrame mit Faktoren (columns: timestamp, date, symbol, <feature_columns>)
- oder `None` wenn Faktoren nicht gefunden

**Notes:**
- PIT Safety: Filters factors by date range (start/end)
- Combines all year partitions that overlap with date range
- Returns None if panel directory doesn't exist or is empty
- Ensures timestamp column is UTC-aware
- Sorts by timestamp, then symbol

**Example:**
```python
from src.assembled_core.data.factor_store import load_factors_parquet

factors = load_factors_parquet(
    group="core_ta",
    freq="1d",
    universe="universe_watchlist",
    start="2024-01-01",
    end="2024-12-31"
)
```

#### `list_factor_partitions(group=None, freq=None, root=None) -> list[dict]`

Listet verfuegbare Factor-Partitionen im Store.

**Parameters:**
- `group`: Optional filter by factor group
- `freq`: Optional filter by frequency
- `root`: Optional root directory (default: `get_factor_store_root()`)

**Returns:**
- List of dicts with keys:
  - `factor_group`, `freq`, `universe_key`
  - `date_range` (start, end)
  - `years` (list of available years)
  - `computed_at`, `config_hash` (if manifest exists)
  - `factor_columns` (if manifest exists)

**Notes:**
- Reads `_metadata.json` files if available
- Falls back to directory structure if metadata not found
- Scans `year=*.parquet` files to determine year coverage

**Example:**
```python
from src.assembled_core.data.factor_store import list_factor_partitions

panels = list_factor_partitions(group="core_ta", freq="1d")
for panel in panels:
    print(f"{panel['universe_key']}: {panel['years']}")
```

### Helper Functions

#### `get_factor_store_root(settings=None) -> Path`

Gibt das Root-Verzeichnis fuer den Factor Store zurueck.

**Returns:**
- Path zu Factor Store Root (`data/factors/`)

#### `compute_universe_key(symbols=None, universe_file=None) -> str`

Generiert deterministischen Universe Key aus Symbol-Liste.

**Parameters:**
- `symbols`: Optional list of symbols (z.B. ["AAPL", "MSFT", "GOOGL"])
- `universe_file`: Optional path to universe file (read symbols from file)

**Returns:**
- Universe key string (z.B. "universe_AAPL_GOOGL_MSFT_a1b2c3d4" oder "universe_e5f6g7h8")

**Notes:**
- For small universes (<=20 symbols): includes symbol names for readability
- For large universes: uses hash only
- Deterministic: same symbols -> same key

---

## Data Contracts

### Factor DataFrame Contract

**Required Columns:**
- `timestamp`: `pd.Timestamp` (UTC, tz-aware) - Zeitstempel
- `symbol`: `string` - Ticker-Symbol

**Optional Columns:**
- `date`: `string` (YYYY-MM-DD) - Redundant but useful for filtering
- `<feature_columns>`: Feature-Spalten (z.B. `log_return`, `ma_20`, `rsi_14`, etc.)

**Sortierung:**
- **Primaer:** `timestamp` (aufsteigend)
- **Sekundaer:** `symbol` (aufsteigend)

**TZ-Policy:**
- **Intern:** Alle Timestamps sind **UTC** (timezone-aware)
- **Extern:** Bei Ingestion werden Zeitzonen auf UTC normalisiert
- **Ausgabe:** Alle Dateien enthalten UTC-Timestamps

**Deduplication:**
- Bei `mode="append"`: Duplikate werden entfernt basierend auf `(timestamp, symbol)` mit `keep="last"`

**Determinismusregeln:**
- Sortierung: immer `(timestamp, symbol)` aufsteigend
- UTC-Policy: alle Timestamps UTC-normalisiert
- Dedupe: `(symbol, timestamp)` mit `keep="last"`

---

## Error Handling

### FileNotFoundError

Wird ausgeloest wenn:
- Partition-Datei nicht gefunden (bei `load_factors_parquet` mit `strict=True`)
- Universe file nicht gefunden (bei `compute_universe_key`)

### ValueError

Wird ausgeloest wenn:
- Required columns fehlen (z.B. `timestamp`, `symbol`)
- `mode` ungültig ist (nicht "replace" oder "append")
- `symbols` oder `universe_file` nicht bereitgestellt (bei `compute_universe_key`)

### OSError

Wird ausgeloest wenn:
- Atomic write fehlschlägt (temp file + replace)
- Parquet-Datei nicht gelesen werden kann

---

## Integration

### EOD Pipeline (`run_daily.py`)

```bash
# Mit Factor Store (aktiviert Caching)
python scripts/run_daily.py --date 2024-12-31 --use-factor-store

# Mit custom Factor Store Root
python scripts/run_daily.py --date 2024-12-31 --use-factor-store \
  --factor-store-root /custom/path/to/factors

# Mit custom Factor Group
python scripts/run_daily.py --date 2024-12-31 --use-factor-store \
  --factor-group core_ta
```

### Backtest (`run_backtest_strategy.py`)

```bash
# Mit Factor Store
python scripts/run_backtest_strategy.py \
  --freq 1d \
  --use-factor-store \
  --start-date 2020-01-01 \
  --end-date 2024-12-31
```

### Python API

```python
from src.assembled_core.data.factor_store import (
    factor_partition_path,
    store_factors_parquet,
    load_factors_parquet,
    list_factor_partitions,
    compute_universe_key,
)
from src.assembled_core.features.ta_features import add_all_features

# Compute factors
prices = load_prices(...)  # DataFrame mit timestamp, symbol, close, ...
factors = add_all_features(prices)

# Store factors
universe_key = compute_universe_key(symbols=prices["symbol"].unique().tolist())
store_factors_parquet(
    df=factors,
    group="core_ta",
    freq="1d",
    universe=universe_key,
    mode="append"
)

# Load factors
factors_loaded = load_factors_parquet(
    group="core_ta",
    freq="1d",
    universe=universe_key,
    start="2024-01-01",
    end="2024-12-31"
)
```

---

## Layering

**Layer:** `data/` (Bottom Layer)

**Imports:**
- Nur `pandas`, `pathlib`, `tempfile`, `hashlib`, `json`, `logging`
- `src.assembled_core.config.settings` (fuer `get_factor_store_root()`)

**Exports:**
- Functions fuer loading/storing factors
- Keine pipeline/qa imports (layering-konform)

---

## Incremental Updates (Sprint 5 / F3)

### Semantics

Der Factor Store unterstuetzt **incremental updates** via `mode="append"`:

1. **Append Mode**: Laedt existing partition, merged mit neuen Daten
2. **Deduplication**: `(symbol, timestamp)` mit `keep="last"` (konsistent mit Snapshot-ID + Panel Store)
3. **Sortierung**: `(timestamp, symbol)` aufsteigend (deterministisch)
4. **Atomic Rewrite**: Temp file + rename (Datenintegritaet)

### Use Case: Daily EOD Pipeline

Statt full recompute aller historischen Features:

```python
from src.assembled_core.features.incremental_updates import compute_only_last_session
from src.assembled_core.data.factor_store import store_factors_parquet
from src.assembled_core.features.ta_features import add_all_features

# Load prices (full history)
prices = load_prices(...)

# Compute only last session
factors_last_session = compute_only_last_session(
    prices=prices,
    builder_fn=add_all_features,
    as_of=target_date,
)

# Append to store (deduplicates automatically)
store_factors_parquet(
    df=factors_last_session,
    group="core_ta",
    freq="1d",
    universe=universe_key,
    mode="append",  # Incremental update
)
```

### Append-Day Equals Recompute-Last-Day

**Guarantee**: Append-day Ergebnis ist aequivalent zur Full-Recompute-Version (fuer den letzten Tag).

**Test**: 
1. Full recompute factors for synthetic panel (D1..D10)
2. Split data: store base (D1..D9), then append last day (D10)
3. Load final partition und vergleiche mit full recompute (identische rows/values)

**Overlap Handling**: Wenn last day bereits existiert -> overwrite via `keep="last"` deterministisch.

### Implementation Details

**Append Mode Logic** (in `store_factors()`):
1. Load existing partition (if exists)
2. Concat existing + new data
3. Deduplicate: `drop_duplicates(subset=["timestamp", "symbol"], keep="last")`
4. Sort: `sort_values(["timestamp", "symbol"])`
5. Atomic rewrite: `_write_parquet_atomic()` (temp file + rename)

**Consistency Rules**:
- Same deduplication rule as Snapshot-ID: `(symbol, timestamp)` with `keep="last"`
- Same deduplication rule as Panel Store: `(symbol, timestamp)` with `keep="last"`
- Deterministic sorting: always `(timestamp, symbol)` ascending

---

## Preferred Path & Fallback Behavior (Sprint 5 / F4)

### Daily Pipeline (`run_daily.py`)

**Preferred Path:**
1. Load Price Panel (from panel_store or provider)
2. Compute features for last session (incremental update)
3. Store factors in Factor Store (`mode="append"`)

**Implementation:**
- Features are computed for last session only (incremental)
- Stored via `store_factors_parquet(..., mode="append")`
- Factor Store is updated incrementally (no full recompute)

**Usage:**
```bash
# Daily run with factor store (default: enabled if --use-factor-store)
python scripts/run_daily.py --date 2024-12-31 --use-factor-store
```

### Backtest (`run_backtest_strategy.py`)

**Preferred Path:**
1. Try to load factors from Factor Store (`load_factors_parquet`)
2. If found: use loaded factors (no computation)
3. If NOT found: fallback to feature computation (local-only, no external fetches)

**Fallback Behavior:**
- If Factor Store miss: compute features directly (local-only)
- Hard Gate: No external fetches in backtest mode (D3 Hard Gate remains intact)
- Features computed from local price data only

**Usage:**
```bash
# Backtest with factor store (preferred, but fallback if not available)
python scripts/run_backtest_strategy.py --freq 1d --use-factor-store

# Backtest without factor store (fallback to direct computation, local-only)
python scripts/run_backtest_strategy.py --freq 1d
```

### Pre-generating Factors

**For Backtests:**
To ensure Factor Store is available for backtests, pre-generate factors:

```python
from src.assembled_core.data.factor_store import (
    compute_universe_key,
    store_factors_parquet,
)
from src.assembled_core.features.ta_features import add_all_features

# Load prices (local-only)
prices = load_prices(...)

# Compute features
factors = add_all_features(prices)

# Store factors
universe_key = compute_universe_key(symbols=prices["symbol"].unique().tolist())
store_factors_parquet(
    df=factors,
    group="core_ta",
    freq="1d",
    universe=universe_key,
    mode="replace",  # Full recompute for initial build
)
```

**Or use Daily Pipeline:**
The daily pipeline automatically stores factors incrementally:
```bash
# Run daily pipeline (stores factors automatically)
python scripts/run_daily.py --date 2024-12-31 --use-factor-store
```

### Integration Notes

**Daily:**
- Factor Store is written to (incremental append)
- Features computed for last session only
- Compatible with incremental update semantics

**Backtest:**
- Factor Store is read from (preferred)
- Fallback to direct computation if not available
- Hard Gate: No external fetches (local-only)

**Hard Gate Compliance:**
- Backtests never fetch from external providers (Yahoo, Finnhub, etc.)
- Factor Store is local-only (no network calls)
- Fallback computation uses local price data only

---

## References

- Design: `docs/FACTOR_STORE_P2_DESIGN.md`
- Contracts: `docs/CONTRACTS.md`
- Panel Store: `src/assembled_core/data/panel_store.py` (aehnliche Struktur)
- Factor Store Integration: `src/assembled_core/features/factor_store_integration.py`
- Incremental Updates: `src/assembled_core/features/incremental_updates.py`