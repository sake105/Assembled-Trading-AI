# Security Master Lite: Symbol Metadata Mapping

## Purpose

Security Master Lite provides deterministic mapping from trading symbols to metadata required for risk controls and exposure calculations.

**Use Cases:**
- Sector/Region exposure limits (Sprint 9)
- Currency exposure tracking
- Asset type filtering
- Risk aggregation by sector/region

**Layering:** Data layer (no dependencies on pipeline/qa/execution).

---

## Schema

### Required Columns

| Column      | Type   | Description                    |
|-------------|--------|--------------------------------|
| symbol      | str    | Trading symbol (e.g., "AAPL") |
| sector      | str    | Sector classification          |
| region      | str    | Geographic region              |
| currency    | str    | Trading currency (e.g., "USD") |
| asset_type  | str    | Asset type (e.g., "EQUITY")   |

### Optional Columns

| Column    | Type | Description              |
|-----------|------|--------------------------|
| exchange  | str  | Exchange code            |
| timezone  | str  | Timezone (e.g., "America/New_York") |
| country   | str  | Country code             |
| industry  | str  | Industry classification  |

---

## Storage Location

**Default Path:** `data/security_master/security_master.parquet`

**Format Conventions:**
- Primary format: `.parquet` (recommended for performance)
- Alternative formats: `.csv`, `.json` (supported for compatibility)
- File naming: `security_master.{ext}`

---

## Missing-Handling Policy

**Default Policy: `raise` (fail-fast)**

When resolving metadata for symbols:
- If symbol is missing from security master: raise `ValueError` with list of missing symbols
- This ensures data quality and prevents silent failures

**Alternative Policy: `default`**

If `missing_policy="default"` is used:
- Missing symbols get default values: `UNKNOWN` for all required fields
- Use with caution: may hide data quality issues

---

## Examples

### Example Security Master Data

```
symbol | sector      | region | currency | asset_type | exchange
-------|-------------|--------|----------|------------|----------
AAPL   | Technology  | US     | USD      | EQUITY     | NASDAQ
MSFT   | Technology  | US     | USD      | EQUITY     | NASDAQ
GOOGL  | Technology  | US     | USD      | EQUITY     | NASDAQ
TSLA   | Consumer    | US     | USD      | EQUITY     | NASDAQ
```

### Usage Example

```python
from pathlib import Path
from src.assembled_core.data.security_master import (
    load_security_master,
    resolve_security_meta,
    get_default_security_master_path,
)

# Load security master
master_path = get_default_security_master_path()
master_df = load_security_master(master_path)

# Resolve metadata for symbols
symbols = ["AAPL", "MSFT", "GOOGL"]
meta_df = resolve_security_meta(symbols, master_df)

# Result: DataFrame with sector, region, currency, asset_type for each symbol
```

### Error Handling

```python
# Missing symbols (default: raise ValueError)
try:
    meta_df = resolve_security_meta(["AAPL", "INVALID"], master_df)
except ValueError as e:
    print(f"Missing symbols: {e}")

# Use default policy for missing symbols
meta_df = resolve_security_meta(
    ["AAPL", "INVALID"],
    master_df,
    missing_policy="default",
)
```

---

## API Reference

### `load_security_master(path: Path) -> pd.DataFrame`

Load security master from file.

**Parameters:**
- `path`: Path to security master file (.parquet, .csv, or .json)

**Returns:**
- DataFrame with required columns, sorted by symbol (ascending)

**Raises:**
- `FileNotFoundError`: If file does not exist
- `ValueError`: If required columns are missing or format is unsupported

**Behavior:**
- Normalizes strings (strip whitespace)
- Deterministic sorting (symbol ascending)
- Validates required columns

### `store_security_master(df: pd.DataFrame, path: Path) -> None`

Store security master to file (atomic write).

**Parameters:**
- `df`: DataFrame with required columns
- `path`: Path to output file

**Raises:**
- `ValueError`: If required columns are missing or format is unsupported
- `OSError`: If write fails

**Behavior:**
- Atomic write (temp file -> rename, no partial files)
- Creates parent directory if needed
- Deterministic sorting (symbol ascending)

### `resolve_security_meta(symbols: list[str], master_df: pd.DataFrame, *, missing_policy: Literal["raise", "default"] = "raise", ...) -> pd.DataFrame`

Resolve security metadata for given symbols.

**Parameters:**
- `symbols`: List of symbols to resolve
- `master_df`: Security master DataFrame
- `missing_policy`: How to handle missing symbols ("raise" or "default")
- `default_*`: Default values for missing symbols (if `missing_policy="default"`)

**Returns:**
- DataFrame with metadata for requested symbols, sorted by symbol (ascending)

**Raises:**
- `ValueError`: If `missing_policy="raise"` and symbols are missing

**Behavior:**
- Normalizes input symbols (strip, convert to string)
- Deterministic sorting (symbol ascending)
- Fail-fast by default (raises on missing symbols)

### `get_default_security_master_path() -> Path`

Get default path for security master file.

**Returns:**
- Path to `data/security_master/security_master.parquet`

---

## Determinism

All functions are deterministic:
- Sorting: Always by symbol (ascending)
- String normalization: Strip whitespace
- No random order or non-deterministic behavior

---

## File Format Support

### Parquet (Recommended)

- Fast read/write
- Preserves data types
- Compressed

### CSV

- Human-readable
- Easy to edit
- Compatible with Excel

### JSON

- Human-readable
- Easy to parse
- Compatible with web APIs

---

## Integration Points

**Current Usage:**
- Not yet integrated (Sprint 9)

**Future Integration:**
- Risk controls: Sector/Region exposure limits
- Exposure engine: Aggregate by sector/region
- Portfolio reporting: Sector/Region breakdowns

---

## Data Quality

**Validation:**
- Required columns must be present
- Symbol column must not contain NaN
- Strings are normalized (strip)

**Fail-Fast:**
- Missing symbols raise `ValueError` by default
- Clear error messages with list of missing symbols

---

## Future Enhancements (Out of Scope)

- Real-time updates from external sources
- Versioning/history
- Multi-source aggregation
- Complex hierarchies (sector -> industry -> sub-industry)
