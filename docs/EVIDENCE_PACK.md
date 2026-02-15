# Evidence Pack Exporter Design

## Purpose

The Evidence Pack Exporter creates a deterministic, portable ZIP archive containing all accounting-related artifacts for a given run and date. This enables:

- **Audit trails**: Complete snapshot of all accounting evidence
- **Portability**: Single ZIP file can be moved/shared without path dependencies
- **Reproducibility**: Deterministic ZIPs enable byte-level verification
- **Ops workflows**: Easy export for compliance, support, or archival

## Overview

The exporter reads the Evidence Index JSON to discover all relevant files, then packages them into a ZIP archive with:

- All referenced files (snapshot, ledger, reconcile, accounting reports, manifest)
- Deterministic file ordering and timestamps
- SHA256 checksums and file manifest
- Portable POSIX paths inside ZIP

## Input Sources

### Source -> required/optional keys (single source of truth)

Required and optional path keys are defined in `evidence_pack.REQUIRED_KEYS_BY_SOURCE` and `OPTIONAL_KEYS_BY_SOURCE`. Pack manifest fields `required_missing` and `optional_missing` contain **keys** (not paths).

| Source            | Required keys                                                                 | Optional keys                                                                                          |
|-------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `evidence_index`  | `ledger_pack_path`                                                            | `broker_snapshot_path`, `reconcile_report_path`, `accounting_report_path`, `manifest_path`             |
| `manifest`        | `ledger_pack_path`, `reconcile_report_path`, `accounting_report_path`         | `broker_snapshot_path`, `evidence_index_path`                                                          |

- **Evidence Index:** Only the ledger pack is required; missing reconcile/accounting are reported in `optional_missing`, pack is still created (unless `strict=True`).
- **Manifest fallback:** Ledger, reconcile and accounting are required; missing any of them raises `ValueError` and no pack is created.

### Primary: Evidence Index JSON

**Location:** `output/evidence_<run_id>/evidence_<YYYY-MM-DD>.json`

**Schema:**
```json
{
  "schema_version": 1,
  "run_id": "ledger_eod_1d",
  "as_of_date": "2025-01-15T00:00:00+00:00",
  "paths": {
    "broker_snapshot_path": "broker_snapshot_ops_20250115/snapshot_2025-01-15.json",
    "ledger_pack_path": "ledger_ledger_eod_1d/ledger_events.parquet",
    "reconcile_report_path": "reconcile_report_ledger_eod_1d/reconcile_2025-01-15.json",
    "accounting_report_path": "accounting_report_ledger_eod_1d/accounting_2025-01-15.json",
    "manifest_path": "run_manifest_1d.json"
  },
  "broker_meta": {...},
  "reconciliation_ok": true
}
```

**Path Resolution:**
- All paths in `paths` are relative to `output_dir`
- Paths may be `None` (optional files)
- Paths use POSIX slashes (`/`) for portability

**Manifest via Evidence Index:**
- When `paths.manifest_path` is set and the referenced file exists, the Evidence Pack exporter:
  - Treats `manifest_path` as an **optional** file for the `evidence_index` source.
  - Includes the manifest JSON (`run_manifest_<freq>.json`) directly from the Evidence Index (no need to fall back to manifest discovery).
  - Marks the corresponding entry in the pack manifest with `source_type: "manifest"`.
- If `paths.manifest_path` is missing or `None`, the exporter can still fall back to the orchestrator manifest discovery logic (manifest fallback source) when no usable Evidence Index is available.

**Note:** The current implementation only supports Evidence Index as input source. Manifest fallback is not implemented in the initial version.

## Output Structure

### ZIP Archive

**Location:** `output/evidence_<run_id>/pack_<YYYY-MM-DD>.zip`

**Contents:**
- All files referenced in Evidence Index (or Manifest)
- Pack manifest JSON (see below)
- Evidence Index JSON itself (if used as source)

**Internal ZIP Structure:**
```
pack_2025-01-15.zip
├── evidence_2025-01-15.json          # Evidence Index (if used as source)
├── broker_snapshot_ops_20250115/
│   └── snapshot_2025-01-15.json
├── ledger_ledger_eod_1d/
│   └── ledger_events.parquet
├── reconcile_report_ledger_eod_1d/
│   ├── reconcile_2025-01-15.json
│   ├── reconcile_2025-01-15.csv
│   └── reconcile_2025-01-15.md
├── accounting_report_ledger_eod_1d/
│   ├── accounting_2025-01-15.json
│   └── accounting_2025-01-15.csv
└── run_manifest_1d.json
```

**Path Rules:**
- All paths inside ZIP use POSIX slashes (`/`)
- Paths are relative to `output_dir` (no absolute paths)
- Directory structure preserved (e.g., `ledger_<run_id>/ledger_events.parquet`)

### Pack Manifest JSON

**Location:** Inside ZIP as `pack_manifest_<YYYY-MM-DD>.json` (also written to `output/evidence_<run_id>/pack_manifest_<YYYY-MM-DD>.json`)

**Schema:**
```json
{
  "schema_version": 1,
  "run_id": "ledger_eod_1d",
  "as_of_date": "2025-01-15T00:00:00+00:00",
  "pack_created_utc": "2025-01-15T10:30:00+00:00",
  "source": "evidence_index",
  "source_path": "evidence_ledger_eod_1d/evidence_2025-01-15.json",
  "files": [
    {
      "path": "evidence_2025-01-15.json",
      "size_bytes": 1234,
      "sha256": "abc123...",
      "source_type": "evidence_index"
    },
    {
      "path": "broker_snapshot_ops_20250115/snapshot_2025-01-15.json",
      "size_bytes": 5678,
      "sha256": "def456...",
      "source_type": "broker_snapshot"
    },
    {
      "path": "ledger_ledger_eod_1d/ledger_events.parquet",
      "size_bytes": 9012,
      "sha256": "ghi789...",
      "source_type": "ledger_pack"
    }
  ],
  "tool_version": "0.1.0"
}
```

**Fields:**
- `schema_version`: Schema version (currently `1`)
- `run_id`: Run identifier
- `as_of_date`: Report date (ISO 8601 UTC)
- `pack_created_utc`: Pack creation timestamp (ISO 8601 UTC) - **Note**: For byte-stable packs, this should be fixed or omitted
- `source`: Source type (`"evidence_index"` or `"manifest"`)
- `source_path`: Relative path to source file
- `files`: Array of file entries with:
  - `path`: Relative POSIX path inside ZIP
  - `size_bytes`: File size in bytes
  - `sha256`: SHA256 hash (hex string, lowercase)
  - `source_type`: Type hint (`"evidence_index"`, `"broker_snapshot"`, `"ledger_pack"`, `"reconcile_report"`, `"accounting_report"`, `"manifest"`, `"pack_manifest"`)
- `tool_version`: Version of exporter tool

### Pack manifest schema (v1)

Top-level keys (all ASCII). Single source of truth for manifest contract.

| Key | Type | Description |
|-----|------|-------------|
| `schema_version` | int | Always `1` |
| `run_id` | string | Run identifier |
| `as_of_date` | string | Report date ISO 8601 UTC |
| `source` | string | `"evidence_index"` or `"manifest"` |
| `source_path` | string \| null | Relative POSIX path to source JSON in ZIP |
| `files` | array | File entries (see below) |
| `required_missing` | array of string | Keys of required paths that were missing |
| `optional_missing` | array of string | Keys of optional paths that were missing |
| `required_present_count` | int | Number of required paths included |
| `optional_present_count` | int | Number of optional paths included |
| `required_missing_count` | int | Length of `required_missing` |
| `optional_missing_count` | int | Length of `optional_missing` |
| `zip_compression` | string | ZIP compression (e.g. `"zip_deflated"`) |
| `required_keys` | array of string | Keys considered required for this source (lexicographic sort) |
| `optional_keys` | array of string | Keys considered optional for this source (lexicographic sort) |
| `zip_entries` | array of string | Sorted list of all ZIP entry paths (POSIX) |
| `zip_entries_count` | int | Must equal `len(zip_entries)`; `pack_manifest_*.json` must be in `zip_entries` |
| `files_count` | int | Must equal `len(files)` |
| `tool_version` | string | Exporter version |

**files[] entry keys (each element):**

| Key | Type | Description |
|-----|------|-------------|
| `path` | string | Relative POSIX path inside ZIP |
| `sha256` | string \| null | SHA256 hex lowercase |
| `size_bytes` | int | File size in bytes |
| `source_type` | string | One of: `evidence_index`, `broker_snapshot`, `ledger_pack`, `reconcile_report`, `accounting_report`, `manifest`, `pack_manifest`, `other` |

### Source semantics

- `source` is either `evidence_index` or `manifest` (which input was used to build the pack).
- `source_path` is the relative POSIX path to the source file inside the ZIP (e.g. the evidence index JSON or the run manifest JSON).
- Export and Verify JSON outputs both include `source` and `source_path` for logs and automation.
- The pack manifest inside the ZIP is the single source of truth for these values.

## Verify Evidence Pack (offline)

The script `scripts/verify_evidence_pack.py` validates an Evidence Pack ZIP offline (manifest present, schema ok, checksums, no illegal paths). Exit codes: 0 = ok, 1 = fail or error. ASCII-only output. Option `--fail-on-warn` exits 1 if any of bad_paths_count, missing_entries_count, paths_not_in_zip_entries_count, or checksum_mismatches_count is greater than zero (even when ok is true).

### verify_evidence_pack --json output schema

With `--json`, the CLI prints a single JSON object to stdout. Schema version: **1**. Keys are stable for automation and parsing.

| Key | Type | Description |
|-----|------|-------------|
| `schema_version` | int | JSON schema version (currently `1`) |
| `zip_path` | string | Path to the ZIP as provided on CLI |
| `ok` | boolean | `true` if verification passed |
| `error_code` | string | `""` when ok; otherwise one of `MISSING_MANIFEST`, `BAD_PATHS`, `CHECKSUM_MISMATCH`, `UNSUPPORTED_SCHEMA`, `FILE_NOT_FOUND`, `UNEXPECTED_ERROR` |
| `missing_manifest` | boolean | `true` if no pack_manifest_*.json in ZIP root |
| `n_files` | int | Number of ZIP entries (0 if unknown) |
| `bad_paths_count` | int | Number of illegal paths |
| `missing_entries_count` | int | Number of manifest files[] paths missing from ZIP |
| `paths_not_in_zip_entries_count` | int | Number of files[] paths not listed in zip_entries |
| `checksum_mismatches_count` | int | Number of checksum mismatches |
| `details` | object | Optional debug: `bad_paths`, `missing_entries`, `paths_not_in_zip_entries`, `checksum_mismatches` (up to 20 each); empty when ok |

- **error_code priority:** When multiple issues exist, a single code is chosen in order: `MISSING_MANIFEST` > `BAD_PATHS` > `CHECKSUM_MISMATCH`. Details lists are capped at 20 entries each.

Output is deterministic: `sort_keys=True`, `indent=2`, trailing newline. Two runs on the same ZIP produce identical bytes.

**Example (ok):**
```json
{
  "bad_paths_count": 0,
  "checksum_mismatches_count": 0,
  "details": {},
  "error_code": "",
  "missing_manifest": false,
  "n_files": 5,
  "ok": true,
  "schema_version": 1,
  "zip_path": "output/evidence_run/pack_2025-01-15.zip"
}
```

## Determinism Rules

### File Ordering

**Rule:** Files are added to ZIP in lexicographic order by path (POSIX, case-sensitive).

**Implementation:**
```python
# Collect all file paths
file_paths = sorted(all_paths, key=lambda p: p.as_posix())
```

**Rationale:**
- Consistent ordering across runs
- Reproducible ZIP structure
- Easy to verify/diff

### ZIP Timestamps

**Rule:** All files in ZIP use fixed timestamp (e.g., `2025-01-15 00:00:00 UTC`).

**Implementation:**
```python
import zipfile
from datetime import datetime

# Fixed timestamp for all files
FIXED_TIMESTAMP = (2025, 1, 15, 0, 0, 0)  # YYYY, MM, DD, HH, MM, SS

# Set timestamp when adding file
zip_info = zipfile.ZipInfo(filename=path_in_zip)
zip_info.date_time = FIXED_TIMESTAMP
```

**Rationale:**
- ZIP file timestamps affect ZIP file hash
- Fixed timestamps enable byte-stable ZIPs
- Timestamp should match `as_of_date` (or be fixed constant)

**Note:** If `pack_created_utc` is included in manifest, it should be fixed for byte-stability (or omitted).

### Path Normalization

**Rule:** All paths inside ZIP use POSIX slashes (`/`), relative to `output_dir`.

**Implementation:**
```python
# Normalize path to POSIX, relative to output_dir
def normalize_zip_path(file_path: Path, output_dir: Path) -> str:
    rel = file_path.relative_to(output_dir)
    return rel.as_posix()  # Always POSIX slashes
```

**Rationale:**
- Portable across Windows/Linux/Mac
- Consistent ZIP structure
- No OS-specific path separators

### Strict determinism checklist (Ops modules)

- **JSON:** Where output must be byte-deterministic: always `sort_keys=True`, `indent=2`, trailing newline (`+ "\n"` or `write("\n")` after `json.dump`).
- **Paths:** Relative + POSIX only: use `.as_posix()` for paths in manifests/returns; no backslashes, no absolute paths.
- **Evidence index / pack manifest / broker snapshot JSON:** All follow the above; return schemas are documented in docstrings as stable.

### Checksum Calculation

**Rule:** SHA256 hashes are calculated from file contents (before adding to ZIP).

**Implementation:**
```python
import hashlib

def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest().lower()
```

**Rationale:**
- Verify file integrity
- Detect corruption
- Enable content-based deduplication

## File Inclusion Rules

### Required Files

**Always Included (if present in source):**
- Evidence Index JSON (if used as source)
- Ledger Pack (`ledger_events.parquet`)
- Reconciliation Report JSON
- Accounting Report JSON
- Broker Snapshot JSON (if present)
- Manifest JSON (if present)

### Optional Files

**Included if Present:**
- Reconciliation Report CSV (if JSON exists, check for CSV in same directory)
- Reconciliation Report Markdown (if JSON exists, check for MD in same directory)
- Accounting Report CSV (if JSON exists, check for CSV in same directory)
- Broker Snapshot Parquet positions (if JSON exists, check for Parquet in same directory)

**Exclusion Rules:**
- Files outside `output_dir` are skipped (with warning)
- Missing files are skipped (with warning, not error)
- Only files explicitly referenced in Evidence Index/Manifest are included

### Directory Structure

**Rule:** Preserve relative directory structure inside ZIP.

**Example:**
- Source: `output/reconcile_report_ledger_eod_1d/reconcile_2025-01-15.json`
- ZIP path: `reconcile_report_ledger_eod_1d/reconcile_2025-01-15.json`

**Rationale:**
- Maintains organization
- Prevents filename collisions
- Matches original layout

## Error Handling

### Fail-Fast Errors

**Missing Evidence Index:**
- Error: `ValueError: No evidence files found for run_id=<id>, as_of_date=<date>. Evidence index may be missing or empty.`
- Action: Fail immediately

**Invalid Evidence Index Schema:**
- Error: `ValueError` or `KeyError` when reading evidence index
- Action: Fail immediately

**Output Directory Not Writable:**
- Error: `PermissionError: Cannot write to output directory: <path>`
- Action: Fail immediately

### Best-Effort Warnings

**Missing Referenced File:**
- Warning: `File not found: <path> (skipping)`
- Action: Continue, skip file, log warning

**File Outside Output Directory:**
- Warning: `File outside output_dir: <path> (skipping)`
- Action: Continue, skip file, log warning

**Checksum Calculation Failure:**
- Warning: `Failed to calculate checksum for <path>: <error>`
- Action: Continue, set `sha256: null` in manifest

**Evidence Index Write Failure:**
- Warning: `Failed to write pack manifest: <error>`
- Action: Continue, ZIP is still created (manifest write is best-effort)

## API Design

### Function Signature

```python
def build_evidence_pack(
    output_dir: Path | str,
    run_id: str,
    as_of_date: str | pd.Timestamp,
    *,
    include_optional: bool = True,
    fixed_timestamp: tuple[int, int, int, int, int, int] | None = None,
) -> dict[str, Any]:
    """Build evidence pack (ZIP + manifest) from Evidence Index.
    
    Args:
        output_dir: Base output directory
        run_id: Run identifier
        as_of_date: Report date (YYYY-MM-DD string or pd.Timestamp)
        include_optional: If True, include optional files (with warnings if missing).
                         If False, skip optional files silently.
        fixed_timestamp: Fixed ZIP timestamp (default: 1980-01-01 00:00:00)
        
    Returns:
        Dictionary with:
            - pack_path: Relative path to ZIP file
            - pack_manifest_path: Relative path to pack manifest JSON
            - n_files: Number of files included in pack
            - missing_optional: List of missing optional file paths
            - checksums: Dict mapping zip_entry_path to SHA256 hash
            - source: Source type ("evidence_index" or None)
    """
```

### Return Value

```python
{
    "pack_path": "evidence_ledger_eod_1d/pack_2025-01-15.zip",
    "pack_manifest_path": "evidence_ledger_eod_1d/pack_manifest_2025-01-15.json",
    "n_files": 5,
    "missing_optional": [
        "broker_snapshot_ops_20250115/snapshot_2025-01-15.json"
    ],
    "checksums": {
        "evidence_2025-01-15.json": "abc123...",
        "broker_snapshot_ops_20250115/snapshot_2025-01-15.json": "def456...",
        "ledger_ledger_eod_1d/ledger_events.parquet": "ghi789...",
        ...
    },
    "source": "evidence_index"
}
```

## CLI Usage

### Standalone CLI Tool

**Script:** `scripts/export_evidence_pack.py`

**Usage:**
```bash
# Export pack from evidence index
python scripts/export_evidence_pack.py \
  --run-id ledger_eod_1d \
  --as-of-date 2025-01-15 \
  --output-dir output

# Require evidence index (fail if missing)
python scripts/export_evidence_pack.py \
  --run-id ledger_eod_1d \
  --as-of-date 2025-01-15 \
  --require-evidence-index

# Fallback to manifest if evidence index missing
python scripts/export_evidence_pack.py \
  --run-id ledger_eod_1d \
  --as-of-date 2025-01-15 \
  --fallback-to-manifest \
  --manifest-path output/run_manifest_1d.json

# Custom fixed timestamp for byte-stability
python scripts/export_evidence_pack.py \
  --run-id ledger_eod_1d \
  --as-of-date 2025-01-15 \
  --fixed-timestamp "2025-01-15 00:00:00"
```

**Arguments:**
- `--run-id <id>`: Run identifier (required)
- `--as-of-date YYYY-MM-DD`: Report date (required, strict validation)
- `--output-dir <path>`: Output directory (default: `output`)
- `--strict`: Fail if optional files are missing (default: warn and continue)
- `--no-optional`: Exclude optional files from pack (default: include optional files)
- `--print-pack-path`: Print only the resolved pack path (one line) to stdout; mutually exclusive with `--text`; useful for cmd/PowerShell pipes. Logs go to stderr only.

**Exit Codes:**
- `0`: Success
- `1`: Error (missing required files, invalid schema, etc.)

## Examples

### Example 1: Export from Evidence Index

**Input:**
- Evidence Index: `output/evidence_ledger_eod_1d/evidence_2025-01-15.json`
- Contains paths to: snapshot, ledger, reconcile, accounting reports

**Output:**
- ZIP: `output/evidence_ledger_eod_1d/pack_2025-01-15.zip`
- Manifest: `output/evidence_ledger_eod_1d/pack_manifest_2025-01-15.json`

**ZIP Contents:**
```
pack_2025-01-15.zip
├── evidence_2025-01-15.json
├── broker_snapshot_ops_20250115/snapshot_2025-01-15.json
├── ledger_ledger_eod_1d/ledger_events.parquet
├── reconcile_report_ledger_eod_1d/reconcile_2025-01-15.json
├── reconcile_report_ledger_eod_1d/reconcile_2025-01-15.csv
├── accounting_report_ledger_eod_1d/accounting_2025-01-15.json
└── accounting_report_ledger_eod_1d/accounting_2025-01-15.csv
```

### Example 2: Missing Optional Files

**Input:**
- Evidence Index references: snapshot, ledger, reconcile, accounting
- Missing: Broker snapshot file (not found)

**Output:**
- ZIP: Created successfully
- Warnings: `Optional file not found: broker_snapshot_ops_20250115/snapshot_2025-01-15.json`
- Pack manifest: Lists all included files with checksums
- Return value: `missing_optional` contains list of missing files

**With --strict:**
- Exit code: 1 (failure)
- Error message: `Missing optional files (--strict mode): [...]`

**With --no-optional:**
- ZIP: Created without optional files
- Missing files are silently excluded (no warnings)

## Implementation Notes

### Dependencies

**Standard Library Only:**
- `zipfile`: ZIP creation
- `hashlib`: SHA256 checksums
- `pathlib`: Path handling
- `json`: JSON serialization
- `logging`: Logging
- `shutil`: Atomic file operations
- `tempfile`: Temporary file creation

**Required:**
- `pandas`: For timestamp handling and date normalization

**No External Dependencies:**
- No `pyarrow`, `fastparquet`, etc. (read-only, no writing)

### Atomic Writes

**ZIP File:**
- Write to temp file: `pack_<date>.zip.tmp`
- Calculate checksum
- Atomic rename: `os.replace(temp_path, final_path)` (Windows-safe)

**Pack Manifest:**
- Write to temp file: `pack_manifest_<date>.json.tmp`
- Atomic rename: `os.replace(temp_path, final_path)` (Windows-safe)

### Deterministic JSON

**Pack Manifest JSON:**
- `sort_keys=True`
- `indent=2`
- Trailing newline
- Consistent field order

**Rationale:**
- Byte-stable JSON
- Reproducible packs
- Easy to diff/verify

## Testing Requirements

### Unit Tests

1. **Evidence Pack Creation:**
   - Create dummy artifacts and evidence index
   - Build evidence pack
   - Verify ZIP and manifest exist
   - Verify ZIP contains expected files

2. **Missing Optional Files:**
   - Evidence index references missing optional files
   - Verify pack created successfully
   - Verify `missing_optional` in return value
   - Verify warnings logged

3. **Determinism:**
   - Create pack twice with same inputs
   - Verify pack manifest bytes are identical
   - Verify ZIP namelist is identical
   - Verify checksums match

4. **ZIP Sorting:**
   - Create files with non-alphabetical names
   - Verify ZIP entries are sorted lexicographically

5. **CLI Tests:**
   - Subprocess tests with `sys.executable`
   - Verify exit codes (0 success, 1 error)
   - Verify ASCII-only error messages
   - Verify `--strict` mode behavior

## Future Enhancements

### Optional Features (Not in Initial Implementation)

1. **Compression Level:**
   - Configurable ZIP compression (default: `zipfile.ZIP_DEFLATED`)
   - Trade-off: Size vs. speed

2. **Filtering:**
   - Include/exclude patterns (e.g., `--include "*.json"`)
   - File size limits

3. **Incremental Packs:**
   - Only include changed files
   - Delta packs

4. **Pack Verification:**
   - CLI tool to verify pack integrity
   - Compare checksums

5. **Pack Extraction:**
   - CLI tool to extract pack
   - Restore directory structure

## Summary

The Evidence Pack Exporter provides a deterministic, portable way to package all accounting artifacts for a run/date. Key features:

- **Deterministic**: Fixed timestamps, sorted file order, stable JSON
- **Portable**: POSIX paths, relative to output_dir
- **Verifiable**: SHA256 checksums, pack manifest
- **Robust**: Best-effort for optional files, fail-fast for required errors
- **Ops-Ready**: ASCII-only errors, clear warnings, CLI tool

The exporter enables audit trails, compliance workflows, and easy artifact sharing without path dependencies.
