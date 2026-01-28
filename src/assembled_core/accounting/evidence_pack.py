"""Evidence pack exporter for accounting artifacts.

This module provides functions to create deterministic ZIP archives containing
all accounting-related artifacts for a given run and date.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.assembled_core import __version__ as CORE_VERSION

logger = logging.getLogger(__name__)

# Fixed timestamp for ZIP entries (1980-01-01 00:00:00 UTC)
# This ensures byte-stable ZIP files
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

# Required and optional path keys per source (single source of truth for Ops/CI).
# Evidence Index: only ledger pack is required; reconcile/accounting are optional.
# Manifest fallback: ledger + reconcile + accounting are required.
REQUIRED_KEYS_BY_SOURCE: dict[str, list[str]] = {
    "evidence_index": ["ledger_pack_path"],
    "manifest": ["ledger_pack_path", "reconcile_report_path", "accounting_report_path"],
}
OPTIONAL_KEYS_BY_SOURCE: dict[str, list[str]] = {
    "evidence_index": ["broker_snapshot_path", "reconcile_report_path", "accounting_report_path", "manifest_path"],
    "manifest": ["broker_snapshot_path", "evidence_index_path"],
}

def _ascii_only(msg: str) -> str:
    """Return msg stripped to ASCII-only (lossy)."""
    return msg.encode("ascii", errors="ignore").decode("ascii")


def _sha256_file(path: Path) -> str:
    """Calculate SHA256 hash of a file.
    
    Args:
        path: Path to file
        
    Returns:
        SHA256 hash as lowercase hex string
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest().lower()


def _sha256_bytes(data: bytes) -> str:
    """Calculate SHA256 hash of in-memory bytes."""
    sha256 = hashlib.sha256()
    sha256.update(data)
    return sha256.hexdigest().lower()


def _normalize_zip_path(file_path: Path, base_dir: Path) -> str:
    """Normalize file path to POSIX relative path for ZIP entry.
    
    Args:
        file_path: Absolute or relative file path
        base_dir: Base directory (output_dir)
        
    Returns:
        POSIX-style relative path (e.g., "ledger_run1/ledger_events.parquet")
        
    Raises:
        ValueError: If file_path is outside base_dir or contains '..' segments
    """
    try:
        rel = file_path.relative_to(base_dir)
        posix_path = rel.as_posix()
        
        # Ensure no '..' segments and no absolute paths
        if ".." in posix_path or posix_path.startswith("/"):
            raise ValueError(f"Path contains '..' or is absolute: {posix_path}")
        
        # Ensure no backslashes (Windows paths)
        if "\\" in posix_path:
            raise ValueError(f"Path contains backslashes: {posix_path}")
        
        return posix_path
    except ValueError as e:
        # Re-raise if it's our validation error
        if "contains" in str(e) or "absolute" in str(e):
            raise
        # Otherwise, file is outside base_dir
        raise ValueError(f"File outside output_dir: {file_path}") from e


def _write_zip_deterministic(
    zip_path: Path,
    files: list[tuple[Path, str]],
    base_dir: Path,
    fixed_timestamp: tuple[int, int, int, int, int, int] = FIXED_ZIP_TIMESTAMP,
) -> None:
    """Write ZIP file with deterministic ordering and timestamps.
    
    Args:
        zip_path: Path to output ZIP file
        files: List of (file_path, zip_entry_path) tuples (already sorted)
        base_dir: Base directory for relative paths
        fixed_timestamp: Fixed timestamp for all ZIP entries (default: 1980-01-01)
    """
    # Files should already be sorted by caller, but ensure deterministic order
    sorted_files = sorted(files, key=lambda x: x[1])
    
    # Write to temp file first (Windows-safe atomic write)
    tmp_path = zip_path.with_suffix(".tmp.zip")
    
    # Fixed compression type for all entries
    COMPRESS_TYPE = zipfile.ZIP_DEFLATED
    
    try:
        with zipfile.ZipFile(tmp_path, "w", COMPRESS_TYPE) as zf:
            for file_path, zip_entry_path in sorted_files:
                if not file_path.exists():
                    logger.warning(f"File not found, skipping: {file_path}")
                    continue
                
                # Validate zip_entry_path is relative and POSIX
                if ".." in zip_entry_path or zip_entry_path.startswith("/"):
                    raise ValueError(f"Invalid ZIP entry path (contains '..' or absolute): {zip_entry_path}")
                if "\\" in zip_entry_path:
                    raise ValueError(f"Invalid ZIP entry path (contains backslashes): {zip_entry_path}")
                
                # Create ZipInfo with fixed timestamp and compression
                zip_info = zipfile.ZipInfo(filename=zip_entry_path)
                zip_info.date_time = fixed_timestamp
                zip_info.compress_type = COMPRESS_TYPE
                
                # Add file to ZIP
                with open(file_path, "rb") as f:
                    zf.writestr(zip_info, f.read())
        
        # Atomic rename (Windows-safe)
        shutil.move(str(tmp_path), str(zip_path))
    except Exception:
        # Clean up temp file on error
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _find_related_files(base_path: Path) -> list[Path]:
    """Find related files in the same directory (CSV, MD, Parquet variants).
    
    Args:
        base_path: Base file path (e.g., reconcile_2025-01-15.json)
        
    Returns:
        List of related files found (e.g., reconcile_2025-01-15.csv, reconcile_2025-01-15.md)
    """
    related = []
    base_name = base_path.stem
    parent_dir = base_path.parent
    
    # Check for CSV variant
    csv_path = parent_dir / f"{base_name}.csv"
    if csv_path.exists():
        related.append(csv_path)
    
    # Check for Markdown variant
    md_path = parent_dir / f"{base_name}.md"
    if md_path.exists():
        related.append(md_path)
    
    # Check for Parquet variant (for broker snapshots: positions_YYYY-MM-DD.parquet)
    # Extract date from filename (e.g., snapshot_2025-01-15.json -> 2025-01-15)
    if "snapshot_" in base_path.name:
        # Extract date part (YYYY-MM-DD)
        date_part = base_path.name.replace("snapshot_", "").replace(".json", "")
        if len(date_part) == 10 and date_part.count("-") == 2:  # YYYY-MM-DD format
            parquet_path = parent_dir / f"positions_{date_part}.parquet"
            if parquet_path.exists():
                related.append(parquet_path)
    
    return related


def _find_manifest_for_run(base_dir: Path, run_id: str) -> Path | None:
    """Find orchestrator manifest JSON for a run (fallback when no evidence index).
    
    Search order (deterministic):
    1. manifest_<run_id>.json (if present)
    2. lexicographically smallest run_manifest_*.json
    """
    # 1) manifest_<run_id>.json in base_dir
    direct_manifest = base_dir / f"manifest_{run_id}.json"
    if direct_manifest.exists():
        return direct_manifest
    
    # 2) Any run_manifest_*.json in base_dir, pick lexicographically smallest name
    candidates = sorted(base_dir.glob("run_manifest_*.json"), key=lambda p: p.name)
    if candidates:
        return candidates[0]
    
    return None


def collect_evidence_files(
    output_dir: Path | str,
    run_id: str,
    as_of_date: str | pd.Timestamp,
) -> dict[str, Any]:
    """Collect all evidence files referenced in Evidence Index or manifest.
    
    Args:
        output_dir: Base output directory
        run_id: Run identifier
        as_of_date: Report date (YYYY-MM-DD string or pd.Timestamp)
        
    Returns:
        Dictionary with:
            - files: List of (file_path, zip_entry_path) tuples
            - missing_required: List of missing required keys (not paths)
            - missing_optional: List of missing optional keys (not paths)
            - optional_zip_paths: List of zip_entry_paths that came from optional keys (for filtering when include_optional=False)
            - evidence_index_path: Path to evidence index JSON (if found)
            - manifest_path: Path to orchestrator manifest JSON (if used)
            - source: Source type ("evidence_index", "manifest", or None)
            - source_path: Relative POSIX path to source JSON (or None)
    """
    base_dir = Path(output_dir)
    
    # Normalize as_of_date
    if isinstance(as_of_date, str):
        as_of_ts = pd.to_datetime(as_of_date, utc=True)
    else:
        as_of_ts = as_of_date
    if as_of_ts.tz is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    date_str = as_of_ts.strftime("%Y-%m-%d")
    
    # Try Evidence Index first
    evidence_dir = base_dir / f"evidence_{run_id}"
    evidence_index_path = evidence_dir / f"evidence_{date_str}.json"
    evidence_index_used = False
    manifest_path: Path | None = None
    source: str | None = None
    source_path_rel: str | None = None
    
    paths: dict[str, Any] = {}
    files: list[tuple[Path, str]] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []
    optional_zip_paths: list[str] = []
    
    # First preference: Evidence Index JSON
    if evidence_index_path.exists():
        try:
            with open(evidence_index_path) as f:
                evidence = json.load(f)
            evidence_index_used = True
            paths = evidence.get("paths", {})
            source = "evidence_index"
            # Source path is the evidence index JSON itself
            source_path_rel = _normalize_zip_path(evidence_index_path, base_dir)
            # Add Evidence Index itself to files
            evidence_zip_path = source_path_rel
            files.append((evidence_index_path, evidence_zip_path))
        except Exception as exc:
            logger.error(f"Failed to read evidence index {evidence_index_path}: {exc}")
            # Fall through to manifest fallback
    
    # Fallback: orchestrator manifest if no usable evidence index
    if not evidence_index_used:
        manifest_path = _find_manifest_for_run(base_dir, run_id)
        if manifest_path is None:
            logger.warning(
                f"Evidence index not found and no manifest found for run_id={run_id}, "
                f"as_of_date={date_str}"
            )
            return {
                "files": [],
                "missing_required": [],
                "missing_optional": [],
                "optional_zip_paths": [],
                "evidence_index_path": None,
                "manifest_path": None,
                "source": None,
                "source_path": None,
            }
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception as exc:
            logger.error(f"Failed to read manifest {manifest_path}: {exc}")
            return {
                "files": [],
                "missing_required": [],
                "missing_optional": [],
                "optional_zip_paths": [],
                "evidence_index_path": None,
                "manifest_path": manifest_path,
                "source": None,
                "source_path": None,
            }
        
        source = "manifest"
        source_path_rel = _normalize_zip_path(manifest_path, base_dir)
        # Add manifest itself as a file in the pack (optional evidence)
        files.append((manifest_path, source_path_rel))
        
        # Extract paths directly from manifest top-level
        paths = {
            "ledger_pack_path": manifest.get("ledger_pack_path"),
            "reconcile_report_path": manifest.get("reconcile_report_path"),
            "accounting_report_path": manifest.get("accounting_report_path"),
            "broker_snapshot_path": manifest.get("broker_snapshot_path"),
            "evidence_index_path": manifest.get("evidence_index_path"),
        }
    
    # Use centralized required/optional keys per source
    required_keys = REQUIRED_KEYS_BY_SOURCE.get(source, [])
    optional_keys = OPTIONAL_KEYS_BY_SOURCE.get(source, [])
    
    # Process required files (missing -> keys in missing_required)
    for key in required_keys:
        rel_path_str = paths.get(key)
        if not rel_path_str:
            missing_required.append(key)
            logger.error(f"Required path missing in source={source}: {key}")
            continue
        
        file_path = (base_dir / rel_path_str).resolve()
        if not file_path.exists():
            missing_required.append(key)
            logger.error(f"Required file not found: {key} -> {rel_path_str}")
            continue
        if file_path.is_dir():
            missing_required.append(key)
            logger.warning(
                _ascii_only(
                    "Required path points to a directory; treating as missing: "
                    f"run_id={run_id}, as_of_date={date_str}, key={key}, path={rel_path_str}"
                )
            )
            continue
        
        zip_entry_path = _normalize_zip_path(file_path, base_dir)
        files.append((file_path, zip_entry_path))
        
        related = _find_related_files(file_path)
        for related_path in related:
            try:
                related_path.relative_to(base_dir)
                related_zip_path = _normalize_zip_path(related_path, base_dir)
                files.append((related_path, related_zip_path))
            except ValueError:
                logger.warning(f"Related file outside output_dir, skipping: {related_path}")
    
    # Process optional files (missing -> keys in missing_optional; included -> zip paths in optional_zip_paths)
    for key in optional_keys:
        rel_path_str = paths.get(key)
        if rel_path_str is None:
            continue
        
        file_path = (base_dir / rel_path_str).resolve()
        if not file_path.exists():
            missing_optional.append(key)
            logger.warning(f"Optional file not found: {key} -> {rel_path_str}")
            continue
        if file_path.is_dir():
            missing_optional.append(key)
            logger.warning(
                _ascii_only(
                    "Optional path points to a directory; treating as missing: "
                    f"run_id={run_id}, as_of_date={date_str}, key={key}, path={rel_path_str}"
                )
            )
            continue
        
        zip_entry_path = _normalize_zip_path(file_path, base_dir)
        optional_zip_paths.append(zip_entry_path)
        files.append((file_path, zip_entry_path))
        
        related = _find_related_files(file_path)
        for related_path in related:
            try:
                related_path.relative_to(base_dir)
                related_zip_path = _normalize_zip_path(related_path, base_dir)
                optional_zip_paths.append(related_zip_path)
                files.append((related_path, related_zip_path))
            except ValueError:
                logger.warning(f"Related file outside output_dir, skipping: {related_path}")
    
    return {
        "files": files,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "optional_zip_paths": optional_zip_paths,
        "evidence_index_path": evidence_index_path if evidence_index_used else None,
        "manifest_path": manifest_path,
        "source": source,
        "source_path": source_path_rel,
    }


def build_evidence_pack(
    output_dir: Path | str,
    run_id: str,
    as_of_date: str | pd.Timestamp,
    *,
    include_optional: bool = True,
    strict: bool = False,
    fixed_timestamp: tuple[int, int, int, int, int, int] | None = None,
) -> dict[str, Any]:
    """Build evidence pack (ZIP + manifest) from Evidence Index.
    
    Policy: Required missing -> always fail. Optional missing -> fail if strict=True,
    else warning (and excluded from pack if include_optional=False).
    
    Args:
        output_dir: Base output directory
        run_id: Run identifier
        as_of_date: Report date (YYYY-MM-DD string or pd.Timestamp)
        include_optional: If True, include optional files (with warnings if missing).
                         If False, skip optional files silently.
        strict: If True, raise ValueError when any optional file is missing.
        fixed_timestamp: Fixed ZIP timestamp (default: 1980-01-01 00:00:00)
        
    Returns:
        Dictionary (return schema stable):
            - pack_path: Relative path to ZIP file (POSIX)
            - pack_manifest_path: Relative path to pack manifest JSON (POSIX)
            - n_files: Number of files included in pack
            - missing_optional: List of missing optional keys (not paths)
            - checksums: Dict mapping zip_entry_path to SHA256 hash
            - source: Source type ("evidence_index" or "manifest" or None)

    Determinism: Pack manifest and ZIP use sort_keys=True, indent=2, trailing newline; paths in manifest are POSIX.
    """
    base_dir = Path(output_dir)
    
    # Normalize as_of_date
    if isinstance(as_of_date, str):
        as_of_ts = pd.to_datetime(as_of_date, utc=True)
    else:
        as_of_ts = as_of_date
    if as_of_ts.tz is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    date_str = as_of_ts.strftime("%Y-%m-%d")
    
    # Collect evidence files (from evidence index or manifest fallback)
    collection = collect_evidence_files(base_dir, run_id, as_of_date)
    
    if not collection["files"]:
        raise ValueError(
            f"No evidence files found for run_id={run_id}, as_of_date={date_str}. "
            f"Evidence index and manifest may be missing or empty."
        )
    
    # Check for missing required files (fail-fast)
    missing_required = collection.get("missing_required", [])
    if missing_required:
        raise ValueError(
            f"Required files missing for run_id={run_id}, as_of_date={date_str}: {missing_required}"
        )
    
    # Strict: fail if any optional file is missing (ASCII-only message)
    missing_optional = collection.get("missing_optional", [])
    if strict and missing_optional:
        msg = (
            f"Strict mode: optional files missing for run_id={run_id}, as_of_date={date_str}: "
            f"{missing_optional}"
        )
        raise ValueError(msg.encode("ascii", errors="ignore").decode("ascii"))
    
    # Filter out optional files from ZIP when include_optional=False (policy: still report optional_missing keys in manifest)
    files = list(collection["files"])
    if not include_optional:
        optional_set = set(collection.get("optional_zip_paths", []))
        files = [(fp, zp) for fp, zp in files if zp not in optional_set]
    
    # Use fixed timestamp or default
    zip_timestamp = fixed_timestamp if fixed_timestamp is not None else FIXED_ZIP_TIMESTAMP
    
    # Create output directory
    evidence_dir = base_dir / f"evidence_{run_id}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    # Build ZIP file
    zip_path = evidence_dir / f"pack_{date_str}.zip"
    
    # Calculate checksums before writing ZIP
    checksums: dict[str, str] = {}
    files_to_zip: list[tuple[Path, str]] = []
    
    for file_path, zip_entry_path in files:
        if not file_path.exists():
            logger.warning(f"File not found, skipping: {file_path}")
            continue
        
        try:
            checksum = _sha256_file(file_path)
            checksums[zip_entry_path] = checksum
            files_to_zip.append((file_path, zip_entry_path))
        except Exception as e:
            logger.warning(f"Failed to calculate checksum for {file_path}: {e}")
            # Still include file, but without checksum
            files_to_zip.append((file_path, zip_entry_path))
    
    # Build pack manifest (before writing ZIP, so we can include it)
    pack_manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "as_of_date": as_of_ts.isoformat(),
        "source": collection["source"],
        "source_path": collection.get("source_path"),
        "files": [
            {
                "path": zip_entry_path,
                "size_bytes": file_path.stat().st_size,
                "sha256": checksums.get(zip_entry_path),
                "source_type": _infer_source_type(zip_entry_path),
            }
            for file_path, zip_entry_path in files_to_zip
        ],
        "required_missing": collection.get("missing_required", []),
        "optional_missing": collection.get("missing_optional", []),
        "tool_version": CORE_VERSION,
    }
    # Invariant: the source artifact (evidence index or manifest JSON) must be
    # present in files[] and explicitly typed via source_type == source.
    source = collection.get("source")
    source_path = collection.get("source_path")
    if source and source_path:
        for entry in pack_manifest["files"]:
            if entry.get("path") == source_path:
                entry["source_type"] = source
                break
    
    # Write pack manifest JSON to disk (deterministic)
    manifest_path = evidence_dir / f"pack_manifest_{date_str}.json"
    manifest_zip_path = f"pack_manifest_{date_str}.json"
    
    # First, write manifest without self-reference (temporary)
    manifest_json_content_temp = json.dumps(pack_manifest, sort_keys=True, indent=2, default=str) + "\n"
    
    # Atomic write (Windows-safe) - temporary version
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=manifest_path.parent,
        delete=False,
        suffix=".tmp.json",
    ) as tmp_file:
        tmp_file.write(manifest_json_content_temp)
        tmp_path = Path(tmp_file.name)
    
    tmp_path.replace(manifest_path)
    
    # Calculate checksum for manifest (without self-reference)
    manifest_checksum = _sha256_file(manifest_path)
    
    # Add manifest to files list (for ZIP)
    files_to_zip.append((manifest_path, manifest_zip_path))
    checksums[manifest_zip_path] = manifest_checksum
    
    # Update manifest to include itself (with checksum)
    pack_manifest["files"].append({
        "path": manifest_zip_path,
        "size_bytes": manifest_path.stat().st_size,
        "sha256": manifest_checksum,
        "source_type": "pack_manifest",
    })
    
    # Re-serialize manifest with self-reference (final version)
    manifest_json_content_final = json.dumps(pack_manifest, sort_keys=True, indent=2, default=str) + "\n"
    
    # Re-write manifest to disk (final version with self-reference)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=manifest_path.parent,
        delete=False,
        suffix=".tmp.json",
    ) as tmp_file:
        tmp_file.write(manifest_json_content_final)
        tmp_path = Path(tmp_file.name)
    
    tmp_path.replace(manifest_path)
    
    # Re-sort files (including manifest) - manifest should be in sorted position
    files_to_zip = sorted(files_to_zip, key=lambda x: x[1])
    
    # Write ZIP deterministically (including manifest with self-reference)
    _write_zip_deterministic(zip_path, files_to_zip, base_dir, zip_timestamp)
    
    # Return relative paths
    pack_path_rel = zip_path.relative_to(base_dir)
    manifest_path_rel = manifest_path.relative_to(base_dir)
    
    logger.info(
        f"Evidence pack created: {pack_path_rel} "
        f"({len(files_to_zip)} files, {zip_path.stat().st_size} bytes)"
    )
    
    return {
        "pack_path": str(pack_path_rel.as_posix()),
        "pack_manifest_path": str(manifest_path_rel.as_posix()),
        "n_files": len(files_to_zip),  # Includes manifest
        "missing_required": collection.get("missing_required", []),
        "missing_optional": collection.get("missing_optional", []),
        "checksums": checksums,
        "source": collection["source"],
    }


def _infer_source_type(zip_entry_path: str) -> str:
    """Infer source type from ZIP entry path.
    
    Args:
        zip_entry_path: POSIX path inside ZIP
        
    Returns:
        Source type string (e.g., "evidence_index", "broker_snapshot", "ledger_pack")
    """
    path_lower = zip_entry_path.lower()
    
    if "evidence_" in path_lower and path_lower.endswith(".json"):
        return "evidence_index"
    elif "broker_snapshot" in path_lower:
        return "broker_snapshot"
    elif "ledger_" in path_lower:
        return "ledger_pack"
    elif "reconcile" in path_lower:
        return "reconcile_report"
    elif "accounting" in path_lower:
        return "accounting_report"
    elif "pack_manifest" in path_lower:
        return "pack_manifest"
    elif "manifest" in path_lower:
        return "manifest"
    else:
        return "other"


def verify_evidence_pack_zip(zip_path: Path | str) -> dict[str, Any]:
    """Verify an evidence pack ZIP file offline (no repo context required).
    
    Checks:
    - ZIP contains a pack_manifest_*.json file in the root
    - Manifest schema_version is supported (currently: 1)
    - SHA256 checksums in manifest match actual file contents in ZIP
    - No illegal ZIP paths (no '..', no absolute paths, no backslashes)
    
    Args:
        zip_path: Path to evidence pack ZIP file
    
    Returns:
        Dictionary (return schema stable):
            - ok: True if manifest present, schema ok, checksums match, no illegal paths
            - n_files: Number of entries in ZIP
            - bad_paths: List of illegal entry paths (.., absolute, backslashes)
            - checksum_mismatches: List of paths with checksum mismatch
            - missing_manifest: True if no pack_manifest_*.json in ZIP root
    """
    zpath = Path(zip_path)
    if not zpath.exists():
        raise FileNotFoundError(f"ZIP file not found: {zpath}")
    
    bad_paths: list[str] = []
    checksum_mismatches: list[str] = []
    missing_manifest = False
    
    with zipfile.ZipFile(zpath, "r") as zf:
        namelist = zf.namelist()
        n_files = len(namelist)
        
        # Validate all entry paths (no '..', no absolute, no backslashes)
        for name in namelist:
            if "\\" in name or ".." in name or name.startswith("/") or name.startswith("\\"):
                bad_paths.append(name)
        
        # Find pack_manifest_*.json in root of ZIP
        manifest_names = [
            name
            for name in namelist
            if name.startswith("pack_manifest_") and name.endswith(".json") and "/" not in name.strip("/")
        ]
        
        if not manifest_names:
            missing_manifest = True
            return {
                "ok": False,
                "n_files": n_files,
                "bad_paths": bad_paths,
                "checksum_mismatches": [],
                "missing_manifest": True,
            }
        
        # Deterministic choice if multiple: lexicographically smallest
        manifest_name = sorted(manifest_names)[0]
        
        # Read and parse manifest JSON
        with zf.open(manifest_name) as mf:
            manifest_bytes = mf.read()
        
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse pack manifest JSON: {exc}") from exc
        
        # Validate schema_version
        schema_version = manifest.get("schema_version")
        if schema_version != 1:
            raise ValueError(f"Unsupported pack manifest schema_version: {schema_version}")
        
        files_meta = manifest.get("files", [])
        if not isinstance(files_meta, list):
            raise ValueError("Invalid pack manifest: 'files' must be a list")
        
        # Build lookup for manifest checksums, skipping the pack manifest itself
        manifest_checksums: dict[str, str] = {}
        for entry in files_meta:
            path = entry.get("path")
            checksum = entry.get("sha256")
            source_type = entry.get("source_type")
            # Skip self-reference for pack manifest to avoid circular checksum issues
            if source_type == "pack_manifest":
                continue
            if isinstance(path, str) and isinstance(checksum, str):
                manifest_checksums[path] = checksum.lower()
        
        # Verify checksums for all paths that have a checksum in manifest
        for rel_path, expected_hash in manifest_checksums.items():
            if rel_path not in namelist:
                checksum_mismatches.append(rel_path)
                continue
            
            with zf.open(rel_path) as f:
                data = f.read()
            actual_hash = _sha256_bytes(data)
            if actual_hash != expected_hash:
                checksum_mismatches.append(rel_path)
        
        ok = not missing_manifest and not bad_paths and not checksum_mismatches
        
        return {
            "ok": ok,
            "n_files": n_files,
            "bad_paths": bad_paths,
            "checksum_mismatches": checksum_mismatches,
            "missing_manifest": missing_manifest,
        }
