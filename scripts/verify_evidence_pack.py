"""Standalone CLI to validate an Evidence Pack ZIP offline.

Validates: manifest present, schema ok, checksums ok, no illegal paths.
Exit codes: 0 = ok, 1 = fail or error. ASCII-only output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # needed for importlib.metadata to find egg-info

from src.assembled_core import __version__ as TOOL_VERSION
from src.assembled_core.accounting.evidence_pack import verify_evidence_pack_zip


def _ascii(s: str) -> str:
    """Normalize string to ASCII-only (drop non-ASCII)."""
    return s.encode("ascii", errors="ignore").decode("ascii")


# Stable error codes for Ops/Automation (ASCII, no Unicode, no new deps).
ERROR_CODE_OK = ""
ERROR_CODE_MISSING_MANIFEST = "MISSING_MANIFEST"
ERROR_CODE_BAD_PATHS = "BAD_PATHS"
ERROR_CODE_MISSING_ENTRIES = "MISSING_ENTRIES"
ERROR_CODE_PATHS_NOT_IN_ZIP_ENTRIES = "PATHS_NOT_IN_ZIP_ENTRIES"
ERROR_CODE_CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
ERROR_CODE_UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
ERROR_CODE_FILE_NOT_FOUND = "FILE_NOT_FOUND"
ERROR_CODE_UNEXPECTED = "UNEXPECTED_ERROR"

JSON_SCHEMA_VERSION = 1


def _error_code_from_result(result: dict) -> str:
    """Derive single error_code from verify result ("" when ok).
    Priority: MISSING_MANIFEST > BAD_PATHS > MISSING_ENTRIES > PATHS_NOT_IN_ZIP_ENTRIES > CHECKSUM_MISMATCH.
    """
    if result.get("ok"):
        return ERROR_CODE_OK
    if result.get("missing_manifest"):
        return ERROR_CODE_MISSING_MANIFEST
    if result.get("bad_paths"):
        return ERROR_CODE_BAD_PATHS
    if result.get("missing_entries"):
        return ERROR_CODE_MISSING_ENTRIES
    if result.get("paths_not_in_zip_entries"):
        return ERROR_CODE_PATHS_NOT_IN_ZIP_ENTRIES
    if result.get("checksum_mismatches"):
        return ERROR_CODE_CHECKSUM_MISMATCH
    return ERROR_CODE_UNEXPECTED


def _result_dict_for_output(
    result: dict,
    zip_path_str: str,
    error_code: str | None = None,
    zip_path_resolved: str | None = None,
) -> dict:
    """Build output dict for JSON output (stable schema).

    Fields:
        - schema_version: JSON schema version for this CLI
        - zip_path: Path as provided on CLI
        - zip_path_resolved: (optional) Absolute path, POSIX-style, ASCII-only
        - ok: True if verification passed
        - error_code: Stable error code ("" or one of ERROR_CODE_*)
        - n_files: Number of ZIP entries (0 if unknown)
        - missing_manifest: True if pack_manifest_* is missing
        - bad_paths_count: Number of illegal paths
        - checksum_mismatches_count: Number of checksum mismatches
        - details: Optional debug fields (small, bounded lists):
            - bad_paths: up to 20 illegal paths
            - checksum_mismatches: up to 20 mismatched paths
        - tool_version: from assembled_core.__version__ (single authority)
    """
    if error_code is not None:
        ec = error_code
    else:
        ec = _error_code_from_result(result)
    details_raw = result.get("details")
    details: dict[str, object] = details_raw if isinstance(details_raw, dict) else {}
    out: dict[str, object] = {
        "schema_version": JSON_SCHEMA_VERSION,
        "zip_path": zip_path_str,
        "ok": result.get("ok", False),
        "error_code": ec,
        "n_files": result.get("n_files", 0),
        "zip_entries_count": result.get("zip_entries_count"),
        "manifest_files_count": result.get("manifest_files_count"),
        "zip_compression": result.get("zip_compression"),
        "source": result.get("source"),
        "source_path": result.get("source_path"),
        "missing_manifest": result.get("missing_manifest", False),
        "bad_paths_count": len(result.get("bad_paths", [])),
        "missing_entries_count": len(result.get("missing_entries", []) or []),
        "paths_not_in_zip_entries_count": len(
            result.get("paths_not_in_zip_entries", [])
        ),
        "checksum_mismatches_count": len(result.get("checksum_mismatches", [])),
        "details": details if details is not None else {},
        "tool_version": TOOL_VERSION,
    }
    if zip_path_resolved is not None:
        out["zip_path_resolved"] = zip_path_resolved.encode(
            "ascii", errors="ignore"
        ).decode("ascii")
    if out.get("source_path") is not None:
        out["source_path"] = (
            str(out["source_path"]).encode("ascii", errors="ignore").decode("ascii")
        )
    return out


def main() -> int:
    """CLI entry: validate evidence pack ZIP; exit 0 if ok, 1 if fail or error."""
    parser = argparse.ArgumentParser(
        description="Validate an Evidence Pack ZIP offline (manifest, schema, checksums, paths)"
    )
    parser.add_argument(
        "--zip",
        type=str,
        required=True,
        metavar="path",
        help="Path to evidence pack ZIP file",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Human-readable OK/FAIL/ERROR lines (default: output is JSON)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON (default; kept for backward compatibility)",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        default=False,
        help="Exit 1 if any of bad_paths_count, missing_entries_count, paths_not_in_zip_entries_count, or checksum_mismatches_count > 0 (even when ok=True)",
    )
    args = parser.parse_args()
    json_output = not args.text  # Default: JSON. --text for human lines.

    zip_path_str = args.zip
    zip_path = Path(zip_path_str)
    zip_path_resolved_str = zip_path.resolve().as_posix() if json_output else None

    try:
        result = verify_evidence_pack_zip(zip_path)
    except FileNotFoundError as exc:
        if json_output:
            out = _result_dict_for_output(
                {
                    "ok": False,
                    "n_files": 0,
                    "missing_manifest": False,
                    "bad_paths": [],
                    "checksum_mismatches": [],
                },
                zip_path_str,
                error_code=ERROR_CODE_FILE_NOT_FOUND,
                zip_path_resolved=zip_path_resolved_str,
            )
            print(json.dumps(out, sort_keys=True, indent=2) + "\n", end="")
        else:
            msg = _ascii(str(exc))
            print(
                f"ERROR: error_code={ERROR_CODE_FILE_NOT_FOUND} {msg}", file=sys.stderr
            )
        return 1
    except ValueError as exc:
        exc_str = str(exc)
        ec = (
            ERROR_CODE_UNSUPPORTED_SCHEMA
            if "schema" in exc_str.lower() or "schema_version" in exc_str
            else ERROR_CODE_UNEXPECTED
        )
        if json_output:
            out = _result_dict_for_output(
                {
                    "ok": False,
                    "n_files": 0,
                    "missing_manifest": False,
                    "bad_paths": [],
                    "checksum_mismatches": [],
                },
                zip_path_str,
                error_code=ec,
                zip_path_resolved=zip_path_resolved_str,
            )
            print(json.dumps(out, sort_keys=True, indent=2) + "\n", end="")
        else:
            msg = _ascii(exc_str)
            print(f"ERROR: error_code={ec} {msg}", file=sys.stderr)
        return 1
    except Exception as exc:
        if json_output:
            out = _result_dict_for_output(
                {
                    "ok": False,
                    "n_files": 0,
                    "missing_manifest": False,
                    "bad_paths": [],
                    "checksum_mismatches": [],
                },
                zip_path_str,
                error_code=ERROR_CODE_UNEXPECTED,
                zip_path_resolved=zip_path_resolved_str,
            )
            print(json.dumps(out, sort_keys=True, indent=2) + "\n", end="")
        else:
            msg = _ascii(f"Unexpected error: {exc}")
            print(f"ERROR: error_code={ERROR_CODE_UNEXPECTED} {msg}", file=sys.stderr)
        return 1

    out = _result_dict_for_output(
        result, zip_path_str, zip_path_resolved=zip_path_resolved_str
    )

    if json_output:
        json_str = json.dumps(out, sort_keys=True, indent=2) + "\n"
        print(json_str, end="")
    else:
        if result["ok"]:
            line = (
                f"OK: ok=True n_files={out['n_files']} "
                f"missing_manifest={out['missing_manifest']} "
                f"bad_paths_count={out['bad_paths_count']} "
                f"checksum_mismatches_count={out['checksum_mismatches_count']}"
            )
        else:
            ec = out.get("error_code", _error_code_from_result(result))
            line = (
                f"FAIL: ok=False error_code={ec} n_files={out['n_files']} "
                f"missing_manifest={out['missing_manifest']} "
                f"bad_paths_count={out['bad_paths_count']} "
                f"checksum_mismatches_count={out['checksum_mismatches_count']}"
            )
        print(_ascii(line))

    if not result["ok"]:
        return 1
    if getattr(args, "fail_on_warn", False):
        n_bad = len(result.get("bad_paths") or [])
        n_missing = len(result.get("missing_entries") or [])
        n_paths_not_in = len(result.get("paths_not_in_zip_entries") or [])
        n_checksum = len(result.get("checksum_mismatches") or [])
        if n_bad > 0 or n_missing > 0 or n_paths_not_in > 0 or n_checksum > 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
