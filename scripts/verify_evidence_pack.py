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

from src.assembled_core.accounting.evidence_pack import verify_evidence_pack_zip


def _ascii(s: str) -> str:
    """Normalize string to ASCII-only (drop non-ASCII)."""
    return s.encode("ascii", errors="ignore").decode("ascii")


# Stable error codes for Ops/Automation (ASCII, no Unicode, no new deps).
ERROR_CODE_OK = ""
ERROR_CODE_MISSING_MANIFEST = "MISSING_MANIFEST"
ERROR_CODE_BAD_PATHS = "BAD_PATHS"
ERROR_CODE_CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
ERROR_CODE_UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
ERROR_CODE_FILE_NOT_FOUND = "FILE_NOT_FOUND"
ERROR_CODE_UNEXPECTED = "UNEXPECTED_ERROR"

JSON_SCHEMA_VERSION = 1


def _error_code_from_result(result: dict) -> str:
    """Derive single error_code from verify result ("" when ok)."""
    if result.get("ok"):
        return ERROR_CODE_OK
    if result.get("missing_manifest"):
        return ERROR_CODE_MISSING_MANIFEST
    if result.get("bad_paths"):
        return ERROR_CODE_BAD_PATHS
    if result.get("checksum_mismatches"):
        return ERROR_CODE_CHECKSUM_MISMATCH
    return ERROR_CODE_UNEXPECTED


def _result_dict_for_output(
    result: dict,
    zip_path_str: str,
    error_code: str | None = None,
) -> dict:
    """Build output dict: schema_version, zip_path, ok, error_code, n_files, missing_manifest, bad_paths_count, checksum_mismatches_count."""
    if error_code is not None:
        ec = error_code
    else:
        ec = _error_code_from_result(result)
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "zip_path": zip_path_str,
        "ok": result.get("ok", False),
        "error_code": ec,
        "n_files": result.get("n_files", 0),
        "missing_manifest": result.get("missing_manifest", False),
        "bad_paths_count": len(result.get("bad_paths", [])),
        "checksum_mismatches_count": len(result.get("checksum_mismatches", [])),
    }


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
        "--json",
        action="store_true",
        help="Output result as deterministic JSON (sort_keys, indent=2, trailing newline)",
    )
    args = parser.parse_args()
    zip_path_str = args.zip
    zip_path = Path(zip_path_str)

    try:
        result = verify_evidence_pack_zip(zip_path)
    except FileNotFoundError as exc:
        if args.json:
            out = _result_dict_for_output(
                {"ok": False, "n_files": 0, "missing_manifest": False, "bad_paths": [], "checksum_mismatches": []},
                zip_path_str,
                error_code=ERROR_CODE_FILE_NOT_FOUND,
            )
            print(json.dumps(out, sort_keys=True, indent=2) + "\n", end="")
        else:
            msg = _ascii(str(exc))
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1
    except ValueError as exc:
        exc_str = str(exc)
        ec = ERROR_CODE_UNSUPPORTED_SCHEMA if "schema" in exc_str.lower() or "schema_version" in exc_str else ERROR_CODE_UNEXPECTED
        if args.json:
            out = _result_dict_for_output(
                {"ok": False, "n_files": 0, "missing_manifest": False, "bad_paths": [], "checksum_mismatches": []},
                zip_path_str,
                error_code=ec,
            )
            print(json.dumps(out, sort_keys=True, indent=2) + "\n", end="")
        else:
            msg = _ascii(exc_str)
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1
    except Exception as exc:
        if args.json:
            out = _result_dict_for_output(
                {"ok": False, "n_files": 0, "missing_manifest": False, "bad_paths": [], "checksum_mismatches": []},
                zip_path_str,
                error_code=ERROR_CODE_UNEXPECTED,
            )
            print(json.dumps(out, sort_keys=True, indent=2) + "\n", end="")
        else:
            msg = _ascii(f"Unexpected error: {exc}")
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    out = _result_dict_for_output(result, zip_path_str)

    if args.json:
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
            line = (
                f"FAIL: ok=False n_files={out['n_files']} "
                f"missing_manifest={out['missing_manifest']} "
                f"bad_paths_count={out['bad_paths_count']} "
                f"checksum_mismatches_count={out['checksum_mismatches_count']}"
            )
        print(_ascii(line))

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
