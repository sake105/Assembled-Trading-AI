"""Standalone CLI tool for exporting evidence packs (Sprint 13).

This tool creates a deterministic ZIP archive containing all accounting-related
artifacts for a given run and date, based on the Evidence Index.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # needed for importlib.metadata to find egg-info

from src.assembled_core import __version__ as TOOL_VERSION
from src.assembled_core.accounting.evidence_pack import (
    build_evidence_pack,
    verify_evidence_pack_zip,
)
from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.logging_utils import setup_logging

# Stable error codes for --json (machine-readable, like verify CLI)
EXPORT_ERROR_MISSING_REQUIRED = "MISSING_REQUIRED"
EXPORT_ERROR_OPTIONAL_MISSING_STRICT = "OPTIONAL_MISSING_STRICT"
EXPORT_ERROR_NO_SOURCE = "NO_SOURCE"
EXPORT_ERROR_UNEXPECTED = "UNEXPECTED_ERROR"


def _atomic_copy(src: Path, dest: Path) -> None:
    """Copy src to dest atomically (temp file + replace)."""
    dest = Path(dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dest)


def _ascii_path(s: str) -> str:
    """Return path string as ASCII-only (lossy) for JSON output."""
    return s.encode("ascii", errors="ignore").decode("ascii")


def _export_json_result(
    *,
    ok: bool,
    error_code: str,
    details: dict,
    pack_path: str | None = None,
    pack_manifest_path: str | None = None,
    source: str | None = None,
    source_path: str | None = None,
    n_files: int = 0,
    required_missing_count: int = 0,
    optional_missing_count: int = 0,
    out_zip_path: str | None = None,
    out_manifest_path: str | None = None,
    output_dir: str | None = None,
    output_dir_resolved: str | None = None,
    pack_path_resolved: str | None = None,
    pack_manifest_path_resolved: str | None = None,
    zip_entries_count: int | None = None,
    files_count: int | None = None,
    pack_manifest_schema_version: int | None = None,
) -> dict:
    """Build deterministic JSON output dict (schema_version, ok, error_code, details, tool_version, plus pack/counts, paths, pack_manifest_schema_version, etc.)."""
    out: dict = {
        "schema_version": 1,
        "ok": ok,
        "error_code": error_code,
        "details": details,
        "tool_version": TOOL_VERSION,
        "pack_path": pack_path,
        "pack_manifest_path": pack_manifest_path,
        "pack_manifest_schema_version": pack_manifest_schema_version,
        "source": source,
        "source_path": source_path,
        "n_files": n_files,
        "required_missing_count": required_missing_count,
        "optional_missing_count": optional_missing_count,
        "out_zip_path": out_zip_path,
        "out_manifest_path": out_manifest_path,
        "output_dir": output_dir,
        "output_dir_resolved": output_dir_resolved,
        "pack_path_resolved": pack_path_resolved,
        "pack_manifest_path_resolved": pack_manifest_path_resolved,
        "zip_entries_count": zip_entries_count,
        "files_count": files_count,
    }
    # ASCII-only for path fields (machine-readable logs)
    for key in (
        "output_dir",
        "output_dir_resolved",
        "pack_path_resolved",
        "pack_manifest_path_resolved",
    ):
        if out.get(key) is not None:
            out[key] = _ascii_path(str(out[key]))
    return out


def main() -> int:
    """CLI entry point for evidence pack export."""
    parser = argparse.ArgumentParser(
        description="Export evidence pack (ZIP + manifest) from Evidence Index"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run identifier (e.g., ledger_eod_1d)",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        required=True,
        help="Report date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail if optional files are missing (default: warn and continue)",
    )
    parser.add_argument(
        "--no-optional",
        action="store_true",
        default=False,
        help="Exclude optional files from pack (default: include optional files)",
    )
    parser.add_argument(
        "--verify-after-build",
        action="store_true",
        default=False,
        help="Run verify on the built ZIP and fail if verification fails (default: False)",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        default=False,
        help="Human-readable single-line status (default: output is JSON for machine parsing)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output JSON (default; kept for backward compatibility)",
    )
    parser.add_argument(
        "--out-zip",
        type=str,
        default=None,
        metavar="path",
        help="After build, copy ZIP to this path (atomic). JSON will include out_zip_path.",
    )
    parser.add_argument(
        "--out-manifest",
        type=str,
        default=None,
        metavar="path",
        help="After build, copy pack manifest to this path (atomic). JSON will include out_manifest_path.",
    )
    parser.add_argument(
        "--print-pack-path",
        action="store_true",
        default=False,
        help="Print only pack_path_resolved (one line) to stdout; mutually exclusive with --text.",
    )

    args = parser.parse_args()
    if args.text and args.print_pack_path:
        logger_err = logging.getLogger()
        logger_err.error("--text and --print-pack-path are mutually exclusive")
        return 1
    json_output = (
        not args.text and not args.print_pack_path
    )  # Default: JSON. --text or --print-pack-path override.

    # Setup logging: always send logs to stderr so stdout is clean (JSON or single status line)
    logger = setup_logging(level="INFO")
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.root.addHandler(logging.StreamHandler(sys.stderr))

    output_dir_passed: str | None = None
    output_dir_resolved_str: str | None = None
    try:
        # Parse and validate as-of-date (strict YYYY-MM-DD format)
        try:
            # Validate format first (YYYY-MM-DD)
            parts = args.as_of_date.split("-")
            if (
                len(parts) != 3
                or len(parts[0]) != 4
                or len(parts[1]) != 2
                or len(parts[2]) != 2
            ):
                msg = f"Invalid date format: {args.as_of_date}. Use YYYY-MM-DD"
                logger.error(msg)
                if json_output:
                    out = _export_json_result(
                        ok=False,
                        error_code=EXPORT_ERROR_UNEXPECTED,
                        details={},
                        pack_path=None,
                        pack_manifest_path=None,
                        source=None,
                        source_path=None,
                    )
                    print(
                        json.dumps(out, sort_keys=True, indent=2, default=str) + "\n",
                        end="",
                    )
                else:
                    print(f"ERROR: {msg}", file=sys.stderr)
                return 1
            # Validate date is parseable
            _ = pd.Timestamp(args.as_of_date, tz="UTC")
        except (ValueError, TypeError):
            msg = f"Invalid date format: {args.as_of_date}. Use YYYY-MM-DD"
            logger.error(msg)
            if json_output:
                out = _export_json_result(
                    ok=False,
                    error_code=EXPORT_ERROR_UNEXPECTED,
                    details={},
                    pack_path=None,
                    pack_manifest_path=None,
                    source=None,
                    source_path=None,
                )
                print(
                    json.dumps(out, sort_keys=True, indent=2, default=str) + "\n",
                    end="",
                )
            else:
                print(f"ERROR: {msg}", file=sys.stderr)
            return 1

        # Determine output directory (keep "as passed" for JSON output_dir)
        if args.output_dir:
            output_dir = Path(args.output_dir).resolve()
            output_dir_passed: str | None = args.output_dir
        else:
            output_dir = Path(OUTPUT_DIR)
            output_dir_passed = str(OUTPUT_DIR)
        output_dir_resolved_str = str(output_dir.resolve().as_posix())
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build evidence pack (library enforces strict: optional_missing -> ValueError)
        try:
            result = build_evidence_pack(
                output_dir=output_dir,
                run_id=args.run_id,
                as_of_date=args.as_of_date,
                include_optional=not args.no_optional,
                strict=args.strict,
            )
        except ValueError as exc:
            exc_str = str(exc)
            if (
                "No evidence files found" in exc_str
                or "Evidence index and manifest may be missing" in exc_str
            ):
                error_code = EXPORT_ERROR_NO_SOURCE
            elif "Required files missing" in exc_str:
                error_code = EXPORT_ERROR_MISSING_REQUIRED
            elif "Strict mode: optional files missing" in exc_str:
                error_code = EXPORT_ERROR_OPTIONAL_MISSING_STRICT
            else:
                error_code = EXPORT_ERROR_UNEXPECTED
            msg = exc_str.encode("ascii", errors="ignore").decode("ascii")
            logger.error("Failed to build evidence pack: %s", msg)
            if json_output:
                out = _export_json_result(
                    ok=False,
                    error_code=error_code,
                    details={},
                    pack_path=None,
                    pack_manifest_path=None,
                    source=None,
                    source_path=None,
                    output_dir=output_dir_passed,
                    output_dir_resolved=output_dir_resolved_str,
                    pack_path_resolved=None,
                    pack_manifest_path_resolved=None,
                )
                print(
                    json.dumps(out, sort_keys=True, indent=2, default=str) + "\n",
                    end="",
                )
            else:
                print(f"ERROR: {msg}", file=sys.stderr)
            return 1
        except Exception as exc:
            msg = str(exc).encode("ascii", errors="ignore").decode("ascii")
            logger.error("Unexpected error while building evidence pack: %s", msg)
            if json_output:
                out = _export_json_result(
                    ok=False,
                    error_code=EXPORT_ERROR_UNEXPECTED,
                    details={},
                    pack_path=None,
                    pack_manifest_path=None,
                    source=None,
                    source_path=None,
                    output_dir=output_dir_passed,
                    output_dir_resolved=output_dir_resolved_str,
                    pack_path_resolved=None,
                    pack_manifest_path_resolved=None,
                )
                print(
                    json.dumps(out, sort_keys=True, indent=2, default=str) + "\n",
                    end="",
                )
            else:
                print(f"ERROR: {msg}", file=sys.stderr)
            return 1

        missing_required = result.get("missing_required") or []
        missing_optional = result.get("missing_optional") or []

        pack_path = result["pack_path"]
        pack_manifest_path = result["pack_manifest_path"]
        n_files = result["n_files"]
        source = result.get("source") or "unknown"
        source_path = result.get("source_path")

        out_zip_path_str: str | None = None
        out_manifest_path_str: str | None = None
        if getattr(args, "out_zip", None):
            zip_src = output_dir / pack_path.replace("\\", "/")
            _atomic_copy(zip_src, Path(args.out_zip))
            out_zip_path_str = str(Path(args.out_zip).resolve().as_posix())
        if getattr(args, "out_manifest", None):
            manifest_src = output_dir / pack_manifest_path.replace("\\", "/")
            _atomic_copy(manifest_src, Path(args.out_manifest))
            out_manifest_path_str = str(Path(args.out_manifest).resolve().as_posix())

        pack_path_resolved_str = str(
            (output_dir / pack_path.replace("\\", "/")).resolve().as_posix()
        )
        pack_manifest_path_resolved_str = str(
            (output_dir / pack_manifest_path.replace("\\", "/")).resolve().as_posix()
        )
        if getattr(args, "print_pack_path", False):
            # stdout: only pack path (one line); logs already go to stderr
            print(pack_path_resolved_str)
            return 0
        # Load manifest for zip_entries_count, files_count, source, source_path, pack_manifest_schema_version (single source of truth)
        manifest_path_full = output_dir / pack_manifest_path.replace("\\", "/")
        zip_entries_count_val: int | None = None
        files_count_val: int | None = None
        pack_manifest_schema_version_val: int | None = None
        manifest_source: str | None = source
        manifest_source_path: str | None = source_path
        if manifest_path_full.exists():
            try:
                manifest_data = json.loads(
                    manifest_path_full.read_text(encoding="utf-8")
                )
                zip_entries_count_val = manifest_data.get("zip_entries_count")
                files_count_val = manifest_data.get("files_count")
                if not isinstance(zip_entries_count_val, int):
                    zip_entries_count_val = None
                if not isinstance(files_count_val, int):
                    files_count_val = None
                pm_sv = manifest_data.get("schema_version")
                pack_manifest_schema_version_val = (
                    pm_sv if isinstance(pm_sv, int) else None
                )
                manifest_source = (
                    manifest_data.get("source")
                    if isinstance(manifest_data.get("source"), str)
                    else source
                )
                sp = manifest_data.get("source_path")
                manifest_source_path = sp if isinstance(sp, str) else source_path
            except (json.JSONDecodeError, OSError):
                pass
        # Guard: pack manifest must have valid schema_version (int)
        if json_output and pack_manifest_schema_version_val is None:
            logger.error("Pack manifest missing or invalid schema_version")
            out = _export_json_result(
                ok=False,
                error_code=EXPORT_ERROR_UNEXPECTED,
                details={"pack_manifest_schema_version_invalid": True},
                pack_path=pack_path,
                pack_manifest_path=pack_manifest_path,
                output_dir=output_dir_passed,
                output_dir_resolved=output_dir_resolved_str,
                pack_path_resolved=pack_path_resolved_str,
                pack_manifest_path_resolved=pack_manifest_path_resolved_str,
            )
            print(json.dumps(out, sort_keys=True, indent=2, default=str) + "\n", end="")
            return 1
        # Mismatch guard: export JSON must match pack manifest (ZIP is truth)
        if manifest_source is not None and manifest_source_path is not None:
            if manifest_source != source or manifest_source_path != (source_path or ""):
                msg = _ascii_path(
                    f"Pack manifest source/source_path mismatch build result; run_id={args.run_id} as_of_date={args.as_of_date}"
                )
                logger.error(msg)
                if json_output:
                    out = _export_json_result(
                        ok=False,
                        error_code=EXPORT_ERROR_UNEXPECTED,
                        details={"source_mismatch": True},
                        pack_path=pack_path,
                        pack_manifest_path=pack_manifest_path,
                        output_dir=output_dir_passed,
                        output_dir_resolved=output_dir_resolved_str,
                        pack_path_resolved=pack_path_resolved_str,
                        pack_manifest_path_resolved=pack_manifest_path_resolved_str,
                    )
                    print(
                        json.dumps(out, sort_keys=True, indent=2, default=str) + "\n",
                        end="",
                    )
                return 1
        if json_output:
            out = _export_json_result(
                ok=True,
                error_code="",
                details={},
                pack_path=pack_path,
                pack_manifest_path=pack_manifest_path,
                pack_manifest_schema_version=pack_manifest_schema_version_val,
                source=manifest_source,
                source_path=manifest_source_path,
                n_files=n_files,
                required_missing_count=len(missing_required),
                optional_missing_count=len(missing_optional),
                out_zip_path=out_zip_path_str,
                out_manifest_path=out_manifest_path_str,
                output_dir=output_dir_passed,
                output_dir_resolved=output_dir_resolved_str,
                pack_path_resolved=pack_path_resolved_str,
                pack_manifest_path_resolved=pack_manifest_path_resolved_str,
                zip_entries_count=zip_entries_count_val,
                files_count=files_count_val,
            )
            print(json.dumps(out, sort_keys=True, indent=2, default=str) + "\n", end="")
        else:
            status_line = (
                f"OK: pack_path={pack_path} "
                f"pack_manifest_path={pack_manifest_path} "
                f"source={source} "
                f"source_path={source_path or ''} "
                f"n_files={n_files} "
                f"required_missing={len(missing_required)} "
                f"optional_missing={len(missing_optional)}"
            )
            status_line_ascii = status_line.encode("ascii", errors="ignore").decode(
                "ascii"
            )
            print(status_line_ascii)

        # Warn about missing optional files (if any; strict would have raised already)
        if missing_optional:
            msg_raw = (
                f"Missing optional files (not included in pack): {missing_optional}"
            )
            msg = msg_raw.encode("ascii", errors="ignore").decode("ascii")
            logger.warning(msg)

        # Optional: verify built ZIP and fail-fast on mismatch
        if getattr(args, "verify_after_build", False):
            zip_path = output_dir / pack_path
            try:
                verify_result = verify_evidence_pack_zip(zip_path)
                if not verify_result.get("ok", False):
                    msg_raw = (
                        f"Verify-after-build failed: n_files={verify_result.get('n_files')} "
                        f"missing_manifest={verify_result.get('missing_manifest')} "
                        f"bad_paths_count={len(verify_result.get('bad_paths', []))} "
                        f"checksum_mismatches_count={len(verify_result.get('checksum_mismatches', []))}"
                    )
                    msg = msg_raw.encode("ascii", errors="ignore").decode("ascii")
                    logger.error(msg)
                    if json_output:
                        out = _export_json_result(
                            ok=False,
                            error_code=EXPORT_ERROR_UNEXPECTED,
                            details={"verify_after_build_failed": True},
                            pack_path=pack_path,
                            pack_manifest_path=pack_manifest_path,
                            source=source,
                            source_path=source_path,
                            n_files=n_files,
                            required_missing_count=len(missing_required),
                            optional_missing_count=len(missing_optional),
                            output_dir=output_dir_passed,
                            output_dir_resolved=output_dir_resolved_str,
                            pack_path_resolved=pack_path_resolved_str,
                            pack_manifest_path_resolved=pack_manifest_path_resolved_str,
                        )
                        print(
                            json.dumps(out, sort_keys=True, indent=2, default=str)
                            + "\n",
                            end="",
                        )
                    else:
                        print(f"ERROR: {msg}", file=sys.stderr)
                    return 1
            except (ValueError, FileNotFoundError) as exc:
                msg = str(exc).encode("ascii", errors="ignore").decode("ascii")
                logger.error("Verify-after-build error: %s", msg)
                if json_output:
                    out = _export_json_result(
                        ok=False,
                        error_code=EXPORT_ERROR_UNEXPECTED,
                        details={},
                        pack_path=pack_path,
                        pack_manifest_path=pack_manifest_path,
                        source=source,
                        source_path=source_path,
                        n_files=n_files,
                        required_missing_count=len(missing_required),
                        optional_missing_count=len(missing_optional),
                        output_dir=output_dir_passed,
                        output_dir_resolved=output_dir_resolved_str,
                        pack_path_resolved=pack_path_resolved_str,
                        pack_manifest_path_resolved=pack_manifest_path_resolved_str,
                    )
                    print(
                        json.dumps(out, sort_keys=True, indent=2, default=str) + "\n",
                        end="",
                    )
                else:
                    print(f"ERROR: {msg}", file=sys.stderr)
                return 1

        return 0

    except KeyboardInterrupt:
        msg = "Interrupted by user"
        logger.error(msg)
        if json_output:
            out = _export_json_result(
                ok=False,
                error_code=EXPORT_ERROR_UNEXPECTED,
                details={},
                pack_path=None,
                pack_manifest_path=None,
                source=None,
                source_path=None,
                output_dir=output_dir_passed,
                output_dir_resolved=output_dir_resolved_str,
                pack_path_resolved=None,
                pack_manifest_path_resolved=None,
            )
            print(json.dumps(out, sort_keys=True, indent=2, default=str) + "\n", end="")
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1
    except Exception as exc:
        msg = str(exc).encode("ascii", errors="ignore").decode("ascii")
        logger.error("Unexpected error in CLI: %s", msg)
        if json_output:
            out = _export_json_result(
                ok=False,
                error_code=EXPORT_ERROR_UNEXPECTED,
                details={},
                pack_path=None,
                pack_manifest_path=None,
                source=None,
                source_path=None,
                output_dir=output_dir_passed,
                output_dir_resolved=output_dir_resolved_str,
                pack_path_resolved=None,
                pack_manifest_path_resolved=None,
            )
            print(json.dumps(out, sort_keys=True, indent=2, default=str) + "\n", end="")
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
