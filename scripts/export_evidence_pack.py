"""Standalone CLI tool for exporting evidence packs (Sprint 13).

This tool creates a deterministic ZIP archive containing all accounting-related
artifacts for a given run and date, based on the Evidence Index.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_pack import (
    build_evidence_pack,
    verify_evidence_pack_zip,
)
from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.logging_utils import setup_logging


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

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(level="INFO")

    try:
        # Parse and validate as-of-date (strict YYYY-MM-DD format)
        try:
            # Validate format first (YYYY-MM-DD)
            parts = args.as_of_date.split("-")
            if len(parts) != 3 or len(parts[0]) != 4 or len(parts[1]) != 2 or len(parts[2]) != 2:
                msg = f"Invalid date format: {args.as_of_date}. Use YYYY-MM-DD"
                logger.error(msg)
                print(f"ERROR: {msg}", file=sys.stderr)
                return 1
            # Validate date is parseable
            _ = pd.Timestamp(args.as_of_date, tz="UTC")
        except (ValueError, TypeError):
            msg = f"Invalid date format: {args.as_of_date}. Use YYYY-MM-DD"
            logger.error(msg)
            print(f"ERROR: {msg}", file=sys.stderr)
            return 1

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir).resolve()
        else:
            output_dir = OUTPUT_DIR
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
            # ValueErrors: missing required, or strict + missing optional
            msg_raw = f"Failed to build evidence pack: {exc}"
            msg = msg_raw.encode("ascii", errors="ignore").decode("ascii")
            logger.error(msg)
            print(f"ERROR: {msg}", file=sys.stderr)
            return 1
        except Exception as exc:
            msg_raw = f"Unexpected error while building evidence pack: {exc}"
            msg = msg_raw.encode("ascii", errors="ignore").decode("ascii")
            logger.error(msg)
            print(f"ERROR: {msg}", file=sys.stderr)
            return 1

        missing_required = result.get("missing_required") or []
        missing_optional = result.get("missing_optional") or []

        # Print success message (ASCII-only, single status line)
        pack_path = result["pack_path"]
        pack_manifest_path = result["pack_manifest_path"]
        n_files = result["n_files"]
        source = result.get("source") or "unknown"
        source_path = result.get("source_path") or ""
        status_line = (
            f"OK: pack_path={pack_path} "
            f"pack_manifest_path={pack_manifest_path} "
            f"source={source} "
            f"source_path={source_path} "
            f"n_files={n_files} "
            f"required_missing={len(missing_required)} "
            f"optional_missing={len(missing_optional)}"
        )
        # Ensure ASCII-only output
        status_line_ascii = status_line.encode("ascii", errors="ignore").decode("ascii")
        print(status_line_ascii)

        # Warn about missing optional files (if any; strict would have raised already)
        if missing_optional:
            msg_raw = f"Missing optional files (not included in pack): {missing_optional}"
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
                    print(f"ERROR: {msg}", file=sys.stderr)
                    return 1
            except (ValueError, FileNotFoundError) as exc:
                msg_raw = f"Verify-after-build error: {exc}"
                msg = msg_raw.encode("ascii", errors="ignore").decode("ascii")
                logger.error(msg)
                print(f"ERROR: {msg}", file=sys.stderr)
                return 1

        return 0

    except KeyboardInterrupt:
        msg = "Interrupted by user"
        logger.error(msg)
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1
    except Exception as exc:
        msg_raw = f"Unexpected error in CLI: {exc}"
        msg = msg_raw.encode("ascii", errors="ignore").decode("ascii")
        logger.error(msg)
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
