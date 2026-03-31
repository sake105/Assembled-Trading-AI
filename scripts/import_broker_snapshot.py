"""Standalone CLI tool for importing external broker snapshots (Sprint 13).

This tool imports external broker snapshots (JSON/CSV) into the standardized
layout: output/broker_snapshot_<run_id>/snapshot_<YYYY-MM-DD>.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.broker_snapshot_importer import (
    import_broker_snapshot,
)
from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.logging_utils import setup_logging


def main() -> int:
    """CLI entry point for broker snapshot import."""
    parser = argparse.ArgumentParser(
        description="Import external broker snapshot (JSON/CSV) into standardized layout"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file (JSON or CSV)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID for snapshot namespace",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        required=True,
        help="Snapshot date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=None,
        help="Cash value override (required for CSV without cash column, optional override for CSV with cash)",
    )
    parser.add_argument(
        "--store-parquet",
        action="store_true",
        default=False,
        help="Store positions as Parquet file (optional)",
    )
    parser.add_argument(
        "--qty-tol",
        type=float,
        default=1e-8,
        help="Quantity tolerance for filtering tiny residuals (default: 1e-8)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(level="INFO")

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
                raise ValueError(
                    f"Invalid date format: {args.as_of_date}. Use YYYY-MM-DD"
                )
            snapshot_date = pd.Timestamp(args.as_of_date, tz="UTC")
        except (ValueError, TypeError):
            logger.error(f"Invalid date format: {args.as_of_date}. Use YYYY-MM-DD")
            return 1

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir).resolve()
        else:
            output_dir = OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate input file exists
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1

        logger.info(f"Importing broker snapshot from: {input_path}")
        logger.info(f"Run ID: {args.run_id}")
        logger.info(f"Snapshot date: {snapshot_date.strftime('%Y-%m-%d')}")
        logger.info(f"Output directory: {output_dir}")

        # Import snapshot
        result = import_broker_snapshot(
            snapshot_path=input_path,
            run_id=args.run_id,
            snapshot_date=snapshot_date,
            output_dir=output_dir,
            qty_tol=args.qty_tol,
            store_parquet=args.store_parquet,
            cash_override=args.cash,
        )

        logger.info("Import successful:")
        logger.info(f"  Snapshot path: {result['broker_snapshot_path']}")
        if result.get("broker_positions_path"):
            logger.info(f"  Positions path: {result['broker_positions_path']}")
        logger.info(f"  Cash: {result['cash']:.2f}")

        return 0

    except ValueError as e:
        # Ensure error message is ASCII-only
        error_msg = str(e)
        logger.error(f"Import failed: {error_msg}")
        return 1
    except Exception as e:
        # Ensure error message is ASCII-only
        error_msg = str(e)
        logger.error(f"Unexpected error: {error_msg}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
