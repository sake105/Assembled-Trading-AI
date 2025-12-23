"""Validate Alt-Daten snapshot directory.

This script validates all Parquet files in an Alt-Daten snapshot directory,
checking for required columns, date ranges, and data completeness.

Usage:
    python scripts/validate_altdata_snapshot.py --root "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025" --interval 1d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tabulate import tabulate

# Required columns for Alt-Daten Parquet files (compatible with LocalParquetPriceDataSource)
REQUIRED_COLS = ["timestamp", "symbol", "close"]
OPTIONAL_COLS = ["open", "high", "low", "adj_close", "volume"]


def validate_parquet_file(parquet_path: Path) -> dict:
    """Validate a single Parquet file.

    Args:
        parquet_path: Path to Parquet file

    Returns:
        Dictionary with validation results:
        {
            "symbol": str,
            "status": "ok" | "error" | "warning",
            "rows": int,
            "date_min": Optional[str],
            "date_max": Optional[str],
            "missing_cols": list[str],
            "message": str
        }
    """
    symbol = parquet_path.stem.upper()
    result = {
        "symbol": symbol,
        "status": "ok",
        "rows": 0,
        "date_min": None,
        "date_max": None,
        "missing_cols": [],
        "message": "",
    }

    try:
        # Load Parquet file
        df = pd.read_parquet(parquet_path)

        if df.empty:
            result["status"] = "error"
            result["message"] = "EMPTY FILE"
            return result

        result["rows"] = len(df)

        # Check required columns
        missing_required = [col for col in REQUIRED_COLS if col not in df.columns]
        if missing_required:
            result["status"] = "error"
            result["missing_cols"] = missing_required
            result["message"] = f"MISSING REQUIRED COLUMNS: {missing_required}"
            return result

        # Check optional columns (warn if missing)
        missing_optional = [col for col in OPTIONAL_COLS if col not in df.columns]
        if missing_optional:
            result["status"] = "warning"
            result["missing_cols"] = missing_optional
            result["message"] = f"Missing optional columns: {missing_optional}"

        # Validate timestamp column
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            valid_timestamps = df["timestamp"].notna()

            if not valid_timestamps.any():
                result["status"] = "error"
                result["message"] = "No valid timestamps found"
                return result

            date_min = df["timestamp"].min()
            date_max = df["timestamp"].max()
            result["date_min"] = (
                date_min.strftime("%Y-%m-%d") if pd.notna(date_min) else None
            )
            result["date_max"] = (
                date_max.strftime("%Y-%m-%d") if pd.notna(date_max) else None
            )

            # Check for data gaps (heuristic: less than 2000 rows for 2000-2025 period)
            if result["rows"] < 2000:
                if result["status"] == "ok":
                    result["status"] = "warning"
                result["message"] = (
                    f"Low row count ({result['rows']}) - possible data gaps"
                )

        # Validate symbol column
        if "symbol" in df.columns:
            unique_symbols = df["symbol"].unique()
            if len(unique_symbols) > 1:
                result["status"] = "warning"
                result["message"] = f"Multiple symbols in file: {list(unique_symbols)}"
            elif len(unique_symbols) == 1 and str(unique_symbols[0]).upper() != symbol:
                result["status"] = "warning"
                result["message"] = (
                    f"Symbol mismatch: file={unique_symbols[0]}, filename={symbol}"
                )

        if result["status"] == "ok" and not result["message"]:
            result["message"] = "OK"

    except Exception as e:
        result["status"] = "error"
        result["message"] = f"ERROR: {str(e)}"

    return result


def validate_snapshot_directory(root: Path, interval: str) -> list[dict]:
    """Validate all Parquet files in snapshot directory.

    Args:
        root: Root directory of Alt-Daten snapshot
        interval: Data interval (e.g., "1d")

    Returns:
        List of validation result dictionaries
    """
    snapshot_dir = root / interval

    if not snapshot_dir.exists():
        print(f"ERROR: Snapshot directory not found: {snapshot_dir}")
        return []

    # Find all Parquet files
    parquet_files = sorted(snapshot_dir.glob("*.parquet"))

    if not parquet_files:
        print(f"WARNING: No Parquet files found in {snapshot_dir}")
        return []

    print(f"Validating {len(parquet_files)} Parquet file(s) in {snapshot_dir}")
    print("=" * 80)

    results = []
    for parquet_file in parquet_files:
        result = validate_parquet_file(parquet_file)
        results.append(result)

    return results


def print_validation_report(results: list[dict]) -> None:
    """Print validation report as table.

    Args:
        results: List of validation result dictionaries
    """
    if not results:
        print("No results to display")
        return

    # Prepare table data
    table_data = []
    for r in results:
        table_data.append(
            [
                r["symbol"],
                r["rows"],
                r["date_min"] or "N/A",
                r["date_max"] or "N/A",
                r["status"].upper(),
                r["message"],
            ]
        )

    headers = ["Symbol", "Rows", "Date Min", "Date Max", "Status", "Message"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Summary
    print("\n" + "=" * 80)
    ok_count = sum(1 for r in results if r["status"] == "ok")
    warning_count = sum(1 for r in results if r["status"] == "warning")
    error_count = sum(1 for r in results if r["status"] == "error")

    print("Summary:")
    print(f"  OK:      {ok_count}/{len(results)}")
    print(f"  Warning: {warning_count}/{len(results)}")
    print(f"  Error:   {error_count}/{len(results)}")
    print("=" * 80)

    # Show errors and warnings
    if warning_count > 0 or error_count > 0:
        print("\nDetails:")
        for r in results:
            if r["status"] != "ok":
                print(f"  {r['symbol']}: {r['status'].upper()} - {r['message']}")


def main() -> None:
    """Main entry point for Alt-Daten snapshot validation."""
    parser = argparse.ArgumentParser(
        description="Validate Alt-Daten snapshot directory (Parquet files).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate default path
    python scripts/validate_altdata_snapshot.py
    
    # Validate custom path
    python scripts/validate_altdata_snapshot.py \\
      --root "F:\\Python_Projekt\\Aktiengerüst\\datensammlungen\\altdaten\\stand 3-12-2025" \\
      --interval 1d
        """,
    )

    parser.add_argument(
        "--root",
        type=str,
        default=r"F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
        help="Root directory of Alt-Daten snapshot (default: F:\\Python_Projekt\\Aktiengerüst\\datensammlungen\\altdaten\\stand 3-12-2025)",
    )

    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval (default: 1d)",
    )

    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        print(f"ERROR: Root directory not found: {root}")
        exit(1)

    # Validate snapshot directory
    results = validate_snapshot_directory(root, args.interval)

    if not results:
        exit(1)

    # Print report
    print_validation_report(results)

    # Exit with error code if any errors found
    error_count = sum(1 for r in results if r["status"] == "error")
    if error_count > 0:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
