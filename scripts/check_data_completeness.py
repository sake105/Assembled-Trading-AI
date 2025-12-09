"""
Check completeness and quality of downloaded historical data.

Prints a report on:
- File existence and sizes
- Date ranges (start/end dates)
- Number of rows per symbol
- Missing dates (gaps)
- Duplicates
- Column completeness
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.assembled_core.config.settings import Settings


def check_file_completeness(
    file_path: Path,
    expected_start: str | None = None,
    expected_end: str | None = None,
    freq: str = "1d",
) -> dict[str, Any]:
    """Check completeness of a single parquet file."""
    result = {
        "exists": False,
        "size_kb": 0,
        "rows": 0,
        "start_date": None,
        "end_date": None,
        "date_range_days": 0,
        "expected_days": None,
        "missing_days": 0,
        "duplicates": 0,
        "columns": [],
        "has_timestamp": False,
        "has_symbol": False,
        "has_close": False,
        "errors": [],
    }

    if not file_path.exists():
        result["errors"].append("File does not exist")
        return result

    result["exists"] = True
    result["size_kb"] = round(file_path.stat().st_size / 1024, 1)

    try:
        df = pd.read_parquet(file_path)

        if len(df) == 0:
            result["errors"].append("File is empty")
            return result

        result["rows"] = len(df)
        result["columns"] = list(df.columns)

        # Check required columns
        result["has_timestamp"] = "timestamp" in df.columns
        result["has_symbol"] = "symbol" in df.columns
        result["has_close"] = "close" in df.columns

        if not result["has_timestamp"]:
            result["errors"].append("Missing 'timestamp' column")
            return result

        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        else:
            if not df["timestamp"].dt.tz:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        result["start_date"] = df["timestamp"].min()
        result["end_date"] = df["timestamp"].max()
        result["date_range_days"] = (result["end_date"] - result["start_date"]).days

        # Check for duplicates
        if result["has_symbol"]:
            duplicates = df.duplicated(subset=["timestamp", "symbol"]).sum()
        else:
            duplicates = df.duplicated(subset=["timestamp"]).sum()
        result["duplicates"] = int(duplicates)

        # Check for missing dates (gaps)
        if expected_start and expected_end:
            expected_start_dt = pd.to_datetime(expected_start, utc=True)
            expected_end_dt = pd.to_datetime(expected_end, utc=True)

            # Create expected date range
            if freq == "1d":
                expected_dates = pd.date_range(
                    expected_start_dt, expected_end_dt, freq="D", tz="UTC"
                )
                # Filter to trading days (Mon-Fri)
                expected_dates = expected_dates[
                    expected_dates.weekday < 5
                ]  # 0=Mon, 4=Fri
            else:
                expected_dates = pd.date_range(
                    expected_start_dt, expected_end_dt, freq=freq, tz="UTC"
                )

            result["expected_days"] = len(expected_dates)

            # Find missing dates
            actual_dates = set(df["timestamp"].dt.date)
            expected_dates_set = set(expected_dates.date)
            missing_dates = expected_dates_set - actual_dates
            result["missing_days"] = len(missing_dates)

            # Store first few missing dates for debugging
            if missing_dates:
                result["missing_dates_sample"] = sorted(list(missing_dates))[:10]

    except Exception as e:
        result["errors"].append(f"Error reading file: {str(e)}")

    return result


def check_universe_completeness(
    universe_file: Path,
    target_root: Path,
    interval: str = "1d",
    expected_start: str | None = None,
    expected_end: str | None = None,
) -> dict[str, Any]:
    """Check completeness for all symbols in a universe file."""
    if not universe_file.exists():
        return {"error": f"Universe file not found: {universe_file}"}

    # Load symbols
    symbols = []
    with open(universe_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line.upper())

    if not symbols:
        return {"error": "No symbols found in universe file"}

    target_dir = target_root / interval
    results = {}

    for symbol in symbols:
        file_path = target_dir / f"{symbol}.parquet"
        results[symbol] = check_file_completeness(
            file_path, expected_start, expected_end, freq=interval
        )

    return {
        "universe_file": str(universe_file),
        "symbols": symbols,
        "results": results,
        "expected_start": expected_start,
        "expected_end": expected_end,
    }


def print_completeness_report(report: dict[str, Any]) -> None:
    """Print a formatted completeness report."""
    if "error" in report:
        print(f"❌ Error: {report['error']}")
        return

    print("=" * 80)
    print("DATA COMPLETENESS REPORT")
    print("=" * 80)
    print(f"Universe: {report['universe_file']}")
    print(f"Symbols: {len(report['symbols'])}")
    if report.get("expected_start"):
        print(f"Expected Date Range: {report['expected_start']} to {report['expected_end']}")
    print("=" * 80)
    print()

    results = report["results"]
    symbols = report["symbols"]

    # Summary statistics
    existing = [s for s in symbols if results[s]["exists"]]
    missing = [s for s in symbols if not results[s]["exists"]]
    with_errors = [s for s in symbols if results[s].get("errors")]

    print("SUMMARY")
    print("-" * 80)
    print(f"Total Symbols: {len(symbols)}")
    print(f"✅ Files Exist: {len(existing)}")
    print(f"❌ Files Missing: {len(missing)}")
    print(f"⚠️  Files with Errors: {len(with_errors)}")
    print()

    # Missing files
    if missing:
        print("MISSING FILES")
        print("-" * 80)
        for sym in missing:
            print(f"  ❌ {sym}")
        print()

    # Files with errors
    if with_errors:
        print("FILES WITH ERRORS")
        print("-" * 80)
        for sym in with_errors:
            errors = results[sym].get("errors", [])
            print(f"  ⚠️  {sym}: {', '.join(errors)}")
        print()

    # Detailed report for existing files
    if existing:
        print("DETAILED REPORT (Existing Files)")
        print("-" * 80)
        print(
            f"{'Symbol':<12} {'Size':<10} {'Rows':<10} {'Start Date':<12} {'End Date':<12} "
            f"{'Range Days':<12} {'Missing':<10} {'Duplicates':<10} {'Status':<10}"
        )
        print("-" * 80)

        for sym in sorted(existing):
            r = results[sym]
            size_str = f"{r['size_kb']} KB" if r["exists"] else "N/A"
            rows_str = str(r["rows"]) if r["rows"] > 0 else "0"
            start_str = (
                r["start_date"].strftime("%Y-%m-%d")
                if r["start_date"] is not None
                else "N/A"
            )
            end_str = (
                r["end_date"].strftime("%Y-%m-%d")
                if r["end_date"] is not None
                else "N/A"
            )
            range_str = str(r["date_range_days"]) if r["date_range_days"] > 0 else "0"
            missing_str = str(r["missing_days"]) if r.get("missing_days") is not None else "N/A"
            dup_str = str(r["duplicates"])

            # Status
            status = "✅ OK"
            if r.get("errors"):
                status = "⚠️  ERROR"
            elif r["missing_days"] and r["missing_days"] > 10:
                status = "⚠️  GAPS"
            elif r["duplicates"] > 0:
                status = "⚠️  DUPS"

            print(
                f"{sym:<12} {size_str:<10} {rows_str:<10} {start_str:<12} {end_str:<12} "
                f"{range_str:<12} {missing_str:<10} {dup_str:<10} {status:<10}"
            )

        print()

        # Column completeness
        print("COLUMN COMPLETENESS")
        print("-" * 80)
        required_cols = ["timestamp", "symbol", "close"]
        for sym in sorted(existing):
            r = results[sym]
            cols_status = []
            for col in required_cols:
                col_key = f"has_{col}" if col != "close" else "has_close"
                if r.get(col_key):
                    cols_status.append(f"✅ {col}")
                else:
                    cols_status.append(f"❌ {col}")
            print(f"  {sym}: {', '.join(cols_status)}")
        print()

        # Missing dates details (first 5 symbols with gaps)
        gaps_found = [
            (sym, r)
            for sym, r in results.items()
            if r["exists"] and r.get("missing_days", 0) > 0
        ]
        if gaps_found:
            print("MISSING DATES (Sample)")
            print("-" * 80)
            for sym, r in sorted(gaps_found, key=lambda x: x[1]["missing_days"], reverse=True)[:5]:
                print(f"  {sym}: {r['missing_days']} missing days")
                if r.get("missing_dates_sample"):
                    sample = r["missing_dates_sample"]
                    print(f"    Sample: {sample[0]} ... {sample[-1] if len(sample) > 1 else ''}")
            print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Check completeness of downloaded historical data"
    )
    parser.add_argument(
        "--universe-file",
        type=str,
        help="Path to universe ticker file (e.g., config/macro_world_etfs_tickers.txt)",
    )
    parser.add_argument(
        "--target-root",
        type=str,
        help="Root directory for downloaded data (default: from settings)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval (default: 1d)",
    )
    parser.add_argument(
        "--expected-start",
        type=str,
        help="Expected start date (YYYY-MM-DD) for completeness check",
    )
    parser.add_argument(
        "--expected-end",
        type=str,
        help="Expected end date (YYYY-MM-DD) for completeness check",
    )
    parser.add_argument(
        "--all-universes",
        action="store_true",
        help="Check all universe files in config/",
    )

    args = parser.parse_args()

    # Load settings
    settings = Settings()
    target_root = Path(args.target_root) if args.target_root else Path(settings.local_data_root)

    if args.all_universes:
        # Check all universe files
        config_dir = Path("config")
        universe_files = list(config_dir.glob("*_tickers.txt"))
        
        if not universe_files:
            print("❌ No universe files found in config/")
            sys.exit(1)

        print(f"Found {len(universe_files)} universe files\n")
        
        for universe_file in sorted(universe_files):
            print(f"\n{'='*80}")
            print(f"Checking: {universe_file.name}")
            print(f"{'='*80}\n")
            
            report = check_universe_completeness(
                universe_file,
                target_root,
                interval=args.interval,
                expected_start=args.expected_start,
                expected_end=args.expected_end,
            )
            print_completeness_report(report)
    elif args.universe_file:
        # Check single universe
        universe_file = Path(args.universe_file)
        report = check_universe_completeness(
            universe_file,
            target_root,
            interval=args.interval,
            expected_start=args.expected_start,
            expected_end=args.expected_end,
        )
        print_completeness_report(report)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

