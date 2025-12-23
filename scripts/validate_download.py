"""Quick validation script for downloaded Parquet files."""

import sys
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_download.py <path_to_parquet>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"ERROR: File does not exist: {file_path}")
        sys.exit(1)

    try:
        df = pd.read_parquet(file_path)

        print("=" * 60)
        print("Download Validation Report")
        print("=" * 60)
        print(f"File: {file_path}")
        print(f"Size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
        print()
        print(f"Total rows: {len(df):,}")
        print(f"Columns: {list(df.columns)}")
        print()
        print("Date range:")
        print(f"  Start: {df['timestamp'].min()}")
        print(f"  End:   {df['timestamp'].max()}")
        print()
        print("Data quality:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("  Missing values:")
            for col, count in missing[missing > 0].items():
                print(f"    {col}: {count} ({count / len(df) * 100:.1f}%)")
        else:
            print("  No missing values ✓")
        print()
        print("Sample data (first 3 rows):")
        print(df.head(3).to_string())
        print()
        print("Sample data (last 3 rows):")
        print(df.tail(3).to_string())
        print()
        print("Statistics:")
        print(f"  Symbol: {df['symbol'].unique()}")
        if "close" in df.columns:
            print(
                f"  Close price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}"
            )
        if "volume" in df.columns:
            print(
                f"  Volume range: {df['volume'].min():,.0f} - {df['volume'].max():,.0f}"
            )
        print()
        print("=" * 60)
        print("✓ Validation complete - File looks good!")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: Failed to read or validate file: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
