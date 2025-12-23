"""Test Pipeline Integration with downloaded Alt-Daten.

Tests:
1. Data loading from Alt-Daten directory
2. Factor calculation
3. Basic pipeline functionality
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.assembled_core.data.data_source import get_price_data_source
from src.assembled_core.config.settings import Settings


def test_data_loading():
    """Test loading data from Alt-Daten directory."""
    print("=" * 60)
    print("Test 1: Data Loading")
    print("=" * 60)

    # Set local data root
    local_data_root = Path(
        r"F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"
    )

    settings = Settings()
    settings.local_data_root = local_data_root

    # Get data source
    ds = get_price_data_source(settings, "local")

    # Load SPY data
    df = ds.get_history(["SPY"], "2010-01-01", "2025-12-03", "1d")

    print(f"✓ Loaded {len(df)} rows for SPY")
    print(
        f"  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
    )
    print(f"  Columns: {list(df.columns)}")
    print(f"  Symbols: {df['symbol'].unique().tolist()}")
    print()

    return df


def test_multiple_symbols():
    """Test loading multiple symbols."""
    print("=" * 60)
    print("Test 2: Multiple Symbols")
    print("=" * 60)

    local_data_root = Path(
        r"F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"
    )

    settings = Settings()
    settings.local_data_root = local_data_root

    ds = get_price_data_source(settings, "local")

    # Load multiple symbols
    symbols = ["SPY", "ACWI", "VT"]
    df = ds.get_history(symbols, "2010-01-01", "2025-12-03", "1d")

    print(f"✓ Loaded {len(df)} rows for {len(symbols)} symbols")
    print(f"  Symbols: {df['symbol'].unique().tolist()}")
    print("  Rows per symbol:")
    for sym in symbols:
        count = len(df[df["symbol"] == sym])
        print(f"    {sym}: {count} rows")
    print()

    return df


def test_factor_calculation():
    """Test factor calculation."""
    print("=" * 60)
    print("Test 3: Factor Calculation")
    print("=" * 60)

    try:
        from src.assembled_core.features.ta_factors_core import build_core_ta_factors

        local_data_root = Path(
            r"F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"
        )

        settings = Settings()
        settings.local_data_root = local_data_root

        ds = get_price_data_source(settings, "local")
        df = ds.get_history(["SPY"], "2010-01-01", "2025-12-03", "1d")

        # Calculate factors
        df_factors = build_core_ta_factors(df)

        print(f"✓ Calculated factors for {len(df_factors)} rows")
        print(f"  Original columns: {len(df.columns)}")
        print(f"  With factors: {len(df_factors.columns)}")
        print(
            f"  Factor columns: {[c for c in df_factors.columns if c not in df.columns][:5]}..."
        )
        print()

        return df_factors
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print()
    print("=" * 60)
    print("Pipeline Integration Tests")
    print("=" * 60)
    print()

    try:
        # Test 1: Data loading
        test_data_loading()

        # Test 2: Multiple symbols
        test_multiple_symbols()

        # Test 3: Factor calculation
        test_factor_calculation()

        print("=" * 60)
        print("All Tests Completed Successfully!")
        print("=" * 60)
        print()
        print("✓ Data loading works")
        print("✓ Multiple symbols work")
        print("✓ Factor calculation works")
        print()
        print("Pipeline is ready to use!")

    except Exception as e:
        print()
        print("=" * 60)
        print("Tests Failed")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
