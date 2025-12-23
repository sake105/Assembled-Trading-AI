"""End-to-End Smoketests for Factor Analysis Pipeline.

This script runs three comprehensive end-to-end tests to verify that all components
of the factor analysis pipeline work correctly together:

1. Plain Core-Factors (TA/Price only)
2. Core + Alt B1 (Earnings/Insider)
3. Core + Alt Full (B1 + B2: Earnings/Insider + News/Macro)

All tests use local Parquet data only (data_source="local") and do not make any
direct API calls.

Usage:
    python scripts/run_factor_analysis_smoketests.py

Prerequisites:
    - Local alt-data available (ASSEMBLED_LOCAL_DATA_ROOT environment variable set)
    - Universe files: config/macro_world_etfs_tickers.txt, config/universe_ai_tech_tickers.txt
    - For tests 2 & 3: Alt-data files in output/altdata/ (download via download scripts first)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
# Script is in scripts/, so parents[1] gives us the repo root
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "scripts" / "cli.py").exists():
    # Fallback: try current working directory
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


def run_analyze_factors(
    freq: str,
    symbols_file: str,
    start_date: str,
    end_date: str,
    factor_set: str,
    horizon_days: int,
    data_source: str = "local",
    output_dir: str | None = None,
) -> tuple[int, list[str]]:
    """Run analyze_factors CLI command and return exit code and output.

    Args:
        freq: Frequency string (e.g., "1d")
        symbols_file: Path to symbols file
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        factor_set: Factor set identifier
        horizon_days: Forward return horizon in days
        data_source: Data source type (default: "local")
        output_dir: Optional output directory

    Returns:
        Tuple of (exit_code, output_lines)
    """
    cmd = [
        sys.executable,
        "scripts/cli.py",
        "analyze_factors",
        "--freq",
        freq,
        "--symbols-file",
        symbols_file,
        "--data-source",
        data_source,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--factor-set",
        factor_set,
        "--horizon-days",
        str(horizon_days),
    ]

    if output_dir:
        cmd.extend(["--output-dir", output_dir])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )

        # Combine stdout and stderr
        output_lines = []
        if result.stdout:
            output_lines.extend(result.stdout.strip().split("\n"))
        if result.stderr:
            output_lines.extend(result.stderr.strip().split("\n"))

        return result.returncode, output_lines

    except Exception as e:
        return 1, [f"Error running command: {e}"]


def check_report_files(
    factor_set: str, horizon_days: int, freq: str, output_dir: Path | None = None
) -> dict[str, bool]:
    """Check if expected report files were created.

    Args:
        factor_set: Factor set identifier
        horizon_days: Horizon days
        freq: Frequency string
        output_dir: Optional output directory (default: output/factor_analysis)

    Returns:
        Dictionary mapping file names to existence (bool)
    """
    if output_dir is None:
        output_dir = ROOT / "output" / "factor_analysis"
    else:
        output_dir = Path(output_dir)

    base_name = f"factor_analysis_{factor_set}_{horizon_days}d_{freq}"

    expected_files = {
        "ic_summary": output_dir / f"{base_name}_ic_summary.csv",
        "rank_ic_summary": output_dir / f"{base_name}_rank_ic_summary.csv",
        "portfolio_summary": output_dir / f"{base_name}_portfolio_summary.csv",
        "report": output_dir / f"{base_name}_report.md",
    }

    return {name: file_path.exists() for name, file_path in expected_files.items()}


def main() -> int:
    """Run all smoketests and report results."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end smoketests for factor analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-test-1",
        action="store_true",
        help="Skip test 1 (Plain Core-Factors)",
    )
    parser.add_argument(
        "--skip-test-2",
        action="store_true",
        help="Skip test 2 (Core + Alt B1)",
    )
    parser.add_argument(
        "--skip-test-3",
        action="store_true",
        help="Skip test 3 (Core + Alt Full)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for reports (default: output/factor_analysis)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Factor Analysis Pipeline - End-to-End Smoketests")
    print("=" * 80)
    print()

    # Check prerequisites
    required_universe_files = [
        "config/macro_world_etfs_tickers.txt",
        "config/universe_ai_tech_tickers.txt",
    ]

    missing_files = []
    for universe_file in required_universe_files:
        if not (ROOT / universe_file).exists():
            missing_files.append(universe_file)

    if missing_files:
        print(f"❌ ERROR: Missing required universe files: {', '.join(missing_files)}")
        return 1

    # Check ASSEMBLED_LOCAL_DATA_ROOT
    import os

    local_data_root = os.environ.get("ASSEMBLED_LOCAL_DATA_ROOT")
    if not local_data_root:
        print("⚠️  WARNING: ASSEMBLED_LOCAL_DATA_ROOT environment variable not set.")
        print("   The tests may fail if local data is not available.")
        print()
    else:
        print(f"✅ Local data root: {local_data_root}")
        print()

    results = []

    # Test 1: Plain Core-Factors
    if not args.skip_test_1:
        print("=" * 80)
        print("TEST 1: Plain Core-Factors")
        print("=" * 80)
        print("Universe: config/macro_world_etfs_tickers.txt")
        print("Factor-Set: core")
        print("Date Range: 2010-01-01 to 2025-12-03")
        print()

        exit_code, output = run_analyze_factors(
            freq="1d",
            symbols_file="config/macro_world_etfs_tickers.txt",
            start_date="2010-01-01",
            end_date="2025-12-03",
            factor_set="core",
            horizon_days=20,
            data_source="local",
            output_dir=args.output_dir,
        )

        report_files = check_report_files("core", 20, "1d", args.output_dir)

        if exit_code == 0:
            print("✅ Test 1 PASSED")
            print(
                f"   Reports generated: {sum(report_files.values())}/{len(report_files)}"
            )
            for name, exists in report_files.items():
                status = "✓" if exists else "✗"
                print(f"     {status} {name}")
        else:
            print("❌ Test 1 FAILED")
            print("   Last output lines:")
            for line in output[-10:]:
                print(f"     {line}")

        results.append(("Test 1: Plain Core-Factors", exit_code == 0, report_files))
        print()

    # Test 2: Core + Alt B1 (Earnings/Insider)
    if not args.skip_test_2:
        print("=" * 80)
        print("TEST 2: Core + Alt B1 (Earnings/Insider)")
        print("=" * 80)
        print("Universe: config/universe_ai_tech_tickers.txt")
        print("Factor-Set: core+alt")
        print("Date Range: 2015-01-01 to 2025-12-03")
        print()
        print("⚠️  Note: This test requires alt-data files:")
        print("   - output/altdata/events_earnings.parquet")
        print("   - output/altdata/events_insider.parquet")
        print("   If missing, the test will continue but without alt-data factors.")
        print()

        exit_code, output = run_analyze_factors(
            freq="1d",
            symbols_file="config/universe_ai_tech_tickers.txt",
            start_date="2015-01-01",
            end_date="2025-12-03",
            factor_set="core+alt",
            horizon_days=20,
            data_source="local",
            output_dir=args.output_dir,
        )

        report_files = check_report_files("core+alt", 20, "1d", args.output_dir)

        if exit_code == 0:
            print("✅ Test 2 PASSED")
            print(
                f"   Reports generated: {sum(report_files.values())}/{len(report_files)}"
            )
            for name, exists in report_files.items():
                status = "✓" if exists else "✗"
                print(f"     {status} {name}")
        else:
            print("❌ Test 2 FAILED")
            print("   Last output lines:")
            for line in output[-10:]:
                print(f"     {line}")

        results.append(("Test 2: Core + Alt B1", exit_code == 0, report_files))
        print()

    # Test 3: Core + Alt Full (B1 + B2)
    if not args.skip_test_3:
        print("=" * 80)
        print("TEST 3: Core + Alt Full (B1 + B2: Earnings/Insider + News/Macro)")
        print("=" * 80)
        print("Universe: config/universe_ai_tech_tickers.txt")
        print("Factor-Set: core+alt_full")
        print("Date Range: 2015-01-01 to 2025-12-03")
        print()
        print("⚠️  Note: This test requires alt-data files:")
        print("   - output/altdata/events_earnings.parquet")
        print("   - output/altdata/events_insider.parquet")
        print("   - output/altdata/news_sentiment_daily.parquet")
        print("   - output/altdata/macro_series.parquet")
        print(
            "   If missing, the test will continue but without those alt-data factors."
        )
        print()

        exit_code, output = run_analyze_factors(
            freq="1d",
            symbols_file="config/universe_ai_tech_tickers.txt",
            start_date="2015-01-01",
            end_date="2025-12-03",
            factor_set="core+alt_full",
            horizon_days=20,
            data_source="local",
            output_dir=args.output_dir,
        )

        report_files = check_report_files("core+alt_full", 20, "1d", args.output_dir)

        if exit_code == 0:
            print("✅ Test 3 PASSED")
            print(
                f"   Reports generated: {sum(report_files.values())}/{len(report_files)}"
            )
            for name, exists in report_files.items():
                status = "✓" if exists else "✗"
                print(f"     {status} {name}")
        else:
            print("❌ Test 3 FAILED")
            print("   Last output lines:")
            for line in output[-10:]:
                print(f"     {line}")

        results.append(("Test 3: Core + Alt Full", exit_code == 0, report_files))
        print()

    # Final Summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()

    all_passed = all(result[1] for result in results)
    total_tests = len(results)
    passed_tests = sum(1 for result in results if result[1])

    print(f"Tests run: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {total_tests - passed_tests}")
    print()

    for test_name, passed, report_files in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        reports_count = sum(report_files.values())
        print(f"{status} {test_name} ({reports_count}/{len(report_files)} reports)")

    print()

    if all_passed:
        print(
            "🎉 All smoketests passed! The factor analysis pipeline is working correctly."
        )
        print()
        print("Next steps:")
        print("  - Review generated reports in output/factor_analysis/")
        print("  - Check IC summaries and portfolio summaries for factor rankings")
        print("  - Create a 'Top-Factors' list based on IC-IR and Sharpe ratios")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")
        print()
        print("Common issues:")
        print("  - Missing local data (check ASSEMBLED_LOCAL_DATA_ROOT)")
        print("  - Missing universe files")
        print("  - Missing alt-data files (for tests 2 & 3)")
        print("  - Insufficient data for the specified date range")
        return 1


if __name__ == "__main__":
    sys.exit(main())
