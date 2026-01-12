# tests/test_data_quality_qc.py
"""Tests for Data Quality Control (QC) module.

This test suite verifies:
1. Invalid prices (negative, zero, NaN) -> FAIL
2. Duplicate rows -> FAIL
3. Missing sessions -> WARN/FAIL (with Calendar)
4. Outlier returns -> WARN/FAIL
5. Stale prices -> WARN
6. Zero volume anomalies -> WARN/FAIL
7. Timezone normalization (UTC)
8. Deterministic issue ordering
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.data_qc import (
    run_price_panel_qc,
    write_qc_report_json,
    write_qc_summary_md,
)


def test_negative_price_fails() -> None:
    """Test that negative prices result in FAIL issues."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "close": [150.0, -10.0, 152.0, 0.0, 154.0],  # Negative and zero prices
    })

    report = run_price_panel_qc(prices, freq="1d")

    assert not report.ok, "Report should not be OK (has FAIL issues)"
    assert report.summary["fail_count"] >= 2, "Should have at least 2 FAIL issues (negative and zero)"

    # Check for negative_price issues
    negative_issues = [issue for issue in report.issues if issue.check == "negative_price"]
    assert len(negative_issues) >= 2, "Should have at least 2 negative_price issues"

    # Verify severity
    for issue in negative_issues:
        assert issue.severity == "FAIL", "Negative prices should be FAIL"


def test_duplicate_symbol_timestamp_fails() -> None:
    """Test that duplicate (symbol,timestamp) rows result in FAIL issues."""
    prices = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),  # Duplicate
            pd.Timestamp("2024-01-02", tz="UTC"),  # Duplicate
        ],
        "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
        "close": [150.0, 151.0, 150.5, 151.5],
    })

    report = run_price_panel_qc(prices, freq="1d")

    assert not report.ok, "Report should not be OK (has FAIL issues)"
    assert report.summary["fail_count"] >= 2, "Should have at least 2 FAIL issues (duplicates)"

    # Check for duplicate_rows issues
    duplicate_issues = [issue for issue in report.issues if issue.check == "duplicate_rows"]
    assert len(duplicate_issues) >= 2, "Should have at least 2 duplicate_rows issues"

    # Verify severity
    for issue in duplicate_issues:
        assert issue.severity == "FAIL", "Duplicate rows should be FAIL"


def test_missing_sessions_warns_or_fails() -> None:
    """Test that missing trading sessions result in WARN/FAIL issues (with Calendar)."""
    try:
        from src.assembled_core.data.calendar import trading_sessions
    except ImportError:
        pytest.skip("exchange_calendars not installed")

    # Create prices with missing sessions (skip some trading days)
    # Use actual trading days from calendar
    start_date = pd.Timestamp("2024-01-02", tz="UTC").date()  # Tuesday (trading day)
    end_date = pd.Timestamp("2024-01-31", tz="UTC").date()

    expected_sessions = trading_sessions(start_date, end_date)
    if expected_sessions.empty:
        pytest.skip("No trading sessions found in date range")

    # Take only every other trading day (creates gaps)
    actual_sessions = expected_sessions[::2]

    prices = pd.DataFrame({
        "timestamp": pd.to_datetime(actual_sessions, utc=True),
        "symbol": ["AAPL"] * len(actual_sessions),
        "close": [150.0] * len(actual_sessions),
    })

    report = run_price_panel_qc(prices, freq="1d")

    # Should have missing_sessions issues
    missing_issues = [issue for issue in report.issues if issue.check == "missing_sessions"]
    assert len(missing_issues) > 0, "Should have missing_sessions issues"

    # Verify severity (should be WARN or FAIL depending on threshold)
    for issue in missing_issues:
        assert issue.severity in ["WARN", "FAIL"], "Missing sessions should be WARN or FAIL"
        assert issue.symbol == "AAPL", "Symbol should be AAPL"


def test_outlier_return_flagged() -> None:
    """Test that outlier returns are flagged as WARN/FAIL."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "close": [150.0, 180.0, 152.0, 200.0, 154.0],  # 20% and 33% returns
    })

    report = run_price_panel_qc(prices, freq="1d")

    # Should have outlier_return issues
    outlier_issues = [issue for issue in report.issues if issue.check == "outlier_return"]
    assert len(outlier_issues) >= 2, "Should have at least 2 outlier_return issues"

    # Verify severity (20% -> WARN, 33% -> FAIL)
    warn_issues = [issue for issue in outlier_issues if issue.severity == "WARN"]
    fail_issues = [issue for issue in outlier_issues if issue.severity == "FAIL"]
    assert len(warn_issues) >= 1, "Should have at least 1 WARN issue (20% return)"
    assert len(fail_issues) >= 1, "Should have at least 1 FAIL issue (33% return)"


def test_stale_price_flagged() -> None:
    """Test that stale prices (unchanged for >=3 sessions) are flagged as WARN."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "close": [150.0, 150.0, 150.0, 150.0, 152.0, 152.0, 152.0, 152.0, 154.0, 154.0],  # 4x 150, 4x 152, 2x 154
    })

    report = run_price_panel_qc(prices, freq="1d")

    # Should have stale_price issues
    stale_issues = [issue for issue in report.issues if issue.check == "stale_price"]
    assert len(stale_issues) >= 2, "Should have at least 2 stale_price issues (4x 150, 4x 152)"

    # Verify severity
    for issue in stale_issues:
        assert issue.severity == "WARN", "Stale prices should be WARN"
        assert issue.symbol == "AAPL", "Symbol should be AAPL"


def test_zero_volume_anomalies() -> None:
    """Test that zero volume anomalies are flagged as WARN/FAIL."""
    # Create prices with many zero volume days (>10%)
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 20,
        "close": [150.0] * 20,
        "volume": [1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0] + [1000.0] * 9,  # 10/20 = 50% zero
    })

    report = run_price_panel_qc(prices, freq="1d")

    # Should have zero_volume issues
    zero_volume_issues = [issue for issue in report.issues if issue.check == "zero_volume"]
    assert len(zero_volume_issues) > 0, "Should have zero_volume issues"

    # Verify severity (50% -> FAIL)
    for issue in zero_volume_issues:
        assert issue.severity == "FAIL", "50% zero volume should be FAIL"
        assert issue.symbol == "AAPL", "Symbol should be AAPL"


def test_timezone_normalization_utc() -> None:
    """Test that timestamps are normalized to UTC (naive and tz-aware input)."""
    # Test with naive timestamps
    prices_naive = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d"),
        "symbol": ["AAPL"] * 5,
        "close": [150.0] * 5,
    })

    report_naive = run_price_panel_qc(prices_naive, freq="1d")

    # Check that issues have UTC timestamps
    for issue in report_naive.issues:
        if issue.timestamp is not None:
            assert issue.timestamp.tz is not None, "Timestamp should be timezone-aware"
            assert issue.timestamp.tz.zone == "UTC", "Timestamp should be UTC"

    # Test with timezone-aware timestamps (non-UTC)
    prices_tz = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="America/New_York"),
        "symbol": ["AAPL"] * 5,
        "close": [150.0] * 5,
    })

    report_tz = run_price_panel_qc(prices_tz, freq="1d")

    # Check that issues have UTC timestamps
    for issue in report_tz.issues:
        if issue.timestamp is not None:
            assert issue.timestamp.tz is not None, "Timestamp should be timezone-aware"
            assert issue.timestamp.tz.zone == "UTC", "Timestamp should be UTC (converted from ET)"


def test_deterministic_issue_ordering() -> None:
    """Test that issues are sorted deterministically."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT", "GOOGL"] * 3 + ["AAPL"],
        "close": [150.0, -10.0, 200.0, 151.0, 152.0, 201.0, 152.0, 152.0, 152.0, 153.0],  # Mix of issues
    })

    report = run_price_panel_qc(prices, freq="1d")

    # Verify ordering: FAIL before WARN
    issues = report.issues
    fail_indices = [i for i, issue in enumerate(issues) if issue.severity == "FAIL"]
    warn_indices = [i for i, issue in enumerate(issues) if issue.severity == "WARN"]

    if fail_indices and warn_indices:
        assert max(fail_indices) < min(warn_indices), "FAIL issues should come before WARN issues"

    # Verify ordering within same severity: check name, then symbol, then timestamp
    for i in range(len(issues) - 1):
        issue1 = issues[i]
        issue2 = issues[i + 1]

        if issue1.severity == issue2.severity:
            # Same severity: check name
            if issue1.check != issue2.check:
                assert issue1.check < issue2.check, "Issues should be sorted by check name"
            elif issue1.symbol != issue2.symbol:
                # Same check: check symbol
                sym1 = issue1.symbol or ""
                sym2 = issue2.symbol or ""
                assert sym1 <= sym2, "Issues should be sorted by symbol"
            elif issue1.timestamp is not None and issue2.timestamp is not None:
                # Same check and symbol: check timestamp
                assert issue1.timestamp <= issue2.timestamp, "Issues should be sorted by timestamp"


def test_qc_report_json_serialization(tmp_path: Path) -> None:
    """Test that QC report can be serialized to JSON."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "close": [150.0] * 5,
    })

    report = run_price_panel_qc(prices, freq="1d")

    # Write to JSON
    json_path = tmp_path / "qc_report.json"
    write_qc_report_json(report, json_path)

    # Verify file exists and is valid JSON
    assert json_path.exists(), "JSON file should exist"
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "ok" in data, "JSON should contain 'ok' key"
        assert "summary" in data, "JSON should contain 'summary' key"
        assert "issues" in data, "JSON should contain 'issues' key"
        assert "created_at_utc" in data, "JSON should contain 'created_at_utc' key"


def test_qc_summary_md_ascii_only(tmp_path: Path) -> None:
    """Test that QC summary Markdown is ASCII-only."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "close": [150.0] * 5,
    })

    report = run_price_panel_qc(prices, freq="1d")

    # Write to Markdown
    md_path = tmp_path / "qc_summary.md"
    write_qc_summary_md(report, md_path)

    # Verify file exists and is ASCII-only
    assert md_path.exists(), "Markdown file should exist"
    content = md_path.read_bytes()
    assert all(b < 128 for b in content), "Markdown file should be ASCII-only"


def test_valid_prices_no_issues() -> None:
    """Test that valid prices result in no issues (OK report)."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "close": [150.0 + i * 0.5 for i in range(10)],  # Valid, increasing prices
        "volume": [1000.0] * 10,  # Valid volume
    })

    report = run_price_panel_qc(prices, freq="1d")

    # Should be OK (no FAIL issues)
    # Note: May have WARN issues (e.g., missing_sessions if calendar check runs)
    # But should have no FAIL issues for valid prices
    fail_issues = [issue for issue in report.issues if issue.severity == "FAIL"]
    assert len(fail_issues) == 0, "Valid prices should have no FAIL issues"
