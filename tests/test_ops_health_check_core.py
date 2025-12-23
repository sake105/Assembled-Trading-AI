"""Tests for Health Check Core Module.

Tests the core health check data structures and utilities:
HealthCheck, HealthCheckResult, status aggregation, serialization, and rendering.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.ops.health_check import (
    HealthCheck,
    HealthCheckResult,
    aggregate_overall_status,
    health_result_from_dict,
    health_result_to_dict,
    render_health_summary_text,
)


@pytest.fixture
def sample_checks() -> list[HealthCheck]:
    """Create sample health checks for testing."""
    timestamp = pd.Timestamp("2025-01-15 10:00:00", tz="UTC")

    return [
        HealthCheck(
            name="max_drawdown",
            status="OK",
            value=-0.125,
            expected="[-0.25, -0.05]",
            details="Within expected range",
            last_updated_at=timestamp,
        ),
        HealthCheck(
            name="daily_volatility",
            status="WARN",
            value=0.08,
            expected="[0.01, 0.05]",
            details="Above expected range",
            last_updated_at=timestamp,
        ),
        HealthCheck(
            name="sharpe_ratio",
            status="OK",
            value=1.25,
            expected="[0.5, 3.0]",
            details="Within expected range",
            last_updated_at=timestamp,
        ),
    ]


@pytest.mark.advanced
def test_aggregate_overall_status_priority_order():
    """Test that aggregate_overall_status respects priority order: CRITICAL > WARN > OK > SKIP."""
    # Test 1: CRITICAL has highest priority
    checks_critical = [
        HealthCheck(name="check1", status="OK", value=1.0),
        HealthCheck(name="check2", status="WARN", value=2.0),
        HealthCheck(name="check3", status="CRITICAL", value=3.0),
    ]
    assert aggregate_overall_status(checks_critical) == "CRITICAL"

    # Test 2: WARN takes precedence over OK
    checks_warn = [
        HealthCheck(name="check1", status="OK", value=1.0),
        HealthCheck(name="check2", status="WARN", value=2.0),
        HealthCheck(name="check3", status="OK", value=3.0),
    ]
    assert aggregate_overall_status(checks_warn) == "WARN"

    # Test 3: All OK returns OK
    checks_ok = [
        HealthCheck(name="check1", status="OK", value=1.0),
        HealthCheck(name="check2", status="OK", value=2.0),
    ]
    assert aggregate_overall_status(checks_ok) == "OK"

    # Test 4: All SKIP returns SKIP
    checks_skip = [
        HealthCheck(name="check1", status="SKIP", value=None),
        HealthCheck(name="check2", status="SKIP", value=None),
    ]
    assert aggregate_overall_status(checks_skip) == "SKIP"

    # Test 5: Empty list returns OK
    assert aggregate_overall_status([]) == "OK"

    # Test 6: Mix of OK and SKIP returns OK (not SKIP)
    checks_mixed = [
        HealthCheck(name="check1", status="OK", value=1.0),
        HealthCheck(name="check2", status="SKIP", value=None),
    ]
    assert aggregate_overall_status(checks_mixed) == "OK"


@pytest.mark.advanced
def test_health_result_roundtrip_dict():
    """Test that health_result_to_dict and health_result_from_dict are inverse operations."""
    timestamp = pd.Timestamp("2025-01-15 10:00:00", tz="UTC")
    check_timestamp = pd.Timestamp("2025-01-14 15:30:00", tz="UTC")

    original = HealthCheckResult(
        overall_status="WARN",
        timestamp=timestamp,
        checks=[
            HealthCheck(
                name="max_drawdown",
                status="OK",
                value=-0.125,
                expected="[-0.25, -0.05]",
                details="Within expected range",
                last_updated_at=check_timestamp,
            ),
            HealthCheck(
                name="daily_volatility",
                status="WARN",
                value=0.08,
                expected="[0.01, 0.05]",
                details="Above expected range",
                last_updated_at=None,
            ),
        ],
        notes=["Test note 1", "Test note 2"],
        meta={"lookback_days": 60, "backtest_dir": "output/backtests/experiment_123"},
    )

    # Convert to dict
    result_dict = health_result_to_dict(original)

    # Verify it's JSON-serializable
    json_str = json.dumps(result_dict)
    assert isinstance(json_str, str)

    # Convert back from dict
    restored = health_result_from_dict(result_dict)

    # Verify all fields match
    assert restored.overall_status == original.overall_status
    assert restored.timestamp == original.timestamp
    assert len(restored.checks) == len(original.checks)
    assert restored.notes == original.notes
    assert restored.meta == original.meta

    # Verify check details
    for i, (restored_check, original_check) in enumerate(
        zip(restored.checks, original.checks)
    ):
        assert restored_check.name == original_check.name
        assert restored_check.status == original_check.status
        assert restored_check.value == original_check.value
        assert restored_check.expected == original_check.expected
        assert restored_check.details == original_check.details
        if original_check.last_updated_at is not None:
            assert restored_check.last_updated_at == original_check.last_updated_at
        else:
            assert restored_check.last_updated_at is None


@pytest.mark.advanced
def test_render_health_summary_text_contains_status_and_names():
    """Test that render_health_summary_text includes overall status and all check names."""
    timestamp = pd.Timestamp("2025-01-15 10:00:00", tz="UTC")

    result = HealthCheckResult(
        overall_status="WARN",
        timestamp=timestamp,
        checks=[
            HealthCheck(
                name="max_drawdown",
                status="OK",
                value=-0.125,
                expected="[-0.25, -0.05]",
            ),
            HealthCheck(
                name="daily_volatility",
                status="WARN",
                value=0.08,
                expected="[0.01, 0.05]",
            ),
        ],
        notes=["Test note"],
        meta={"lookback_days": 60},
    )

    text = render_health_summary_text(result)

    # Check header
    assert "Health Check Summary" in text

    # Check overall status
    assert "Overall Status:" in text
    assert "[WARN]" in text
    assert "WARN" in text

    # Check timestamp
    assert "Timestamp:" in text
    assert "2025-01-15" in text

    # Check summary statistics
    assert "Summary:" in text
    assert "OK: 1 checks" in text
    assert "WARN: 1 checks" in text

    # Check metadata
    assert "Metadata:" in text
    assert "lookback_days: 60" in text

    # Check check names
    assert "max_drawdown" in text
    assert "daily_volatility" in text

    # Check notes
    assert "Notes:" in text
    assert "Test note" in text


@pytest.mark.advanced
def test_render_health_summary_text_empty_checks():
    """Test rendering with empty checks list."""
    timestamp = pd.Timestamp("2025-01-15 10:00:00", tz="UTC")

    result = HealthCheckResult(
        overall_status="OK",
        timestamp=timestamp,
        checks=[],
    )

    text = render_health_summary_text(result)

    assert "Overall Status:" in text
    assert "[OK]" in text
    assert "No checks performed." in text


@pytest.mark.advanced
def test_health_check_with_none_values():
    """Test HealthCheck with None values."""
    check = HealthCheck(
        name="missing_file",
        status="SKIP",
        value=None,
        expected=None,
        details="File not found",
        last_updated_at=None,
    )

    assert check.name == "missing_file"
    assert check.status == "SKIP"
    assert check.value is None
    assert check.expected is None
    assert check.details == "File not found"
    assert check.last_updated_at is None


@pytest.mark.advanced
def test_health_result_to_dict_handles_none_timestamps():
    """Test that health_result_to_dict handles None timestamps correctly."""
    timestamp = pd.Timestamp("2025-01-15 10:00:00", tz="UTC")

    result = HealthCheckResult(
        overall_status="OK",
        timestamp=timestamp,
        checks=[
            HealthCheck(
                name="check1",
                status="OK",
                value=1.0,
                last_updated_at=None,
            ),
            HealthCheck(
                name="check2",
                status="OK",
                value=2.0,
                last_updated_at=timestamp,
            ),
        ],
    )

    result_dict = health_result_to_dict(result)

    # Verify timestamp is ISO string
    assert isinstance(result_dict["timestamp"], str)
    assert "2025-01-15" in result_dict["timestamp"]

    # Verify check with None timestamp
    assert result_dict["checks"][0]["last_updated_at"] is None

    # Verify check with timestamp
    assert isinstance(result_dict["checks"][1]["last_updated_at"], str)
    assert "2025-01-15" in result_dict["checks"][1]["last_updated_at"]

    # Verify JSON serialization works
    json_str = json.dumps(result_dict)
    assert isinstance(json_str, str)


@pytest.mark.advanced
def test_health_result_from_dict_handles_missing_fields():
    """Test that health_result_from_dict handles missing optional fields."""
    data = {
        "overall_status": "OK",
        "timestamp": "2025-01-15T10:00:00Z",
        "checks": [
            {
                "name": "check1",
                "status": "OK",
                "value": 1.0,
            },
        ],
    }

    result = health_result_from_dict(data)

    assert result.overall_status == "OK"
    assert isinstance(result.timestamp, pd.Timestamp)
    assert len(result.checks) == 1
    assert result.checks[0].name == "check1"
    assert result.notes is None
    assert result.meta is None
