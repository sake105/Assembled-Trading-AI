"""Health Check Core Module.

This module provides core data structures and utilities for health checks,
including HealthCheck and HealthCheckResult dataclasses, status aggregation,
and report rendering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd

HealthCheckStatus = Literal["OK", "WARN", "CRITICAL", "SKIP"]


@dataclass
class HealthCheck:
    """Result of a single health check.

    Attributes:
        name: Check name (e.g., "max_drawdown", "factor_exposures_momentum_beta")
        status: Check status ("OK", "WARN", "CRITICAL", "SKIP")
        value: Actual value (float, int, str, or None)
        expected: Expected range or value as string (e.g., "[-0.25, -0.05]" for Drawdown)
        details: Optional human-readable details about the check
        last_updated_at: Optional timestamp when underlying data was last updated
    """

    name: str
    status: HealthCheckStatus
    value: float | int | str | None
    expected: str | None = None
    details: str | None = None
    last_updated_at: pd.Timestamp | None = None


@dataclass
class HealthCheckResult:
    """Result of a health check run.

    Attributes:
        overall_status: Overall status ("OK", "WARN", "CRITICAL", "SKIP")
        timestamp: Timestamp when check was run (UTC)
        checks: List of individual check results
        notes: Optional list of notes or messages
        meta: Optional metadata dictionary (e.g., lookback_days, backtest_dir)
    """

    overall_status: HealthCheckStatus
    timestamp: pd.Timestamp
    checks: list[HealthCheck] = field(default_factory=list)
    notes: list[str] | None = None
    meta: dict[str, Any] | None = None


def aggregate_overall_status(checks: list[HealthCheck]) -> HealthCheckStatus:
    """Aggregate overall status from list of checks.

    Rules:
    - If at least one check is CRITICAL, overall status is CRITICAL
    - Else, if at least one check is WARN, overall status is WARN
    - Else, if all checks are SKIP, overall status is SKIP
    - Else, overall status is OK

    Args:
        checks: List of HealthCheck instances

    Returns:
        Aggregated status ("OK", "WARN", "CRITICAL", or "SKIP")
    """
    if not checks:
        return "OK"

    statuses = [check.status for check in checks]

    # Priority: CRITICAL > WARN > OK > SKIP
    if "CRITICAL" in statuses:
        return "CRITICAL"
    elif "WARN" in statuses:
        return "WARN"
    elif all(status == "SKIP" for status in statuses):
        return "SKIP"
    else:
        return "OK"


def health_result_to_dict(result: HealthCheckResult) -> dict[str, Any]:
    """Convert HealthCheckResult to JSON-serializable dictionary.

    Args:
        result: HealthCheckResult instance

    Returns:
        Dictionary with ISO-formatted timestamps

    Note:
        Timestamps are converted to ISO format strings for JSON serialization.
    """
    # Convert checks to dicts
    checks_dicts = []
    for check in result.checks:
        check_dict = asdict(check)
        # Convert pd.Timestamp to ISO string if present
        if check_dict.get("last_updated_at") is not None:
            check_dict["last_updated_at"] = check.last_updated_at.isoformat()
        checks_dicts.append(check_dict)

    # Convert result timestamp to ISO string
    result_dict = {
        "overall_status": result.overall_status,
        "timestamp": result.timestamp.isoformat(),
        "checks": checks_dicts,
    }

    if result.notes is not None:
        result_dict["notes"] = result.notes

    if result.meta is not None:
        result_dict["meta"] = result.meta

    return result_dict


def health_result_from_dict(data: dict[str, Any]) -> HealthCheckResult:
    """Create HealthCheckResult from dictionary (for re-loads).

    Args:
        data: Dictionary with HealthCheckResult data (from JSON or health_result_to_dict)

    Returns:
        HealthCheckResult instance

    Note:
        Timestamps are parsed from ISO format strings.
    """
    # Parse timestamp
    timestamp = pd.Timestamp(data["timestamp"])

    # Parse checks
    checks = []
    for check_dict in data.get("checks", []):
        # Parse last_updated_at if present
        last_updated_at = None
        if check_dict.get("last_updated_at") is not None:
            last_updated_at = pd.Timestamp(check_dict["last_updated_at"])

        check = HealthCheck(
            name=check_dict["name"],
            status=check_dict["status"],
            value=check_dict.get("value"),
            expected=check_dict.get("expected"),
            details=check_dict.get("details"),
            last_updated_at=last_updated_at,
        )
        checks.append(check)

    # Create result
    result = HealthCheckResult(
        overall_status=data["overall_status"],
        timestamp=timestamp,
        checks=checks,
        notes=data.get("notes"),
        meta=data.get("meta"),
    )

    return result


def render_health_summary_text(result: HealthCheckResult) -> str:
    """Render health check result as human-readable text summary.

    Args:
        result: HealthCheckResult instance

    Returns:
        Formatted text string with header, status, and check details
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("Health Check Summary")
    lines.append("=" * 60)
    lines.append("")

    # Overall status
    status_symbol = {
        "OK": "[OK]",
        "WARN": "[WARN]",
        "CRITICAL": "[CRITICAL]",
        "SKIP": "[SKIP]",
    }.get(result.overall_status, "[?]")

    lines.append(f"Overall Status: {status_symbol} {result.overall_status}")
    lines.append(f"Timestamp: {result.timestamp.isoformat()}")
    lines.append("")

    # Summary statistics
    status_counts = {}
    for check in result.checks:
        status_counts[check.status] = status_counts.get(check.status, 0) + 1

    if status_counts:
        lines.append("Summary:")
        for status in ["OK", "WARN", "CRITICAL", "SKIP"]:
            count = status_counts.get(status, 0)
            if count > 0:
                lines.append(f"  {status}: {count} checks")
        lines.append("")

    # Metadata (if present)
    if result.meta:
        lines.append("Metadata:")
        for key, value in result.meta.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

    # Checks
    if result.checks:
        lines.append("Checks:")
        lines.append("")

        for check in result.checks:
            status_symbol_check = {
                "OK": "[OK]",
                "WARN": "[WARN]",
                "CRITICAL": "[CRITICAL]",
                "SKIP": "[SKIP]",
            }.get(check.status, "[?]")

            lines.append(f"  {status_symbol_check} {check.name}")

            if check.value is not None:
                value_str = str(check.value)
                if isinstance(check.value, float):
                    value_str = f"{check.value:.4f}"
                lines.append(f"    Value: {value_str}")

            if check.expected:
                lines.append(f"    Expected: {check.expected}")

            if check.details:
                lines.append(f"    Details: {check.details}")

            if check.last_updated_at is not None:
                lines.append(f"    Last Updated: {check.last_updated_at.isoformat()}")

            lines.append("")
    else:
        lines.append("No checks performed.")
        lines.append("")

    # Notes (if present)
    if result.notes:
        lines.append("Notes:")
        for note in result.notes:
            lines.append(f"  - {note}")
        lines.append("")

    return "\n".join(lines)
