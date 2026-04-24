"""Operations & Monitoring Module.

This module provides operations and monitoring capabilities for the backend,
including health checks, status reporting, and operational insights.
"""

from src.assembled_core.ops.health_check import (
    HealthCheck,
    HealthCheckResult,
    HealthCheckStatus,
    aggregate_overall_status,
    health_result_from_dict,
    health_result_to_dict,
    render_health_summary_text,
)
from src.assembled_core.ops.daily_scheduler import (
    DailyScheduler,
    WorkerResult,
    run_daily_cycle,
    build_cycle_summary,
)

# Wired 2026-04-22: previously orphan ops modules
from src.assembled_core.ops.compare import compare_summaries  # noqa: F401
from src.assembled_core.ops.heartbeat import (  # noqa: F401
    check_liveness,
    heartbeat_age_seconds,
    read_heartbeat,
    write_heartbeat,
)
from src.assembled_core.ops.inspect_data import inspect_eod_prices  # noqa: F401
from src.assembled_core.ops.intel_activity_summary import build_intel_activity_summary  # noqa: F401

__all__ = [
    "HealthCheck",
    "HealthCheckResult",
    "HealthCheckStatus",
    "aggregate_overall_status",
    "health_result_from_dict",
    "health_result_to_dict",
    "render_health_summary_text",
    "DailyScheduler",
    "WorkerResult",
    "run_daily_cycle",
    "build_cycle_summary",
    "AlertSink",
    "EmailSink",
    "SlackWebhookSink",
    "dispatch_alerts",
    "GrafanaPanel",
    "portfolio_performance_dashboard",
    "risk_metrics_dashboard",
    "execution_quality_dashboard",
    "system_health_dashboard",
    "export_all_dashboards",
]
