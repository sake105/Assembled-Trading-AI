"""Operations & Monitoring Module.

This module provides operations and monitoring capabilities for the backend,
including health checks, status reporting, and operational insights.
"""

# Wired 2026-04-22: previously orphan ops modules
# daily_scheduler-Reexport ENTFERNT 2026-08-17 (Audit-Plan 6.5): Modul nach
# archive/orphaned_code_2026-08-17/ops/ archiviert (Scheduler-Kette ohne
# Launcher; kein Paket-Import nutzte die Reexporte — repo-weit 0 Treffer).
from src.assembled_core.ops.compare import compare_summaries  # noqa: F401
from src.assembled_core.ops.health_check import (
    HealthCheck,
    HealthCheckResult,
    HealthCheckStatus,
    aggregate_overall_status,
    health_result_from_dict,
    health_result_to_dict,
    render_health_summary_text,
)
from src.assembled_core.ops.heartbeat import (  # noqa: F401
    check_liveness,
    heartbeat_age_seconds,
    read_heartbeat,
    write_heartbeat,
)
from src.assembled_core.ops.inspect_data import inspect_eod_prices  # noqa: F401
from src.assembled_core.ops.intel_activity_summary import (
    build_intel_activity_summary,  # noqa: F401
)

__all__ = [
    "HealthCheck",
    "HealthCheckResult",
    "HealthCheckStatus",
    "aggregate_overall_status",
    "health_result_from_dict",
    "health_result_to_dict",
    "render_health_summary_text",
]
