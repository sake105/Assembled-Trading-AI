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
from src.assembled_core.ops.alert_sinks import (
    AlertSink,
    EmailSink,
    SlackWebhookSink,
    dispatch_alerts,
)
from src.assembled_core.ops.grafana_dashboards import (
    GrafanaPanel,
    execution_quality_dashboard,
    export_all_dashboards,
    portfolio_performance_dashboard,
    risk_metrics_dashboard,
    system_health_dashboard,
)
from src.assembled_core.ops.certification import (  # noqa: F401
    CertificationReport,
    CertificationRunner,
    CheckResult,
    build_default_runner,
    check_imports_ok,
    check_numpy_scipy,
)
from src.assembled_core.ops.compare import compare_summaries  # noqa: F401
from src.assembled_core.ops.dashboard_data import (  # noqa: F401
    DashboardSnapshot,
    build_pnl_curve,
    build_position_table,
    build_signal_heatmap,
    compute_exposure,
    compute_risk_snapshot,
)
from src.assembled_core.ops.heartbeat import (  # noqa: F401
    check_liveness,
    heartbeat_age_seconds,
    read_heartbeat,
    write_heartbeat,
)
from src.assembled_core.ops.inspect_data import inspect_eod_prices  # noqa: F401
from src.assembled_core.ops.intel_activity_summary import build_intel_activity_summary  # noqa: F401
from src.assembled_core.ops.self_healing import (  # noqa: F401
    DataSourceCascade,
    EscalationLevel,
    EscalationState,
    HealingAction,
    RiskEscalationLadder,
)

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
