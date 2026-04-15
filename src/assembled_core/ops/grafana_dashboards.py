"""Grafana Dashboard Definitions (M31 Task 31.3).

Generates Grafana-compatible JSON dashboard definitions for:
1. Portfolio performance overview
2. Risk metrics panel
3. Execution quality metrics
4. System health / pipeline status

These definitions can be imported into Grafana directly or served
via the dashboard_data API endpoint.

Reference: Grafana Dashboard JSON Model v9+
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GrafanaPanel:
    """Single Grafana panel configuration."""
    title: str
    panel_type: str   # "timeseries", "gauge", "stat", "table", "heatmap"
    targets: list[dict]
    grid_pos: dict     # {"x": 0, "y": 0, "w": 12, "h": 8}
    thresholds: list[dict] | None = None
    unit: str = ""
    description: str = ""


def _make_panel(panel_id: int, panel: GrafanaPanel) -> dict:
    """Convert GrafanaPanel to Grafana JSON panel dict."""
    p = {
        "id": panel_id,
        "title": panel.title,
        "type": panel.panel_type,
        "gridPos": panel.grid_pos,
        "targets": panel.targets,
        "description": panel.description,
        "fieldConfig": {
            "defaults": {
                "unit": panel.unit,
            },
            "overrides": [],
        },
        "options": {},
    }

    if panel.thresholds:
        p["fieldConfig"]["defaults"]["thresholds"] = {
            "mode": "absolute",
            "steps": panel.thresholds,
        }

    return p


def portfolio_performance_dashboard() -> dict:
    """Generate portfolio performance overview dashboard."""
    panels = [
        GrafanaPanel(
            title="Equity Curve",
            panel_type="timeseries",
            targets=[{"expr": "portfolio_equity", "legendFormat": "Equity"}],
            grid_pos={"x": 0, "y": 0, "w": 16, "h": 8},
            unit="currencyUSD",
            description="Portfolio equity over time",
        ),
        GrafanaPanel(
            title="Daily Returns",
            panel_type="timeseries",
            targets=[{"expr": "portfolio_daily_return", "legendFormat": "Return"}],
            grid_pos={"x": 16, "y": 0, "w": 8, "h": 8},
            unit="percentunit",
        ),
        GrafanaPanel(
            title="Sharpe Ratio (Rolling 252d)",
            panel_type="stat",
            targets=[{"expr": "portfolio_rolling_sharpe_252d", "legendFormat": "Sharpe"}],
            grid_pos={"x": 0, "y": 8, "w": 6, "h": 4},
            thresholds=[
                {"color": "red", "value": None},
                {"color": "yellow", "value": 0.5},
                {"color": "green", "value": 1.0},
            ],
        ),
        GrafanaPanel(
            title="Max Drawdown",
            panel_type="gauge",
            targets=[{"expr": "portfolio_max_drawdown", "legendFormat": "MaxDD"}],
            grid_pos={"x": 6, "y": 8, "w": 6, "h": 4},
            unit="percentunit",
            thresholds=[
                {"color": "green", "value": None},
                {"color": "yellow", "value": -0.10},
                {"color": "red", "value": -0.20},
            ],
        ),
        GrafanaPanel(
            title="Annualized Return",
            panel_type="stat",
            targets=[{"expr": "portfolio_annual_return", "legendFormat": "CAGR"}],
            grid_pos={"x": 12, "y": 8, "w": 6, "h": 4},
            unit="percentunit",
        ),
        GrafanaPanel(
            title="Win Rate",
            panel_type="stat",
            targets=[{"expr": "portfolio_win_rate", "legendFormat": "Win%"}],
            grid_pos={"x": 18, "y": 8, "w": 6, "h": 4},
            unit="percentunit",
        ),
        GrafanaPanel(
            title="Position Heatmap",
            panel_type="heatmap",
            targets=[{"expr": "portfolio_positions", "legendFormat": "{{ticker}}"}],
            grid_pos={"x": 0, "y": 12, "w": 24, "h": 8},
        ),
    ]

    return _build_dashboard("Portfolio Performance", "portfolio-perf", panels, tags=["portfolio", "performance"])


def risk_metrics_dashboard() -> dict:
    """Generate risk monitoring dashboard."""
    panels = [
        GrafanaPanel(
            title="Portfolio VaR (99%)",
            panel_type="timeseries",
            targets=[
                {"expr": "risk_var_99_normal", "legendFormat": "Normal VaR"},
                {"expr": "risk_var_99_stressed", "legendFormat": "Stressed VaR"},
                {"expr": "risk_var_99_combined", "legendFormat": "Combined VaR"},
            ],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
            unit="currencyUSD",
        ),
        GrafanaPanel(
            title="Risk State",
            panel_type="stat",
            targets=[{"expr": "risk_state_current", "legendFormat": "State"}],
            grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
            thresholds=[
                {"color": "green", "value": None},
                {"color": "yellow", "value": 1},
                {"color": "red", "value": 2},
            ],
        ),
        GrafanaPanel(
            title="Capital Scale",
            panel_type="gauge",
            targets=[{"expr": "risk_capital_scale", "legendFormat": "Scale"}],
            grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
            unit="percentunit",
            thresholds=[
                {"color": "red", "value": None},
                {"color": "yellow", "value": 0.5},
                {"color": "green", "value": 0.8},
            ],
        ),
        GrafanaPanel(
            title="Drawdown",
            panel_type="timeseries",
            targets=[{"expr": "risk_drawdown", "legendFormat": "Drawdown"}],
            grid_pos={"x": 12, "y": 4, "w": 12, "h": 4},
            unit="percentunit",
        ),
        GrafanaPanel(
            title="Exposure by Sector",
            panel_type="table",
            targets=[{"expr": "risk_sector_exposure", "legendFormat": "{{sector}}"}],
            grid_pos={"x": 0, "y": 8, "w": 12, "h": 8},
            unit="percentunit",
        ),
        GrafanaPanel(
            title="Factor Exposures",
            panel_type="timeseries",
            targets=[
                {"expr": "risk_factor_market", "legendFormat": "Market"},
                {"expr": "risk_factor_size", "legendFormat": "Size"},
                {"expr": "risk_factor_value", "legendFormat": "Value"},
                {"expr": "risk_factor_momentum", "legendFormat": "Momentum"},
            ],
            grid_pos={"x": 12, "y": 8, "w": 12, "h": 8},
        ),
        GrafanaPanel(
            title="Kill Switch Status",
            panel_type="stat",
            targets=[{"expr": "risk_kill_switch_active", "legendFormat": "Kill Switch"}],
            grid_pos={"x": 0, "y": 16, "w": 6, "h": 4},
            thresholds=[
                {"color": "green", "value": None},
                {"color": "red", "value": 1},
            ],
        ),
    ]

    return _build_dashboard("Risk Metrics", "risk-metrics", panels, tags=["risk", "monitoring"])


def execution_quality_dashboard() -> dict:
    """Generate execution quality / TCA dashboard."""
    panels = [
        GrafanaPanel(
            title="Slippage (bps)",
            panel_type="timeseries",
            targets=[{"expr": "exec_slippage_bps", "legendFormat": "Slippage"}],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
            unit="bps",
        ),
        GrafanaPanel(
            title="Implementation Shortfall",
            panel_type="timeseries",
            targets=[{"expr": "exec_impl_shortfall", "legendFormat": "IS"}],
            grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
            unit="bps",
        ),
        GrafanaPanel(
            title="Fill Rate",
            panel_type="gauge",
            targets=[{"expr": "exec_fill_rate", "legendFormat": "Fill%"}],
            grid_pos={"x": 0, "y": 8, "w": 6, "h": 4},
            unit="percentunit",
        ),
        GrafanaPanel(
            title="Avg Order Latency",
            panel_type="stat",
            targets=[{"expr": "exec_avg_latency_ms", "legendFormat": "Latency"}],
            grid_pos={"x": 6, "y": 8, "w": 6, "h": 4},
            unit="ms",
        ),
        GrafanaPanel(
            title="Daily Turnover",
            panel_type="timeseries",
            targets=[{"expr": "exec_daily_turnover", "legendFormat": "Turnover"}],
            grid_pos={"x": 12, "y": 8, "w": 12, "h": 4},
            unit="percentunit",
        ),
        GrafanaPanel(
            title="VWAP vs Arrival Price",
            panel_type="timeseries",
            targets=[
                {"expr": "exec_vwap_deviation", "legendFormat": "VWAP Dev"},
                {"expr": "exec_arrival_deviation", "legendFormat": "Arrival Dev"},
            ],
            grid_pos={"x": 0, "y": 12, "w": 24, "h": 6},
            unit="bps",
        ),
    ]

    return _build_dashboard("Execution Quality", "exec-quality", panels, tags=["execution", "tca"])


def system_health_dashboard() -> dict:
    """Generate system health / pipeline status dashboard."""
    panels = [
        GrafanaPanel(
            title="Pipeline Status",
            panel_type="stat",
            targets=[{"expr": "pipeline_last_run_status", "legendFormat": "Status"}],
            grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
            thresholds=[
                {"color": "green", "value": None},
                {"color": "red", "value": 1},
            ],
        ),
        GrafanaPanel(
            title="Data Freshness (hours)",
            panel_type="gauge",
            targets=[{"expr": "data_staleness_hours", "legendFormat": "Staleness"}],
            grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
            unit="h",
            thresholds=[
                {"color": "green", "value": None},
                {"color": "yellow", "value": 6},
                {"color": "red", "value": 24},
            ],
        ),
        GrafanaPanel(
            title="Model Drift Score",
            panel_type="gauge",
            targets=[{"expr": "model_drift_score", "legendFormat": "Drift"}],
            grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
            thresholds=[
                {"color": "green", "value": None},
                {"color": "yellow", "value": 0.3},
                {"color": "red", "value": 0.6},
            ],
        ),
        GrafanaPanel(
            title="QA Gate Pass Rate",
            panel_type="stat",
            targets=[{"expr": "qa_gate_pass_rate", "legendFormat": "Pass%"}],
            grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
            unit="percentunit",
        ),
        GrafanaPanel(
            title="Pipeline Run History",
            panel_type="timeseries",
            targets=[{"expr": "pipeline_run_duration_s", "legendFormat": "Duration"}],
            grid_pos={"x": 0, "y": 4, "w": 24, "h": 6},
            unit="s",
        ),
        GrafanaPanel(
            title="Alert History",
            panel_type="table",
            targets=[{"expr": "alerts_recent", "legendFormat": "{{severity}}"}],
            grid_pos={"x": 0, "y": 10, "w": 24, "h": 6},
        ),
    ]

    return _build_dashboard("System Health", "system-health", panels, tags=["system", "health", "pipeline"])


def _build_dashboard(title: str, uid: str, panels: list[GrafanaPanel], tags: list[str] | None = None) -> dict:
    """Build complete Grafana dashboard JSON."""
    grafana_panels = [_make_panel(i + 1, p) for i, p in enumerate(panels)]

    return {
        "dashboard": {
            "id": None,
            "uid": uid,
            "title": title,
            "tags": tags or [],
            "timezone": "browser",
            "schemaVersion": 39,
            "version": 1,
            "refresh": "30s",
            "time": {"from": "now-7d", "to": "now"},
            "panels": grafana_panels,
            "templating": {"list": []},
            "annotations": {"list": []},
        },
        "overwrite": True,
    }


def export_all_dashboards(output_dir: str | None = None) -> dict[str, dict]:
    """Export all dashboard definitions.

    Args:
        output_dir: Optional directory to write JSON files.

    Returns:
        {name: dashboard_dict} mapping.
    """
    dashboards = {
        "portfolio_performance": portfolio_performance_dashboard(),
        "risk_metrics": risk_metrics_dashboard(),
        "execution_quality": execution_quality_dashboard(),
        "system_health": system_health_dashboard(),
    }

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        for name, dash in dashboards.items():
            path = os.path.join(output_dir, f"{name}.json")
            with open(path, "w") as f:
                json.dump(dash, f, indent=2)
            logger.info("[Grafana] Exported %s → %s", name, path)

    return dashboards


__all__ = [
    "GrafanaPanel",
    "portfolio_performance_dashboard",
    "risk_metrics_dashboard",
    "execution_quality_dashboard",
    "system_health_dashboard",
    "export_all_dashboards",
]
