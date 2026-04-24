"""Paper Trading Monitor.

Lightweight monitoring layer for the unified paper engine. Aggregates
PaperDayResult objects and exposes summary stats, alerts, and equity-curve
snapshots suitable for ops dashboards and CI smoke checks.

All imports are defensive -- the monitor degrades gracefully when optional
accounting or reporting modules are unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:  # pragma: no cover
    _HAS_PANDAS = False
    logger.warning("[UNIFIED-PAPER] pandas unavailable -- monitoring degraded")

try:
    from src.assembled_core.execution.unified_paper_engine import PaperDayResult
    _HAS_ENGINE = True
except Exception:  # pragma: no cover
    _HAS_ENGINE = False
    PaperDayResult = Any  # type: ignore[assignment,misc]
    logger.warning("[UNIFIED-PAPER] unified_paper_engine unavailable -- some features disabled")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PaperMonitorAlert:
    """Single monitoring alert.

    Attributes:
        level: "INFO" | "WARN" | "ERROR"
        code:  Short machine-readable code (e.g. "HIGH_COST_BPS").
        message: Human-readable description.
        date:  Trading date the alert applies to (YYYY-MM-DD).
    """

    level: str
    code: str
    message: str
    date: str


@dataclass
class PaperMonitorSummary:
    """Aggregated monitoring summary over a run or period.

    Attributes:
        n_days:        Total days processed.
        n_success:     Days with status == "success".
        n_error:       Days with status == "error".
        n_kill_switch: Days blocked by kill switch.
        total_fills:   Cumulative fill count.
        avg_cost_bps:  Average total execution cost (bps).
        equity_start:  Equity at beginning of period.
        equity_end:    Equity at end of period.
        total_return:  Arithmetic return over the period (equity_end/equity_start - 1).
        alerts:        List of PaperMonitorAlert raised during the period.
    """

    n_days: int = 0
    n_success: int = 0
    n_error: int = 0
    n_kill_switch: int = 0
    total_fills: int = 0
    avg_cost_bps: float = 0.0
    equity_start: float = 0.0
    equity_end: float = 0.0
    total_return: float = 0.0
    alerts: list[PaperMonitorAlert] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Alert thresholds (conservative defaults)
# ---------------------------------------------------------------------------

DEFAULT_ALERT_THRESHOLDS: dict[str, float] = {
    "max_daily_loss_pct": -0.03,      # -3 % single-day loss triggers WARN
    "max_cost_bps": 50.0,             # >50 bps total cost triggers WARN
    "min_fill_rate": 0.0,             # 0 fills on a non-kill-switch day triggers INFO
}


# ---------------------------------------------------------------------------
# Core monitor class
# ---------------------------------------------------------------------------

class PaperMonitor:
    """Collects PaperDayResult objects and surfaces monitoring information.

    Usage::

        monitor = PaperMonitor()
        for result in engine.run_paper_period("2025-01-02", "2025-01-31"):
            monitor.record(result)

        summary = monitor.get_summary()
        print(summary)
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        alert_log_path: Path | None = None,
    ) -> None:
        self._results: list[Any] = []
        self._thresholds = thresholds or DEFAULT_ALERT_THRESHOLDS.copy()
        self._alert_log_path = alert_log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, result: Any) -> None:
        """Ingest one PaperDayResult and run alert checks.

        Args:
            result: PaperDayResult from UnifiedPaperEngine.run_paper_day().
        """
        self._results.append(result)
        alerts = self._check_alerts(result)
        for alert in alerts:
            log_fn = logger.warning if alert.level in ("WARN", "ERROR") else logger.info
            log_fn(
                "[UNIFIED-PAPER] Alert [%s] %s: %s",
                alert.level,
                alert.code,
                alert.message,
            )
        if self._alert_log_path and alerts:
            self._append_alerts_to_log(alerts)

    def get_summary(self) -> PaperMonitorSummary:
        """Compute and return the aggregated monitoring summary.

        Returns:
            PaperMonitorSummary over all recorded results.
        """
        if not self._results:
            return PaperMonitorSummary()

        n_success = sum(1 for r in self._results if getattr(r, "status", "") == "success")
        n_error = sum(1 for r in self._results if getattr(r, "status", "") == "error")
        n_kill = sum(1 for r in self._results if getattr(r, "status", "") == "kill_switch")
        total_fills = sum(int(getattr(r, "n_fills", 0)) for r in self._results)

        cost_list = [
            float(getattr(r, "total_cost_bps", 0.0))
            for r in self._results
            if getattr(r, "n_fills", 0) > 0
        ]
        avg_cost_bps = sum(cost_list) / len(cost_list) if cost_list else 0.0

        equity_start = float(getattr(self._results[0], "equity_before", 0.0))
        equity_end = float(getattr(self._results[-1], "equity_after", 0.0))
        total_return = (
            (equity_end / equity_start - 1.0) if equity_start > 0 else 0.0
        )

        all_alerts: list[PaperMonitorAlert] = []
        for result in self._results:
            all_alerts.extend(self._check_alerts(result))

        return PaperMonitorSummary(
            n_days=len(self._results),
            n_success=n_success,
            n_error=n_error,
            n_kill_switch=n_kill,
            total_fills=total_fills,
            avg_cost_bps=avg_cost_bps,
            equity_start=equity_start,
            equity_end=equity_end,
            total_return=total_return,
            alerts=all_alerts,
        )

    def get_equity_curve(self) -> "Any":
        """Return equity curve as a DataFrame (requires pandas).

        Returns:
            pd.DataFrame with columns: date, equity_before, equity_after, daily_return.
            Returns an empty DataFrame if pandas is unavailable.
        """
        if not _HAS_PANDAS:
            logger.warning("[UNIFIED-PAPER] pandas unavailable -- equity_curve not available")
            return None

        rows = []
        for r in self._results:
            rows.append({
                "date": getattr(r, "date", ""),
                "equity_before": float(getattr(r, "equity_before", 0.0)),
                "equity_after": float(getattr(r, "equity_after", 0.0)),
                "daily_return": float(getattr(r, "daily_return", 0.0)),
                "n_fills": int(getattr(r, "n_fills", 0)),
                "total_cost_bps": float(getattr(r, "total_cost_bps", 0.0)),
                "status": str(getattr(r, "status", "")),
            })

        if not rows:
            return pd.DataFrame(
                columns=["date", "equity_before", "equity_after",
                         "daily_return", "n_fills", "total_cost_bps", "status"]
            )

        df = pd.DataFrame(rows)
        return df.sort_values("date").reset_index(drop=True)

    def reset(self) -> None:
        """Clear all recorded results."""
        self._results = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_alerts(self, result: Any) -> list[PaperMonitorAlert]:
        """Run configured alert checks against a single PaperDayResult.

        Args:
            result: PaperDayResult to inspect.

        Returns:
            List of PaperMonitorAlert (may be empty).
        """
        alerts: list[PaperMonitorAlert] = []
        date = str(getattr(result, "date", "unknown"))
        status = str(getattr(result, "status", ""))
        daily_return = float(getattr(result, "daily_return", 0.0))
        cost_bps = float(getattr(result, "total_cost_bps", 0.0))
        n_fills = int(getattr(result, "n_fills", 0))
        n_orders = int(getattr(result, "n_orders", 0))

        # Daily loss alert
        max_loss = float(self._thresholds.get("max_daily_loss_pct", -0.03))
        if daily_return < max_loss:
            alerts.append(PaperMonitorAlert(
                level="WARN",
                code="HIGH_DAILY_LOSS",
                message=f"Daily return {daily_return*100:.2f}% below threshold {max_loss*100:.2f}%",
                date=date,
            ))

        # High cost alert
        max_cost = float(self._thresholds.get("max_cost_bps", 50.0))
        if cost_bps > max_cost and n_fills > 0:
            alerts.append(PaperMonitorAlert(
                level="WARN",
                code="HIGH_COST_BPS",
                message=f"Execution cost {cost_bps:.1f} bps above threshold {max_cost:.1f} bps",
                date=date,
            ))

        # Zero fills when orders were generated (non-kill-switch)
        if status not in ("kill_switch",) and n_orders > 0 and n_fills == 0:
            alerts.append(PaperMonitorAlert(
                level="INFO",
                code="ZERO_FILLS",
                message=f"{n_orders} orders generated but 0 fills (cash gate or risk controls?)",
                date=date,
            ))

        # Error status alert
        if status == "error":
            err_msgs = getattr(result, "errors", [])
            alerts.append(PaperMonitorAlert(
                level="ERROR",
                code="DAY_ERROR",
                message=f"Day ended with error status: {err_msgs}",
                date=date,
            ))

        return alerts

    def _append_alerts_to_log(self, alerts: list[PaperMonitorAlert]) -> None:
        """Append alert lines to the optional alert log file."""
        try:
            self._alert_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._alert_log_path, "a", encoding="utf-8") as fh:
                for alert in alerts:
                    fh.write(
                        f"{alert.date}\t{alert.level}\t{alert.code}\t{alert.message}\n"
                    )
        except Exception as exc:
            logger.warning("[UNIFIED-PAPER] Could not write alert log: %s", exc)


# ---------------------------------------------------------------------------
# Convenience function: monitor a full period run
# ---------------------------------------------------------------------------

def monitor_paper_period(
    results: list[Any],
    thresholds: dict[str, float] | None = None,
    alert_log_path: Path | None = None,
) -> PaperMonitorSummary:
    """Run monitoring over a list of PaperDayResult objects.

    Convenience wrapper around PaperMonitor for single-call usage.

    Args:
        results:        List of PaperDayResult from run_paper_period().
        thresholds:     Optional override for alert thresholds.
        alert_log_path: Optional path to append alert TSV lines.

    Returns:
        PaperMonitorSummary with aggregated stats and all raised alerts.
    """
    monitor = PaperMonitor(thresholds=thresholds, alert_log_path=alert_log_path)
    for r in results:
        monitor.record(r)
    return monitor.get_summary()


__all__ = [
    "PaperMonitorAlert",
    "PaperMonitorSummary",
    "PaperMonitor",
    "monitor_paper_period",
    "DEFAULT_ALERT_THRESHOLDS",
]
