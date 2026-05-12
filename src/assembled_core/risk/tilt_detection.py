"""Behavioral tilt detection (audit C2-073).

Operator-tilt is the human-decision-quality side of risk: a trader
making short-time-window decisions after a string of losses, a deep
intraday drawdown, or a sequence of consecutive losing days is
empirically a worse decision-maker than the same trader rested. This
module surfaces four mechanical signals so the system can pause new
order generation and require an explicit operator unlock.

The signals are **rule-based**, not statistical — they are designed
to be visible in any post-trade review and trivial to argue against
when a false positive blocks legitimate work. Thresholds are sourced
from configs/policy.yaml under ``risk.tilt`` so they can be tuned
without a code change.

Signals (any one is sufficient to flag tilt):

1. **Consecutive-loss-days**: ≥ N losing realized PnL days in the
   last K calendar days (default N=3 in K=7 days).
2. **Day-loss magnitude**: realized PnL < -X% of equity in the last
   24h (default X=3%).
3. **Weekly drawdown**: equity peak-to-trough drop ≥ Y% over the
   last 7 calendar days (default Y=8%).
4. **Monthly drawdown**: equity peak-to-trough drop ≥ Z% over the
   last 30 calendar days (default Z=15%).

The output is a typed ``TiltState`` so callers can branch on
specifically which rule fired. ``TiltState.is_tilted`` is True iff at
least one rule fired.

This module is pure: it does NOT read files, write audit logs, or
trigger alerts. The caller is expected to wire it into the pipeline,
log the decision via the standard audit chain, and gate order
generation accordingly. That separation keeps the rules unit-testable
in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Default thresholds (override via TiltConfig)
# ---------------------------------------------------------------------------

DEFAULT_CONSECUTIVE_LOSSES_N = 3
DEFAULT_CONSECUTIVE_LOSSES_WINDOW_DAYS = 7
DEFAULT_DAY_LOSS_PCT = 0.03
DEFAULT_WEEKLY_DD_PCT = 0.08
DEFAULT_MONTHLY_DD_PCT = 0.15


@dataclass(frozen=True)
class TiltConfig:
    """Tilt-rule thresholds (sourced from configs/policy.yaml)."""

    consecutive_losses_n: int = DEFAULT_CONSECUTIVE_LOSSES_N
    consecutive_losses_window_days: int = DEFAULT_CONSECUTIVE_LOSSES_WINDOW_DAYS
    day_loss_pct: float = DEFAULT_DAY_LOSS_PCT
    weekly_dd_pct: float = DEFAULT_WEEKLY_DD_PCT
    monthly_dd_pct: float = DEFAULT_MONTHLY_DD_PCT


@dataclass(frozen=True)
class DailyPnLPoint:
    """One observation: end-of-day timestamp + realized PnL + equity."""

    ts: datetime
    realized_pnl: float
    equity: float


@dataclass(frozen=True)
class TiltState:
    """Result of a tilt-detection pass.

    ``triggered_rules`` lists the names of any rules that fired.
    Order is stable (alphabetical) so audit logs are diffable.
    """

    is_tilted: bool
    triggered_rules: tuple[str, ...] = field(default_factory=tuple)
    consecutive_losses_count: int = 0
    day_loss_pct: float = 0.0
    weekly_dd_pct: float = 0.0
    monthly_dd_pct: float = 0.0


def _ensure_utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _window(
    history: Sequence[DailyPnLPoint],
    *,
    now: datetime,
    days: int,
) -> list[DailyPnLPoint]:
    cutoff = _ensure_utc(now) - timedelta(days=days)
    return [p for p in history if _ensure_utc(p.ts) >= cutoff]


def _peak_to_trough_drawdown(points: Iterable[DailyPnLPoint]) -> float:
    """Return |trough-after-peak| / peak, 0.0 if fewer than 2 points."""
    pts = list(points)
    if len(pts) < 2:
        return 0.0
    peak = pts[0].equity
    max_dd = 0.0
    for p in pts:
        if p.equity > peak:
            peak = p.equity
        if peak > 0:
            dd = (peak - p.equity) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def detect_tilt(
    history: Sequence[DailyPnLPoint],
    *,
    config: TiltConfig | None = None,
    now: datetime | None = None,
) -> TiltState:
    """Run all four tilt rules against ``history`` and return the result.

    Args:
        history: end-of-day PnL points, sorted by ``ts`` ascending. An
            empty history returns ``is_tilted=False`` and zero metrics.
        config: thresholds; defaults from policy.
        now: reference time for window calculations; defaults to UTC now.

    Returns:
        A ``TiltState`` describing which (if any) rules fired.
    """
    cfg = config or TiltConfig()
    now_ts = _ensure_utc(now or datetime.now(timezone.utc))

    if not history:
        return TiltState(is_tilted=False)

    triggered: list[str] = []

    # Rule 1 — consecutive losing days in window
    window_pts = _window(history, now=now_ts, days=cfg.consecutive_losses_window_days)
    losing = [p for p in window_pts if p.realized_pnl < 0]
    cons_count = len(losing)
    if cons_count >= cfg.consecutive_losses_n:
        triggered.append("consecutive_loss_days")

    # Rule 2 — single-day loss magnitude (last 24h)
    last_24h = _window(history, now=now_ts, days=1)
    day_loss_pct = 0.0
    if last_24h:
        total_pnl_24h = sum(p.realized_pnl for p in last_24h)
        ref_equity = last_24h[-1].equity if last_24h[-1].equity > 0 else 0.0
        if ref_equity > 0:
            day_loss_pct = max(0.0, -total_pnl_24h / ref_equity)
            if day_loss_pct >= cfg.day_loss_pct:
                triggered.append("day_loss_magnitude")

    # Rule 3 — weekly drawdown
    weekly_pts = _window(history, now=now_ts, days=7)
    weekly_dd = _peak_to_trough_drawdown(weekly_pts)
    if weekly_dd >= cfg.weekly_dd_pct:
        triggered.append("weekly_drawdown")

    # Rule 4 — monthly drawdown
    monthly_pts = _window(history, now=now_ts, days=30)
    monthly_dd = _peak_to_trough_drawdown(monthly_pts)
    if monthly_dd >= cfg.monthly_dd_pct:
        triggered.append("monthly_drawdown")

    triggered_sorted = tuple(sorted(triggered))
    return TiltState(
        is_tilted=bool(triggered_sorted),
        triggered_rules=triggered_sorted,
        consecutive_losses_count=cons_count,
        day_loss_pct=day_loss_pct,
        weekly_dd_pct=weekly_dd,
        monthly_dd_pct=monthly_dd,
    )


__all__ = [
    "DailyPnLPoint",
    "TiltConfig",
    "TiltState",
    "detect_tilt",
    "DEFAULT_CONSECUTIVE_LOSSES_N",
    "DEFAULT_CONSECUTIVE_LOSSES_WINDOW_DAYS",
    "DEFAULT_DAY_LOSS_PCT",
    "DEFAULT_WEEKLY_DD_PCT",
    "DEFAULT_MONTHLY_DD_PCT",
]
