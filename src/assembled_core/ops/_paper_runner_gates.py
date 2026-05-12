"""Pre-cycle gates wired into paper_runner (audit C2-073 + halt-cache).

Two gates run BEFORE :func:`run_trading_cycle` is called:

1. **Halt-cache gate** — refreshes a file-backed halt-set and assigns it
   to ``ctx.halted_symbols`` so the existing filter at
   ``_tc_sizing.size_positions`` (line ~1716) drops halted symbols from
   the final target-positions DataFrame. The cache itself is
   :class:`utils.halt_cache.HaltCache` (60s TTL, fail-soft).

2. **Tilt-detection gate** — builds a daily-PnL history from the paper
   ledger's equity curve, calls
   :func:`risk.tilt_detection.detect_tilt`, and on a fired rule:

   - emits a structured ``[TILT]`` warning log,
   - stamps ``ctx.tilt_state`` so downstream code / reports can see it,
   - if ``policy.tilt.block_orders=true`` returns a sentinel that the
     caller uses to skip the trading cycle and return early.

Both gates are **default-off**. With ``policy.halt_cache.enabled=false``
(default) and ``policy.tilt.enabled=false`` (default), this module is
a no-op and existing paper-runner behavior is unchanged.

Audit log entries: every gate decision (refresh count, tilt-fire) gets
a structured one-line WARNING in the standard logger so they show up in
paper-track artifacts without needing a new sink.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# Module-level cache. Constructed once per process and re-used across
# daily ticks so the TTL semantics actually mean something. The supplier
# is a function over the latest ``policy.halt_cache.symbols_file`` path
# read at refresh-time, so editing the file in-place gets picked up
# within one TTL window.
_HALT_CACHE: Any = None


@dataclass(frozen=True)
class TiltDecision:
    """Outcome of the tilt-gate."""

    is_tilted: bool
    triggered_rules: tuple[str, ...]
    blocked: bool  # block_orders policy fired
    raw_state: Any


def _build_halt_supplier(symbols_file: Path):
    def _supplier() -> list[str]:
        if not symbols_file.exists():
            return []
        try:
            data = json.loads(symbols_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "[halt-cache] failed to read %s: %s — returning empty set",
                symbols_file,
                exc,
            )
            return []
        if isinstance(data, list):
            return [str(s) for s in data if s]
        if isinstance(data, dict) and "halted" in data:
            return [str(s) for s in (data["halted"] or []) if s]
        return []

    return _supplier


def apply_halt_cache_gate(
    ctx: Any,
    *,
    paper_cfg: dict[str, Any],
    root: Path,
) -> int:
    """Populate ``ctx.halted_symbols`` from the configured halt feed.

    Returns the number of halted symbols loaded into the context (0
    when disabled or feed is empty). Pure side-effect on ctx; no
    return value other than the diagnostic count.
    """
    global _HALT_CACHE
    hc_cfg = paper_cfg.get("halt_cache") or {}
    if not hc_cfg.get("enabled", False):
        return 0

    from src.assembled_core.utils.halt_cache import HaltCache

    symbols_file_rel = hc_cfg.get("symbols_file") or "data/halts/halted_symbols.json"
    symbols_file = (
        root / symbols_file_rel
        if not Path(symbols_file_rel).is_absolute()
        else Path(symbols_file_rel)
    )
    ttl = float(hc_cfg.get("ttl_seconds", 60.0) or 60.0)

    if (
        _HALT_CACHE is None
        or getattr(_HALT_CACHE, "_symbols_file", None) != symbols_file
    ):
        _HALT_CACHE = HaltCache(
            supplier=_build_halt_supplier(symbols_file),
            ttl_seconds=ttl,
        )
        _HALT_CACHE._symbols_file = symbols_file  # type: ignore[attr-defined]

    snap = _HALT_CACHE.snapshot()
    ctx.halted_symbols = frozenset(snap)
    if snap:
        logger.warning(
            "[halt-cache] %d halted symbol(s) loaded from %s: %s",
            len(snap),
            symbols_file,
            sorted(snap),
        )
    else:
        logger.debug(
            "[halt-cache] no halted symbols (file=%s, exists=%s)",
            symbols_file,
            symbols_file.exists(),
        )
    return len(snap)


def _equity_curve_to_pnl_points(
    equity_curve: list[dict[str, Any]] | None,
):
    """Convert paper-ledger equity_curve list into DailyPnLPoint objects."""
    from src.assembled_core.risk.tilt_detection import DailyPnLPoint

    if not equity_curve:
        return []
    pts: list[DailyPnLPoint] = []
    prev_equity: float | None = None
    for entry in equity_curve:
        ts_raw = entry.get("timestamp") or entry.get("ts") or entry.get("date")
        eq_raw = entry.get("equity")
        if ts_raw is None or eq_raw is None:
            continue
        try:
            ts = pd.to_datetime(ts_raw, utc=True).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            eq = float(eq_raw)
        except (ValueError, TypeError):
            continue
        # Realized PnL = today's equity - yesterday's equity. First row gets 0.
        realized = (eq - prev_equity) if prev_equity is not None else 0.0
        pts.append(DailyPnLPoint(ts=ts, realized_pnl=realized, equity=eq))
        prev_equity = eq
    return pts


def apply_tilt_gate(
    ctx: Any,
    *,
    paper_cfg: dict[str, Any],
    ledger_state: dict[str, Any] | None,
    now: datetime | None = None,
) -> TiltDecision:
    """Run tilt-detection on the paper ledger's equity curve.

    Returns a :class:`TiltDecision` describing what (if anything)
    fired. ``blocked=True`` indicates the caller should NOT call
    ``run_trading_cycle`` and instead bail with a zero-orders result.
    """
    tilt_cfg = paper_cfg.get("tilt") or {}
    if not tilt_cfg.get("enabled", False) or ledger_state is None:
        return TiltDecision(False, (), False, None)

    from src.assembled_core.risk.tilt_detection import TiltConfig, detect_tilt

    cfg = TiltConfig(
        consecutive_losses_n=int(tilt_cfg.get("consecutive_losses_n", 3) or 3),
        consecutive_losses_window_days=int(
            tilt_cfg.get("consecutive_losses_window_days", 7) or 7
        ),
        day_loss_pct=float(tilt_cfg.get("day_loss_pct", 0.03) or 0.03),
        weekly_dd_pct=float(tilt_cfg.get("weekly_dd_pct", 0.08) or 0.08),
        monthly_dd_pct=float(tilt_cfg.get("monthly_dd_pct", 0.15) or 0.15),
    )

    history = _equity_curve_to_pnl_points(ledger_state.get("equity_curve"))
    state = detect_tilt(history, config=cfg, now=now)

    ctx.tilt_state = {
        "is_tilted": state.is_tilted,
        "triggered_rules": list(state.triggered_rules),
        "consecutive_losses_count": state.consecutive_losses_count,
        "day_loss_pct": state.day_loss_pct,
        "weekly_dd_pct": state.weekly_dd_pct,
        "monthly_dd_pct": state.monthly_dd_pct,
    }

    if not state.is_tilted:
        return TiltDecision(False, (), False, state)

    block = bool(tilt_cfg.get("block_orders", False))
    logger.warning(
        "[tilt] fired rules=%s (block_orders=%s) — cons_losses=%d day_loss_pct=%.3f "
        "weekly_dd=%.3f monthly_dd=%.3f",
        state.triggered_rules,
        block,
        state.consecutive_losses_count,
        state.day_loss_pct,
        state.weekly_dd_pct,
        state.monthly_dd_pct,
    )
    return TiltDecision(True, state.triggered_rules, block, state)


__all__ = [
    "TiltDecision",
    "apply_halt_cache_gate",
    "apply_tilt_gate",
]
