"""Soft Profit Lock overlay: reduce exposure after strong gains (policy-driven).

Combines multiplicatively with GeoRisk: final_multiplier = geo * profit_lock.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd


def compute_profit_lock_multiplier(
    equity_curve: pd.Series,
    policy: Dict[str, Any],
    now_idx: int,
    state: Dict[str, Any] | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute profit lock multiplier from equity curve and policy.

    If lookback return >= trigger_return, return multiplier_on_trigger (clamped by floor)
    for cooldown_days; else 1.0.

    Args:
        equity_curve: Series of equity values (index = bar index or time).
        policy: dict with enabled, lookback_days, trigger_return, multiplier_on_trigger,
                floor, cooldown_days.
        now_idx: Current bar index (integer) into equity_curve.
        state: Optional state dict to persist trigger index for cooldown. Keys:
               trigger_idx (int). If None, cooldown does not persist across calls.

    Returns:
        (multiplier, updated_state). multiplier in [floor, 1.0].
        If policy disabled or curve too short, returns (1.0, state or {}).
    """
    out_state: Dict[str, Any] = dict(state) if state is not None else {}

    pl = policy or {}
    if not pl.get("enabled", False):
        return 1.0, out_state

    if equity_curve is None or not isinstance(equity_curve, pd.Series):
        return 1.0, out_state
    if equity_curve.empty or len(equity_curve) < 2:
        return 1.0, out_state

    lookback_days = int(pl.get("lookback_days", 20) or 20)
    trigger_return = float(pl.get("trigger_return", 0.08) or 0.08)
    multiplier_on_trigger = float(pl.get("multiplier_on_trigger", 0.80) or 0.80)
    floor = float(pl.get("floor", 0.50) or 0.50)
    cooldown_days = int(pl.get("cooldown_days", 10) or 10)

    # Clamp multiplier to [floor, 1.0]
    mult_val = max(floor, min(1.0, multiplier_on_trigger))

    n = len(equity_curve)
    if now_idx < 0 or now_idx >= n:
        return 1.0, out_state

    # Need enough history for lookback
    if now_idx < lookback_days:
        return 1.0, out_state

    # Cooldown: if we triggered recently, stay at reduced multiplier
    trigger_idx = out_state.get("trigger_idx")
    if trigger_idx is not None:
        if now_idx - trigger_idx <= cooldown_days:
            return mult_val, out_state
        out_state.pop("trigger_idx", None)

    # Lookback return
    start_idx = max(0, now_idx - lookback_days)
    start_val = float(equity_curve.iloc[start_idx])
    now_val = float(equity_curve.iloc[now_idx])
    if start_val <= 0:
        return 1.0, out_state
    ret = (now_val / start_val) - 1.0

    if ret >= trigger_return:
        out_state["trigger_idx"] = now_idx
        return mult_val, out_state
    return 1.0, out_state


__all__ = ["compute_profit_lock_multiplier"]
