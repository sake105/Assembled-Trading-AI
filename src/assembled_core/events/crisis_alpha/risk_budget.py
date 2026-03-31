"""Crisis-Alpha risk budget — daily loss guard and position size limits.

Risk budget controls for the crisis sub-portfolio:

1. Daily loss guard: if abs(daily_pnl) >= daily_loss_limit → PAUSE state.
   This is checked in the state machine and independently here for pre-trade use.

2. Max sub-portfolio exposure: total |weight| across all crisis positions
   is capped at max_gross_exposure.

3. Per-instrument weight limit: enforced per basket definition (max_weight).

4. Max position count: optional limit on number of simultaneous crisis positions.

All functions are pure (no I/O) for testability.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------


def _get(d: dict, *keys, default=None):
    node = d
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
        if node is None:
            return default
    return node


# ---------------------------------------------------------------------------
# Daily loss guard
# ---------------------------------------------------------------------------


def check_daily_loss(ctx: CrisisAlphaContext) -> tuple[bool, str]:
    """Return (ok, reason). ok=False means daily loss limit breached → must PAUSE.

    Args:
        ctx: CrisisAlphaContext with daily_pnl and daily_loss_limit.

    Returns:
        (True, "OK") if within limit, (False, reason) if breached.
    """
    if ctx.daily_loss_breached():
        return (
            False,
            f"daily loss breached: pnl={ctx.daily_pnl:.4f} limit={ctx.daily_loss_limit:.4f}",
        )
    return (
        True,
        f"daily loss OK: pnl={ctx.daily_pnl:.4f} limit={ctx.daily_loss_limit:.4f}",
    )


# ---------------------------------------------------------------------------
# Gross exposure cap
# ---------------------------------------------------------------------------


def check_gross_exposure(
    target_weights: dict[str, float],
    max_gross_exposure: float = 0.30,
) -> tuple[bool, str]:
    """Return (ok, reason). ok=False if total |weight| exceeds cap.

    Args:
        target_weights: Dict of {symbol: weight} for crisis positions.
        max_gross_exposure: Maximum allowed sum of |weights| (default 0.30 = 30%).

    Returns:
        (True, reason) if within cap, (False, reason) if over.
    """
    gross = sum(abs(w) for w in target_weights.values())
    if gross > max_gross_exposure:
        return (
            False,
            f"gross exposure {gross:.4f} > cap {max_gross_exposure:.4f}",
        )
    return True, f"gross exposure OK: {gross:.4f} <= {max_gross_exposure:.4f}"


# ---------------------------------------------------------------------------
# Apply weight caps from basket definitions
# ---------------------------------------------------------------------------


def apply_weight_caps(
    target_weights: dict[str, float],
    baskets: list[dict[str, Any]],
) -> dict[str, float]:
    """Cap each position weight at its basket-defined max_weight.

    Args:
        target_weights: Dict of {symbol: weight} (signed weights allowed).
        baskets: List of basket definition dicts (each with "symbol" and "max_weight").

    Returns:
        New dict with weights capped at max_weight per instrument.
        If a symbol is not in baskets, it is removed (unknown instrument).
    """
    cap_map = {b["symbol"]: b.get("max_weight", 0.10) for b in baskets}
    capped: dict[str, float] = {}
    for symbol, weight in target_weights.items():
        if symbol not in cap_map:
            continue  # skip unknown instruments
        cap = cap_map[symbol]
        capped[symbol] = max(-cap, min(cap, weight))
    return capped


# ---------------------------------------------------------------------------
# Scale to gross exposure cap
# ---------------------------------------------------------------------------


def scale_to_gross_cap(
    target_weights: dict[str, float],
    max_gross_exposure: float = 0.30,
) -> dict[str, float]:
    """Proportionally scale all weights so total |weight| <= max_gross_exposure.

    Args:
        target_weights: Dict of {symbol: weight}.
        max_gross_exposure: Maximum allowed gross exposure.

    Returns:
        Scaled weights (proportional reduction if over cap, no change if within).
    """
    gross = sum(abs(w) for w in target_weights.values())
    if gross <= max_gross_exposure or gross == 0.0:
        return dict(target_weights)
    scale = max_gross_exposure / gross
    return {sym: w * scale for sym, w in target_weights.items()}


# ---------------------------------------------------------------------------
# Full risk budget application
# ---------------------------------------------------------------------------


def apply_risk_budget(
    target_weights: dict[str, float],
    baskets: list[dict[str, Any]],
    policy: dict | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Apply all risk budget constraints and return (final_weights, audit_reasons).

    Steps:
    1. Filter to known basket symbols only.
    2. Apply per-instrument weight caps.
    3. Scale to gross exposure cap.

    Args:
        target_weights: Raw proposed weights {symbol: weight}.
        baskets: Active basket definitions.
        policy: Policy dict (optional; reads crisis_alpha.risk_budget.*).

    Returns:
        (final_weights, reasons) — reasons is a list of audit log strings.
    """
    reasons: list[str] = []
    cfg = _get(policy or {}, "crisis_alpha", "risk_budget", default={})
    max_gross = float(_get(cfg, "max_gross_exposure", default=0.30))

    # Step 1: cap per-instrument
    capped = apply_weight_caps(target_weights, baskets)
    removed = set(target_weights) - set(capped)
    if removed:
        reasons.append(f"removed unknown instruments: {sorted(removed)}")

    # Step 2: scale to gross cap
    final = scale_to_gross_cap(capped, max_gross)
    gross_after = sum(abs(w) for w in final.values())
    reasons.append(
        f"risk_budget applied: gross={gross_after:.4f} cap={max_gross:.4f} "
        f"instruments={len(final)}"
    )

    return final, reasons
