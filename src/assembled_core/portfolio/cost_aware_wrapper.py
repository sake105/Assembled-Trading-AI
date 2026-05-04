"""Cost-aware sizing wrapper (Sprint 3 / Plan W12).

A light wrapper that shrinks trade deltas when the expected transaction cost
becomes large relative to the portfolio notional. Designed as a drop-in post-
processing step after score / HRP / Black-Litterman sizing has produced
``target_weights``.

Contract:
  - pure function, never mutates inputs
  - returns a new ``(adjusted_weights, reasons)`` tuple
  - no optional deps; O(n) in number of symbols
  - fully deterministic per call

Math:
    delta_s      = target_s - current_s                (per symbol)
    turnover     = sum(|delta_s|)
    weighted_cost = sum(|delta_s| * cost_bps(s) / 1e4)
    raw_shrink   = penalty_factor * weighted_cost / max(turnover, eps)
    shrink       = clip(1 - raw_shrink, min_shrink, 1.0)
    adjusted_s   = current_s + shrink * delta_s

``penalty_factor`` scales how aggressively costs reduce the trade. With
``penalty_factor = 0.5 * current_invested_pct`` (the heuristic from the plan)
a heavily invested portfolio shrinks trades more than an empty one.
"""

from __future__ import annotations

from typing import Any

DEFAULT_COST_BPS = 6.0  # one-way commission + spread proxy
_MIN_SHRINK = 0.0  # allow full cancellation if costs dominate


def apply_cost_aware_wrapper(
    target_weights: dict[str, float],
    current_weights: dict[str, float] | None,
    *,
    cost_bps_per_symbol: dict[str, float] | None = None,
    penalty_factor: float = 0.5,
    min_shrink: float = _MIN_SHRINK,
) -> tuple[dict[str, float], list[str]]:
    """Shrink trade deltas proportionally to aggregated transaction cost.

    Args:
        target_weights: Desired portfolio weights (symbol -> weight).
        current_weights: Current portfolio weights (symbol -> weight). ``None``
            or empty dict is treated as zero holdings.
        cost_bps_per_symbol: Optional per-symbol one-way cost in basis points.
            Missing symbols fall back to :data:`DEFAULT_COST_BPS`.
        penalty_factor: Scales the cost drag. Typical range [0, 1]. The plan
            recommends ``0.5 * current_invested_pct``.
        min_shrink: Lower bound for the shrink factor. Defaults to 0 (full
            cancellation allowed).

    Returns:
        ``(adjusted_weights, reasons)`` where ``reasons`` is a list of
        human-readable explanation strings. When no shrinkage is needed
        the reasons list is empty and ``adjusted_weights`` equals a copy of
        ``target_weights``.
    """
    if not target_weights:
        return {}, []

    current = dict(current_weights or {})
    cost_map = cost_bps_per_symbol or {}

    symbols = list(target_weights.keys())
    deltas = {s: float(target_weights[s]) - float(current.get(s, 0.0)) for s in symbols}
    turnover = sum(abs(d) for d in deltas.values())

    if turnover <= 1e-12 or penalty_factor <= 0.0:
        return dict(target_weights), []

    weighted_cost = 0.0
    for sym, d in deltas.items():
        cost_bps = float(cost_map.get(sym, DEFAULT_COST_BPS))
        weighted_cost += abs(d) * (cost_bps / 10_000.0)

    raw_shrink = penalty_factor * weighted_cost / turnover
    shrink = max(min_shrink, min(1.0, 1.0 - raw_shrink))

    reasons: list[str] = []
    if shrink >= 1.0 - 1e-9:
        # no effective shrinkage
        return dict(target_weights), []

    adjusted: dict[str, float] = {}
    for sym in symbols:
        start = float(current.get(sym, 0.0))
        adjusted[sym] = start + shrink * deltas[sym]

    reasons.append(
        f"cost_aware_wrapper: turnover={turnover:.4f} "
        f"weighted_cost={weighted_cost:.6f} "
        f"shrink={shrink:.4f} penalty_factor={penalty_factor:.3f}"
    )
    return adjusted, reasons


def apply_cost_aware_from_policy(
    target_weights: dict[str, float],
    current_weights: dict[str, float] | None,
    policy: dict[str, Any],
    *,
    cost_bps_per_symbol: dict[str, float] | None = None,
    current_invested_pct: float | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Convenience entry point that reads config from ``policy``.

    Reads ``policy['cost_aware_wrapper']`` for:
      - ``enabled`` (bool, default False)
      - ``penalty_factor`` (float). If omitted and ``current_invested_pct`` is
        provided, defaults to ``0.5 * current_invested_pct`` per plan W12.
      - ``min_shrink`` (float, default 0.0)

    When disabled or absent, returns ``(copy(target_weights), [])``.
    """
    cfg = (policy or {}).get("cost_aware_wrapper") or {}
    if not cfg.get("enabled", False):
        return dict(target_weights), []

    penalty = cfg.get("penalty_factor")
    if penalty is None:
        invested = float(current_invested_pct or 0.0)
        penalty = 0.5 * invested
    penalty = float(penalty)

    min_shrink = float(cfg.get("min_shrink", _MIN_SHRINK))

    return apply_cost_aware_wrapper(
        target_weights,
        current_weights,
        cost_bps_per_symbol=cost_bps_per_symbol,
        penalty_factor=penalty,
        min_shrink=min_shrink,
    )


__all__ = [
    "DEFAULT_COST_BPS",
    "apply_cost_aware_from_policy",
    "apply_cost_aware_wrapper",
]
