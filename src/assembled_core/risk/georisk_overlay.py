from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from src.assembled_core.pipeline.trading_cycle_shared import TradingContext


def compute_exposure_multiplier(ctx: "TradingContext", policy: Dict[str, Any]) -> float:
    """Compute exposure multiplier from GeoRisk overlay policy and TradingContext.

    Rules (v1):
    - If georisk_overlay.enabled is false -> 1.0
    - If ctx.news_geo is None -> 1.0 (unless intel degraded handling applies)
    - If intel_geo_score or intel_news_triggers is DEGRADED:
        - Use qc.if_intel_degraded state_hint (default WATCH)
        - geo_score=0, geo_confidence=0.0
    - If geo_confidence < confidence_floor -> 1.0
    - state_hint: from ctx.risk_state.state if present, else ctx.news_geo.state_hint
    - If by_geo_score has key for geo_score -> use it; else mapping[state_hint].multiplier
    - Clamp multiplier to [0.0, 1.0]
    """
    overlay = (policy or {}).get("georisk_overlay") or {}
    if not overlay.get("enabled", False):
        return 1.0

    news_geo = getattr(ctx, "news_geo", None)
    intel_flags = getattr(ctx, "intel_health_flags", {}) or {}
    risk_state = getattr(ctx, "risk_state", None)

    # Intel degraded handling
    if (
        intel_flags.get("intel_geo_score") == "DEGRADED"
        or intel_flags.get("intel_news_triggers") == "DEGRADED"
    ):
        qc = overlay.get("qc") or {}
        degraded_state = str(qc.get("if_intel_degraded", "WATCH"))
        state_hint = degraded_state
        geo_score = 0
        geo_conf = 0.0
    else:
        if not news_geo:
            return 1.0
        geo_score = int(news_geo.get("geo_score", 0))
        geo_conf = float(news_geo.get("geo_confidence", 0.0))
        # Prefer persisted risk_state.state over news_geo.state_hint
        state_from_risk = None
        if risk_state is not None:
            if isinstance(risk_state, dict):
                state_from_risk = risk_state.get("state")
            else:
                state_from_risk = getattr(risk_state, "state", None)
        if state_from_risk in ("WATCH", "ACTIVE", "COOLDOWN", "PAUSE"):
            state_hint = str(state_from_risk)
        else:
            state_hint = str(news_geo.get("state_hint", "WATCH"))

    # Confidence floor
    conf_floor = float(overlay.get("confidence_floor", 0.60) or 0.60)
    if geo_conf < conf_floor:
        return 1.0

    # Multiplier by geo_score overrides mapping/state
    by_geo = overlay.get("by_geo_score") or {}
    key = str(geo_score)
    if key in by_geo:
        multiplier = float(by_geo[key])
    else:
        mapping = overlay.get("mapping") or {}
        state_cfg = mapping.get(state_hint) or {}
        multiplier = float(state_cfg.get("multiplier", 1.0))

    if multiplier < 0.0:
        multiplier = 0.0
    if multiplier > 1.0:
        multiplier = 1.0
    return multiplier


def apply_exposure_multiplier_to_targets(
    target_positions: Any,
    multiplier: float,
    cash_symbol: str = "CASH",
) -> Any:
    """Apply exposure multiplier to target positions DataFrame.

    Scales target_weight and target_qty (if present) for all non-cash symbols.
    Optionally lets cash absorb the freed-up weight if a cash row is present.
    """
    if target_positions is None:
        return target_positions

    # No-op if multiplier is >= 1.0
    if multiplier >= 1.0:
        return target_positions

    # Import pandas lazily to avoid hard dependency at module import time
    try:
        import pandas as pd  # type: ignore[import]
    except Exception:
        return target_positions

    # Expect a pandas-like DataFrame with .columns and .copy()
    if not hasattr(target_positions, "columns"):
        return target_positions

    df = target_positions.copy()
    if df.empty:
        return df

    has_symbol = "symbol" in df.columns
    has_weight = "target_weight" in df.columns
    has_qty = "target_qty" in df.columns

    # Nothing to scale
    if not has_weight and not has_qty:
        return df

    if has_symbol:
        cash_mask = df["symbol"] == cash_symbol
        has_cash = bool(cash_mask.any())
        if has_cash:
            risky_mask = ~cash_mask
        else:
            risky_mask = pd.Series(True, index=df.index)
    else:
        has_cash = False
        cash_mask = None
        risky_mask = pd.Series(True, index=df.index)

    # Track original risky and cash weights (if available)
    risky_sum_before = 0.0
    if has_weight:
        risky_weights_before = pd.to_numeric(
            df.loc[risky_mask, "target_weight"],
            errors="coerce",
        ).fillna(0.0)
        risky_sum_before = float(risky_weights_before.sum())

        # Scale risky weights
        df.loc[risky_mask, "target_weight"] = risky_weights_before * multiplier

    # Scale quantities if present
    if has_qty:
        risky_qty_before = pd.to_numeric(
            df.loc[risky_mask, "target_qty"],
            errors="coerce",
        )
        df.loc[risky_mask, "target_qty"] = risky_qty_before * multiplier

    # Let cash absorb the freed-up weight (if weights and cash are present)
    if has_weight and has_cash and risky_sum_before != 0.0 and cash_mask is not None:
        risky_sum_after = risky_sum_before * multiplier
        delta_to_cash = risky_sum_before - risky_sum_after
        cash_before = (
            pd.to_numeric(
                df.loc[cash_mask, "target_weight"],
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        )
        df.loc[cash_mask, "target_weight"] = cash_before + delta_to_cash

    return df


__all__ = ["compute_exposure_multiplier", "apply_exposure_multiplier_to_targets"]
