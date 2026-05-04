from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.assembled_core.pipeline.trading_cycle_shared import TradingContext


def compute_exposure_multiplier(ctx: "TradingContext", policy: Dict[str, Any]) -> float:
    """Compute exposure multiplier from GeoRisk overlay policy and TradingContext.

    Rules (v2 — EDCL-aware):
    - If georisk_overlay.enabled is false -> 1.0
    - If ctx.news_geo is None -> 1.0 (unless intel degraded handling applies)
    - If intel_geo_score or intel_news_triggers is DEGRADED:
        - Use qc.if_intel_degraded state_hint (default WATCH)
        - geo_score=0, geo_confidence=0.0
    - If geo_confidence < confidence_floor -> 1.0
    - state_hint: from ctx.risk_state.state if present, else ctx.news_geo.state_hint
    - If by_geo_score has key for geo_score -> use it; else mapping[state_hint].multiplier
    - Clamp multiplier to [0.0, max_geo_multiplier] (default 2.0 to allow EDCL upscaling)
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

    max_geo_mult = float(overlay.get("max_geo_multiplier", 2.0))
    if multiplier < 0.0:
        multiplier = 0.0
    if multiplier > max_geo_mult:
        multiplier = max_geo_mult
    return multiplier


def apply_exposure_multiplier_to_targets(
    target_positions: Any,
    multiplier: float,
    cash_symbol: str = "CASH",
    max_gross_exposure: float | None = None,
) -> Any:
    """Apply exposure multiplier to target positions DataFrame (bidirectional).

    Scales target_weight and target_qty (if present) for all non-cash symbols.
    - Downscaling (multiplier < 1.0): cash absorbs freed weight.
    - Upscaling (multiplier > 1.0): risky weights are increased proportionally;
      if max_gross_exposure is set and would be exceeded, weights are normalized
      to that ceiling.
    """
    if target_positions is None:
        return target_positions

    if abs(multiplier - 1.0) < 1e-9:
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

    # Track original risky weights
    risky_sum_before = 0.0
    if has_weight:
        risky_weights_before = pd.to_numeric(
            df.loc[risky_mask, "target_weight"],
            errors="coerce",
        ).fillna(0.0)
        risky_sum_before = float(risky_weights_before.sum())
        df.loc[risky_mask, "target_weight"] = risky_weights_before * multiplier

    # Scale quantities if present
    if has_qty:
        risky_qty_before = pd.to_numeric(
            df.loc[risky_mask, "target_qty"],
            errors="coerce",
        )
        df.loc[risky_mask, "target_qty"] = risky_qty_before * multiplier

    if has_weight:
        risky_sum_after = risky_sum_before * multiplier

        if multiplier < 1.0:
            # Downscaling: cash absorbs the freed-up weight
            if has_cash and risky_sum_before != 0.0 and cash_mask is not None:
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
            elif not has_cash and risky_sum_before != 0.0:
                freed = risky_sum_before - risky_sum_after
                log.warning(
                    "[WARN] apply_exposure_multiplier_to_targets: multiplier=%.4f < 1.0 "
                    "but no CASH row found in target_positions — %.6f freed weight silently lost. "
                    "Add a CASH row or ensure the caller handles the weight gap.",
                    multiplier,
                    freed,
                )

        elif multiplier > 1.0 and max_gross_exposure is not None:
            # Upscaling: enforce max_gross_exposure ceiling
            total_abs = float(
                pd.to_numeric(df.loc[risky_mask, "target_weight"], errors="coerce")
                .fillna(0.0)
                .abs()
                .sum()
            )
            if total_abs > max_gross_exposure:
                scale_down = max_gross_exposure / total_abs
                df.loc[risky_mask, "target_weight"] = (
                    pd.to_numeric(
                        df.loc[risky_mask, "target_weight"], errors="coerce"
                    ).fillna(0.0)
                    * scale_down
                )

    return df


def compute_edcl_conviction_multiplier(
    ctx: "TradingContext", policy: Dict[str, Any]
) -> float:
    """Compute EDCL conviction-based exposure multiplier [1.0, max_multiplier].

    Only fires when edcl_conviction_overlay.enabled=true and ctx.edcl_state.conviction
    exceeds conviction_threshold. Returns 1.0 (no-op) in all other cases.
    """
    edcl_cfg = (policy or {}).get("edcl_conviction_overlay") or {}
    if not edcl_cfg.get("enabled", False):
        return 1.0

    # By default EDCL conviction upscaling only applies live/paper — not backtest.
    mode = getattr(ctx, "mode", "backtest")
    if mode not in ("live", "paper") and not edcl_cfg.get("allow_in_backtest", False):
        return 1.0

    edcl_state = getattr(ctx, "edcl_state", None) or {}
    conviction = float(edcl_state.get("conviction", 0.0))
    threshold = float(edcl_cfg.get("conviction_threshold", 0.70))
    if conviction < threshold:
        return 1.0

    max_mult = float(edcl_cfg.get("max_multiplier", 2.0))
    denom = 1.0 - threshold if threshold < 1.0 else 1.0
    scale = (conviction - threshold) / denom
    multiplier = 1.0 + scale * (max_mult - 1.0)
    return min(multiplier, max_mult)


def get_market_implied_geo_signal(
    policy: dict | None = None,
    use_polymarket: bool = True,
    use_kalshi: bool = True,
    poly_weight: float = 0.6,
) -> dict:
    """Aggregate prediction-market geo-risk signals from Polymarket and/or Kalshi.

    Fetches live market probabilities from CFTC-regulated prediction markets and
    returns a blended geo-risk signal dict suitable for use in exposure calculations.

    Args:
        policy: Policy dict (unused, kept for interface parity).
        use_polymarket: Include Polymarket (T1.5 source).
        use_kalshi: Include Kalshi (T1.5 source).
        poly_weight: Polymarket weight in blended signal (Kalshi = 1 - poly_weight).

    Returns:
        Dict with keys: signal [0,1], source, n_sources, poly_signal, kals_signal.
    """
    poly_sig = None
    kals_sig = None

    if use_polymarket:
        try:
            from assembled_core.data.sources.polymarket_source import (
                get_market_implied_geo_signal as _poly_signal,
            )

            poly_sig = _poly_signal(policy=policy)
        except Exception as _exc:
            log.debug("[GeoRisk] polymarket signal failed: %s", _exc)

    if use_kalshi:
        try:
            from assembled_core.data.sources.kalshi_source import (
                fetch_combined_prediction_signal,
            )
            from assembled_core.data.sources.kalshi_source import (
                get_market_implied_geo_signal as _kals_signal,
            )

            kals_sig = _kals_signal()
        except Exception as _exc:
            log.debug("[GeoRisk] kalshi signal failed: %s", _exc)

    if poly_sig is None and kals_sig is None:
        return {"signal": 0.0, "source": "prediction_markets_combined", "n_sources": 0}

    try:
        from assembled_core.data.sources.kalshi_source import (
            fetch_combined_prediction_signal,
        )

        return fetch_combined_prediction_signal(poly_sig, kals_sig, poly_weight)
    except Exception:
        # Fallback: use whichever signal is available
        available = poly_sig or kals_sig
        return available or {
            "signal": 0.0,
            "source": "prediction_markets_combined",
            "n_sources": 0,
        }


__all__ = [
    "compute_exposure_multiplier",
    "apply_exposure_multiplier_to_targets",
    "compute_edcl_conviction_multiplier",
    "get_market_implied_geo_signal",
]
