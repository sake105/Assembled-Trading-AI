"""EDCL Phase C — Conviction-Score Engine.

Aggregates the raw TriggerBasket conviction from Phase B with:
  1. Historical event-type betas (via FeatureStore ASOF-join, PIT-safe)
  2. Optional market-confirmation signal (market_confirmation.py)
  3. Source-diversity bonus (N distinct source tiers)

Output is a calibrated conviction score [0, 1] suitable for use in:
  - edcl_conviction_multiplier in _sp_compute_final_multiplier (Phase A)
  - compute_news_dim_with_edcl in composite_score (Phase D)
  - Triple-confirmation logic (Phase H)

All FeatureStore calls are optional-dep guarded — engine degrades gracefully
to basket.conviction when no historical data is available.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_conviction_score(
    basket: Any,
    as_of: datetime | None = None,
    policy: dict[str, Any] | None = None,
    feature_store_root: str | None = None,
    options_iv_skew_z: float | None = None,
) -> float:
    """Compute a calibrated EDCL conviction score [0, 1].

    Algorithm (in order of reliability):
    1. Start with basket.conviction (raw geo-trigger score from Phase B).
    2. If FeatureStore is available: blend with median absolute historical beta
       for each fired trigger type (asset response in 5-day window).
    3. Apply diversity bonus: more fired triggers → slightly higher conviction.
    4. Apply source-diversity bonus if basket has n_high_conviction > 1.
    5. (Item 106) IV-skew pillar: elevated put/call skew signals institutional
       hedging → boost conviction by up to 3% when skew_z > 1.0.
    6. Clamp to [0, 1].

    Args:
        basket: TriggerBasket from build_trigger_basket() (Phase B).
        as_of: Inference timestamp for PIT-safe feature lookup. Defaults to now.
        policy: Policy dict — reads edcl_conviction_overlay sub-dict.
        feature_store_root: Override for FeatureStore root path.
        options_iv_skew_z: Optional IV put/call skew as z-score vs. 20d baseline.
            Positive = puts expensive vs calls (bearish hedging). Boosts conviction
            when > 1.0 (institutional hedging activity detected). Pass via
            ctx.options_iv_skew_z in the trading cycle.

    Returns:
        Calibrated conviction score in [0, 1].
    """
    if basket is None or basket.conviction == 0.0:
        return 0.0

    cfg = (policy or {}).get("edcl_conviction_overlay") or {}
    base = float(basket.conviction)

    # Historical beta enrichment (optional — needs FeatureStore + event-beta view)
    beta_boost = _try_fetch_beta_boost(basket, as_of, feature_store_root, cfg)

    # Trigger diversity bonus: each additional unique trigger type adds 2% (cap 10%)
    n_triggers = len(basket.fired_triggers)
    diversity_bonus = min(0.02 * max(n_triggers - 1, 0), 0.10)

    # Corroboration bonus: multiple high-conviction events
    n_high = getattr(basket, "n_high_conviction", 0)
    corroboration = 0.05 * min(n_high, 3) if n_high > 1 else 0.0

    # 5th pillar (Item 106): IV-skew — institutional hedging via options market
    # Elevated put/call skew (skew_z > 1.0) indicates market fear → conviction boost
    iv_skew_bonus = 0.0
    _iv_z = options_iv_skew_z
    if _iv_z is None:
        # Also accept from basket context (set by trading cycle via ctx.options_iv_skew_z)
        _iv_z = getattr(basket, "options_iv_skew_z", None)
    if _iv_z is not None and _iv_z == _iv_z:  # not NaN
        iv_skew_bonus = min(0.03, max(0.0, (float(_iv_z) - 1.0) * 0.015))

    raw = base * (1.0 + beta_boost) + diversity_bonus + corroboration + iv_skew_bonus
    score = min(1.0, max(0.0, raw))

    log.debug(
        "[EDCL-CONVICTION] base=%.3f beta_boost=%.3f diversity=%.3f "
        "corroboration=%.3f iv_skew_bonus=%.3f → %.3f",
        base,
        beta_boost,
        diversity_bonus,
        corroboration,
        iv_skew_bonus,
        score,
    )
    return score


def compute_event_beta(
    trigger_type_name: str,
    asset: str,
    lookback_days: int = 5,
    as_of: datetime | None = None,
    feature_store_root: str | None = None,
) -> float | None:
    """Return the median absolute return of `asset` in the `lookback_days` window
    following historical occurrences of `trigger_type_name`.

    Uses the 'event_beta' view in the FeatureStore (must be pre-computed by a
    training script). Returns None if data is not available.

    Args:
        trigger_type_name: TriggerType.name string (e.g. "ENERGY_SUPPLY_RISK").
        asset: Ticker symbol (e.g. "XLE").
        lookback_days: Forward-return window in trading days.
        as_of: PIT boundary — only use betas computed before this date.
        feature_store_root: Override FeatureStore root.

    Returns:
        Median absolute 5-day beta or None.
    """
    try:
        import pandas as pd
        from src.assembled_core.data.feature_store import read_features_asof

        ts = as_of if as_of is not None else datetime.now(timezone.utc)
        entities = pd.DataFrame([{"ticker": asset, "inference_ts": ts}])
        result = read_features_asof(
            view="event_beta",
            entities=entities,
            inference_ts_col="inference_ts",
            embargo_minutes=0,
            root=feature_store_root,
        )
        if result is None or result.empty:
            return None
        col = f"beta_{trigger_type_name}_{lookback_days}d"
        if col not in result.columns:
            return None
        val = result[col].dropna()
        return float(val.median()) if not val.empty else None
    except Exception as exc:
        log.debug(
            "event_beta lookup skipped (%s/%s): %s", trigger_type_name, asset, exc
        )
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_fetch_beta_boost(
    basket: Any,
    as_of: datetime | None,
    feature_store_root: str | None,
    cfg: dict[str, Any],
) -> float:
    """Return a [0, 0.3] beta-evidence boost, or 0.0 if unavailable."""
    if not basket.fired_triggers or not basket.affected_assets:
        return 0.0

    use_beta = cfg.get("use_historical_beta", True)
    if not use_beta:
        return 0.0

    betas: list[float] = []
    for ttype, _ in basket.fired_triggers[:3]:  # top-3 triggers only
        for asset in basket.affected_assets[:5]:  # top-5 assets
            b = compute_event_beta(
                trigger_type_name=ttype.name,
                asset=asset,
                as_of=as_of,
                feature_store_root=feature_store_root,
            )
            if b is not None:
                betas.append(abs(b))

    if not betas:
        return 0.0

    # Median beta → scale to [0, 0.3] boost
    sorted_betas = sorted(betas)
    n = len(sorted_betas)
    median_beta = (
        sorted_betas[n // 2]
        if n % 2 == 1
        else (sorted_betas[n // 2 - 1] + sorted_betas[n // 2]) / 2.0
    )
    # Typical asset event beta ~ 0.02–0.10 (2–10% move). Cap boost at beta >= 0.10.
    boost = min(median_beta / 0.10, 1.0) * 0.30
    log.debug(
        "[EDCL-CONVICTION] beta evidence: n=%d median=%.4f boost=%.3f",
        len(betas),
        median_beta,
        boost,
    )
    return boost


def compute_edcl_position_size(
    conviction: float,
    policy: dict[str, Any] | None = None,
    feature_row: "pd.Series | None" = None,
    conformal_model_path: str | None = None,
) -> dict[str, float]:
    """Phase E — Compute position size and stop-loss for an EDCL-triggered trade.

    Uses the conformal position model to derive:
    - A dynamic position size factor based on forecast uncertainty
    - A stop-loss level as the lower bound of the conformal prediction interval

    The final max_weight is:
        base_max_weight * conformal_factor * conviction_scale_factor

    Args:
        conviction: EDCL conviction score [0, 1] from compute_conviction_score().
        policy: Policy dict — reads edcl_conviction_overlay.edcl_sizing sub-dict.
        feature_row: Optional feature vector (pd.Series) for conformal inference.
            If None, falls back to median_interval_width from model artifact.
        conformal_model_path: Override model path. Falls back to policy.edcl_sizing.model_path.

    Returns:
        Dict with keys:
            max_weight: float — max portfolio weight for this EDCL trade [0, base_max]
            stop_loss_pct: float — stop-loss as fraction below entry (e.g. 0.05 = 5%)
            size_factor: float — combined conformal × conviction scaling [0, 1]
            conformal_factor: float — conformal uncertainty discount [0, 1]
    """
    edcl_cfg = (policy or {}).get("edcl_conviction_overlay") or {}
    sizing_cfg = edcl_cfg.get("edcl_sizing") or {}
    base_max = float(sizing_cfg.get("max_edcl_weight", 0.30))
    _target_coverage = float(
        sizing_cfg.get("target_coverage", 0.85)
    )  # used by conformal model if loaded

    # Default: no scaling (conformal model unavailable)
    conformal_factor = 1.0
    # ATR-aware fallback: stop_loss_pct = max(0.05, atr_20d * 2.5)
    # Volatile symbols get wider stops (e.g. BBAI 12% ATR → 30%), stable get tighter
    _atr_20d: float | None = None
    if feature_row is not None:
        for _atr_col in ("ta_atr_20_v1", "ta_atr_14_v1", "atr_norm", "ta_atr"):
            _v = feature_row.get(_atr_col) if hasattr(feature_row, "get") else None
            if _v is not None and float(_v) == float(_v) and float(_v) > 0:
                _atr_20d = float(_v)
                break
    if _atr_20d is not None:
        stop_loss_pct = max(0.05, _atr_20d * 2.5)
    else:
        stop_loss_pct = 0.05  # flat fallback when no ATR feature available

    model_path = conformal_model_path or sizing_cfg.get("model_path")
    if model_path:
        try:
            from pathlib import Path

            _path = Path(model_path)
            if not _path.is_absolute():
                # Resolve relative to project root (3 levels up from this file)
                _path = Path(__file__).parents[3] / model_path
            if _path.exists():
                # F-C4-N-1 R5: hash-verified load.
                from src.assembled_core.ml.model_registry import safe_load_model

                bundle = safe_load_model(_path, strict=False)
                med_width = float(bundle.get("median_interval_width", 0.05))

                if feature_row is not None:
                    # Full inference path using conformal model
                    try:
                        from src.assembled_core.portfolio.conformal_position import (
                            conformal_size_factor,
                        )

                        feat_cols = bundle.get("feature_cols", [])
                        row_aligned = feature_row.reindex(feat_cols, fill_value=0.0)
                        X = row_aligned.values.reshape(1, -1)
                        model = bundle.get("model")
                        if model is not None and hasattr(model, "predict"):
                            _y_pred = float(model.predict(X)[0])  # noqa: F841 — future: use for interval shift
                            # Use median_interval_width as proxy (no MAPIE re-inference here)
                            conformal_factor = conformal_size_factor(
                                interval_width=med_width,
                                max_width=float(
                                    bundle.get("max_interval_width", med_width * 2)
                                ),
                                min_factor=0.20,
                            )
                            stop_loss_pct = (
                                med_width / 2.0
                            )  # symmetric interval → half-width
                    except Exception as e:
                        log.debug("EDCL conformal inference skipped: %s", e)
                else:
                    # Fallback: use pre-computed median width for discount
                    from src.assembled_core.portfolio.conformal_position import (
                        conformal_size_factor,
                    )

                    conformal_factor = conformal_size_factor(
                        interval_width=med_width,
                        max_width=float(
                            bundle.get("max_interval_width", med_width * 2)
                        ),
                        min_factor=0.20,
                    )
                    stop_loss_pct = med_width / 2.0
        except Exception as e:
            log.debug("EDCL conformal model load skipped (%s): %s", model_path, e)

    # Conviction scaling: linear from threshold to 1.0
    threshold = float(edcl_cfg.get("conviction_threshold", 0.70))
    denom = 1.0 - threshold if threshold < 1.0 else 1.0
    conviction_scale = (
        min(1.0, max(0.0, (conviction - threshold) / denom))
        if conviction > threshold
        else 0.0
    )

    size_factor = conformal_factor * conviction_scale
    max_weight = base_max * size_factor

    log.debug(
        "[EDCL-SIZING] conviction=%.3f scale=%.3f conf_factor=%.3f → size_factor=%.3f max_weight=%.3f stop_loss=%.3f",
        conviction,
        conviction_scale,
        conformal_factor,
        size_factor,
        max_weight,
        stop_loss_pct,
    )
    return {
        "max_weight": float(max_weight),
        "stop_loss_pct": float(stop_loss_pct),
        "size_factor": float(size_factor),
        "conformal_factor": float(conformal_factor),
    }


__all__ = [
    "compute_conviction_score",
    "compute_event_beta",
    "compute_edcl_position_size",
]
