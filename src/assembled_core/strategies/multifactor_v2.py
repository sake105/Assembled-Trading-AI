"""STRATEGY-V2: Multi-factor strategy — delegating skeleton (Sprint 2 / Plan §4).

v2 is introduced as a feature-flagged, opt-in alternative to v1 so the production
paper-runner path remains fully stable while new layers (meta-model confidence
filter, regime-conditional factor weights, ATR-regime exits, sector-rotation
booster, crash-prediction integration) are added incrementally in later sprints.

This file is the **minimum viable skeleton**:
    * Reuses v1's `compute_signals` / `compute_target_positions` / `check_exit_signals`
      unchanged so behavior is identical when no v2-only features are enabled.
    * Adds a post-signal **meta-model confidence filter** hook:
        - gated by `strategy_cfg["meta_model"]["enabled"]` (default False)
        - loads a sklearn-compatible classifier from `model_path`
        - calls `predict_proba` on the factor columns and multiplies / filters
          `score` by confidence (>= `min_confidence`, default 0.55)
        - on any failure (missing file, column mismatch, import error) falls back
          silently to v1 output (defensive rollout)

Later sprints will:
    * replace the v1 delegation with a native 30-factor computation
    * wire regime-conditional weights (trained offline)
    * wire ATR-regime exits in `check_exit_signals_v2`
    * wire sector-rotation booster pre-scoring

Selection of v2 is done via `strategy.name: multifactor_v2` in `configs/app.yaml`
(the paper_runner strategy dispatch picks up the name; see paper_runner.py).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.assembled_core.strategies.multifactor_v1 import (
    check_exit_signals as _v1_check_exit_signals,
)
from src.assembled_core.strategies.multifactor_v1 import (
    compute_signals as _v1_compute_signals,
)
from src.assembled_core.strategies.multifactor_v1 import (
    compute_target_positions as _v1_compute_target_positions,
)

logger = logging.getLogger(__name__)

STRATEGY_VERSION = "v2"
# Sprint 2/3 scaffold marker — current module is a pass-through to v1. The
# 15 new factors, regime-conditional weights, and meta-model filter will be
# added in follow-up commits. Keep this string ending in ".skeleton" until
# the native v2 factor computation lands.
VERSION = "multifactor_v2.skeleton"


# ---------------------------------------------------------------------------
# 30-factor target contract (Sprint 2 plan §4.1)
# ---------------------------------------------------------------------------
#
# This list encodes the TARGET CONTRACT for v2 — it is documentation-in-code,
# NOT used in the current pass-through compute path. It mirrors the table in
# section 4.1 of the Sprint 2 plan (lively-hugging-dolphin.md lines ~410-455):
# factors 1-15 are the existing v1 factors (re-weighted), factors 16-29 are
# new additive alpha sources, and factor 30 is the meta-model confidence
# which is applied as a MULTIPLIER on the composite, not as an additive
# summand. The additive weights are normalized to sum to 1.0 on return.
#
# The plan text explicitly notes a minor normalization is needed (raw sums
# do not land exactly at 1.0); we resolve that here via a single proportional
# rescale of the 29 additive weights. The meta_model factor carries
# weight=0.0 and kind="multiplicative" so downstream code can branch on kind.
_FACTOR_LIST_V2_RAW: list[dict[str, Any]] = [
    # --- existing v1 factors (re-calibrated weights) -----------------------
    {"id": 1, "name": "trend_ema_spread", "dimension": "trend", "kind": "additive", "weight": 0.10},
    {"id": 2, "name": "trend_ma200_position", "dimension": "trend", "kind": "additive", "weight": 0.07},
    {"id": 3, "name": "trend_adx_strength", "dimension": "trend", "kind": "additive", "weight": 0.05},
    {"id": 4, "name": "trend_macd_hist", "dimension": "trend", "kind": "additive", "weight": 0.05},
    {"id": 5, "name": "mom_rsi_centered", "dimension": "momentum", "kind": "additive", "weight": 0.06},
    {"id": 6, "name": "mom_volume_weighted", "dimension": "momentum", "kind": "additive", "weight": 0.05},
    {"id": 7, "name": "mom_obv_trend", "dimension": "momentum", "kind": "additive", "weight": 0.03},
    {"id": 8, "name": "mr_bollinger_pctb", "dimension": "mean_reversion", "kind": "additive", "weight": 0.04},
    {"id": 9, "name": "mr_stoch_oversold", "dimension": "mean_reversion", "kind": "additive", "weight": 0.03},
    {"id": 10, "name": "vol_abnormal", "dimension": "volume", "kind": "additive", "weight": 0.03},
    {"id": 11, "name": "vol_tick_imbalance", "dimension": "volume", "kind": "additive", "weight": 0.03},
    {"id": 12, "name": "vola_regime_score", "dimension": "volatility", "kind": "additive", "weight": 0.03},
    {"id": 13, "name": "vola_vov_penalty", "dimension": "volatility", "kind": "additive", "weight": 0.03},
    {"id": 14, "name": "breadth_above_ma", "dimension": "breadth", "kind": "additive", "weight": 0.04},
    {"id": 15, "name": "breadth_ad_line", "dimension": "breadth", "kind": "additive", "weight": 0.03},
    # --- new v2 factors ----------------------------------------------------
    {"id": 16, "name": "mr_zscore_reversal_3d", "dimension": "mean_reversion", "kind": "additive", "weight": 0.03},
    {"id": 17, "name": "mr_rsi_extreme_uptrend", "dimension": "mean_reversion", "kind": "additive", "weight": 0.03},
    {"id": 18, "name": "sector_rotation_bias", "dimension": "sector", "kind": "additive", "weight": 0.05},
    {"id": 19, "name": "earnings_surprise_z", "dimension": "event", "kind": "additive", "weight": 0.04},
    {"id": 20, "name": "insider_activity_score", "dimension": "event", "kind": "additive", "weight": 0.03},
    {"id": 21, "name": "news_sentiment_7d", "dimension": "news", "kind": "additive", "weight": 0.03},
    {"id": 22, "name": "news_volume_spike", "dimension": "news", "kind": "additive", "weight": 0.02},
    {"id": 23, "name": "macro_growth_momentum", "dimension": "macro", "kind": "additive", "weight": 0.02},
    {"id": 24, "name": "macro_inflation_surprise", "dimension": "macro", "kind": "additive", "weight": 0.02},
    {"id": 25, "name": "intermarket_bond_equity", "dimension": "intermarket", "kind": "additive", "weight": 0.02},
    {"id": 26, "name": "intermarket_credit_spread", "dimension": "intermarket", "kind": "additive", "weight": 0.01},
    {"id": 27, "name": "options_put_call_extreme", "dimension": "options", "kind": "additive", "weight": 0.02},
    {"id": 28, "name": "vix_regime_score", "dimension": "options", "kind": "additive", "weight": 0.02},
    {"id": 29, "name": "crash_probability_inverse", "dimension": "risk", "kind": "additive", "weight": 0.03},
    # --- multiplicative meta-model filter (NOT additive) -------------------
    {"id": 30, "name": "meta_model_confidence", "dimension": "ml", "kind": "multiplicative", "weight": 0.0},
]


def _get_factor_list_v2() -> list[dict[str, Any]]:
    """Return the 30-factor target contract for multifactor_v2.

    This is documentation-in-code. The current compute path is a pass-through
    to v1 and does NOT use this list. It exists so downstream work (regime
    weight training, meta-model training, factor-store coverage checks) has
    a single authoritative reference for the target 30-factor stack.

    Additive weights are proportionally rescaled so that their sum equals
    1.0 exactly, matching the plan's stated "normalize to 1.0" intent. The
    multiplicative factor (id=30, meta_model_confidence) always carries
    weight=0.0 and must be applied as a multiplier on the composite.

    Returns:
        A fresh list of 30 dict entries, each with keys:
            id (int), name (str), dimension (str), kind (str), weight (float).
    """
    factors = [dict(f) for f in _FACTOR_LIST_V2_RAW]
    additive = [f for f in factors if f["kind"] == "additive"]
    raw_sum = sum(f["weight"] for f in additive)
    if raw_sum > 0:
        for f in additive:
            f["weight"] = f["weight"] / raw_sum
    return factors


def compute_signals(
    prices_with_features: pd.DataFrame,
    strategy_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """v2 entry point — delegates to v1 and optionally applies meta-model filter.

    The signature and return schema are identical to v1, so this function is a
    drop-in replacement from the paper-runner / trading_cycle perspective.

    Meta-model filter (opt-in):
        strategy_cfg["meta_model"] = {
            "enabled": True,
            "model_path": "models/meta_model_v1.joblib",
            "min_confidence": 0.55,
        }
    """
    signals = _v1_compute_signals(prices_with_features, strategy_cfg)
    if signals is None or signals.empty:
        return signals

    cfg = strategy_cfg or {}
    meta_cfg = cfg.get("meta_model") or {}
    if not meta_cfg.get("enabled", False):
        return signals

    try:
        signals = _apply_meta_model_filter(signals, meta_cfg)
    except Exception as exc:  # noqa: BLE001 — defensive fallback
        logger.debug("[multifactor_v2] meta_model filter skipped: %s", exc)
        return signals

    return signals


def _apply_meta_model_filter(
    signals: pd.DataFrame,
    meta_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Load classifier, compute confidence, filter / scale signals.

    Defensive: any failure (missing joblib, missing file, column mismatch,
    predict error) raises and the caller falls back to unfiltered signals.
    """
    import os

    model_path = meta_cfg.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"meta_model path missing: {model_path}")

    min_conf = float(meta_cfg.get("min_confidence", 0.55))

    import joblib  # local import — optional dep

    model = joblib.load(model_path)

    # Feature columns: prefer explicit list from cfg, else use all numeric
    # columns that start with factor prefixes known from v1.
    feature_cols = meta_cfg.get("feature_cols")
    if not feature_cols:
        feature_cols = [
            c
            for c in signals.columns
            if c.startswith(("trend_", "mom_", "mr_", "vol_", "vola_", "breadth_"))
        ]
    if not feature_cols:
        raise ValueError("meta_model: no feature columns available on signals")

    missing = [c for c in feature_cols if c not in signals.columns]
    if missing:
        raise ValueError(f"meta_model: missing feature columns: {missing}")

    X = signals[feature_cols].fillna(0.0).to_numpy()
    proba = model.predict_proba(X)
    # binary classifier: positive-class probability is column 1
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(f"meta_model: unexpected predict_proba shape {proba.shape}")
    confidence = proba[:, 1]

    out = signals.copy()
    out["meta_confidence"] = confidence
    mask_keep = confidence >= min_conf
    out = out.loc[mask_keep].copy()
    if out.empty:
        return out
    # Scale score by confidence so downstream sizing prefers high-confidence picks
    if "score" in out.columns:
        out["score"] = out["score"].astype(float) * out["meta_confidence"].astype(float)
    return out.reset_index(drop=True)


def compute_target_positions(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """v2 target positions — delegates to v1 unchanged."""
    return _v1_compute_target_positions(*args, **kwargs)


def check_exit_signals(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """v2 exit signals — delegates to v1 unchanged.

    Sprint 3 will replace this with an ATR-regime-aware exit engine.
    """
    return _v1_check_exit_signals(*args, **kwargs)
