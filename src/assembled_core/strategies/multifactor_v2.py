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
