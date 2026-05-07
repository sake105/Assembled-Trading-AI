"""hrp_sizing — apply_hrp_sizing() adapter for _tc_sizing dispatch.

Bridges the trading-cycle sizing dispatcher (_tc_sizing._sp_dispatch_sizing,
``sizing_method == "hrp"``) to the core HRP algorithm in
``hierarchical_risk_parity.compute_hrp_weights()``.

The dispatcher passes:
  - ``score_weights``    : dict[symbol, float]  — score-based baseline weights
  - ``prices``           : pd.DataFrame         — wide price panel (dates × symbols)
  - ``lookback_days``    : int                  — rolling window for return computation
  - ``blend``            : float in [0,1]       — HRP share (1-blend = score share)
  - ``target_invested_pct`` : float             — normalisation target
  - ``min_weight``       : float
  - ``max_weight``       : float

Returns: (blended_weights: dict[str, float], meta: dict)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_hrp_sizing(
    score_weights: dict[str, float],
    prices: pd.DataFrame,
    *,
    lookback_days: int = 60,
    blend: float = 0.7,
    target_invested_pct: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Apply HRP-blended sizing.

    Computes HRP weights from a rolling return window, then blends with
    score-based baseline weights::

        final_w = blend * hrp_w + (1 - blend) * score_w

    Normalises the result to ``target_invested_pct`` and clips to
    ``[min_weight, max_weight]``.

    Args:
        score_weights: Baseline per-symbol weights (from score-based sizing).
        prices: Wide-format price DataFrame (date index, symbol columns).
            Must contain at least the symbols in ``score_weights``.
        lookback_days: Number of trading days for return computation.
        blend: HRP weight share in [0, 1]. Default 0.7 (70% HRP, 30% score).
        target_invested_pct: Target gross exposure (default 1.0 = fully invested).
        min_weight: Minimum per-asset weight floor (default 0.0).
        max_weight: Maximum per-asset weight cap (default 1.0).

    Returns:
        Tuple of:
        - blended_weights: dict[symbol, float] (normalised, clipped)
        - meta: diagnostic dict with keys ``hrp_computed``, ``n_assets``,
          ``blend``, ``hrp_weights`` (dict or None)
    """
    from src.assembled_core.portfolio.hierarchical_risk_parity import (
        compute_hrp_weights,
    )

    meta: dict[str, Any] = {
        "hrp_computed": False,
        "n_assets": len(score_weights),
        "blend": blend,
        "hrp_weights": None,
    }

    symbols = list(score_weights.keys())
    if not symbols:
        return {}, meta

    # ------------------------------------------------------------------ #
    # Build return series for the requested symbols                        #
    # ------------------------------------------------------------------ #
    available_cols = [s for s in symbols if s in prices.columns]
    if len(available_cols) < 2:
        logger.warning(
            "[hrp_sizing] only %d symbols available in price panel — falling back to score weights",
            len(available_cols),
        )
        return dict(score_weights), meta

    # Slice lookback window
    price_slice = prices[available_cols].tail(lookback_days + 1)
    returns = price_slice.pct_change().dropna()

    if len(returns) < 30:
        logger.warning(
            "[hrp_sizing] insufficient return history (%d rows) — falling back to score weights",
            len(returns),
        )
        return dict(score_weights), meta

    # ------------------------------------------------------------------ #
    # Compute HRP weights                                                  #
    # ------------------------------------------------------------------ #
    hrp_raw = compute_hrp_weights(
        returns,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    if not hrp_raw:
        logger.warning("[hrp_sizing] compute_hrp_weights returned empty — falling back to score weights")
        return dict(score_weights), meta

    meta["hrp_computed"] = True
    meta["hrp_weights"] = hrp_raw

    # ------------------------------------------------------------------ #
    # Blend HRP with score weights                                         #
    # ------------------------------------------------------------------ #
    all_symbols = set(score_weights) | set(hrp_raw)
    blended: dict[str, float] = {}

    for sym in all_symbols:
        hrp_w = hrp_raw.get(sym, 0.0)
        score_w = score_weights.get(sym, 0.0)
        blended[sym] = blend * hrp_w + (1.0 - blend) * score_w

    # Apply weight bounds
    blended = {s: float(np.clip(w, min_weight, max_weight)) for s, w in blended.items()}

    # Normalise to target_invested_pct
    total = sum(blended.values())
    if total > 1e-8:
        scale = target_invested_pct / total
        blended = {s: round(w * scale, 6) for s, w in blended.items()}

    logger.info(
        "[hrp_sizing] blend=%.2f HRP-computed=%s n=%d total_w=%.4f",
        blend,
        meta["hrp_computed"],
        len(blended),
        sum(blended.values()),
    )

    return blended, meta


__all__ = ["apply_hrp_sizing"]
