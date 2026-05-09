"""hrp_sizing — apply_hrp_sizing() adapter for _tc_sizing dispatch.

Bridges the trading-cycle sizing dispatcher (_tc_sizing._sp_dispatch_sizing,
``sizing_method == "hrp"``) to the core HRP algorithm in
``hierarchical_risk_parity.compute_hrp_weights()``.

The dispatcher passes:
  - ``score_weights``    : dict[symbol, float]  — score-based baseline weights
  - ``prices``           : pd.DataFrame         — price panel (long or wide format)
  - ``lookback_days``    : int                  — rolling window for return computation
  - ``blend``            : float in [0,1]       — HRP share (1-blend = score share)
  - ``target_invested_pct`` : float             — normalisation target
  - ``min_weight``       : float
  - ``max_weight``       : float

Returns: (blended_weights: dict[str, float], reasons: list[str])
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_wide_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format price panel (with 'symbol'+'close' cols) to wide format."""
    if prices.empty:
        return prices
    if "symbol" in prices.columns and "close" in prices.columns:
        ts_col = next(
            (c for c in ["timestamp", "date", "datetime"] if c in prices.columns),
            None,
        )
        pivot_index = ts_col if ts_col else prices.index
        wide = prices.pivot_table(index=pivot_index, columns="symbol", values="close")
        wide.index = pd.to_datetime(wide.index)
        return wide
    return prices


def apply_hrp_sizing(
    score_weights: dict[str, float],
    prices: pd.DataFrame,
    *,
    lookback_days: int = 60,
    blend: float = 0.7,
    target_invested_pct: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> tuple[dict[str, float], list[str]]:
    """Apply HRP-blended sizing.

    Computes HRP weights from a rolling return window, then blends with
    score-based baseline weights::

        final_w = blend * hrp_w + (1 - blend) * score_w

    Normalises the result to ``target_invested_pct`` and clips to
    ``[min_weight, max_weight]``.

    Args:
        score_weights: Baseline per-symbol weights (from score-based sizing).
        prices: Price DataFrame — either wide (date × symbol) or long-format
            with 'symbol' and 'close' columns (auto-pivoted).
        lookback_days: Number of trading days for return computation.
        blend: HRP weight share in [0, 1]. Default 0.7 (70% HRP, 30% score).
        target_invested_pct: Target gross exposure (default 1.0 = fully invested).
        min_weight: Minimum per-asset weight floor (default 0.0).
        max_weight: Maximum per-asset weight cap (default 1.0).

    Returns:
        Tuple of (blended_weights dict, reasons list).
    """
    from src.assembled_core.portfolio.hierarchical_risk_parity import (
        compute_hrp_weights,
    )

    reasons: list[str] = []
    symbols = list(score_weights.keys())
    if not symbols:
        return {}, reasons

    wide = _to_wide_prices(prices)

    def _fallback_normalized() -> tuple[dict[str, float], list[str]]:
        total = sum(score_weights.values())
        if total > 1e-8:
            scale = target_invested_pct / total
            return {s: round(w * scale, 6) for s, w in score_weights.items()}, reasons
        return dict(score_weights), reasons

    available_cols = [s for s in symbols if s in wide.columns]
    if len(available_cols) < 2:
        reasons.append(
            f"insufficient price data: {len(available_cols)} symbols in panel, "
            "falling back to score weights"
        )
        return _fallback_normalized()

    price_slice = wide[available_cols].tail(lookback_days + 1)
    returns = price_slice.pct_change().dropna()

    if len(returns) < 30:
        reasons.append(
            f"insufficient return history ({len(returns)} rows), "
            "falling back to score weights"
        )
        return _fallback_normalized()

    hrp_raw = compute_hrp_weights(returns, min_weight=min_weight, max_weight=max_weight)

    if not hrp_raw:
        reasons.append(
            "compute_hrp_weights returned empty, falling back to score weights"
        )
        return _fallback_normalized()

    all_symbols = set(score_weights)  # restrict to signal-selected symbols only
    blended: dict[str, float] = {}
    for sym in all_symbols:
        hrp_w = hrp_raw.get(sym, 0.0)
        score_w = score_weights.get(sym, 0.0)
        blended[sym] = blend * hrp_w + (1.0 - blend) * score_w

    blended = {s: float(np.clip(w, min_weight, max_weight)) for s, w in blended.items()}

    total = sum(blended.values())
    if total > 1e-8:
        scale = target_invested_pct / total
        blended = {s: round(w * scale, 6) for s, w in blended.items()}

    reasons.append(f"blended HRP weights (blend={blend:.2f}, n={len(blended)})")
    logger.info(
        "[hrp_sizing] blend=%.2f n=%d total_w=%.4f",
        blend,
        len(blended),
        sum(blended.values()),
    )
    return blended, reasons


def apply_hrp_sizing_from_policy(
    score_weights: dict[str, float],
    prices: pd.DataFrame,
    policy: dict,
) -> tuple[dict[str, float], list[str]]:
    """Apply HRP sizing controlled by a policy dict.

    Args:
        score_weights: Baseline per-symbol weights.
        prices: Price DataFrame (wide or long format).
        policy: Dict with optional ``hrp_sizing`` key containing:
            enabled, lookback_days, blend, target_invested_pct, min_weight, max_weight.

    Returns:
        (weights, reasons) — unchanged score copy when disabled.
    """
    cfg = policy.get("hrp_sizing", {})
    if not cfg.get("enabled", False):
        return dict(score_weights), []

    return apply_hrp_sizing(
        score_weights,
        prices,
        lookback_days=int(cfg.get("lookback_days", 60)),
        blend=float(cfg.get("blend", 0.7)),
        target_invested_pct=float(cfg.get("target_invested_pct", 1.0)),
        min_weight=float(cfg.get("min_weight", 0.0)),
        max_weight=float(cfg.get("max_weight", 1.0)),
    )


def compute_hrp_target_weights(
    returns_panel: pd.DataFrame,
    *,
    target_gross: float = 1.0,
    min_history: int = 30,
) -> pd.Series:
    """Compute HRP weights directly from a wide-format returns panel.

    Args:
        returns_panel: Wide DataFrame of daily returns (dates × symbols).
        target_gross: Target sum of weights (default 1.0 = fully invested).
        min_history: Minimum number of rows required (raises ValueError if short).

    Returns:
        pd.Series with name="hrp_weight", scaled to ``target_gross``.

    Raises:
        ValueError: For invalid inputs (not DataFrame, single symbol,
            insufficient history, non-positive target_gross).
    """
    from src.assembled_core.portfolio.hierarchical_risk_parity import (
        compute_hrp_weights,
    )

    if not isinstance(returns_panel, pd.DataFrame):
        raise ValueError(
            f"returns_panel must be a DataFrame, got {type(returns_panel).__name__}"
        )
    if returns_panel.shape[1] < 2:
        raise ValueError(
            f"at least 2 symbols required for HRP computation, got {returns_panel.shape[1]}"
        )
    if len(returns_panel) < min_history:
        raise ValueError(
            f"insufficient history: {len(returns_panel)} rows < {min_history} required"
        )
    if target_gross <= 0:
        raise ValueError(f"target_gross must be positive, got {target_gross}")

    hrp_raw = compute_hrp_weights(returns_panel)
    if not hrp_raw:
        raise ValueError("HRP computation returned empty weights")

    series = pd.Series(hrp_raw, name="hrp_weight")
    total = series.sum()
    if total > 1e-8:
        series = series * (target_gross / total)
    return series


def blend_hrp_with_score(
    hrp: pd.Series,
    score: pd.Series,
    *,
    hrp_alpha: float = 0.7,
) -> pd.Series:
    """Blend HRP weights with score-based weights.

    Handles disjoint symbol sets by filling missing values with 0.
    Re-normalises the output to ``max(hrp.sum(), score.sum())``.

    Args:
        hrp: HRP weights series (name="hrp_weight" convention).
        score: Score-based weights series.
        hrp_alpha: Weight given to HRP in [0, 1]. Default 0.7.

    Returns:
        Blended pd.Series indexed over the union of both series.

    Raises:
        ValueError: When ``hrp_alpha`` is outside [0, 1].
    """
    if not (0.0 <= hrp_alpha <= 1.0):
        raise ValueError(f"hrp_alpha must be in [0, 1], got {hrp_alpha}")

    all_symbols = hrp.index.union(score.index)
    hrp_aligned = hrp.reindex(all_symbols, fill_value=0.0)
    score_aligned = score.reindex(all_symbols, fill_value=0.0)

    blended = hrp_alpha * hrp_aligned + (1.0 - hrp_alpha) * score_aligned

    target = max(hrp.sum(), score.sum())
    total = blended.sum()
    if total > 1e-8:
        blended = blended * (target / total)

    return blended


__all__ = [
    "apply_hrp_sizing",
    "apply_hrp_sizing_from_policy",
    "blend_hrp_with_score",
    "compute_hrp_target_weights",
]
