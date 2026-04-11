"""Black-Litterman sizing wrapper (Sprint 3 / Plan W11).

Thin wrapper around :class:`BlackLittermanOptimizer.optimize_from_scores`
that turns a score-based preliminary allocation plus a price panel into BL
posterior weights. Designed as the ``method="bl"`` branch for strategy
sizing callers.

Characteristics:
  - pure function, never mutates inputs
  - scipy is optional; when missing, falls back to the input ``score_weights``
  - builds an annualised sample covariance from the provided price panel
  - scales the final weights to ``target_invested_pct``
  - reuses ``optimize_from_scores`` so views are driven by factor scores,
    not magic constants
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.assembled_core.portfolio.black_litterman import (
    SCIPY_AVAILABLE,
    BlackLittermanOptimizer,
)

_ANNUALISATION = 252.0


def _pivot_returns(
    prices: pd.DataFrame,
    symbols: list[str],
    lookback_days: int,
) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    required = {"timestamp", "symbol", "close"}
    if not required.issubset(prices.columns):
        return pd.DataFrame()

    rows = prices[prices["symbol"].isin(symbols)].copy()
    if rows.empty:
        return pd.DataFrame()

    rows = rows.sort_values(["symbol", "timestamp"])
    pivot = rows.pivot_table(
        index="timestamp",
        columns="symbol",
        values="close",
        aggfunc="last",
    )
    if len(pivot) > lookback_days:
        pivot = pivot.iloc[-lookback_days:]
    return pivot.pct_change().dropna(how="all")


def _sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised sample covariance; safe on empty input."""
    if returns is None or returns.empty:
        return pd.DataFrame()
    sigma_daily = returns.cov()
    sigma_annual = sigma_daily * _ANNUALISATION
    return sigma_annual


def apply_bl_sizing(
    score_weights: dict[str, float],
    prices: pd.DataFrame,
    *,
    lookback_days: int = 60,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    max_position: float = 0.15,
    confidence: float = 0.5,
    return_scale: float = 0.10,
    target_invested_pct: float = 1.0,
) -> tuple[dict[str, float], list[str]]:
    """Produce BL posterior weights from score-based preliminary weights.

    Args:
        score_weights: Preliminary weights from score-based sizing.
        prices: Price DataFrame with ``timestamp``, ``symbol``, ``close``.
        lookback_days: Rolling window for the return panel.
        risk_aversion: BL market risk aversion (``delta``).
        tau: BL prior uncertainty scaling.
        max_position: Per-symbol position cap (passed to the optimiser).
        confidence: Uniform view confidence (passed to
            ``optimize_from_scores``).
        return_scale: Max absolute view magnitude in decimal (passed to
            ``optimize_from_scores``).
        target_invested_pct: Final scaling target.

    Returns:
        ``(adjusted_weights, reasons)`` tuple. When scipy is missing, data is
        insufficient, or optimisation fails, the function returns the scaled
        score weights and records the reason in ``reasons``.
    """
    if not score_weights:
        return {}, []

    reasons: list[str] = []
    symbols = list(score_weights.keys())

    def _fallback_score_only(reason: str) -> tuple[dict[str, float], list[str]]:
        reasons.append(f"bl_sizing: {reason}; falling back to score")
        total = sum(abs(v) for v in score_weights.values())
        if total <= 1e-12:
            return dict(score_weights), reasons
        scale = float(target_invested_pct) / total
        return {s: score_weights[s] * scale for s in symbols}, reasons

    if not SCIPY_AVAILABLE:
        return _fallback_score_only("scipy not available")

    returns = _pivot_returns(prices, symbols, lookback_days)
    usable = [c for c in symbols if c in returns.columns]
    if len(usable) < 2 or len(returns) < 30:
        return _fallback_score_only(
            f"insufficient data (symbols={len(usable)}, rows={len(returns)})"
        )

    sigma = _sample_covariance(returns[usable])
    if sigma.empty or sigma.shape[0] != sigma.shape[1]:
        return _fallback_score_only("covariance computation failed")

    scores_series = pd.Series({s: score_weights.get(s, 0.0) for s in usable})

    try:
        optimiser = BlackLittermanOptimizer(
            risk_aversion=risk_aversion,
            tau=tau,
            max_position=max_position,
        )
        weights_series = optimiser.optimize_from_scores(
            scores_series,
            sigma,
            confidence=confidence,
            return_scale=return_scale,
        )
    except Exception as exc:  # noqa: BLE001 - defensive optimiser wrap
        return _fallback_score_only(f"BL optimisation failed ({exc})")

    bl_weights: dict[str, float] = {}
    for sym in symbols:
        val = weights_series.get(sym, np.nan) if hasattr(weights_series, "get") else np.nan
        if val is None or (isinstance(val, float) and np.isnan(val)):
            # Fall back to score for symbols missing from the BL output
            bl_weights[sym] = float(score_weights.get(sym, 0.0))
        else:
            bl_weights[sym] = float(val)

    total = sum(abs(v) for v in bl_weights.values())
    if total <= 1e-12:
        return _fallback_score_only("BL returned all-zero weights")

    scale = float(target_invested_pct) / total
    bl_weights = {s: w * scale for s, w in bl_weights.items()}
    reasons.append(
        f"bl_sizing: BL posterior on {len(usable)} symbols, "
        f"tau={tau:.3f}, risk_aversion={risk_aversion:.2f}, "
        f"scaled to target_invested_pct={target_invested_pct:.3f}"
    )
    return bl_weights, reasons


def apply_bl_sizing_from_policy(
    score_weights: dict[str, float],
    prices: pd.DataFrame,
    policy: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    """Read BL config from ``policy['bl_sizing']`` and apply."""
    cfg = (policy or {}).get("bl_sizing") or {}
    if not cfg.get("enabled", False):
        return dict(score_weights), []

    return apply_bl_sizing(
        score_weights,
        prices,
        lookback_days=int(cfg.get("lookback_days", 60) or 60),
        risk_aversion=float(cfg.get("risk_aversion", 2.5) or 2.5),
        tau=float(cfg.get("tau", 0.05) or 0.05),
        max_position=float(cfg.get("max_position", 0.15) or 0.15),
        confidence=float(cfg.get("confidence", 0.5) or 0.5),
        return_scale=float(cfg.get("return_scale", 0.10) or 0.10),
        target_invested_pct=float(cfg.get("target_invested_pct", 1.0) or 1.0),
    )


__all__ = ["apply_bl_sizing", "apply_bl_sizing_from_policy"]
