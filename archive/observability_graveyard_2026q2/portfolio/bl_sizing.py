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


def compute_bl_target_weights(
    returns_panel: pd.DataFrame,
    view_scores: pd.Series,
    tau: float = 0.05,
    target_gross: float = 0.80,
    equal_weight_prior: bool = True,
) -> pd.Series:
    """Compute Black-Litterman target weights from a wide returns panel and views.

    Inline BL math (does not call :class:`BlackLittermanOptimizer`) to keep the
    sidecar additive and independent of the dormant optimiser API. The dormant
    module expects a scipy-based Sharpe maximiser which is heavier than needed
    here; this function implements the analytic BL posterior directly via
    ``numpy.linalg.solve`` and then maps the posterior to long-only weights
    through an inverse-variance scaling. That avoids dragging cvxpy/scipy.optimize
    into the call path and keeps the function deterministic.

    Assumptions / simplifications (documented, not implicit):
      - Prior ``w_mkt = 1/n`` for every symbol when ``equal_weight_prior`` is
        True. Real market-cap weights are a future upgrade.
      - Risk aversion ``delta = 2.5`` (Litterman's equity convention, hardcoded).
      - View expected returns = ``z_score * 0.10`` (crude calibration: 10%
        annual return per 1-sigma signal).
      - Pick-matrix ``P`` is the identity (one view per asset, in the order of
        ``returns_panel.columns``).
      - View uncertainty ``Omega = diag(tau * sigma_i^2)`` where ``sigma_i^2``
        is taken from the diagonal of ``tau * Sigma``.
      - Posterior -> weights via ``w_i proportional to max(E[R]_i, 0) /
        sigma_i^2``, then normalised to ``target_gross``.

    Args:
        returns_panel: Wide DataFrame (index=dates, columns=symbols) of
            returns. Sample covariance is built from this directly via
            ``returns_panel.cov()``.
        view_scores: ``pd.Series`` indexed by symbol with factor z-scores.
            Must cover every column in ``returns_panel``.
        tau: Prior uncertainty scaling (Litterman convention ~0.025-0.10).
        target_gross: Target sum of weights after long-only clipping and
            renormalisation.
        equal_weight_prior: If True use ``w_mkt = 1/n``. Currently the only
            supported prior (kept as a kwarg for forward compatibility).

    Returns:
        ``pd.Series`` indexed by the panel symbols, name ``"bl_weight"``,
        summing to ``target_gross`` (modulo float epsilon).

    Raises:
        ValueError: If the panel has fewer than 30 rows, fewer than 2
            symbols, or ``view_scores`` does not cover the panel symbols,
            or ``target_gross <= 0``.
    """
    if target_gross <= 0:
        raise ValueError(f"target_gross must be > 0, got {target_gross}")

    if not isinstance(returns_panel, pd.DataFrame):
        raise ValueError("returns_panel must be a pandas DataFrame")

    if len(returns_panel) < 30:
        raise ValueError(
            f"insufficient history: {len(returns_panel)} rows < 30"
        )

    if returns_panel.shape[1] < 2:
        raise ValueError(
            f"need at least 2 symbols, got {returns_panel.shape[1]}"
        )

    if not isinstance(view_scores, pd.Series):
        raise ValueError("view_scores must be a pandas Series")

    symbols = list(returns_panel.columns)
    missing = [s for s in symbols if s not in view_scores.index]
    if missing:
        raise ValueError(
            f"view_scores missing symbols present in returns_panel: {missing}"
        )

    n = len(symbols)
    sigma = returns_panel.cov().reindex(index=symbols, columns=symbols)
    sigma_arr = sigma.to_numpy(dtype=float)

    # Stabilise the covariance slightly to keep solves well-conditioned on
    # pathological test data (tiny diagonal noise, no fundamental change).
    sigma_arr = sigma_arr + np.eye(n) * 1e-12

    # Prior (equal-weight market portfolio).
    if not equal_weight_prior:
        raise ValueError(
            "equal_weight_prior=False is not yet supported in the sidecar"
        )
    w_mkt = np.full(n, 1.0 / n)

    # Litterman implied excess returns: Pi = delta * Sigma * w_mkt with delta=2.5.
    delta = 2.5
    pi = delta * sigma_arr @ w_mkt

    # Views: z_score -> expected return via crude 10%-per-sigma calibration.
    z = view_scores.reindex(symbols).to_numpy(dtype=float)
    Q = z * 0.10

    # Pick matrix is identity: one view per asset in panel order.
    P = np.eye(n)

    tau_sigma = tau * sigma_arr
    # Omega = diag(tau * sigma_i^2), where sigma_i^2 comes from the diagonal
    # of tau*Sigma (consistent with the BL reference convention here).
    omega_diag = np.diag(tau_sigma).copy()
    # Floor to avoid singular Omega on near-zero-variance columns.
    omega_diag = np.maximum(omega_diag, 1e-12)
    Omega = np.diag(omega_diag)

    # Posterior:
    #   E[R] = [ (tau*Sigma)^-1 + P' Omega^-1 P ]^-1
    #          [ (tau*Sigma)^-1 Pi + P' Omega^-1 Q ]
    try:
        tau_sigma_inv = np.linalg.inv(tau_sigma + np.eye(n) * 1e-10)
        omega_inv = np.linalg.inv(Omega)
        M = tau_sigma_inv + P.T @ omega_inv @ P
        rhs = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
        mu_bl = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"BL linear system failed: {exc}") from exc

    # Inverse-variance-weighted long-only scaling.
    var_diag = np.maximum(np.diag(sigma_arr), 1e-12)
    raw = mu_bl / var_diag
    raw = np.clip(raw, 0.0, None)  # long-only

    total = float(raw.sum())
    if total <= 1e-12:
        # Everything clipped to zero — degenerate view. Fall back to equal
        # weights scaled to target_gross so the caller still gets a usable
        # long-only allocation.
        raw = np.ones(n)
        total = float(raw.sum())

    weights = pd.Series(
        raw * (target_gross / total),
        index=symbols,
        name="bl_weight",
        dtype=float,
    )
    return weights


def blend_bl_with_score(
    bl_weights: pd.Series,
    score_weights: pd.Series,
    bl_alpha: float = 0.7,
) -> pd.Series:
    """Convex blend of BL and score-based weights.

    ``out = bl_alpha * bl + (1 - bl_alpha) * score``

    Inputs are aligned on the union of their symbol indices; missing symbols
    are treated as zero on the side where they are absent. The final result
    is renormalised to the maximum of the two input gross sums so the blend
    cannot inflate exposure. Mirrors :func:`blend_hrp_with_score` in
    ``hrp_sizing``.

    Args:
        bl_weights: BL weights indexed by symbol.
        score_weights: Score-based weights indexed by symbol.
        bl_alpha: Blend coefficient in ``[0, 1]``. ``1.0`` returns BL,
            ``0.0`` returns score (both up to renormalisation).

    Raises:
        ValueError: If ``bl_alpha`` is outside ``[0, 1]``.
    """
    if not 0.0 <= bl_alpha <= 1.0:
        raise ValueError(f"bl_alpha must be in [0, 1], got {bl_alpha}")

    all_symbols = bl_weights.index.union(score_weights.index)
    bl_aligned = bl_weights.reindex(all_symbols, fill_value=0.0).astype(float)
    score_aligned = score_weights.reindex(all_symbols, fill_value=0.0).astype(float)

    blended = bl_alpha * bl_aligned + (1.0 - bl_alpha) * score_aligned

    bl_sum = float(bl_weights.sum())
    score_sum = float(score_weights.sum())
    target = max(bl_sum, score_sum)

    blended_sum = float(blended.sum())
    if blended_sum > 0 and target > 0:
        blended = blended * (target / blended_sum)

    blended.name = "blended_weight"
    return blended


__all__ = [
    "apply_bl_sizing",
    "apply_bl_sizing_from_policy",
    "compute_bl_target_weights",
    "blend_bl_with_score",
]
