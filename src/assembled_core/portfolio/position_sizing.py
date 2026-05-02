"""Position sizing module.

This module provides position sizing strategies for EOD trading.
It determines target positions based on signals and available capital.

Strategies:
- Equal weight: 1/N for each position
- Score-based: Weight proportional to signal score
- Kelly Criterion: Optimal sizing based on win rate and payoff ratio
- Risk Parity: Inverse-volatility weighting for equal risk contribution
- Volatility-scaled: ATR or realized-vol-based position scaling
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_target_positions(
    signals: pd.DataFrame,
    total_capital: float = 1.0,
    top_n: int | None = None,
    equal_weight: bool = True,
) -> pd.DataFrame:
    """Compute target positions from trading signals.

    This function determines target positions (weights or quantities) based on:
    - Signal scores (if available)
    - Top-N selection (if top_n is specified)
    - Equal weighting (if equal_weight=True) or score-based weighting

    Args:
        signals: DataFrame with columns: symbol, direction (and optionally score)
            direction: "LONG" or "FLAT"
            score: Signal strength (0.0 to 1.0), optional
        total_capital: Total capital available (default: 1.0 for normalized weights)
        top_n: Optional maximum number of positions to select (default: None = all LONG signals)
        equal_weight: If True, use equal weights (1/N). If False, use score-based weights (default: True)

    Returns:
        DataFrame with columns: symbol, target_weight, target_qty
        target_weight: Target weight (0.0 to 1.0)
        target_qty: Target quantity (in units, if total_capital represents actual capital)
        Sorted by symbol

    Raises:
        ValueError: If signals DataFrame is empty or missing required columns
    """
    if signals.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])

    # Ensure required columns
    required = ["symbol", "direction"]
    missing = [c for c in required if c not in signals.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available: {list(signals.columns)}"
        )

    # Filter for LONG signals only
    long_signals = signals[signals["direction"] == "LONG"].copy()

    if long_signals.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])

    # Select top N by score if specified
    if top_n is not None and top_n > 0:
        if "score" in long_signals.columns:
            # Sort by score descending, take top N
            long_signals = long_signals.nlargest(top_n, "score")
        else:
            # If no score, just take first N
            long_signals = long_signals.head(top_n)

    # Compute weights
    n_positions = len(long_signals)

    if equal_weight:
        # Equal weighting: 1/N for each position
        long_signals["target_weight"] = 1.0 / n_positions
    else:
        # Score-based weighting (normalize scores to sum to 1.0)
        if "score" in long_signals.columns:
            total_score = long_signals["score"].sum()
            if total_score > 0:
                long_signals["target_weight"] = long_signals["score"] / total_score
            else:
                # Fallback to equal weight if all scores are zero
                long_signals["target_weight"] = 1.0 / n_positions
        else:
            # Fallback to equal weight if no scores
            long_signals["target_weight"] = 1.0 / n_positions

    # Compute target quantities (if total_capital represents actual capital)
    # For normalized weights (total_capital=1.0), target_qty = target_weight
    # For actual capital, target_qty would need current prices (not available here)
    # So we set target_qty = target_weight * total_capital as a placeholder
    long_signals["target_qty"] = long_signals["target_weight"] * total_capital

    # Select and sort output columns
    result = long_signals[["symbol", "target_weight", "target_qty"]].copy()
    result = result.sort_values("symbol").reset_index(drop=True)

    return result


def compute_target_positions_from_trend_signals(
    trend_signals: pd.DataFrame,
    total_capital: float = 1.0,
    top_n: int | None = None,
    min_score: float = 0.0,
) -> pd.DataFrame:
    """Compute target positions from trend signals (convenience function).

    This is a convenience wrapper around compute_target_positions that:
    - Filters signals by minimum score
    - Uses score-based weighting (not equal weight)

    Args:
        trend_signals: DataFrame with columns: symbol, direction, score
            (from signals.rules_trend.generate_trend_signals)
        total_capital: Total capital available (default: 1.0)
        top_n: Optional maximum number of positions (default: None)
        min_score: Minimum score threshold (default: 0.0)

    Returns:
        DataFrame with columns: symbol, target_weight, target_qty
    """
    # Filter by minimum score
    if "score" in trend_signals.columns:
        filtered = trend_signals[trend_signals["score"] >= min_score].copy()
    else:
        filtered = trend_signals.copy()

    # Use score-based weighting (not equal weight)
    return compute_target_positions(
        filtered, total_capital=total_capital, top_n=top_n, equal_weight=False
    )


def compute_kelly_weights(
    signals: pd.DataFrame,
    win_rates: pd.Series | dict[str, float] | None = None,
    payoff_ratios: pd.Series | dict[str, float] | None = None,
    fraction: float = 0.5,
    max_weight: float = 0.25,
    total_capital: float = 1.0,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Compute position weights using Kelly Criterion.

    Kelly formula: f* = (p * b - q) / b
    where p = win probability, b = payoff ratio (avg win / avg loss), q = 1 - p

    Uses fractional Kelly (default 0.5) for more conservative sizing.

    Args:
        signals: DataFrame with columns: symbol, direction (and optionally score)
        win_rates: Per-symbol win rates (0-1). Dict or Series keyed by symbol.
            If None, uses signal score as proxy (default: 0.55)
        payoff_ratios: Per-symbol payoff ratios (avg_win / avg_loss).
            If None, uses 1.5 as default
        fraction: Kelly fraction (0.5 = half-Kelly, default: 0.5)
        max_weight: Maximum weight per position (default: 0.25)
        total_capital: Total capital (default: 1.0)
        top_n: Maximum number of positions

    Returns:
        DataFrame with columns: symbol, target_weight, target_qty, kelly_raw
    """
    if signals.empty:
        return pd.DataFrame(
            columns=["symbol", "target_weight", "target_qty", "kelly_raw"]
        )

    long_signals = signals[signals["direction"] == "LONG"].copy()
    if long_signals.empty:
        return pd.DataFrame(
            columns=["symbol", "target_weight", "target_qty", "kelly_raw"]
        )

    # Select top N
    if top_n is not None and "score" in long_signals.columns:
        long_signals = long_signals.nlargest(top_n, "score")

    # Get win rates per symbol
    if isinstance(win_rates, dict):
        win_rates = pd.Series(win_rates)

    kelly_weights = []
    for _, row in long_signals.iterrows():
        sym = row["symbol"]
        if win_rates is not None and sym in win_rates.index:
            p = float(win_rates[sym])
        elif "score" in row.index and pd.notna(row.get("score")):
            p = 0.5 + float(row["score"]) * 0.1  # Map score to slight edge
        else:
            p = 0.55

        if isinstance(payoff_ratios, pd.Series) and sym in payoff_ratios.index:
            b = float(payoff_ratios[sym])
        elif isinstance(payoff_ratios, dict) and sym in payoff_ratios:
            b = float(payoff_ratios[sym])
        else:
            b = 1.5

        q = 1.0 - p
        kelly_raw = (p * b - q) / b if b > 0 else 0.0
        kelly_frac = max(0.0, kelly_raw * fraction)
        kelly_capped = min(kelly_frac, max_weight)

        kelly_weights.append(
            {"symbol": sym, "kelly_raw": kelly_raw, "kelly_frac": kelly_capped}
        )

    result = pd.DataFrame(kelly_weights)

    # Normalize weights to sum to <=1
    total_w = result["kelly_frac"].sum()
    if total_w > 1.0:
        result["target_weight"] = result["kelly_frac"] / total_w
    else:
        result["target_weight"] = result["kelly_frac"]

    result["target_qty"] = result["target_weight"] * total_capital
    result = result[["symbol", "target_weight", "target_qty", "kelly_raw"]]
    return result.sort_values("symbol").reset_index(drop=True)


def compute_risk_parity_weights(
    signals: pd.DataFrame,
    volatilities: pd.Series | dict[str, float],
    total_capital: float = 1.0,
    top_n: int | None = None,
    max_weight: float = 0.30,
) -> pd.DataFrame:
    """Compute position weights using Risk Parity (inverse-volatility weighting).

    Each position gets weight proportional to 1/volatility, so that each
    position contributes approximately equal risk to the portfolio.

    Args:
        signals: DataFrame with columns: symbol, direction
        volatilities: Per-symbol annualized volatilities. Dict or Series keyed by symbol.
        total_capital: Total capital (default: 1.0)
        top_n: Maximum number of positions
        max_weight: Maximum weight per position (default: 0.30)

    Returns:
        DataFrame with columns: symbol, target_weight, target_qty, volatility
    """
    if signals.empty:
        return pd.DataFrame(
            columns=["symbol", "target_weight", "target_qty", "volatility"]
        )

    long_signals = signals[signals["direction"] == "LONG"].copy()
    if long_signals.empty:
        return pd.DataFrame(
            columns=["symbol", "target_weight", "target_qty", "volatility"]
        )

    if top_n is not None and "score" in long_signals.columns:
        long_signals = long_signals.nlargest(top_n, "score")

    if isinstance(volatilities, dict):
        volatilities = pd.Series(volatilities)

    rows = []
    skipped: list[str] = []
    for _, row in long_signals.iterrows():
        sym = row["symbol"]
        # Drop symbols with missing / non-finite / zero vol — silently defaulting
        # to 0.20 defeats the purpose of risk-parity (all missing symbols would
        # get identical inverse-vol weights).
        vol_raw = volatilities.get(sym, None) if hasattr(volatilities, "get") else None
        vol = float(vol_raw) if vol_raw is not None and pd.notna(vol_raw) else float("nan")
        if not np.isfinite(vol) or vol < 1e-8:
            skipped.append(sym)
            continue
        rows.append({"symbol": sym, "inv_vol": 1.0 / vol, "volatility": vol})
    if skipped:
        logger.warning(
            "[risk_parity] dropped %d symbols with missing/zero volatility: %s",
            len(skipped),
            skipped[:20],
        )

    if not rows:
        return pd.DataFrame(
            columns=["symbol", "target_weight", "target_qty", "volatility"]
        )

    result = pd.DataFrame(rows)
    total_inv_vol = result["inv_vol"].sum()

    if total_inv_vol > 0:
        result["target_weight"] = (result["inv_vol"] / total_inv_vol).clip(
            upper=max_weight
        )
        # Renormalize after clipping
        total_w = result["target_weight"].sum()
        if total_w > 1.0:
            result["target_weight"] = result["target_weight"] / total_w
    else:
        n = len(result)
        result["target_weight"] = 1.0 / n if n > 0 else 0.0

    result["target_qty"] = result["target_weight"] * total_capital
    result = result[["symbol", "target_weight", "target_qty", "volatility"]]
    return result.sort_values("symbol").reset_index(drop=True)


def compute_vol_scaled_weights(
    signals: pd.DataFrame,
    volatilities: pd.Series | dict[str, float],
    target_vol: float = 0.15,
    total_capital: float = 1.0,
    top_n: int | None = None,
    max_weight: float = 0.30,
) -> pd.DataFrame:
    """Compute position weights scaled by target volatility.

    Each position is sized so that its contribution to portfolio volatility
    matches a target level, assuming positions are independent.

    Weight_i = target_vol / (sqrt(N) * vol_i)

    Args:
        signals: DataFrame with columns: symbol, direction
        volatilities: Per-symbol annualized volatilities
        target_vol: Target portfolio volatility (default: 0.15 = 15%)
        total_capital: Total capital (default: 1.0)
        top_n: Maximum number of positions
        max_weight: Maximum weight per position (default: 0.30)

    Returns:
        DataFrame with columns: symbol, target_weight, target_qty, volatility
    """
    if signals.empty:
        return pd.DataFrame(
            columns=["symbol", "target_weight", "target_qty", "volatility"]
        )

    long_signals = signals[signals["direction"] == "LONG"].copy()
    if long_signals.empty:
        return pd.DataFrame(
            columns=["symbol", "target_weight", "target_qty", "volatility"]
        )

    if top_n is not None and "score" in long_signals.columns:
        long_signals = long_signals.nlargest(top_n, "score")

    if isinstance(volatilities, dict):
        volatilities = pd.Series(volatilities)

    n_positions = len(long_signals)
    sqrt_n = np.sqrt(n_positions) if n_positions > 0 else 1.0

    rows = []
    skipped: list[str] = []
    for _, row in long_signals.iterrows():
        sym = row["symbol"]
        vol_raw = volatilities.get(sym, None) if hasattr(volatilities, "get") else None
        vol = float(vol_raw) if vol_raw is not None and pd.notna(vol_raw) else float("nan")
        if not np.isfinite(vol) or vol < 1e-8:
            skipped.append(sym)
            continue

        weight = target_vol / (sqrt_n * vol)
        weight = min(weight, max_weight)

        rows.append({"symbol": sym, "target_weight": weight, "volatility": vol})
    if skipped:
        logger.warning(
            "[vol_scaled] dropped %d symbols with missing/zero volatility: %s",
            len(skipped),
            skipped[:20],
        )

    if not rows:
        return pd.DataFrame(
            columns=["symbol", "target_weight", "target_qty", "volatility"]
        )

    result = pd.DataFrame(rows)

    # Normalize if total exceeds 1.0
    total_w = result["target_weight"].sum()
    if total_w > 1.0:
        result["target_weight"] = result["target_weight"] / total_w

    result["target_qty"] = result["target_weight"] * total_capital
    result = result[["symbol", "target_weight", "target_qty", "volatility"]]
    return result.sort_values("symbol").reset_index(drop=True)


# ---------------------------------------------------------------------------
# TC-Penalized Rebalancing (Plan 5.3)
# ---------------------------------------------------------------------------


def apply_tc_penalized_rebalancing(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    cost_bps: dict[str, float] | float = 10.0,
    dead_zone_pct: float = 0.02,
    tc_penalty_gamma: float = 1.0,
) -> dict[str, float]:
    """Apply transaction-cost-aware rebalancing with dead zones.

    Adjusts target weights to penalise turnover:
    - If ``|w_target - w_current| < dead_zone_pct`` → no change (keep current).
    - Otherwise, shrink the trade toward current by ``gamma × cost``.

    Objective direction:  ``max E[return] - lambda*risk - gamma*TC``
    where ``TC = sum(|w_new - w_old| × cost_bps_i)``.

    Args:
        target_weights: Symbol → target weight from optimizer.
        current_weights: Symbol → current portfolio weight.
        cost_bps: Per-symbol cost in bps (dict) or flat cost for all (float).
        dead_zone_pct: Minimum weight change to execute (default 2%).
        tc_penalty_gamma: TC penalty multiplier. Bull→0.5, Crisis→2.0.

    Returns:
        Adjusted weights dict (symbol → weight).
    """
    all_symbols = set(target_weights.keys()) | set(current_weights.keys())
    adjusted: dict[str, float] = {}

    for sym in all_symbols:
        w_target = target_weights.get(sym, 0.0)
        w_current = current_weights.get(sym, 0.0)
        delta = w_target - w_current

        # Dead zone: skip small adjustments
        if abs(delta) < dead_zone_pct:
            adjusted[sym] = w_current
            continue

        # TC penalty: shrink trade toward current
        if isinstance(cost_bps, dict):
            sym_cost = cost_bps.get(sym, 10.0) / 10000.0
        else:
            sym_cost = float(cost_bps) / 10000.0

        penalty = tc_penalty_gamma * sym_cost
        if delta > 0:
            adjusted[sym] = w_target - penalty
        else:
            adjusted[sym] = w_target + penalty

        # Don't overshoot — keep between current and target
        if delta > 0:
            adjusted[sym] = max(w_current, min(adjusted[sym], w_target))
        else:
            adjusted[sym] = min(w_current, max(adjusted[sym], w_target))

    # Renormalize to sum to <= 1.0 (preserve cash)
    total = sum(max(0.0, w) for w in adjusted.values())
    if total > 1.0:
        adjusted = {s: max(0.0, w) / total for s, w in adjusted.items()}

    return adjusted


# ---------------------------------------------------------------------------
# Liquidity-Constrained Sizing (Plan 7.3)
# ---------------------------------------------------------------------------


def apply_liquidity_constraint(
    target_weights: dict[str, float],
    adv_dollars: dict[str, float],
    total_capital: float,
    max_participation_pct: float = 0.05,
    max_days: int = 3,
) -> dict[str, float]:
    """Constrain position sizes by liquidity (ADV).

    ``max_position = min(target_weight × capital, max_participation × ADV × max_days)``

    Illiquid assets (ADV < $1M) get their weight automatically reduced.

    Args:
        target_weights: Symbol → target weight.
        adv_dollars: Symbol → average daily dollar volume.
        total_capital: Total portfolio capital in dollars.
        max_participation_pct: Max fraction of daily volume (default 5%).
        max_days: Max days to fill position (default 3).

    Returns:
        Adjusted weights (may sum to < 1.0 — residual goes to cash).
    """
    adjusted: dict[str, float] = {}

    for sym, w in target_weights.items():
        adv = adv_dollars.get(sym, 0.0)
        if adv <= 0:
            adjusted[sym] = 0.0
            continue

        max_dollar = max_participation_pct * adv * max_days
        max_weight = max_dollar / total_capital if total_capital > 0 else 0.0
        adjusted[sym] = min(w, max_weight)

    return adjusted


# ── 5.6  Maximum Diversification Portfolio ─────────────────────────────
def compute_max_diversification_weights(
    cov_matrix: np.ndarray,
    vols: np.ndarray | None = None,
    max_iter: int = 500,
) -> np.ndarray:
    """Compute Maximum Diversification Ratio portfolio weights.

    Maximizes DR = sum(w_i * sigma_i) / sqrt(w' Sigma w)
    via iterative inverse-vol reweighting.

    Args:
        cov_matrix: N×N covariance matrix.
        vols: Individual asset volatilities. If None, extracted from cov diagonal.
        max_iter: Maximum iterations for convergence.

    Returns:
        Array of portfolio weights (sum=1, long-only).
    """
    n = cov_matrix.shape[0]
    if vols is None:
        vols = np.sqrt(np.diag(cov_matrix))

    w = np.ones(n) / n  # start equal weight

    for _ in range(max_iter):
        port_vol = float(np.sqrt(w @ cov_matrix @ w))
        if port_vol < 1e-12:
            break
        # Marginal risk contribution
        mrc = cov_matrix @ w / port_vol
        # Update: weight inversely to marginal risk, scaled by asset vol
        w_new = vols / np.maximum(mrc, 1e-12)
        w_new_sum = w_new.sum()
        w_new = w_new / w_new_sum if w_new_sum > 1e-12 else np.ones(len(w_new)) / len(w_new)

        if np.max(np.abs(w_new - w)) < 1e-8:
            break
        w = w_new

    return w


# ── 5.8  Tail Risk Parity (CVaR-based) ────────────────────────────────
def compute_tail_risk_parity_weights(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    max_iter: int = 200,
) -> dict[str, float]:
    """Compute weights so each asset contributes equally to portfolio CVaR.

    Args:
        returns: DataFrame of asset returns (columns = assets).
        alpha: CVaR confidence level (default 5%).
        max_iter: Maximum iterations.

    Returns:
        Dict of asset → weight.
    """
    assets = list(returns.columns)
    n = len(assets)
    w = np.ones(n) / n

    for _ in range(max_iter):
        port_ret = returns.values @ w
        cutoff = np.percentile(port_ret, alpha * 100)
        tail_mask = port_ret <= cutoff
        if tail_mask.sum() < 2:
            break

        # Marginal CVaR: avg contribution in tail
        tail_returns = returns.values[tail_mask]
        marginal_cvar = np.abs(tail_returns.mean(axis=0))
        marginal_cvar = np.maximum(marginal_cvar, 1e-12)

        # Inverse marginal CVaR weighting
        w_new = (1.0 / marginal_cvar)
        w_new = w_new / w_new.sum()

        if np.max(np.abs(w_new - w)) < 1e-8:
            break
        w = 0.5 * w + 0.5 * w_new  # damped update

    return {assets[i]: float(w[i]) for i in range(n)}


def apply_news_sentiment_weight_adjustment(
    target_positions: pd.DataFrame,
    news_events: "pd.DataFrame | None" = None,
    *,
    entity_linker: "object | None" = None,
    sentiment_col: str = "sentiment_score",
    max_adjustment: float = 0.10,
    shadow_only: bool = True,
) -> pd.DataFrame:
    """Shadow-mode (T4.4): adjust target weights by news sentiment via EntityLinker.

    When shadow_only=True (default), logs adjustments but returns positions unchanged.
    When shadow_only=False, applies a capped weight bump/reduction based on sentiment.

    Invariants:
    - Never increases a weight by more than max_adjustment (10pp by default).
    - Never pushes a weight below 0.
    - Weights re-normalized after adjustment.
    """
    if news_events is None or news_events.empty or entity_linker is None:
        return target_positions

    result = target_positions.copy()

    # Build per-symbol sentiment map via EntityLinker
    sentiment_map: dict[str, float] = {}
    for _, row in news_events.iterrows():
        entity = str(row.get("entity") or row.get("symbol") or "")
        score = float(row.get(sentiment_col, 0.0) or 0.0)
        try:
            sym = entity_linker.link(entity)
        except Exception:
            sym = None
        if sym:
            prev = sentiment_map.get(sym.upper(), 0.0)
            sentiment_map[sym.upper()] = (prev + score) / 2.0  # rolling mean

    if not sentiment_map:
        return result

    adjustments: list[dict] = []
    for idx, row in result.iterrows():
        sym = str(row["symbol"]).upper()
        sent = sentiment_map.get(sym)
        if sent is None:
            continue
        # Map sentiment [-1, 1] → weight delta [-max_adjustment, +max_adjustment]
        delta = float(sent) * max_adjustment
        old_w = float(row["target_weight"])
        new_w = max(0.0, min(1.0, old_w + delta))
        adjustments.append({"idx": idx, "symbol": sym, "old_w": old_w, "new_w": new_w, "delta": delta})
        if not shadow_only:
            result.at[idx, "target_weight"] = new_w

    if adjustments:
        logger.debug(
            "[%s-T4.4] news_sentiment adjustments: %d symbols | samples: %s",
            "SHADOW" if shadow_only else "OK",
            len(adjustments),
            [(a["symbol"], round(a["delta"], 3)) for a in adjustments[:5]],
        )

    if not shadow_only and adjustments:
        total_w = result["target_weight"].sum()
        if total_w > 1e-9:
            result["target_weight"] = result["target_weight"] / total_w
        # Distribute old total qty proportionally to new weights (weight × qty is dimensionally wrong)
        if "target_qty" in result.columns:
            old_qty_sum = result["target_qty"].sum()
            result["target_qty"] = result["target_weight"] * (old_qty_sum if old_qty_sum > 1e-9 else 1.0)

    return result


# ---------------------------------------------------------------------------
# Round 7 additive: Turnover-aware position sizing (optional wrapper)
# ---------------------------------------------------------------------------


def compute_target_positions_with_smoothing(
    signals: pd.DataFrame,
    previous_positions: pd.Series | dict | None = None,
    total_capital: float = 1.0,
    top_n: int | None = None,
    equal_weight: bool = True,
    smoothing_alpha: float = 0.3,
    max_turnover: float | None = None,
) -> pd.DataFrame:
    """compute_target_positions + Turnover-Smoothing + optional Budget-Cap.

    Wrappt compute_target_positions (unverändert) und wendet EMA-Smoothing +
    optional Turnover-Budget an.

    Args:
        signals, total_capital, top_n, equal_weight: siehe compute_target_positions
        previous_positions: pd.Series oder dict mit letzten Weights (symbol -> weight).
                            None → identisch zu compute_target_positions.
        smoothing_alpha: EMA-Koeffizient in [0, 1]. 1.0 = kein Smoothing.
        max_turnover: Optional Turnover-Budget in [0, 1].

    Returns:
        DataFrame mit denselben Spalten wie compute_target_positions, Weights
        sind geglättet.
    """
    result = compute_target_positions(
        signals=signals,
        total_capital=total_capital,
        top_n=top_n,
        equal_weight=equal_weight,
    )

    if previous_positions is None or result.empty:
        return result

    try:
        from src.assembled_core.portfolio.turnover_penalty import (
            apply_turnover_smoothing,
            enforce_turnover_budget,
        )
    except ImportError:
        return result

    # Extract current target as Series(symbol → weight)
    sym_col = "symbol" if "symbol" in result.columns else result.columns[0]
    target = pd.Series(
        result["target_weight"].values,
        index=result[sym_col].values,
        name="target_weight",
    )

    smoothed = apply_turnover_smoothing(target, previous_positions, alpha=smoothing_alpha)
    if max_turnover is not None:
        smoothed = enforce_turnover_budget(smoothed, previous_positions, max_turnover=max_turnover)

    # Re-apply smoothed weights back to result
    result = result.copy()
    result["target_weight"] = result[sym_col].map(smoothed).fillna(0.0).values
    if "target_qty" in result.columns:
        # Re-scale qty proportional to new weight, preserving capital scaling.
        # Base function: target_qty = target_weight * total_capital (line 103).
        result["target_qty"] = result["target_weight"] * total_capital

    return result


def compute_kelly_weights_with_uncertainty(
    edges: pd.Series,
    variances: pd.Series,
    conformal_half_widths: pd.Series | None = None,
    reference_half_width: float | None = None,
    fractional_kelly: float = 0.5,
    max_fraction: float = 0.25,
    normalize: bool = True,
) -> pd.Series:
    """Kelly-Weights mit Conformal-Uncertainty-Discount (Round 7F).

    Additive Wrapper-Funktion — ruft kelly_uncertainty.compute_kelly_weights_with_uncertainty
    auf. Bestehendes `compute_kelly_weights` bleibt UNVERÄNDERT.

    Args:
        edges: Erwartete Returns pro Symbol
        variances: Return-Varianzen
        conformal_half_widths: Optional Prediction-Intervall pro Symbol
        reference_half_width: Referenz-Intervall für Uncertainty-Scaling
        fractional_kelly: 0.5 = half-Kelly konservativ
        max_fraction: Max-Position pro Symbol
        normalize: sum(|weights|)=1

    Returns:
        pd.Series der Weights.
    """
    from src.assembled_core.portfolio.kelly_uncertainty import (
        compute_kelly_weights_with_uncertainty as _compute,
    )
    return _compute(
        edges=edges,
        variances=variances,
        conformal_half_widths=conformal_half_widths,
        reference_half_width=reference_half_width,
        fractional_kelly=fractional_kelly,
        max_fraction=max_fraction,
        normalize=normalize,
    )
