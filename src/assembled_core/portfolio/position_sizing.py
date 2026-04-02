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

import numpy as np
import pandas as pd


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
    for _, row in long_signals.iterrows():
        sym = row["symbol"]
        vol = float(volatilities.get(sym, 0.20)) if hasattr(volatilities, "get") else 0.20
        if vol < 1e-8:
            vol = 0.20  # Default if zero/missing
        rows.append({"symbol": sym, "inv_vol": 1.0 / vol, "volatility": vol})

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
    for _, row in long_signals.iterrows():
        sym = row["symbol"]
        vol = float(volatilities.get(sym, 0.20)) if hasattr(volatilities, "get") else 0.20
        if vol < 1e-8:
            vol = 0.20

        weight = target_vol / (sqrt_n * vol)
        weight = min(weight, max_weight)

        rows.append({"symbol": sym, "target_weight": weight, "volatility": vol})

    result = pd.DataFrame(rows)

    # Normalize if total exceeds 1.0
    total_w = result["target_weight"].sum()
    if total_w > 1.0:
        result["target_weight"] = result["target_weight"] / total_w

    result["target_qty"] = result["target_weight"] * total_capital
    result = result[["symbol", "target_weight", "target_qty", "volatility"]]
    return result.sort_values("symbol").reset_index(drop=True)
