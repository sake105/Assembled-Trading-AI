"""Liquidity scoring and liquidity-adjusted position sizing (V15).

Computes per-symbol liquidity scores using:
- Amihud illiquidity lambda: |return| / dollar_volume
- Roll spread estimator: 2 * sqrt(-cov(delta_p_t, delta_p_{t-1}))
- Normalized composite score (0 = illiquid, 1 = mega-liquid)

References:
- Amihud (2002): "Illiquidity and stock returns"
- Roll (1984): "A simple implicit measure of the effective bid-ask spread"
- AQR: standard liquidity screen for factor portfolios
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class LiquidityScore:
    """Per-symbol liquidity assessment."""

    symbol: str
    amihud_lambda: float  # Higher = more illiquid
    roll_spread_bps: float  # Estimated bid-ask spread in bps
    adv_usd: float  # Average daily dollar volume
    score: float  # 0-1 normalized (1 = most liquid)
    tier: str  # "mega", "large", "mid", "small", "micro"


def compute_amihud_lambda(
    returns: np.ndarray,
    dollar_volume: np.ndarray,
) -> float:
    """Compute Amihud illiquidity ratio.

    lambda = mean(|r_t| / DollarVolume_t)

    Higher values indicate less liquid assets.
    """
    mask = dollar_volume > 0
    if mask.sum() < 5:
        return np.inf

    illiq = np.abs(returns[mask]) / dollar_volume[mask]
    return float(np.mean(illiq))


def compute_roll_spread(close_prices: np.ndarray) -> float:
    """Estimate bid-ask spread using Roll (1984) model.

    spread = 2 * sqrt(-cov(delta_p_t, delta_p_{t-1}))

    Returns spread in price units. Returns 0 if covariance is positive
    (model assumption violated).
    """
    if len(close_prices) < 10:
        return 0.0

    dp = np.diff(close_prices)
    if len(dp) < 3:
        return 0.0

    cov_val = np.cov(dp[1:], dp[:-1])[0, 1]

    if cov_val >= 0:
        return 0.0

    return float(2.0 * np.sqrt(-cov_val))


def compute_liquidity_scores(
    prices_df: pd.DataFrame,
    lookback_days: int = 60,
    symbol_col: str = "symbol",
    close_col: str = "close",
    volume_col: str = "volume",
    timestamp_col: str = "timestamp",
) -> list[LiquidityScore]:
    """Compute liquidity scores for all symbols in the price panel.

    Args:
        prices_df: OHLCV panel with symbol, timestamp, close, volume columns.
        lookback_days: Number of trading days for computation.
        symbol_col: Column name for symbol.
        close_col: Column name for close price.
        volume_col: Column name for volume.
        timestamp_col: Column name for timestamp.

    Returns:
        List of LiquidityScore objects, one per symbol.
    """
    if prices_df.empty:
        return []

    scores = []
    for sym, grp in prices_df.groupby(symbol_col):
        grp = grp.sort_values(timestamp_col).tail(lookback_days)

        close = grp[close_col].values.astype(float)
        vol = grp[volume_col].values.astype(float) if volume_col in grp.columns else np.ones(len(grp))

        if len(close) < 10:
            scores.append(LiquidityScore(
                symbol=str(sym), amihud_lambda=np.inf,
                roll_spread_bps=0.0, adv_usd=0.0, score=0.0, tier="micro",
            ))
            continue

        # Returns
        returns = np.diff(np.log(np.maximum(close, 1e-10)))
        dollar_vol = close[1:] * vol[1:]

        # Amihud
        amihud = compute_amihud_lambda(returns, dollar_vol)

        # Roll spread
        roll_spread = compute_roll_spread(close)
        avg_price = float(np.mean(close))
        roll_bps = (roll_spread / avg_price * 10_000) if avg_price > 0 else 0.0

        # ADV
        adv = float(np.mean(dollar_vol)) if len(dollar_vol) > 0 else 0.0

        # Tier classification
        if adv >= 100_000_000:
            tier = "mega"
        elif adv >= 20_000_000:
            tier = "large"
        elif adv >= 5_000_000:
            tier = "mid"
        elif adv >= 1_000_000:
            tier = "small"
        else:
            tier = "micro"

        scores.append(LiquidityScore(
            symbol=str(sym),
            amihud_lambda=amihud,
            roll_spread_bps=round(roll_bps, 2),
            adv_usd=round(adv, 0),
            score=0.0,  # Will be normalized below
            tier=tier,
        ))

    # Normalize scores: rank-based 0-1 (lower Amihud = more liquid = higher score)
    if scores:
        amihud_vals = np.array([s.amihud_lambda if np.isfinite(s.amihud_lambda) else 1e10 for s in scores])
        # Rank: lower Amihud = higher rank = higher score
        ranks = np.argsort(np.argsort(amihud_vals))  # ascending rank
        n = len(scores)
        for i, s in enumerate(scores):
            # Invert: most liquid gets score near 1.0
            s.score = round(1.0 - ranks[i] / max(n - 1, 1), 4)

    _log.info(
        "Liquidity scores: %d symbols, tiers: %s",
        len(scores),
        {t: sum(1 for s in scores if s.tier == t) for t in ["mega", "large", "mid", "small", "micro"]},
    )

    return scores


def apply_liquidity_adjusted_sizing(
    target_weights: dict[str, float],
    liquidity_scores: list[LiquidityScore],
    alpha: float = 0.5,
    min_score_threshold: float = 0.1,
) -> dict[str, float]:
    """Adjust position weights by liquidity score.

    weight_adj = weight * score^alpha

    Assets below min_score_threshold are zeroed out.

    Args:
        target_weights: Symbol -> target weight.
        liquidity_scores: List of LiquidityScore objects.
        alpha: Exponent for liquidity adjustment (0=no effect, 1=full).
        min_score_threshold: Minimum score to keep position.

    Returns:
        Adjusted weights dict.
    """
    score_map = {s.symbol: s.score for s in liquidity_scores}
    adjusted: dict[str, float] = {}

    for sym, w in target_weights.items():
        liq = score_map.get(sym, 0.5)  # Default mid-range if unknown
        if liq < min_score_threshold:
            adjusted[sym] = 0.0
            _log.debug("Liquidity filter: %s zeroed (score=%.3f)", sym, liq)
            continue
        adjusted[sym] = w * (liq ** alpha)

    # Renormalize to preserve gross exposure
    total_orig = sum(abs(v) for v in target_weights.values())
    total_adj = sum(abs(v) for v in adjusted.values())
    if total_adj > 1e-10 and total_orig > 1e-10:
        scale = total_orig / total_adj
        adjusted = {s: w * scale for s, w in adjusted.items()}

    return adjusted


__all__ = [
    "LiquidityScore",
    "compute_amihud_lambda",
    "compute_roll_spread",
    "compute_liquidity_scores",
    "apply_liquidity_adjusted_sizing",
]
