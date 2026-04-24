"""Behavioral Finance Signals — Exploiting Cognitive Biases (M28).

Implements signals that exploit well-documented behavioral biases:
  1. Disposition Effect: Investors sell winners too early, hold losers too long.
     Signal: stocks with large unrealized losses have selling pressure removed.
  2. Anchoring Bias: Investors anchor to round numbers and 52-week extremes.
     Signal: breakouts above/below psychologically significant levels.
  3. Herding Detection: Crowded trades unwind violently.
     Signal: extreme short interest or volume concentration = contrarian opportunity.
  4. Overreaction/Underreaction: Post-earnings and post-news drift.
     Signal: magnitude-adjusted drift continuation after events.

Reference:
    Barberis, N. & Thaler, R. (2003). "A Survey of Behavioral Finance."
    Frazzini, A. (2006). "The Disposition Effect and Underreaction to News."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BehavioralConfig:
    """Configuration for behavioral finance signals.

    Attributes:
        disposition_lookback: Days for unrealized gain/loss computation.
        anchoring_round_levels: Round-number price levels to check.
        high_low_lookback: 52-week high/low lookback in trading days.
        herding_volume_lookback: Days for volume concentration measurement.
        overreaction_lookback: Days for overreaction measurement.
        overreaction_threshold: Absolute return threshold for overreaction (e.g. 0.05 = 5%).
        blend_weights: Weights for combining sub-signals.
    """

    disposition_lookback: int = 60
    anchoring_round_levels: list[int] | None = None
    high_low_lookback: int = 252
    herding_volume_lookback: int = 20
    overreaction_lookback: int = 5
    overreaction_threshold: float = 0.05
    blend_weights: dict[str, float] | None = None

    def __post_init__(self):
        if self.anchoring_round_levels is None:
            self.anchoring_round_levels = [10, 25, 50, 100, 200, 500, 1000]
        if self.blend_weights is None:
            self.blend_weights = {
                "disposition": 0.30,
                "anchoring": 0.20,
                "herding": 0.25,
                "overreaction": 0.25,
            }


@dataclass
class BehavioralSignal:
    """Result of behavioral signal computation for a single symbol.

    Attributes:
        symbol: Ticker symbol.
        disposition_score: Disposition effect score (-1 to +1).
        anchoring_score: Anchoring/breakout score (-1 to +1).
        herding_score: Herding/crowding contrarian score (-1 to +1).
        overreaction_score: Overreaction/drift score (-1 to +1).
        composite_score: Blended final score.
    """

    symbol: str
    disposition_score: float
    anchoring_score: float
    herding_score: float
    overreaction_score: float
    composite_score: float


def compute_disposition_score(
    prices: np.ndarray,
    volumes: np.ndarray | None = None,
    lookback: int = 60,
) -> float:
    """Compute disposition effect score.

    Stocks trading near their lookback lows have fewer willing sellers
    (bagholders anchored to cost basis). This creates a supply/demand
    imbalance that can drive prices higher.

    Score > 0: stock near lows, potential buying opportunity (sellers exhausted).
    Score < 0: stock near highs, potential selling pressure (profit-taking).

    Args:
        prices: Price array (most recent last).
        volumes: Volume array (optional, used for volume-weighted scoring).
        lookback: Lookback period.

    Returns:
        Score in [-1, 1].
    """
    p = np.asarray(prices, dtype=float)
    if len(p) < max(lookback, 10):
        return 0.0

    window = p[-lookback:]
    current = p[-1]
    high = window.max()
    low = window.min()

    if high - low < 1e-10:
        return 0.0

    # Position in range: 0 = at low, 1 = at high
    position = (current - low) / (high - low)

    # Disposition score: negative when near highs (profit-taking pressure),
    # positive when near lows (selling exhaustion)
    score = 1.0 - 2.0 * position

    # Volume confirmation: if volume is declining near lows, selling is exhausted
    if volumes is not None and len(volumes) >= lookback:
        v = np.asarray(volumes, dtype=float)[-lookback:]
        recent_v = v[-5:].mean()
        avg_v = v.mean()
        if avg_v > 0:
            vol_ratio = recent_v / avg_v
            # Low volume at lows = stronger disposition signal
            if score > 0 and vol_ratio < 0.8:
                score *= 1.2
            # High volume at highs = stronger profit-taking signal
            elif score < 0 and vol_ratio > 1.2:
                score *= 1.2

    return float(np.clip(score, -1.0, 1.0))


def compute_anchoring_score(
    prices: np.ndarray,
    round_levels: list[int] | None = None,
    high_low_lookback: int = 252,
) -> float:
    """Compute anchoring/breakout score.

    Investors anchor to round numbers and 52-week highs/lows.
    Breakouts above these levels often trigger momentum cascades.

    Score > 0: breaking above resistance (52-week high or round number).
    Score < 0: breaking below support.

    Args:
        prices: Price array (most recent last).
        round_levels: Round-number levels to check for anchoring.
        high_low_lookback: Lookback for high/low anchors.

    Returns:
        Score in [-1, 1].
    """
    p = np.asarray(prices, dtype=float)
    if len(p) < 20:
        return 0.0

    if round_levels is None:
        round_levels = [10, 25, 50, 100, 200, 500, 1000]

    current = p[-1]
    score = 0.0

    # 52-week high/low proximity
    lookback_window = p[-min(high_low_lookback, len(p)):]
    high_52w = lookback_window.max()
    low_52w = lookback_window.min()

    if high_52w - low_52w > 1e-10:
        # Near 52-week high: bullish breakout potential
        high_proximity = (current - high_52w) / high_52w
        if -0.02 <= high_proximity <= 0.02:
            score += 0.5 * (1.0 + high_proximity / 0.02)

        # Near 52-week low: bearish breakdown
        low_proximity = (current - low_52w) / low_52w
        if -0.02 <= low_proximity <= 0.02:
            score -= 0.5 * (1.0 - low_proximity / 0.02)

    # Round number proximity
    for level in round_levels:
        if level <= 0:
            continue
        nearest_round = round(current / level) * level
        if nearest_round > 0:
            distance_pct = (current - nearest_round) / nearest_round
            if abs(distance_pct) < 0.02:
                # Just above round number = bullish, just below = bearish
                score += 0.3 * np.sign(distance_pct)

    return float(np.clip(score, -1.0, 1.0))


def compute_herding_score(
    volumes: np.ndarray,
    returns: np.ndarray,
    lookback: int = 20,
) -> float:
    """Compute herding/crowding contrarian score.

    Extreme volume concentration relative to recent history suggests
    crowded positioning. Contrarian signal: fade the crowd.

    Score > 0: contrarian buy (crowd is selling/panicking).
    Score < 0: contrarian sell (crowd is euphoric/piling in).

    Args:
        volumes: Volume array.
        returns: Return array.
        lookback: Lookback for volume normalization.

    Returns:
        Score in [-1, 1].
    """
    v = np.asarray(volumes, dtype=float)
    r = np.asarray(returns, dtype=float)

    min_len = min(len(v), len(r))
    if min_len < max(lookback, 10):
        return 0.0

    v = v[-min_len:]
    r = r[-min_len:]

    # Volume z-score
    recent_vol = v[-5:].mean()
    avg_vol = v[-lookback:].mean()
    std_vol = v[-lookback:].std()

    if std_vol < 1e-10:
        return 0.0

    vol_z = (recent_vol - avg_vol) / std_vol

    # Recent return direction
    recent_return = r[-5:].sum()

    # Herding: extreme volume + negative return = panic selling -> contrarian buy
    # Extreme volume + positive return = euphoria -> contrarian sell
    if abs(vol_z) > 1.5:
        score = -np.sign(recent_return) * min(abs(vol_z) / 3.0, 1.0)
    else:
        score = 0.0

    return float(np.clip(score, -1.0, 1.0))


def compute_overreaction_score(
    returns: np.ndarray,
    lookback: int = 5,
    threshold: float = 0.05,
) -> float:
    """Compute overreaction/underreaction score.

    Large short-term moves tend to partially reverse (overreaction),
    while moderate moves tend to continue (underreaction/drift).

    Score > 0: expect upward drift/reversal.
    Score < 0: expect downward drift/reversal.

    Args:
        returns: Daily return array.
        lookback: Recent window for move detection.
        threshold: Absolute cumulative return threshold for overreaction.

    Returns:
        Score in [-1, 1].
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < lookback + 5:
        return 0.0

    recent_cum = r[-lookback:].sum()

    if abs(recent_cum) > threshold:
        # Overreaction: fade the move (partial reversal expected)
        score = -np.sign(recent_cum) * min(abs(recent_cum) / (2 * threshold), 1.0)
    elif abs(recent_cum) > threshold * 0.3:
        # Moderate move: underreaction, expect continuation
        score = np.sign(recent_cum) * min(abs(recent_cum) / threshold, 0.5)
    else:
        score = 0.0

    return float(np.clip(score, -1.0, 1.0))


def generate_behavioral_signals(
    prices_df: pd.DataFrame,
    config: BehavioralConfig | None = None,
) -> list[BehavioralSignal]:
    """Generate behavioral finance signals for all symbols.

    Args:
        prices_df: DataFrame with columns [timestamp, symbol, close, volume].
            Volume column is optional.
        config: Behavioral signal configuration.

    Returns:
        List of BehavioralSignal for each symbol.
    """
    cfg = config or BehavioralConfig()
    has_volume = "volume" in prices_df.columns

    symbols = sorted(prices_df["symbol"].unique())
    signals = []

    for sym in symbols:
        mask = prices_df["symbol"] == sym
        sym_data = prices_df.loc[mask].sort_values("timestamp")
        prices = sym_data["close"].values
        volumes = sym_data["volume"].values if has_volume else None

        if len(prices) < 20:
            continue

        returns = np.diff(prices) / prices[:-1]
        returns = returns[np.isfinite(returns)]

        disp = compute_disposition_score(prices, volumes, cfg.disposition_lookback)
        anch = compute_anchoring_score(
            prices, cfg.anchoring_round_levels, cfg.high_low_lookback,
        )
        herd = compute_herding_score(
            volumes if volumes is not None else np.ones(len(returns)),
            returns,
            cfg.herding_volume_lookback,
        ) if len(returns) >= cfg.herding_volume_lookback else 0.0
        over = compute_overreaction_score(
            returns, cfg.overreaction_lookback, cfg.overreaction_threshold,
        )

        # Blend
        w = cfg.blend_weights
        composite = (
            w["disposition"] * disp
            + w["anchoring"] * anch
            + w["herding"] * herd
            + w["overreaction"] * over
        )
        composite = float(np.clip(composite, -1.0, 1.0))

        signals.append(BehavioralSignal(
            symbol=sym,
            disposition_score=round(disp, 4),
            anchoring_score=round(anch, 4),
            herding_score=round(herd, 4),
            overreaction_score=round(over, 4),
            composite_score=round(composite, 4),
        ))

    logger.info(
        "[BehavioralFinance] Generated signals for %d symbols, "
        "avg composite=%.3f",
        len(signals),
        np.mean([s.composite_score for s in signals]) if signals else 0.0,
    )

    return signals


__all__ = [
    "BehavioralConfig",
    "BehavioralSignal",
    "compute_disposition_score",
    "compute_anchoring_score",
    "compute_herding_score",
    "compute_overreaction_score",
    "generate_behavioral_signals",
]
