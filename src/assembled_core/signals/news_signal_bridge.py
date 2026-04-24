"""News signal bridge — blends news intel overlay with trend signals.

Takes output from rules_trend.generate_trend_signals (technical direction +
score) and modifies it based on an IntelOverlay from intel_signal_adapter.

Blend rules:
  1. If overlay.is_actionable is False: return trend signals unchanged.
  2. For each (timestamp, symbol) row:
     a. Look up ticker_scores[symbol] (falls back to macro_score if missing).
     b. Blend: final_score = (1 - alpha) * trend_score + alpha * news_score
        where alpha is the news blend weight (default 0.20).
     c. If the news score is strongly bearish (< -0.5) and trend is LONG,
        the signal is downgraded to FLAT (news overrides momentum on crisis).
  3. Return the blended signals DataFrame (same schema as trend signals).

This module is intentionally thin: it does NOT re-rank or re-weight positions.
That happens downstream in the portfolio layer.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.assembled_core.signals.intel_signal_adapter import IntelOverlay

logger = logging.getLogger(__name__)

_DEFAULT_NEWS_ALPHA = 0.20   # Weight for news score in blend
_CRISIS_BEARISH_THRESHOLD = -0.5  # News score below this forces LONG → FLAT


def blend_with_news(
    trend_signals: pd.DataFrame,
    overlay: IntelOverlay | None,
    news_alpha: float = _DEFAULT_NEWS_ALPHA,
) -> pd.DataFrame:
    """Blend trend signals with news intel overlay.

    Args:
        trend_signals: DataFrame from rules_trend.generate_trend_signals with
            columns: timestamp, symbol, direction, score.
        overlay: IntelOverlay from intel_signal_adapter.adapt_intel_signal.
            If None or neutral, trend_signals are returned unchanged.
        news_alpha: Blend weight for news score [0, 1]. Higher = more news weight.

    Returns:
        DataFrame with same schema as trend_signals. The ``score`` column is
        blended; ``direction`` may be changed to FLAT in crisis conditions.
    """
    if trend_signals is None or trend_signals.empty:
        return trend_signals if trend_signals is not None else pd.DataFrame()

    if overlay is None or not overlay.is_actionable:
        return trend_signals.copy()

    signals = trend_signals.copy()
    news_alpha = max(0.0, min(1.0, news_alpha))

    def _news_score_for(symbol: str) -> float:
        if symbol in overlay.ticker_scores:
            return overlay.ticker_scores[symbol]
        return overlay.macro_score

    original_count = len(signals)
    downgraded = 0

    for idx, row in signals.iterrows():
        sym = row["symbol"]
        trend_score = float(row["score"])
        news_score = _news_score_for(sym)

        blended = (1.0 - news_alpha) * trend_score + news_alpha * news_score
        signals.at[idx, "score"] = max(0.0, min(1.0, blended))

        # Crisis downgrade: strong bearish news overrides a LONG signal
        if news_score < _CRISIS_BEARISH_THRESHOLD and row["direction"] == "LONG":
            signals.at[idx, "direction"] = "FLAT"
            signals.at[idx, "score"] = 0.0
            downgraded += 1

    if downgraded > 0:
        logger.info(
            "[NewsSignalBridge] %d/%d LONG signals downgraded to FLAT"
            " (news_score < %.2f, risk=%s)",
            downgraded, original_count, _CRISIS_BEARISH_THRESHOLD, overlay.risk_level,
        )

    logger.debug(
        "[NewsSignalBridge] blended %d signals | alpha=%.2f | macro=%.2f | actionable=%s",
        original_count, news_alpha, overlay.macro_score, overlay.is_actionable,
    )

    return signals
