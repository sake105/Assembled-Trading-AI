"""Intel signal adapter — bridges IntelSignal to the signals layer.

Converts an IntelSignal (from news_signal_aggregator) into a format
compatible with the rules_trend signal pipeline:
- per-symbol direction scores in [-1, +1]
- a macro overlay score for broad-market risk

This is the seam between the news/intel world and the trading decisions.
Callers typically fetch an IntelSignal from the aggregator and pass it
here to get a signal overlay they can blend with technical signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# When risk_level is HIGH/CRITICAL, multiply the overlay strength by this factor.
_CRISIS_MULTIPLIER = 1.5

# Maximum absolute overlay score we will inject into any ticker.
_MAX_OVERLAY = 0.5


@dataclass
class IntelOverlay:
    """Per-symbol and macro-level overlay derived from an IntelSignal.

    Fields:
        ticker_scores: {symbol: score} where score ∈ [-1, +1].
            Positive = bullish overlay, negative = bearish overlay.
        macro_score: overall market-level overlay ∈ [-1, +1].
            Derived from net_direction and risk_level.
        risk_level: pass-through from IntelSignal (LOW/MODERATE/HIGH/CRITICAL).
        is_actionable: True if the signal meets actionability criteria.
        generated_at: timestamp of the source IntelSignal.
    """

    ticker_scores: dict[str, float] = field(default_factory=dict)
    macro_score: float = 0.0
    risk_level: str = "LOW"
    is_actionable: bool = False
    generated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @classmethod
    def neutral(cls) -> "IntelOverlay":
        return cls(ticker_scores={}, macro_score=0.0, risk_level="LOW", is_actionable=False)


def adapt_intel_signal(intel_signal: object | None) -> IntelOverlay:
    """Convert an IntelSignal to an IntelOverlay for the signals layer.

    Args:
        intel_signal: IntelSignal from news_signal_aggregator.aggregate_signals,
                      or None if no signal is available.

    Returns:
        IntelOverlay with ticker scores and macro score. Returns neutral overlay
        on None or non-actionable signals.
    """
    if intel_signal is None:
        return IntelOverlay.neutral()

    # Support duck-typing — works with the real IntelSignal dataclass and mocks
    net_direction: str = getattr(intel_signal, "net_direction", "neutral")
    risk_level: str = getattr(intel_signal, "risk_level", "LOW")
    asset_basket: dict[str, float] = getattr(intel_signal, "asset_basket", {})
    is_actionable: bool = getattr(intel_signal, "is_actionable", lambda: False)()
    generated_at: datetime = getattr(
        intel_signal, "generated_at", datetime.now(tz=timezone.utc)
    )

    if not is_actionable:
        return IntelOverlay(
            ticker_scores={},
            macro_score=0.0,
            risk_level=risk_level,
            is_actionable=False,
            generated_at=generated_at,
        )

    # Macro score from net_direction
    direction_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
    raw_macro = direction_map.get(net_direction, 0.0)

    # Scale by risk level
    risk_scale = {"LOW": 0.2, "MODERATE": 0.4, "HIGH": 0.7, "CRITICAL": 1.0}
    scale = risk_scale.get(risk_level, 0.3)
    macro_score = max(-1.0, min(1.0, raw_macro * scale))

    # Per-ticker scores from asset_basket (already in [-1, +1] range)
    ticker_scores: dict[str, float] = {}
    for sym, score in asset_basket.items():
        capped = max(-_MAX_OVERLAY, min(_MAX_OVERLAY, float(score)))
        ticker_scores[sym] = capped

    logger.debug(
        "[IntelAdapter] net_dir=%s risk=%s macro=%.2f tickers=%d",
        net_direction, risk_level, macro_score, len(ticker_scores),
    )

    return IntelOverlay(
        ticker_scores=ticker_scores,
        macro_score=macro_score,
        risk_level=risk_level,
        is_actionable=True,
        generated_at=generated_at,
    )


def overlay_to_dataframe(overlay: IntelOverlay) -> pd.DataFrame:
    """Convert an IntelOverlay to a tidy DataFrame for logging/audit.

    Returns DataFrame with columns: symbol, intel_score, macro_score,
    risk_level, is_actionable, generated_at.
    """
    if not overlay.ticker_scores:
        return pd.DataFrame(
            columns=["symbol", "intel_score", "macro_score", "risk_level",
                     "is_actionable", "generated_at"]
        )

    rows = [
        {
            "symbol": sym,
            "intel_score": score,
            "macro_score": overlay.macro_score,
            "risk_level": overlay.risk_level,
            "is_actionable": overlay.is_actionable,
            "generated_at": overlay.generated_at,
        }
        for sym, score in overlay.ticker_scores.items()
    ]
    return pd.DataFrame(rows)
