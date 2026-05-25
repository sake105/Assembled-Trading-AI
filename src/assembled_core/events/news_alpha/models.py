"""Data models for news alpha signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class NewsAlphaSignal:
    """A directional trade signal generated from a news event.

    Consumed by the news_alpha pipeline and passed to the execution layer.
    """

    # Identity
    event_id: str  # unique ID from trigger item
    topic_id: str  # e.g. "shipping_disruption"
    trigger_type: str  # e.g. "supply_chain"
    source: str  # news source label

    # Trade direction
    symbol: str  # ETF ticker
    direction: Literal["long", "short"]
    is_2x: bool = False  # True if this is a leveraged ETF

    # Sizing
    raw_weight: float = 0.0  # pre-cap target weight in news_alpha sub-portfolio
    severity: int = 2  # 1/2/3 — affects sizing

    # Timing
    signal_utc: str = ""  # ISO UTC string when signal was generated
    entry_day: int = 0  # days since signal (0 = same day / next open)
    hold_days: int = 5  # expected holding period

    # Exit anchors (set at entry, updated by exit_rules)
    entry_price: float = 0.0
    stop_loss_pct: float = 0.08  # default 8% stop loss
    take_profit_pct: float = 0.15  # default 15% take profit

    # Metadata
    rationale: str = ""
    active: bool = True  # False = position should be closed


@dataclass
class NewsAlphaResult:
    """Output of the news_alpha pipeline for one evaluation cycle."""

    timestamp_utc: str
    signals: list[NewsAlphaSignal] = field(default_factory=list)
    target_weights: dict[str, float] = field(default_factory=dict)
    positions_to_exit: list[tuple[NewsAlphaSignal, str]] = field(default_factory=list)
    shadow_only: bool = True
    errors: list[str] = field(default_factory=list)
