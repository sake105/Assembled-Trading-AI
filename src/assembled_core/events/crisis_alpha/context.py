"""CrisisAlphaContext — input contract for the Crisis-Alpha v1 subsystem.

All inputs needed by the crisis state machine, gates, entry logic, and
risk-budget checks are bundled here.  This makes the crisis pipeline a
pure function of this context, enabling deterministic testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class CrisisAlphaContext:
    """Input contract for the Crisis-Alpha pipeline.

    Attributes:
        timestamp_utc: Current UTC timestamp (evaluation point).
        geo_score: Aggregated geo-risk score from news triggers (0.0–3.0 scale).
            Derived from ``news_trigger_items`` by caller.
        geo_sources: Number of distinct news sources contributing to geo_score.
            Used to guard against single-source or social-only activation.
        social_only: True if the geo signal comes exclusively from social media
            (no confirmed Tier-A news or disclosure source).  Must block activation.
        market_stress_ok: True if market stress is confirmed (e.g. vol_z elevated,
            drawdown present, VIX signal).  Required for WATCH→ACTIVE.
        health_ok: True if the news/intel health status is GREEN or DEGRADED
            (not ERROR).  ERROR health must block activation.
        daily_pnl: Today's realised + unrealised P&L for the crisis sub-portfolio
            (negative = loss).  Used by the daily loss guard.
        daily_loss_limit: Maximum allowed daily loss (absolute, positive value).
            If abs(daily_pnl) >= daily_loss_limit and pnl is negative → PAUSE.
        news_trigger_items: Raw trigger items from the NEWS pipeline.  Used by
            evidence rules to count qualifying triggers.
        open_positions: List of dicts with keys ``symbol``, ``side``, ``qty``,
            ``entry_price``, ``entry_ts`` for current crisis-alpha positions.
        metadata: Optional additional context (for logging / debugging).
    """

    timestamp_utc: datetime
    geo_score: float
    geo_sources: int
    social_only: bool
    market_stress_ok: bool
    health_ok: bool
    daily_pnl: float = 0.0
    daily_loss_limit: float = 0.02  # 2 % of portfolio equity as default
    news_trigger_items: list[dict[str, Any]] = field(default_factory=list)
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, timestamp_utc: datetime | None = None) -> "CrisisAlphaContext":
        """Return a neutral context with no geo signal (WATCH-safe defaults)."""
        return cls(
            timestamp_utc=timestamp_utc or datetime.now(timezone.utc),
            geo_score=0.0,
            geo_sources=0,
            social_only=False,
            market_stress_ok=False,
            health_ok=True,
        )

    def daily_loss_breached(self) -> bool:
        """Return True if the daily loss limit has been exceeded."""
        return self.daily_pnl < 0 and abs(self.daily_pnl) >= self.daily_loss_limit
