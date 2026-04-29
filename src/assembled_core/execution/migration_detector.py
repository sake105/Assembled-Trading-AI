"""PDT-to-intraday-margin migration detector.

From 41_PDT_REGEL_INTRADAY_MARGIN.md §6.2.

Monitors whether the broker has migrated away from PDT rules by checking
if 4th-day-trade attempts still result in HTTP 403 blocks.
FINRA effective date: 4 June 2026. Broker phase-in until 20 Oct 2027.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PDTMigrationDetector:
    """Detects if the broker has migrated away from PDT rules.

    If 3+ fourth-day-trade attempts were made in the observation window
    and zero PDT blocks occurred, migration has likely happened.
    """

    def __init__(self, observation_window_days: int = 30) -> None:
        self.observation_window = timedelta(days=observation_window_days)
        self.pdt_blocks: deque[datetime] = deque(maxlen=100)
        self.fourth_day_trade_attempts: deque[datetime] = deque(maxlen=100)

    def record_pdt_block(self, timestamp: datetime | None = None) -> None:
        self.pdt_blocks.append(timestamp or datetime.utcnow())

    def record_fourth_day_trade_attempt(self, timestamp: datetime | None = None) -> None:
        """Record an order that would have been a 4th day-trade.

        If broker allowed it without a 403, migration has occurred.
        """
        self.fourth_day_trade_attempts.append(timestamp or datetime.utcnow())

    def likely_migrated(self) -> bool:
        cutoff = datetime.utcnow() - self.observation_window
        recent_attempts = sum(1 for ts in self.fourth_day_trade_attempts if ts > cutoff)
        recent_blocks = sum(1 for ts in self.pdt_blocks if ts > cutoff)
        if recent_attempts >= 3 and recent_blocks == 0:
            logger.warning(
                "PDT migration likely: %d 4th-day-trade attempts in last %dd, 0 PDT blocks. "
                "Recommend: set pdt_tracker.enabled = False.",
                recent_attempts,
                self.observation_window.days,
            )
            return True
        return False


__all__ = ["PDTMigrationDetector"]
