"""Pipeline Timing (Plan 11.5).

Timer per pipeline step with alert when total exceeds budget.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PipelineTimer:
    """Track execution time per pipeline step."""

    budget_seconds: float = 300.0  # 5 minute default budget
    _timings: dict[str, float] = field(default_factory=dict)
    _current_step: str | None = field(default=None, init=False)
    _step_start: float = field(default=0.0, init=False)

    def start_step(self, name: str) -> None:
        """Start timing a step."""
        self._current_step = name
        self._step_start = time.monotonic()

    def end_step(self) -> float:
        """End current step timing.

        Returns:
            Duration in seconds.
        """
        if self._current_step is None:
            return 0.0

        duration = time.monotonic() - self._step_start
        self._timings[self._current_step] = duration
        logger.info("[Timer] %s: %.2fs", self._current_step, duration)
        self._current_step = None
        return duration

    @property
    def total_seconds(self) -> float:
        return sum(self._timings.values())

    @property
    def over_budget(self) -> bool:
        return self.total_seconds > self.budget_seconds

    def summary(self) -> dict:
        """Get timing summary."""
        return {
            "steps": dict(self._timings),
            "total_seconds": round(self.total_seconds, 2),
            "budget_seconds": self.budget_seconds,
            "over_budget": self.over_budget,
        }


__all__ = ["PipelineTimer"]
