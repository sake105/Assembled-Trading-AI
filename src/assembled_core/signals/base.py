"""BaseSignal plugin interface and SignalOutput.

From 33_EXECUTION_ORDERMANAGEMENT.md §33.13.

Signals are loaded via entry-points group "ata.signals" so new signals
can be added as separate packages without touching the core.

Usage:
    from src.assembled_core.signals.base import BaseSignal, SignalOutput
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SignalOutput:
    """Output of a single signal computation."""

    symbol: str
    score: float  # [-1, +1]
    confidence: float  # [0, 1]
    metadata: dict[str, Any] = field(default_factory=dict)
    features_used: list[str] = field(default_factory=list)
    computed_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    horizon_days: int = 5
    required_data: list[str] = field(default_factory=list)

    def is_actionable(
        self, min_confidence: float = 0.0, min_abs_score: float = 0.0
    ) -> bool:
        return abs(self.score) >= min_abs_score and self.confidence >= min_confidence


class BaseSignal(ABC):
    """Abstract base for all trading signals.

    Subclasses must implement ``compute()``.  The plugin system loads
    concrete implementations via ``importlib.metadata`` entry-points.
    """

    name: str = "base"
    version: str = "0.0.0"
    required_features: list[str] = []
    required_data: list[str] = []  # e.g. ['bars_daily', 'news']
    horizon_days: int = 5

    @abstractmethod
    async def compute(
        self,
        symbol: str,
        feature_store: Any,
        now: datetime,
    ) -> SignalOutput | None:
        """Compute signal for *symbol* at *now*.

        Returns None when data is insufficient or the signal is inactive.
        """
        ...

    async def healthcheck(self) -> bool:
        """Return True if the signal is operational."""
        return True

    def describe(self) -> dict[str, Any]:
        """Machine-readable descriptor for the feature catalog."""
        return {
            "name": self.name,
            "version": self.version,
            "horizon_days": self.horizon_days,
            "required_features": self.required_features,
            "required_data": self.required_data,
            "docstring": self.__class__.__doc__ or "",
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, v={self.version})"


__all__ = ["BaseSignal", "SignalOutput"]
