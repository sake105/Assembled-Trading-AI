# src/assembled_core/pipeline/__init__.py
"""Pipeline modules for trading strategy execution, backtesting, and portfolio simulation."""

# Direct imports (no lazy imports - circular dependencies are resolved)
from src.assembled_core.pipeline.trading_cycle import (
    TradingContext,
    TradingCycleResult,
    run_trading_cycle,
)
from src.assembled_core.pipeline.event_bus import Event, EventBus, EventType
from src.assembled_core.pipeline.graceful_degradation import (  # noqa: F401
    DegradationTracker,
    neutralize_missing_features,
)

__all__ = [
    "TradingContext",
    "TradingCycleResult",
    "run_trading_cycle",
    "Event",
    "EventBus",
    "EventType",
    "DegradationTracker",
    "neutralize_missing_features",
]
