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
from src.assembled_core.pipeline.pipeline_timing import PipelineTimer  # noqa: F401
from src.assembled_core.pipeline.run_metadata import (  # noqa: F401
    collect_run_metadata,
    save_run_metadata,
)
from src.assembled_core.pipeline import backtest_legacy as backtest_legacy  # noqa: F401

__all__ = [
    "TradingContext",
    "TradingCycleResult",
    "run_trading_cycle",
    "Event",
    "EventBus",
    "EventType",
    "DegradationTracker",
    "neutralize_missing_features",
    "PipelineTimer",
    "collect_run_metadata",
    "save_run_metadata",
    "backtest_legacy",
]
