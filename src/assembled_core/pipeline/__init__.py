# src/assembled_core/pipeline/__init__.py
"""Pipeline modules for trading strategy execution, backtesting, and portfolio simulation."""

# Direct imports (no lazy imports - circular dependencies are resolved)
from src.assembled_core.pipeline.trading_cycle import (
    TradingContext,
    TradingCycleResult,
    run_trading_cycle,
)

__all__ = [
    "TradingContext",
    "TradingCycleResult",
    "run_trading_cycle",
]
