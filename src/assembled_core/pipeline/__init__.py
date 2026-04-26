# src/assembled_core/pipeline/__init__.py
"""Pipeline modules for trading strategy execution, backtesting, and portfolio simulation."""

# TradingContext and TradingCycleResult live in trading_cycle (shared types).
# run_trading_cycle is served by trading_cycle_v2 (7-function decomposition).
from src.assembled_core.pipeline.trading_cycle import (
    TradingContext,
    TradingCycleResult,
)
from src.assembled_core.pipeline.trading_cycle_v2 import (
    run_trading_cycle,
)

__all__ = [
    "TradingContext",
    "TradingCycleResult",
    "run_trading_cycle",
]
