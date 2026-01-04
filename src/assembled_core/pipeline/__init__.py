# src/assembled_core/pipeline/__init__.py
"""Pipeline modules for trading strategy execution, backtesting, and portfolio simulation."""

# Lazy import to avoid circular dependencies
# trading_cycle imports execution modules which import qa modules which import backtest_engine
# which imports trading_cycle again. We delay the import until actually needed.

__all__ = [
    "TradingContext",
    "TradingCycleResult",
    "run_trading_cycle",
]


def __getattr__(name: str):
    """Lazy import for trading_cycle to avoid circular dependencies."""
    if name in ("TradingContext", "TradingCycleResult", "run_trading_cycle"):
        from src.assembled_core.pipeline.trading_cycle import (
            TradingContext,
            TradingCycleResult,
            run_trading_cycle,
        )
        if name == "TradingContext":
            return TradingContext
        elif name == "TradingCycleResult":
            return TradingCycleResult
        elif name == "run_trading_cycle":
            return run_trading_cycle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
