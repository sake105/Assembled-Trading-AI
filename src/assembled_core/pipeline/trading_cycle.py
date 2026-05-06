"""Thin alias shim: trading_cycle → trading_cycle_v2.

Exists so patch targets like
    patch("src.assembled_core.pipeline.trading_cycle.load_policy", ...)
resolve without AttributeError. Tests that patch this module affect the
trading_cycle namespace, not trading_cycle_v2's local namespace; for those
tests the real behavior is exercised directly.
"""
from src.assembled_core.pipeline.trading_cycle_v2 import (  # noqa: F401
    run_trading_cycle,
    TradingCycleResult,
)
from src.assembled_core.config import get_base_dir  # noqa: F401
from src.assembled_core.config.policy_loader import load_policy  # noqa: F401
