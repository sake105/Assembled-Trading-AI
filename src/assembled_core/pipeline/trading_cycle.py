"""Unified Trading Cycle Orchestrator (legacy shim).

All active logic lives in trading_cycle_v2.py and trading_cycle_shared.py.
This file exists only for backward-compatible imports.
"""

from __future__ import annotations

import warnings

from src.assembled_core.config import get_base_dir
from src.assembled_core.config.policy_loader import load_policy
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    TradingCycleResult,
    _apply_group_exposure_caps,
    _apply_pre_trade_impact,
    _apply_risk_controls_default,
    _build_features_default,
    _evaluate_auto_dd_kill_switch,
    _evaluate_circuit_breaker,
    _evaluate_circuit_breaker_daily,
    _evaluate_var_gate,
    _estimate_symbol_volatilities,
    _filter_prices_for_as_of,
    _generate_orders_default,
    should_rebalance,
)

__all__ = [
    "get_base_dir",
    "load_policy",
    "TradingContext",
    "TradingCycleResult",
    "run_trading_cycle",
    "_apply_group_exposure_caps",
    "_apply_pre_trade_impact",
    "_apply_risk_controls_default",
    "_build_features_default",
    "_evaluate_auto_dd_kill_switch",
    "_evaluate_circuit_breaker",
    "_evaluate_circuit_breaker_daily",
    "_evaluate_var_gate",
    "_estimate_symbol_volatilities",
    "_filter_prices_for_as_of",
    "_generate_orders_default",
    "should_rebalance",
]


def run_trading_cycle(*args, **kwargs):
    """DEPRECATED: use trading_cycle_v2.run_trading_cycle instead."""
    warnings.warn(
        "trading_cycle.run_trading_cycle is deprecated; "
        "use src.assembled_core.pipeline.trading_cycle_v2.run_trading_cycle",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.assembled_core.pipeline.trading_cycle_v2 import (
        run_trading_cycle as _v2,
    )
    return _v2(*args, **kwargs)
