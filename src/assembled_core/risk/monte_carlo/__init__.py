"""Monte-Carlo path simulation and trade-shuffling for backtest robustness.

Public API
----------
- :func:`shuffle_trades` — bootstrap-shuffle trade P&L to estimate Sharpe/MDD CIs.
- :func:`simulate_paths_gbm` — parametric GBM path simulation.
- :func:`simulate_paths_block_bootstrap` — block bootstrap path simulation.
- :class:`ShuffleResult` — result container for trade-shuffle.
- :class:`PathSimResult` — result container for path simulations.

This module complements ``qa/scenario_engine.py`` (historical stress-replays).
It is standalone and not wired into the live pipeline.
"""

from __future__ import annotations

from assembled_core.risk.monte_carlo.path_simulator import (
    PathSimResult,
    simulate_paths_block_bootstrap,
    simulate_paths_gbm,
)
from assembled_core.risk.monte_carlo.trade_shuffle import (
    ShuffleResult,
    shuffle_trades,
)

__all__ = [
    "PathSimResult",
    "ShuffleResult",
    "simulate_paths_block_bootstrap",
    "simulate_paths_gbm",
    "shuffle_trades",
]
