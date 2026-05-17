"""Monte-Carlo path simulation and trade-shuffling for backtest robustness.

Public API
----------
- :func:`shuffle_trades` — bootstrap-resample trade P&L WITH replacement.
- :func:`permute_trades` — permute trade order WITHOUT replacement (canonical
  replacement for legacy ``qa.monte_carlo_paths.monte_carlo_trade_paths``).
- :func:`simulate_paths_iid_normal` — parametric i.i.d. normal-return path
  simulation (formerly mis-named "gbm"; see F-risk-4).
- :func:`simulate_paths_block_bootstrap` — block bootstrap path simulation.
- :class:`ShuffleResult` — result container for trade-shuffle/permute.
- :class:`PathSimResult` — result container for path simulations.

**Canonical module** for Monte-Carlo robustness analysis (§6.5.3
consolidation, 2026-05-17). Legacy modules ``qa/monte_carlo.py`` and
``qa/monte_carlo_paths.py`` are deprecated — they emit a
``DeprecationWarning`` and point to this module.

Complements ``qa/scenario_engine.py`` (historical stress-replays).
"""

from __future__ import annotations

from src.assembled_core.risk.monte_carlo.path_simulator import (
    PathSimResult,
    simulate_paths_block_bootstrap,
    simulate_paths_iid_normal,
)
from src.assembled_core.risk.monte_carlo.trade_shuffle import (
    ShuffleResult,
    permute_trades,
    shuffle_trades,
)

__all__ = [
    "PathSimResult",
    "ShuffleResult",
    "permute_trades",
    "shuffle_trades",
    "simulate_paths_block_bootstrap",
    "simulate_paths_iid_normal",
]
