"""Strategy modules for Assembled Trading AI Backend.

Exposes:
- EMA trend v0 strategy
- Multi-factor long-short strategy
- Stat-arb (cointegration / pair-signals / PCA arbitrage)
- Strategy discovery (genetic / enumerative)
Wired 2026-04-22 to include previously orphan stat_arb and
strategy_discovery modules.
"""

from __future__ import annotations

from src.assembled_core.strategies.ema_trend_v0 import (
    compute_signals as ema_compute_signals,
    compute_target_positions as ema_compute_target_positions,
)
from src.assembled_core.strategies.multifactor_long_short import (
    MultiFactorStrategyConfig,
    compute_multifactor_long_short_positions,
    generate_multifactor_long_short_signals,
)
from src.assembled_core.strategies import stat_arb
from src.assembled_core.strategies.stat_arb import (
    PCAFactorModel,
    PCASignal,
    PairCandidate,
    PairPosition,
    PairResult,
    PairSignal,
    PairSignalGenerator,
    PairTradeSignal,
    check_cointegration,
    compute_pca_factors,
    compute_spread,
    estimate_half_life,
    estimate_hedge_ratio,
    find_cointegrated_pairs,
    generate_pair_signal,
    generate_pca_signals,
    screen_pairs,
    test_cointegration,
)
from src.assembled_core.strategies.strategy_discovery import (
    DiscoveryResult,
    StrategyCandidate,
    discover_strategies,
)

__all__ = [
    "MultiFactorStrategyConfig",
    "generate_multifactor_long_short_signals",
    "compute_multifactor_long_short_positions",
    "ema_compute_signals",
    "ema_compute_target_positions",
    # Stat-arb module + API
    "stat_arb",
    "PairResult",
    "PairTradeSignal",
    "estimate_hedge_ratio",
    "compute_spread",
    "estimate_half_life",
    "check_cointegration",
    "find_cointegrated_pairs",
    "generate_pair_signal",
    "PairCandidate",
    "test_cointegration",
    "screen_pairs",
    "PairSignalGenerator",
    "PairSignal",
    "PairPosition",
    "compute_pca_factors",
    "generate_pca_signals",
    "PCAFactorModel",
    "PCASignal",
    # Strategy discovery
    "StrategyCandidate",
    "DiscoveryResult",
    "discover_strategies",
]
