"""Strategy Allocator — Multi-Strategy Ensemble Framework (M21.2).

Combines multiple trading strategies into a single blended signal using
configurable combination methods:
  - weighted_average: Capital-weighted signal blending
  - majority_vote: Direction consensus with average score
  - regime_conditional: Different strategy weights per market regime

Usage:
    allocator = StrategyAllocator(
        strategies={"ema_trend": strat1, "multifactor": strat2},
        weights={"ema_trend": 0.3, "multifactor": 0.7},
        method="weighted_average",
    )
    combined = allocator.generate_combined_signals(prices)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.assembled_core.strategies.base import Strategy, StrategySignal

logger = logging.getLogger(__name__)


@dataclass
class AllocationConfig:
    """Configuration for strategy allocation.

    Attributes:
        weights: Strategy name -> capital weight (should sum to 1.0).
        method: Combination method: "weighted_average", "majority_vote",
                or "regime_conditional".
        regime_weights: For regime_conditional method: regime -> {strategy -> weight}.
        min_strategies_required: Minimum number of strategies that must produce
            signals for the ensemble to be valid.
        score_normalization: Whether to z-score normalize individual strategy scores
            before combining. Prevents one strategy from dominating.
    """

    weights: dict[str, float] = field(default_factory=dict)
    method: str = "weighted_average"
    regime_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    min_strategies_required: int = 1
    score_normalization: bool = True


@dataclass
class EnsembleResult:
    """Result of multi-strategy ensemble.

    Attributes:
        combined_signals: Blended signal DataFrame [timestamp, symbol, direction, score].
        per_strategy_signals: Individual strategy outputs.
        strategy_contributions: Weight actually applied to each strategy.
        metadata: Diagnostic information.
    """

    combined_signals: pd.DataFrame
    per_strategy_signals: dict[str, pd.DataFrame]
    strategy_contributions: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


class StrategyAllocator:
    """Combines multiple strategies into a blended ensemble signal.

    Supports three combination methods:
    1. weighted_average: Weighted mean of scores across strategies.
    2. majority_vote: Direction decided by majority, score averaged.
    3. regime_conditional: Weights change based on current market regime.
    """

    def __init__(
        self,
        strategies: dict[str, Strategy],
        config: AllocationConfig | None = None,
    ):
        self._strategies = strategies
        self._config = config or AllocationConfig()

        # Default equal weights if not provided
        if not self._config.weights:
            n = len(strategies)
            w = 1.0 / n if n > 0 else 1.0
            self._config.weights = {name: w for name in strategies}

        # Normalize weights to sum to 1
        total = sum(self._config.weights.values())
        if total > 0:
            self._config.weights = {
                k: v / total for k, v in self._config.weights.items()
            }

    @property
    def strategy_names(self) -> list[str]:
        return list(self._strategies.keys())

    def generate_combined_signals(
        self,
        prices: pd.DataFrame,
        regime: str = "bull",
        **kwargs,
    ) -> EnsembleResult:
        """Run all strategies and combine their signals.

        Args:
            prices: Price data for all strategies.
            regime: Current market regime (used for regime_conditional method).
            **kwargs: Passed to each strategy's generate_signals.

        Returns:
            EnsembleResult with combined and per-strategy signals.
        """
        per_strategy: dict[str, pd.DataFrame] = {}
        active_weights: dict[str, float] = {}

        # Select weights based on method
        if self._config.method == "regime_conditional" and self._config.regime_weights:
            weights = self._config.regime_weights.get(
                regime, self._config.weights
            )
        else:
            weights = self._config.weights

        # Run each strategy
        for name, strategy in self._strategies.items():
            try:
                result = strategy.generate_signals(prices, **kwargs)
                signals = result.signals if isinstance(result, StrategySignal) else result
                if not signals.empty:
                    per_strategy[name] = signals
                    active_weights[name] = weights.get(name, 0.0)
                    logger.debug(
                        "[Allocator] %s produced %d signals", name, len(signals),
                    )
                else:
                    logger.debug("[Allocator] %s produced no signals", name)
            except Exception as exc:
                logger.warning(
                    "[Allocator] %s failed: %s", name, exc,
                )

        # Check minimum strategies
        if len(per_strategy) < self._config.min_strategies_required:
            logger.warning(
                "[Allocator] Only %d/%d strategies produced signals (min=%d)",
                len(per_strategy), len(self._strategies),
                self._config.min_strategies_required,
            )
            return EnsembleResult(
                combined_signals=pd.DataFrame(
                    columns=["timestamp", "symbol", "direction", "score"]
                ),
                per_strategy_signals=per_strategy,
                strategy_contributions=active_weights,
                metadata={"error": "insufficient_strategies"},
            )

        # Normalize weights for active strategies
        total_w = sum(active_weights.values())
        if total_w > 0:
            active_weights = {k: v / total_w for k, v in active_weights.items()}

        # Combine signals
        method = self._config.method
        if method == "majority_vote":
            combined = self._combine_majority_vote(per_strategy, active_weights)
        else:
            combined = self._combine_weighted_average(per_strategy, active_weights)

        return EnsembleResult(
            combined_signals=combined,
            per_strategy_signals=per_strategy,
            strategy_contributions=active_weights,
            metadata={
                "method": method,
                "regime": regime,
                "n_strategies_active": len(per_strategy),
                "n_strategies_total": len(self._strategies),
            },
        )

    def _combine_weighted_average(
        self,
        per_strategy: dict[str, pd.DataFrame],
        weights: dict[str, float],
    ) -> pd.DataFrame:
        """Blend scores via weighted average across strategies.

        For each (timestamp, symbol), the combined score is:
            score = sum(w_i * score_i) for all strategies producing a signal.
        Direction is determined by the sign of the combined score.
        """
        all_signals = []
        for name, signals in per_strategy.items():
            df = signals.copy()
            w = weights.get(name, 0.0)

            # Normalize scores per strategy if configured
            if self._config.score_normalization and len(df) > 1:
                mean = df["score"].mean()
                std = df["score"].std()
                if std > 1e-10:
                    df["score"] = (df["score"] - mean) / std

            # Apply direction sign
            df["signed_score"] = df.apply(
                lambda r: r["score"] if r["direction"] == "LONG"
                else -r["score"] if r["direction"] == "SHORT"
                else 0.0,
                axis=1,
            )
            df["weighted_score"] = df["signed_score"] * w
            df["strategy"] = name
            all_signals.append(df[["timestamp", "symbol", "weighted_score", "strategy"]])

        if not all_signals:
            return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])

        merged = pd.concat(all_signals, ignore_index=True)
        grouped = merged.groupby(["timestamp", "symbol"], as_index=False).agg(
            score=("weighted_score", "sum"),
            n_strategies=("strategy", "nunique"),
        )

        grouped["direction"] = np.where(
            grouped["score"] > 0, "LONG",
            np.where(grouped["score"] < 0, "SHORT", "FLAT"),
        )
        grouped["score"] = grouped["score"].abs()

        return grouped[["timestamp", "symbol", "direction", "score"]].copy()

    def _combine_majority_vote(
        self,
        per_strategy: dict[str, pd.DataFrame],
        weights: dict[str, float],
    ) -> pd.DataFrame:
        """Direction by majority vote, score by weighted average magnitude."""
        all_signals = []
        for name, signals in per_strategy.items():
            df = signals.copy()
            w = weights.get(name, 0.0)
            df["vote"] = df["direction"].map(
                {"LONG": 1, "SHORT": -1, "FLAT": 0}
            ).fillna(0).astype(int)
            df["weighted_vote"] = df["vote"] * w
            df["weighted_score"] = df["score"].abs() * w
            df["strategy"] = name
            all_signals.append(
                df[["timestamp", "symbol", "weighted_vote", "weighted_score", "strategy"]]
            )

        if not all_signals:
            return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])

        merged = pd.concat(all_signals, ignore_index=True)
        grouped = merged.groupby(["timestamp", "symbol"], as_index=False).agg(
            vote_sum=("weighted_vote", "sum"),
            avg_score=("weighted_score", "sum"),
            n_strategies=("strategy", "nunique"),
        )

        grouped["direction"] = np.where(
            grouped["vote_sum"] > 0, "LONG",
            np.where(grouped["vote_sum"] < 0, "SHORT", "FLAT"),
        )
        grouped["score"] = grouped["avg_score"]

        return grouped[["timestamp", "symbol", "direction", "score"]].copy()


__all__ = [
    "AllocationConfig",
    "EnsembleResult",
    "StrategyAllocator",
]
