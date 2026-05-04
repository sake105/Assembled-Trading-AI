"""Tests for M21.2: Strategy Allocator — multi-strategy ensemble."""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.strategies.base import Strategy, StrategySignal

pytest.importorskip("src.assembled_core.portfolio.strategy_allocator")
from src.assembled_core.portfolio.strategy_allocator import (
    AllocationConfig,
    EnsembleResult,
    StrategyAllocator,
)

# -- Test strategies ---------------------------------------------------------


class BullStrategy(Strategy):
    @property
    def name(self) -> str:
        return "bull_strat"

    def generate_signals(self, prices, **kwargs):
        return StrategySignal(
            signals=pd.DataFrame(
                {
                    "timestamp": pd.Timestamp("2024-01-01"),
                    "symbol": ["AAPL", "MSFT", "GOOG"],
                    "direction": ["LONG", "LONG", "LONG"],
                    "score": [0.9, 0.7, 0.5],
                }
            )
        )


class BearStrategy(Strategy):
    @property
    def name(self) -> str:
        return "bear_strat"

    def generate_signals(self, prices, **kwargs):
        return StrategySignal(
            signals=pd.DataFrame(
                {
                    "timestamp": pd.Timestamp("2024-01-01"),
                    "symbol": ["AAPL", "MSFT", "TSLA"],
                    "direction": ["SHORT", "LONG", "SHORT"],
                    "score": [0.8, 0.3, 0.6],
                }
            )
        )


class EmptyStrategy(Strategy):
    @property
    def name(self) -> str:
        return "empty_strat"

    def generate_signals(self, prices, **kwargs):
        return StrategySignal(
            signals=pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
        )


class FailingStrategy(Strategy):
    @property
    def name(self) -> str:
        return "failing_strat"

    def generate_signals(self, prices, **kwargs):
        raise RuntimeError("Strategy crashed")


# -- Tests -------------------------------------------------------------------


PRICES = pd.DataFrame({"timestamp": [1], "symbol": ["X"], "close": [100.0]})


@pytest.mark.phase12
class TestAllocationConfig:
    def test_defaults(self):
        cfg = AllocationConfig()
        assert cfg.method == "weighted_average"
        assert cfg.score_normalization is True
        assert cfg.min_strategies_required == 1


@pytest.mark.phase12
class TestStrategyAllocatorInit:
    def test_equal_weights_default(self):
        alloc = StrategyAllocator(
            {"a": BullStrategy(), "b": BearStrategy()},
        )
        assert len(alloc.strategy_names) == 2

    def test_custom_weights_normalized(self):
        cfg = AllocationConfig(weights={"a": 3.0, "b": 1.0})
        alloc = StrategyAllocator(
            {"a": BullStrategy(), "b": BearStrategy()},
            config=cfg,
        )
        assert alloc._config.weights["a"] == pytest.approx(0.75)
        assert alloc._config.weights["b"] == pytest.approx(0.25)


@pytest.mark.phase12
class TestWeightedAverage:
    def test_single_strategy(self):
        alloc = StrategyAllocator(
            {"bull": BullStrategy()},
            config=AllocationConfig(score_normalization=False),
        )
        result = alloc.generate_combined_signals(PRICES)
        assert isinstance(result, EnsembleResult)
        assert len(result.combined_signals) == 3
        assert all(result.combined_signals["direction"] == "LONG")

    def test_two_strategies_blend(self):
        alloc = StrategyAllocator(
            {"bull": BullStrategy(), "bear": BearStrategy()},
            config=AllocationConfig(
                weights={"bull": 0.6, "bear": 0.4},
                score_normalization=False,
            ),
        )
        result = alloc.generate_combined_signals(PRICES)
        combined = result.combined_signals
        # Should have signals for AAPL, MSFT, GOOG, TSLA
        symbols = set(combined["symbol"])
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_per_strategy_signals_returned(self):
        alloc = StrategyAllocator(
            {"bull": BullStrategy(), "bear": BearStrategy()},
        )
        result = alloc.generate_combined_signals(PRICES)
        assert "bull" in result.per_strategy_signals
        assert "bear" in result.per_strategy_signals

    def test_metadata_populated(self):
        alloc = StrategyAllocator({"bull": BullStrategy()})
        result = alloc.generate_combined_signals(PRICES)
        assert result.metadata["method"] == "weighted_average"
        assert result.metadata["n_strategies_active"] == 1


@pytest.mark.phase12
class TestMajorityVote:
    def test_vote_method(self):
        alloc = StrategyAllocator(
            {"bull": BullStrategy(), "bear": BearStrategy()},
            config=AllocationConfig(method="majority_vote"),
        )
        result = alloc.generate_combined_signals(PRICES)
        assert not result.combined_signals.empty
        # MSFT: both say LONG -> LONG
        msft = result.combined_signals[result.combined_signals["symbol"] == "MSFT"]
        if not msft.empty:
            assert msft.iloc[0]["direction"] == "LONG"


@pytest.mark.phase12
class TestRegimeConditional:
    def test_regime_weights_used(self):
        cfg = AllocationConfig(
            method="regime_conditional",
            weights={"bull": 0.5, "bear": 0.5},
            regime_weights={
                "bull": {"bull": 0.9, "bear": 0.1},
                "crisis": {"bull": 0.1, "bear": 0.9},
            },
        )
        alloc = StrategyAllocator(
            {"bull": BullStrategy(), "bear": BearStrategy()},
            config=cfg,
        )
        result_bull = alloc.generate_combined_signals(PRICES, regime="bull")
        result_crisis = alloc.generate_combined_signals(PRICES, regime="crisis")

        assert result_bull.strategy_contributions.get("bull", 0) > 0.5
        assert result_crisis.strategy_contributions.get("bear", 0) > 0.5


@pytest.mark.phase12
class TestEdgeCases:
    def test_empty_strategy_excluded(self):
        alloc = StrategyAllocator(
            {"bull": BullStrategy(), "empty": EmptyStrategy()},
        )
        result = alloc.generate_combined_signals(PRICES)
        assert "empty" not in result.per_strategy_signals
        assert len(result.combined_signals) > 0

    def test_failing_strategy_handled(self):
        alloc = StrategyAllocator(
            {"bull": BullStrategy(), "fail": FailingStrategy()},
        )
        result = alloc.generate_combined_signals(PRICES)
        assert "fail" not in result.per_strategy_signals
        assert len(result.combined_signals) > 0

    def test_min_strategies_gate(self):
        alloc = StrategyAllocator(
            {"empty": EmptyStrategy()},
            config=AllocationConfig(min_strategies_required=1),
        )
        result = alloc.generate_combined_signals(PRICES)
        assert result.metadata.get("error") == "insufficient_strategies"

    def test_score_normalization(self):
        alloc = StrategyAllocator(
            {"bull": BullStrategy()},
            config=AllocationConfig(score_normalization=True),
        )
        result = alloc.generate_combined_signals(PRICES)
        assert not result.combined_signals.empty
