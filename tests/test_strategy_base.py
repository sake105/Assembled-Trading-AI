"""Tests for M21.1: Strategy base protocol and registry."""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.strategies.base import (
    Strategy,
    StrategySignal,
    FunctionalStrategy,
    StrategyRegistry,
)

# -- Fixtures ---------------------------------------------------------------


class DummyStrategy(Strategy):
    @property
    def name(self) -> str:
        return "dummy"

    def generate_signals(self, prices, **kwargs):
        return StrategySignal(
            signals=pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2024-01-01")],
                    "symbol": ["AAPL"],
                    "direction": ["LONG"],
                    "score": [0.8],
                }
            )
        )


def dummy_func(prices, **kwargs):
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "symbol": ["SPY"],
            "direction": ["LONG"],
            "score": [0.5],
        }
    )


# -- Tests -------------------------------------------------------------------


@pytest.mark.fast
class TestStrategySignal:
    def test_valid_signal(self):
        sig = StrategySignal(
            signals=pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2024-01-01")],
                    "symbol": ["AAPL"],
                    "direction": ["LONG"],
                    "score": [0.5],
                }
            )
        )
        assert len(sig.signals) == 1

    def test_empty_signal_ok(self):
        sig = StrategySignal(signals=pd.DataFrame())
        assert sig.signals.empty

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            StrategySignal(
                signals=pd.DataFrame(
                    {
                        "timestamp": [1],
                        "symbol": ["A"],
                        "score": [0.5],
                    }
                )
            )


@pytest.mark.fast
class TestStrategy:
    def test_dummy_strategy_signals(self):
        strat = DummyStrategy()
        result = strat.generate_signals(pd.DataFrame())
        assert isinstance(result, StrategySignal)
        assert len(result.signals) == 1
        assert result.signals.iloc[0]["direction"] == "LONG"

    def test_name_property(self):
        assert DummyStrategy().name == "dummy"

    def test_description_fallback(self):
        strat = DummyStrategy()
        assert isinstance(strat.description, str)

    def test_validate_inputs(self):
        strat = DummyStrategy()
        good = pd.DataFrame(
            {
                "timestamp": [1],
                "symbol": ["A"],
                "close": [100.0],
            }
        )
        assert strat.validate_inputs(good) is True
        assert strat.validate_inputs(pd.DataFrame()) is False


@pytest.mark.fast
class TestFunctionalStrategy:
    def test_wrap_function(self):
        fs = FunctionalStrategy("test_func", dummy_func)
        assert fs.name == "test_func"
        result = fs.generate_signals(pd.DataFrame())
        assert isinstance(result, StrategySignal)
        assert result.signals.iloc[0]["symbol"] == "SPY"

    def test_bad_return_type_raises(self):
        def bad_func(prices, **kwargs):
            return "not a dataframe"

        fs = FunctionalStrategy("bad", bad_func)
        with pytest.raises(TypeError, match="expected DataFrame"):
            fs.generate_signals(pd.DataFrame())


@pytest.mark.fast
class TestStrategyRegistry:
    def setup_method(self):
        StrategyRegistry.clear()

    def test_register_instance(self):
        strat = DummyStrategy()
        StrategyRegistry.register_instance(strat)
        assert StrategyRegistry.get("dummy") is strat

    def test_register_function(self):
        StrategyRegistry.register_function("my_func", dummy_func)
        strat = StrategyRegistry.get("my_func")
        assert strat is not None
        assert strat.name == "my_func"

    def test_list_strategies(self):
        StrategyRegistry.register_instance(DummyStrategy())
        StrategyRegistry.register_function("func1", dummy_func)
        names = StrategyRegistry.list_strategies()
        assert "dummy" in names
        assert "func1" in names

    def test_get_unknown_returns_none(self):
        assert StrategyRegistry.get("nonexistent") is None

    def test_clear(self):
        StrategyRegistry.register_instance(DummyStrategy())
        assert len(StrategyRegistry.list_strategies()) == 1
        StrategyRegistry.clear()
        assert len(StrategyRegistry.list_strategies()) == 0

    def test_get_all(self):
        StrategyRegistry.register_instance(DummyStrategy())
        all_strats = StrategyRegistry.get_all()
        assert "dummy" in all_strats
