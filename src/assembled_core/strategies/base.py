"""Strategy Base Protocol and Registry (M21.1).

Provides a unified interface for all trading strategies so they can be
composed in a multi-strategy ensemble. Strategies can be functional
(a compute_signals function) or class-based.

Usage:
    from src.assembled_core.strategies.base import Strategy, StrategyRegistry

    # Register a functional strategy
    @StrategyRegistry.register("ema_trend")
    class EMATrendStrategy(Strategy):
        name = "ema_trend"
        def generate_signals(self, prices, **kwargs): ...

    # Lookup
    strat = StrategyRegistry.get("ema_trend")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    """Standardized strategy output.

    All strategies must produce signals in this format for ensemble
    composition to work correctly.

    Attributes:
        signals: DataFrame with columns [timestamp, symbol, direction, score].
            direction: "LONG", "SHORT", or "FLAT".
            score: Numeric signal strength (higher = stronger conviction).
        metadata: Optional strategy-specific metadata.
    """

    signals: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        required = {"timestamp", "symbol", "direction", "score"}
        if not self.signals.empty:
            missing = required - set(self.signals.columns)
            if missing:
                raise ValueError(
                    f"StrategySignal missing required columns: {missing}"
                )


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement generate_signals() and provide a name.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        return self.__class__.__doc__ or self.name

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        **kwargs,
    ) -> StrategySignal:
        """Generate trading signals from price data.

        Args:
            prices: DataFrame with at minimum [timestamp, symbol, close].
            **kwargs: Strategy-specific parameters (factors, regime, etc.).

        Returns:
            StrategySignal with standardized signal DataFrame.
        """
        ...

    def validate_inputs(self, prices: pd.DataFrame) -> bool:
        """Check that prices DataFrame has minimum required columns."""
        required = {"timestamp", "symbol", "close"}
        return required.issubset(set(prices.columns)) and not prices.empty


# ---------------------------------------------------------------------------
# Wrapper: adapt functional strategies to the Strategy protocol
# ---------------------------------------------------------------------------


class FunctionalStrategy(Strategy):
    """Wraps a function with signature (prices, **kwargs) -> DataFrame into Strategy.

    The wrapped function should return a DataFrame with
    [timestamp, symbol, direction, score] columns.
    """

    def __init__(
        self,
        name: str,
        func: callable,
        description: str = "",
    ):
        self._name = name
        self._func = func
        self._description = description or f"Functional strategy: {name}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def generate_signals(
        self,
        prices: pd.DataFrame,
        **kwargs,
    ) -> StrategySignal:
        result = self._func(prices, **kwargs)
        if isinstance(result, pd.DataFrame):
            return StrategySignal(signals=result, metadata={"strategy": self._name})
        if isinstance(result, StrategySignal):
            return result
        raise TypeError(
            f"Strategy function {self._name} returned {type(result)}, "
            "expected DataFrame or StrategySignal"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class StrategyRegistry:
    """Global registry for named strategies.

    Supports both class registration via decorator and runtime registration
    of Strategy instances or functions.
    """

    _strategies: dict[str, Strategy] = {}

    @classmethod
    def register(cls, name: str | None = None):
        """Decorator to register a Strategy subclass.

        Usage:
            @StrategyRegistry.register("my_strat")
            class MyStrategy(Strategy): ...
        """
        def decorator(strategy_cls):
            reg_name = name or getattr(strategy_cls, "name", strategy_cls.__name__)
            if isinstance(reg_name, property):
                # Can't read property from class, instantiate to get name
                instance = strategy_cls()
                reg_name = instance.name
                cls._strategies[reg_name] = instance
            else:
                cls._strategies[reg_name] = strategy_cls()
            logger.debug("[StrategyRegistry] registered: %s", reg_name)
            return strategy_cls
        return decorator

    @classmethod
    def register_instance(cls, strategy: Strategy) -> None:
        """Register an already-instantiated Strategy."""
        cls._strategies[strategy.name] = strategy
        logger.debug("[StrategyRegistry] registered instance: %s", strategy.name)

    @classmethod
    def register_function(
        cls,
        name: str,
        func: callable,
        description: str = "",
    ) -> FunctionalStrategy:
        """Register a plain function as a strategy."""
        wrapper = FunctionalStrategy(name, func, description)
        cls._strategies[name] = wrapper
        logger.debug("[StrategyRegistry] registered function: %s", name)
        return wrapper

    @classmethod
    def get(cls, name: str) -> Strategy | None:
        """Look up a registered strategy by name."""
        return cls._strategies.get(name)

    @classmethod
    def list_strategies(cls) -> list[str]:
        """Return list of registered strategy names."""
        return list(cls._strategies.keys())

    @classmethod
    def get_all(cls) -> dict[str, Strategy]:
        """Return all registered strategies."""
        return dict(cls._strategies)

    @classmethod
    def clear(cls) -> None:
        """Remove all registered strategies (mainly for testing)."""
        cls._strategies.clear()


__all__ = [
    "Strategy",
    "StrategySignal",
    "FunctionalStrategy",
    "StrategyRegistry",
]
