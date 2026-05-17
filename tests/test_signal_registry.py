"""Tests for signals/base.py and signals/registry.py (spec 60)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from src.assembled_core.signals.base import BaseSignal, SignalOutput
from src.assembled_core.signals.registry import SignalRegistry, get_registry

# ---------------------------------------------------------------------------
# SignalOutput
# ---------------------------------------------------------------------------


class TestSignalOutput:
    def test_basic_fields(self):
        out = SignalOutput(
            symbol="AAPL",
            score=0.7,
            confidence=0.85,
            features_used=["rsi", "macd"],
            horizon_days=5,
        )
        assert out.symbol == "AAPL"
        assert out.score == 0.7
        assert out.confidence == 0.85
        assert out.horizon_days == 5

    def test_is_actionable(self):
        out = SignalOutput(symbol="AAPL", score=0.7, confidence=0.85)
        assert out.is_actionable(min_confidence=0.8, min_abs_score=0.5) is True
        assert out.is_actionable(min_confidence=0.9) is False

    def test_default_metadata_empty(self):
        out = SignalOutput(symbol="X", score=0.0, confidence=1.0)
        assert out.metadata == {}

    def test_default_horizon_days(self):
        out = SignalOutput(symbol="X", score=0.0, confidence=1.0)
        assert out.horizon_days == 5


# ---------------------------------------------------------------------------
# BaseSignal — concrete subclass for testing
# ---------------------------------------------------------------------------


class _ConcreteSignal(BaseSignal):
    name = "test_signal"
    version = "1.2.3"
    required_features = ["rsi", "macd"]
    required_data = ["bars_daily"]
    horizon_days = 3

    async def compute(
        self, symbol: str, feature_store: Any, now: datetime
    ) -> SignalOutput | None:
        return SignalOutput(
            symbol=symbol,
            score=0.5,
            confidence=0.9,
            features_used=self.required_features,
            computed_at=now,
            horizon_days=self.horizon_days,
        )


class TestBaseSignal:
    def test_name_and_version(self):
        sig = _ConcreteSignal()
        assert sig.name == "test_signal"
        assert sig.version == "1.2.3"

    def test_required_data(self):
        sig = _ConcreteSignal()
        assert "bars_daily" in sig.required_data

    def test_describe_keys(self):
        sig = _ConcreteSignal()
        desc = sig.describe()
        assert "name" in desc
        assert "version" in desc
        assert "horizon_days" in desc
        assert "required_features" in desc
        assert "required_data" in desc
        assert "docstring" in desc

    def test_describe_values(self):
        sig = _ConcreteSignal()
        desc = sig.describe()
        assert desc["name"] == "test_signal"
        assert desc["horizon_days"] == 3

    def test_healthcheck_default_true(self):
        import asyncio

        sig = _ConcreteSignal()
        result = asyncio.run(sig.healthcheck())
        assert result is True

    def test_repr(self):
        sig = _ConcreteSignal()
        assert "test_signal" in repr(sig)


# ---------------------------------------------------------------------------
# SignalRegistry
# ---------------------------------------------------------------------------


class TestSignalRegistry:
    def test_empty_registry(self):
        reg = SignalRegistry()
        assert len(reg) == 0
        assert reg.all() == []

    def test_register_and_get(self):
        reg = SignalRegistry()
        sig = _ConcreteSignal()
        reg.register(sig)
        assert reg.get("test_signal") is sig

    def test_get_unknown_returns_none(self):
        reg = SignalRegistry()
        assert reg.get("does_not_exist") is None

    def test_duplicate_name_raises(self):
        reg = SignalRegistry()
        reg.register(_ConcreteSignal())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_ConcreteSignal())

    def test_names_sorted(self):
        reg = SignalRegistry()
        reg.register(_ConcreteSignal())
        assert reg.names() == ["test_signal"]

    def test_iter(self):
        reg = SignalRegistry()
        reg.register(_ConcreteSignal())
        signals = list(reg)
        assert len(signals) == 1

    def test_errors_empty_by_default(self):
        reg = SignalRegistry()
        assert reg.errors() == {}

    def test_load_all_no_entry_points(self):
        reg = SignalRegistry()
        count = reg.load_all()
        assert count == 0  # no ata.signals entry-points registered in this package
        assert reg.errors() == {}  # no errors either

    def test_get_registry_returns_registry(self):
        reg = get_registry()
        assert isinstance(reg, SignalRegistry)
