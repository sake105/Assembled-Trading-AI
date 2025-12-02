"""Tests for execution.kill_switch module."""
from __future__ import annotations

import os
from unittest.mock import patch

import pandas as pd
import pytest

pytestmark = pytest.mark.phase10

from src.assembled_core.execution.kill_switch import (
    guard_orders_with_kill_switch,
    is_kill_switch_engaged,
)


@pytest.fixture
def sample_orders() -> pd.DataFrame:
    """Create sample orders DataFrame."""
    return pd.DataFrame({
        "symbol": ["AAPL", "GOOGL"],
        "side": ["BUY", "SELL"],
        "qty": [100, 50],
        "price": [150.0, 2500.0],
    })


class TestKillSwitch:
    """Tests for kill switch functions."""

    def test_kill_switch_not_engaged_by_default(self):
        """Test that kill switch is not engaged when env variable is not set."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove env var if it exists
            os.environ.pop("ASSEMBLED_KILL_SWITCH", None)
            assert is_kill_switch_engaged() is False

    def test_kill_switch_engaged_when_env_set_to_1(self):
        """Test that kill switch is engaged when env variable is set to '1'."""
        with patch.dict(os.environ, {"ASSEMBLED_KILL_SWITCH": "1"}, clear=False):
            assert is_kill_switch_engaged() is True

    def test_kill_switch_engaged_when_env_set_to_true(self):
        """Test that kill switch is engaged when env variable is set to 'true'."""
        for value in ["true", "True", "TRUE"]:
            with patch.dict(os.environ, {"ASSEMBLED_KILL_SWITCH": value}, clear=False):
                assert is_kill_switch_engaged() is True

    def test_kill_switch_engaged_case_insensitive(self):
        """Test that kill switch check is case-insensitive."""
        for value in ["YES", "yes", "Yes", "ON", "on", "On"]:
            with patch.dict(os.environ, {"ASSEMBLED_KILL_SWITCH": value}, clear=False):
                assert is_kill_switch_engaged() is True

    def test_kill_switch_not_engaged_with_other_values(self):
        """Test that kill switch is not engaged with other values."""
        for value in ["0", "false", "False", "no", "off", "disabled", ""]:
            with patch.dict(os.environ, {"ASSEMBLED_KILL_SWITCH": value}, clear=False):
                assert is_kill_switch_engaged() is False

    def test_guard_orders_blocks_all_when_engaged(self, sample_orders):
        """Test that guard_orders_with_kill_switch blocks all orders when engaged."""
        with patch.dict(os.environ, {"ASSEMBLED_KILL_SWITCH": "1"}, clear=False):
            filtered = guard_orders_with_kill_switch(sample_orders)
            
            assert len(filtered) == 0
            assert list(filtered.columns) == list(sample_orders.columns)  # Same columns

    def test_guard_orders_passes_orders_when_not_engaged(self, sample_orders):
        """Test that guard_orders_with_kill_switch passes orders when not engaged."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ASSEMBLED_KILL_SWITCH", None)
            filtered = guard_orders_with_kill_switch(sample_orders)
            
            assert len(filtered) == len(sample_orders)
            assert filtered.equals(sample_orders)

    def test_guard_orders_empty_orders(self):
        """Test that guard_orders handles empty orders DataFrame."""
        empty_orders = pd.DataFrame(columns=["symbol", "side", "qty"])
        
        with patch.dict(os.environ, {"ASSEMBLED_KILL_SWITCH": "1"}, clear=False):
            filtered = guard_orders_with_kill_switch(empty_orders)
            
            assert len(filtered) == 0
            assert list(filtered.columns) == list(empty_orders.columns)

    def test_guard_orders_empty_when_not_engaged(self):
        """Test that empty orders remain empty when kill switch is not engaged."""
        empty_orders = pd.DataFrame(columns=["symbol", "side", "qty"])
        
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ASSEMBLED_KILL_SWITCH", None)
            filtered = guard_orders_with_kill_switch(empty_orders)
            
            assert len(filtered) == 0

