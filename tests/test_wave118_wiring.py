"""Tests for wave-118 module wiring into trading_cycle.py.

Covers:
  Step 5.61  — data.streaming.ws_client (WSConfig / WebSocketClient)
  Step 2.102 — data.universe (get_universe_members)
  Step 8.50  — events.crisis_alpha.baskets (get_baskets / get_basket_symbols)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.data.streaming.ws_client import WSConfig, WebSocketClient
from src.assembled_core.data.universe import get_universe_members
from src.assembled_core.events.crisis_alpha.baskets import get_baskets, get_basket_symbols


# ---------------------------------------------------------------------------
# data.streaming.ws_client (Step 5.61)
# ---------------------------------------------------------------------------

def test_ws_config_creates():
    cfg = WSConfig()
    assert isinstance(cfg, WSConfig)


def test_ws_config_defaults():
    cfg = WSConfig()
    assert cfg.reconnect_delay == 5.0
    assert cfg.max_reconnect_attempts == 10


def test_ws_config_default_url():
    cfg = WSConfig()
    assert "alpaca" in cfg.url or len(cfg.url) > 0


def test_web_socket_client_importable():
    assert WebSocketClient is not None


# ---------------------------------------------------------------------------
# data.universe (Step 2.102)
# ---------------------------------------------------------------------------

def test_get_universe_members_importable():
    assert get_universe_members is not None


# ---------------------------------------------------------------------------
# events.crisis_alpha.baskets (Step 8.50)
# ---------------------------------------------------------------------------

def test_get_baskets_returns_list():
    result = get_baskets()
    assert isinstance(result, list)


def test_get_baskets_not_empty():
    result = get_baskets()
    assert len(result) > 0


def test_get_basket_symbols_returns_list():
    result = get_basket_symbols()
    assert isinstance(result, list)


def test_get_basket_symbols_not_empty():
    result = get_basket_symbols()
    assert len(result) > 0


def test_get_baskets_with_none_policy():
    result = get_baskets(policy=None)
    assert isinstance(result, list)
