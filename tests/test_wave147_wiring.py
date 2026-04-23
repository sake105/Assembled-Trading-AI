"""Tests for wave-147 module wiring into trading_cycle.py.

Covers:
  Step core.1 — errors (AssembledError / KillSwitchActive / PITViolation)
  Step core.2 — ema_config (EmaConfig / get_default_ema_config)
  Step core.3 — logging_utils (get_logger)
"""

from __future__ import annotations

import logging
import pytest

from src.assembled_core.errors import (
    AssembledError,
    KillSwitchActive,
    PITViolation,
    RecoverableError,
    FatalTradingError,
)
from src.assembled_core.ema_config import EmaConfig, get_default_ema_config
from src.assembled_core.logging_utils import get_logger


# ---------------------------------------------------------------------------
# errors (Step core.1)
# ---------------------------------------------------------------------------

def test_assembled_error_importable():
    assert AssembledError is not None


def test_kill_switch_active_is_exception():
    assert issubclass(KillSwitchActive, Exception)


def test_pit_violation_is_exception():
    assert issubclass(PITViolation, Exception)


def test_recoverable_error_is_assembled():
    assert issubclass(RecoverableError, AssembledError)


def test_fatal_trading_error_is_assembled():
    assert issubclass(FatalTradingError, AssembledError)


def test_errors_can_be_raised():
    with pytest.raises(KillSwitchActive):
        raise KillSwitchActive("kill switch triggered")


# ---------------------------------------------------------------------------
# ema_config (Step core.2)
# ---------------------------------------------------------------------------

def test_ema_config_importable():
    assert EmaConfig is not None


def test_ema_config_creates():
    cfg = EmaConfig(fast=10, slow=30)
    assert cfg.fast == 10
    assert cfg.slow == 30


def test_get_default_ema_config_1d():
    cfg = get_default_ema_config("1d")
    assert isinstance(cfg, EmaConfig)
    assert cfg.fast < cfg.slow


def test_get_default_ema_config_5min():
    cfg = get_default_ema_config("5min")
    assert isinstance(cfg, EmaConfig)


# ---------------------------------------------------------------------------
# logging_utils (Step core.3)
# ---------------------------------------------------------------------------

def test_get_logger_importable():
    assert get_logger is not None


def test_get_logger_returns_logger():
    logger = get_logger("test_wave147")
    assert isinstance(logger, logging.Logger)
