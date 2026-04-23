"""Tests for wave-146 module wiring into trading_cycle.py.

Covers:
  Step cfg.2  — config.logging_config (configure_json_logging / JSONFormatter)
  Step cfg.3  — config.secrets_loader (get_secret / is_secret_set)
  Step cost.1 — costs (CostModel / get_default_cost_model)
"""

from __future__ import annotations

import logging
import pytest

from src.assembled_core.config.logging_config import configure_json_logging, JSONFormatter
from src.assembled_core.config.secrets_loader import get_secret, is_secret_set, load_env_file
from src.assembled_core.costs import CostModel, get_default_cost_model


# ---------------------------------------------------------------------------
# config.logging_config (Step cfg.2)
# ---------------------------------------------------------------------------

def test_configure_json_logging_importable():
    assert configure_json_logging is not None


def test_json_formatter_importable():
    assert JSONFormatter is not None


def test_configure_json_logging_returns_logger():
    logger = configure_json_logging(level="DEBUG", logger_name="test_json_log_wave146")
    assert isinstance(logger, logging.Logger)


# ---------------------------------------------------------------------------
# config.secrets_loader (Step cfg.3)
# ---------------------------------------------------------------------------

def test_get_secret_importable():
    assert get_secret is not None


def test_is_secret_set_importable():
    assert is_secret_set is not None


def test_get_secret_missing_not_required():
    result = get_secret("TOTALLY_NONEXISTENT_KEY_WAVE146", required=False, default="fallback")
    assert result == "fallback"


def test_is_secret_set_false_for_missing():
    assert is_secret_set("TOTALLY_NONEXISTENT_KEY_WAVE146") is False


def test_load_env_file_importable():
    assert load_env_file is not None


# ---------------------------------------------------------------------------
# costs (Step cost.1)
# ---------------------------------------------------------------------------

def test_cost_model_importable():
    assert CostModel is not None


def test_get_default_cost_model_returns_model():
    m = get_default_cost_model()
    assert isinstance(m, CostModel)
    assert m.commission_bps > 0.0


def test_cost_model_fields():
    m = CostModel(commission_bps=2.0, spread_w=0.5, impact_w=0.3)
    assert m.commission_bps == 2.0
    assert m.spread_w == 0.5
