"""Tests for wave-78 module wiring into trading_cycle.py.

Covers:
  Step 8.87 — intel.news_language (detect_language / is_english)
  Step 8.88 — intel.news_macro_calendar (MacroCalendar)
  Step 8.89 — intel.central_bank_divergence (compute_policy_divergence_matrix)
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from src.assembled_core.intel.news_language import detect_language, is_english
from src.assembled_core.intel.news_macro_calendar import MacroCalendar, MacroEvent, Proximity
from src.assembled_core.intel.central_bank_divergence import (
    compute_policy_divergence_matrix,
    get_most_divergent_pair,
    get_policy_stance,
    detect_synchronized_tightening,
)


# ---------------------------------------------------------------------------
# news_language (Step 8.87)
# ---------------------------------------------------------------------------

def test_detect_language_returns_str():
    lang = detect_language("Federal Reserve raises interest rates")
    assert isinstance(lang, str)


def test_detect_language_english():
    lang = detect_language("The stock market rallied on positive economic data")
    assert lang in ("en", "english")


def test_is_english_true():
    assert is_english("Federal Reserve raises rates") is True


def test_is_english_short():
    # Very short text — may be uncertain
    result = is_english("ok")
    assert isinstance(result, bool)


def test_detect_language_non_english():
    # German text
    lang = detect_language("Die Aktien steigen aufgrund positiver Wirtschaftsdaten")
    assert isinstance(lang, str)


# ---------------------------------------------------------------------------
# news_macro_calendar (Step 8.88)
# ---------------------------------------------------------------------------

def test_macro_calendar_creates():
    cal = MacroCalendar()
    assert isinstance(cal, MacroCalendar)


def test_macro_calendar_empty():
    cal = MacroCalendar()
    assert len(cal._events) == 0


def test_macro_calendar_add_event():
    cal = MacroCalendar()
    event = MacroEvent(
        event_id="fomc_2024_06",
        kind="central_bank",
        ts=datetime(2024, 6, 12, 14, 0, tzinfo=timezone.utc),
        importance=3,
    )
    cal.add(event)
    assert len(cal._events) == 1


def test_macro_event_creates():
    event = MacroEvent(
        event_id="nfp_2024_07",
        kind="labor_market",
        ts=datetime(2024, 7, 5, 12, 30, tzinfo=timezone.utc),
        importance=3,
    )
    assert event.event_id == "nfp_2024_07"
    assert event.kind == "labor_market"


# ---------------------------------------------------------------------------
# central_bank_divergence (Step 8.89)
# ---------------------------------------------------------------------------

def test_compute_policy_divergence_matrix_returns_dict():
    matrix = compute_policy_divergence_matrix()
    assert isinstance(matrix, dict)


def test_compute_policy_divergence_matrix_has_pairs():
    matrix = compute_policy_divergence_matrix()
    assert len(matrix) > 0
    for key in matrix.keys():
        assert isinstance(key, tuple)
        assert len(key) == 2


def test_get_most_divergent_pair_returns_tuple():
    result = get_most_divergent_pair()
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_get_policy_stance_returns_str():
    # "FED" is a known central bank
    stance = get_policy_stance("FED")
    assert isinstance(stance, str)


def test_detect_synchronized_tightening_returns_bool():
    result = detect_synchronized_tightening()
    assert isinstance(result, bool)
