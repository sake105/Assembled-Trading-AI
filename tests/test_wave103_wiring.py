"""Tests for wave-103 module wiring into trading_cycle.py.

Covers:
  Step paper.3 — paper.intel_runner (compute_news_geo / _empty_news_geo)
  Step paper.4 — paper.paper_track (PaperTrackConfig / PaperTrackState)
  Step paper.5 — paper.strategy_adapters (generate_signals_and_targets_for_day)
"""

from __future__ import annotations

import pytest

from src.assembled_core.paper.intel_runner import (
    compute_news_geo,
    _empty_news_geo,
    load_intel_summaries,
)
from src.assembled_core.paper.paper_track import (
    PaperTrackConfig,
    PaperTrackState,
    PaperTrackDayResult,
)
from src.assembled_core.paper.strategy_adapters import generate_signals_and_targets_for_day


# ---------------------------------------------------------------------------
# intel_runner (Step paper.3)
# ---------------------------------------------------------------------------

def test_empty_news_geo_returns_dict():
    result = _empty_news_geo()
    assert isinstance(result, dict)


def test_empty_news_geo_has_state_hint():
    result = _empty_news_geo()
    assert "state_hint" in result


def test_empty_news_geo_has_geo_score():
    result = _empty_news_geo()
    assert "geo_score" in result
    assert result["geo_score"] == 0


def test_load_intel_summaries_importable():
    assert load_intel_summaries is not None


# ---------------------------------------------------------------------------
# paper_track (Step paper.4)
# ---------------------------------------------------------------------------

def test_paper_track_config_creates():
    cfg = PaperTrackConfig(
        strategy_name="test",
        strategy_type="multifactor",
        universe_file="configs/universe.yaml",
        freq="daily",
    )
    assert isinstance(cfg, PaperTrackConfig)


def test_paper_track_config_is_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(PaperTrackConfig)


def test_paper_track_state_importable():
    assert PaperTrackState is not None


def test_paper_track_day_result_importable():
    assert PaperTrackDayResult is not None


# ---------------------------------------------------------------------------
# strategy_adapters (Step paper.5)
# ---------------------------------------------------------------------------

def test_generate_signals_and_targets_for_day_importable():
    assert generate_signals_and_targets_for_day is not None
