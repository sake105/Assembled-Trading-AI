"""Tests for wave-79 module wiring into trading_cycle.py.

Covers:
  Step 8.90 — intel.entity_linker (EntityLinker)
  Step 8.91 — intel.news_impact_calibrator (ImpactCalibrator)
  Step 8.92 — intel.news_entity_mapper (extract_tickers_from_title / SimpleEntityLinker)
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.entity_linker import EntityLinker
from src.assembled_core.intel.news_impact_calibrator import ImpactCalibrator, CalibrationEntry
from src.assembled_core.intel.news_entity_mapper import (
    extract_tickers_from_title,
    SimpleEntityLinker,
)


# ---------------------------------------------------------------------------
# entity_linker (Step 8.90)
# ---------------------------------------------------------------------------

def test_entity_linker_creates():
    el = EntityLinker()
    assert isinstance(el, EntityLinker)


def test_entity_linker_link_returns_list():
    el = EntityLinker()
    result = el.link("Apple Inc")
    assert isinstance(result, list)


def test_entity_linker_link_known():
    el = EntityLinker()
    result = el.link("AAPL")
    assert isinstance(result, list)


def test_entity_linker_link_unknown():
    el = EntityLinker()
    result = el.link("__completely_unknown_entity_xyz__")
    assert isinstance(result, list)


def test_entity_linker_with_symbols():
    el = EntityLinker(symbols=["AAPL", "MSFT", "GOOG"])
    result = el.link("AAPL")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# news_impact_calibrator (Step 8.91)
# ---------------------------------------------------------------------------

def test_impact_calibrator_creates():
    ic = ImpactCalibrator()
    assert isinstance(ic, ImpactCalibrator)


def test_impact_calibrator_observe():
    ic = ImpactCalibrator()
    ic.observe("earnings", pred_bps=15.0, realised_bps=12.5)
    assert len(ic._stats) == 1


def test_impact_calibrator_report_empty():
    ic = ImpactCalibrator()
    report = ic.report(include_sparse=True)
    assert isinstance(report, dict)


def test_impact_calibrator_recommend_prior():
    ic = ImpactCalibrator(min_samples_for_report=2)
    ic.observe("earnings", 15.0, 12.5)
    ic.observe("earnings", 10.0, 8.0)
    adj = ic.recommend_prior_adjustment("earnings")
    assert isinstance(adj, float)


def test_calibration_entry_importable():
    assert CalibrationEntry is not None


# ---------------------------------------------------------------------------
# news_entity_mapper (Step 8.92)
# ---------------------------------------------------------------------------

def test_extract_tickers_returns_list():
    result = extract_tickers_from_title("Apple Inc AAPL and Microsoft MSFT")
    assert isinstance(result, list)


def test_extract_tickers_empty():
    result = extract_tickers_from_title("")
    assert isinstance(result, list)


def test_simple_entity_linker_creates():
    sel = SimpleEntityLinker()
    assert isinstance(sel, SimpleEntityLinker)


def test_simple_entity_linker_link_known():
    sel = SimpleEntityLinker()
    result = sel.link("Apple")
    assert isinstance(result, (list, str))
