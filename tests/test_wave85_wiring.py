"""Tests for wave-85 module wiring into trading_cycle.py.

Covers:
  Step 8.108 — intel.sanctions_model (get_sanction_package / HISTORICAL_SANCTIONS)
  Step 8.109 — intel.sector_news_overlay (SectorNewsOverlay)
  Step 8.110 — intel.shipping_lanes (LANES_DATABASE / get_lane)
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.sanctions_model import (
    get_sanction_package,
    HISTORICAL_SANCTIONS,
    compute_sanction_cascade,
)
from src.assembled_core.intel.sector_news_overlay import SectorNewsOverlay
from src.assembled_core.intel.shipping_lanes import LANES_DATABASE, get_lane, get_lanes_through_chokepoint


# ---------------------------------------------------------------------------
# sanctions_model (Step 8.108)
# ---------------------------------------------------------------------------

def test_historical_sanctions_not_empty():
    assert len(HISTORICAL_SANCTIONS) > 0


def test_get_sanction_package_known():
    pkg = get_sanction_package("RUSSIA_2022_FULL")
    assert pkg is not None


def test_get_sanction_package_unknown_returns_none():
    pkg = get_sanction_package("DOES_NOT_EXIST_XYZ")
    assert pkg is None


def test_historical_sanctions_keys_are_strings():
    for key in HISTORICAL_SANCTIONS:
        assert isinstance(key, str)


# ---------------------------------------------------------------------------
# sector_news_overlay (Step 8.109)
# ---------------------------------------------------------------------------

def test_sector_news_overlay_creates():
    sno = SectorNewsOverlay()
    assert isinstance(sno, SectorNewsOverlay)


def test_sector_news_overlay_compute_empty():
    sno = SectorNewsOverlay()
    result = sno.compute(clusters=[])
    assert isinstance(result, dict)


def test_sector_news_overlay_empty_returns_empty_dict():
    sno = SectorNewsOverlay()
    result = sno.compute(clusters=[], event_store=None)
    assert len(result) == 0


def test_sector_news_overlay_decay_param():
    sno = SectorNewsOverlay(decay_hours=6.0)
    assert sno._decay_hours == 6.0


# ---------------------------------------------------------------------------
# shipping_lanes (Step 8.110)
# ---------------------------------------------------------------------------

def test_lanes_database_not_empty():
    assert len(LANES_DATABASE) > 0


def test_get_lane_returns_known_lane():
    lane_id = next(iter(LANES_DATABASE))
    lane = get_lane(lane_id)
    assert lane is not None


def test_get_lane_unknown_returns_none():
    lane = get_lane("NONEXISTENT_LANE_XYZ")
    assert lane is None


def test_get_lanes_through_chokepoint_returns_list():
    result = get_lanes_through_chokepoint("HORMUZ")
    assert isinstance(result, list)
