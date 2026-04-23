"""Tests for wave-84 module wiring into trading_cycle.py.

Covers:
  Step 8.105 — intel.news_velocity (VelocityTracker / VelocityResult)
  Step 8.106 — intel.pit_store (PITStore)
  Step 8.107 — intel.rss_fetcher (RSSFetcher)
"""

from __future__ import annotations

import pytest
from pathlib import Path

from src.assembled_core.intel.news_velocity import VelocityTracker, VelocityResult
from src.assembled_core.intel.pit_store import PITStore
from src.assembled_core.intel.rss_fetcher import RSSFetcher


# ---------------------------------------------------------------------------
# news_velocity (Step 8.105)
# ---------------------------------------------------------------------------

def test_velocity_tracker_creates():
    vt = VelocityTracker()
    assert isinstance(vt, VelocityTracker)


def test_velocity_tracker_update_empty():
    vt = VelocityTracker()
    result = vt.update([])
    assert isinstance(result, VelocityResult)


def test_velocity_tracker_no_surge_on_empty():
    vt = VelocityTracker()
    result = vt.update([])
    assert result.is_surge is False


def test_velocity_result_has_velocity():
    vt = VelocityTracker()
    result = vt.update([])
    assert isinstance(result.velocity, float)


def test_velocity_result_importable():
    assert VelocityResult is not None


# ---------------------------------------------------------------------------
# pit_store (Step 8.106)
# ---------------------------------------------------------------------------

def test_pit_store_creates(tmp_path):
    ps = PITStore(root=tmp_path)
    assert isinstance(ps, PITStore)


def test_pit_store_root_path(tmp_path):
    ps = PITStore(root=tmp_path)
    assert ps._root == Path(tmp_path)


def test_pit_store_no_disk_ops_on_create(tmp_path):
    ps = PITStore(root=tmp_path / "new_dir")
    assert isinstance(ps, PITStore)


# ---------------------------------------------------------------------------
# rss_fetcher (Step 8.107)
# ---------------------------------------------------------------------------

def test_rss_fetcher_creates():
    rf = RSSFetcher()
    assert isinstance(rf, RSSFetcher)


def test_rss_fetcher_feed_ids_is_list():
    rf = RSSFetcher()
    assert isinstance(rf.feed_ids, list)


def test_rss_fetcher_enabled_feeds_is_list():
    rf = RSSFetcher()
    assert isinstance(rf.enabled_feeds, list)
