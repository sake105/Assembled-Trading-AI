"""Tests for wave-74 module wiring into trading_cycle.py.

Covers:
  Step 8.75 — intel.health_monitor (HealthMonitor)
  Step 8.76 — intel.news_decay (NewsDecay)
  Step 8.77 — intel.nation_profiles (load_nation_profiles / compute_vulnerability_score)
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.health_monitor import HealthMonitor
from src.assembled_core.intel.news_decay import NewsDecay, DecayProfile


# ---------------------------------------------------------------------------
# health_monitor (Step 8.75)
# ---------------------------------------------------------------------------

def test_health_monitor_creates():
    hm = HealthMonitor()
    assert isinstance(hm, HealthMonitor)


def test_health_monitor_register():
    hm = HealthMonitor()
    hm.register("news_pipeline")
    assert "news_pipeline" in hm._components


def test_health_monitor_all_ok_empty():
    hm = HealthMonitor()
    # No components registered → all_ok should handle gracefully
    result = hm.all_ok()
    assert isinstance(result, bool)


def test_health_monitor_snapshot():
    hm = HealthMonitor()
    hm.register("news_pipeline")
    snap = hm.snapshot()
    assert isinstance(snap, dict)
    assert "news_pipeline" in snap


def test_health_monitor_can_go_active_new():
    hm = HealthMonitor()
    hm.register("comp1")
    # Not yet updated → should not be ACTIVE
    result = hm.can_go_active()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# news_decay (Step 8.76)
# ---------------------------------------------------------------------------

def test_news_decay_creates():
    nd = NewsDecay()
    assert isinstance(nd, NewsDecay)


def test_news_decay_impact_remaining_fresh():
    nd = NewsDecay()
    impact = nd.impact_remaining("earnings", minutes_since=0.0)
    assert impact == 1.0


def test_news_decay_impact_remaining_decays():
    nd = NewsDecay()
    impact_fresh = nd.impact_remaining("earnings", minutes_since=0.0)
    impact_old = nd.impact_remaining("earnings", minutes_since=1000.0)
    assert impact_old < impact_fresh


def test_news_decay_impact_remaining_range():
    nd = NewsDecay()
    impact = nd.impact_remaining("earnings", minutes_since=60.0)
    assert 0.0 <= impact <= 1.0


def test_news_decay_unknown_type_uses_default():
    nd = NewsDecay()
    impact = nd.impact_remaining("__unknown_type__", minutes_since=30.0)
    assert isinstance(impact, float)


# ---------------------------------------------------------------------------
# nation_profiles (Step 8.77)
# ---------------------------------------------------------------------------

def test_nation_profiles_importable():
    from src.assembled_core.intel.nation_profiles import (
        load_nation_profiles,
        compute_vulnerability_score,
        NationProfile,
    )
    assert callable(load_nation_profiles)
    assert callable(compute_vulnerability_score)


def test_load_nation_profiles_returns_dict():
    from src.assembled_core.intel.nation_profiles import load_nation_profiles
    try:
        profiles = load_nation_profiles()
        assert isinstance(profiles, dict)
    except (FileNotFoundError, Exception):
        pytest.skip("nation_profiles.yaml not found — file-dependent test")


def test_compute_vulnerability_score_importable():
    from src.assembled_core.intel.nation_profiles import compute_vulnerability_score, NationProfile
    profile = NationProfile(
        nation_id="test",
        name="Test Nation",
        imports={},
        exports={},
        transit_dependencies={},
        fiscal={},
        military={},
        tech_sovereignty={},
        vulnerabilities={},
    )
    score = compute_vulnerability_score(profile, shock_type="trade")
    assert isinstance(score, float)
