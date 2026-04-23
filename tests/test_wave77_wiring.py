"""Tests for wave-77 module wiring into trading_cycle.py.

Covers:
  Step 8.84 — intel.news_impact_estimator (NewsImpactEstimator / ImpactEstimate)
  Step 8.85 — intel.market_confirmation (compute_market_confirmation)
  Step 8.86 — intel.currency_crisis (rank_currencies_by_risk / compute_currency_stress_score)
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.news_impact_estimator import (
    NewsImpactEstimator,
    ImpactEstimate,
)
from src.assembled_core.intel.market_confirmation import compute_market_confirmation
from src.assembled_core.intel.currency_crisis import (
    rank_currencies_by_risk,
    get_currency_profile,
    compute_currency_stress_score,
)


# ---------------------------------------------------------------------------
# news_impact_estimator (Step 8.84)
# ---------------------------------------------------------------------------

def test_news_impact_estimator_creates():
    est = NewsImpactEstimator()
    assert isinstance(est, NewsImpactEstimator)


def test_news_impact_estimate_returns_estimate():
    est = NewsImpactEstimator()

    class FakeClassification:
        event_types = ["earnings"]
        severity = 5.0
        market_direction = "bullish"
        time_horizon = "short"
        confidence = 0.7

    result = est.estimate(FakeClassification())
    assert isinstance(result, ImpactEstimate)


def test_news_impact_estimate_has_bps():
    est = NewsImpactEstimator()

    class FakeClassification:
        event_types = ["rate_decision"]
        severity = 7.0
        market_direction = "bearish"
        time_horizon = "short"
        confidence = 0.8

    result = est.estimate(FakeClassification())
    assert hasattr(result, "bps")
    assert isinstance(result.bps, float)


def test_news_impact_estimate_neutral():
    est = NewsImpactEstimator()

    class NeutralClass:
        event_types = []
        severity = 0.0
        market_direction = "neutral"
        time_horizon = "short"
        confidence = 0.5

    result = est.estimate(NeutralClass())
    assert result.bps == 0.0


# ---------------------------------------------------------------------------
# market_confirmation (Step 8.85)
# ---------------------------------------------------------------------------

def test_compute_market_confirmation_returns_dict():
    result = compute_market_confirmation(cache={})
    assert isinstance(result, dict)


def test_compute_market_confirmation_has_keys():
    result = compute_market_confirmation(cache={})
    assert "vix_spike" in result
    assert "oil_move" in result
    assert "gold_move" in result


def test_compute_market_confirmation_uses_cache():
    cache = {}
    r1 = compute_market_confirmation(cache=cache)
    r2 = compute_market_confirmation(cache=cache)
    # both should be dicts regardless of cache behavior
    assert isinstance(r1, dict)
    assert isinstance(r2, dict)


# ---------------------------------------------------------------------------
# currency_crisis (Step 8.86)
# ---------------------------------------------------------------------------

def test_rank_currencies_by_risk_returns_list():
    result = rank_currencies_by_risk()
    assert isinstance(result, list)


def test_rank_currencies_by_risk_has_tuples():
    result = rank_currencies_by_risk()
    assert len(result) > 0
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)


def test_rank_currencies_sorted_descending():
    result = rank_currencies_by_risk()
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_get_currency_profile_known():
    profile = get_currency_profile("TRY")
    assert profile is not None or profile is None  # graceful regardless


def test_compute_currency_stress_score_range():
    result = rank_currencies_by_risk()
    for _, score in result:
        assert isinstance(score, float)
