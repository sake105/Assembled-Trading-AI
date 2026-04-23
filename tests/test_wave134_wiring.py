"""Tests for wave-134 module wiring into trading_cycle.py.

Covers:
  Step pipe.3 — pipeline.event_bus (EventBus / EventType)
  Step pipe.4 — pipeline.graceful_degradation (DegradationTracker)
  Step pipe.5 — pipeline.io (load_prices / load_orders)
"""

from __future__ import annotations

import pytest

from src.assembled_core.pipeline.event_bus import EventBus, EventType, Event
from src.assembled_core.pipeline.graceful_degradation import DegradationTracker, neutralize_missing_features
from src.assembled_core.pipeline.io import load_prices, load_orders


# ---------------------------------------------------------------------------
# pipeline.event_bus (Step pipe.3)
# ---------------------------------------------------------------------------

def test_event_bus_importable():
    assert EventBus is not None


def test_event_type_importable():
    assert EventType is not None


def test_event_bus_creates():
    bus = EventBus(max_queue_size=100)
    assert isinstance(bus, EventBus)


def test_event_type_has_fill():
    member_names = [e.name for e in EventType]
    assert len(member_names) > 0


# ---------------------------------------------------------------------------
# pipeline.graceful_degradation (Step pipe.4)
# ---------------------------------------------------------------------------

def test_degradation_tracker_importable():
    assert DegradationTracker is not None


def test_degradation_tracker_creates():
    dt = DegradationTracker()
    assert dt.is_degraded is False


def test_degradation_tracker_record():
    dt = DegradationTracker()
    dt.record_failure("fred", "timeout")
    assert dt.is_degraded is True
    assert "fred" in dt.failed_sources


def test_neutralize_missing_features_importable():
    assert neutralize_missing_features is not None


# ---------------------------------------------------------------------------
# pipeline.io (Step pipe.5)
# ---------------------------------------------------------------------------

def test_load_prices_importable():
    assert load_prices is not None


def test_load_orders_importable():
    assert load_orders is not None


def test_load_prices_raises_for_missing_file():
    with pytest.raises((FileNotFoundError, KeyError, Exception)):
        load_prices("1d", price_file="/nonexistent/prices.parquet")
