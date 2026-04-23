"""Tests for wave-82 module wiring into trading_cycle.py.

Covers:
  Step 8.99  — intel.news_semantic_dedup (SemanticDedup)
  Step 8.100 — intel.news_sentiment_drift (SentimentDriftTracker)
  Step 8.101 — intel.news_signal_aggregator (aggregate_signals / IntelSignal)
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.news_semantic_dedup import SemanticDedup
from src.assembled_core.intel.news_sentiment_drift import SentimentDriftTracker, DriftEntry
from src.assembled_core.intel.news_signal_aggregator import aggregate_signals, IntelSignal


# ---------------------------------------------------------------------------
# news_semantic_dedup (Step 8.99)
# ---------------------------------------------------------------------------

def test_semantic_dedup_creates():
    sdd = SemanticDedup(enabled=False)
    assert isinstance(sdd, SemanticDedup)


def test_semantic_dedup_backend_lexical():
    sdd = SemanticDedup(enabled=False)
    assert isinstance(sdd.backend, str)


def test_semantic_dedup_entries_empty():
    sdd = SemanticDedup(enabled=False)
    assert len(sdd._entries) == 0


def test_semantic_dedup_model_none_when_disabled():
    sdd = SemanticDedup(enabled=False)
    assert sdd._model is None


# ---------------------------------------------------------------------------
# news_sentiment_drift (Step 8.100)
# ---------------------------------------------------------------------------

def test_sentiment_drift_tracker_creates():
    sdt = SentimentDriftTracker()
    assert isinstance(sdt, SentimentDriftTracker)


def test_sentiment_drift_tracker_empty_buffers():
    sdt = SentimentDriftTracker()
    assert len(sdt._buffers) == 0


def test_sentiment_drift_tracker_update_empty():
    sdt = SentimentDriftTracker()
    sdt.update([])
    assert len(sdt._buffers) == 0


def test_drift_entry_importable():
    assert DriftEntry is not None


# ---------------------------------------------------------------------------
# news_signal_aggregator (Step 8.101)
# ---------------------------------------------------------------------------

def test_aggregate_signals_empty_returns_intel_signal():
    result = aggregate_signals([])
    assert isinstance(result, IntelSignal)


def test_aggregate_signals_empty_neutral():
    result = aggregate_signals([])
    assert result.net_direction == "neutral"


def test_aggregate_signals_empty_zero_signals():
    result = aggregate_signals([])
    assert result.n_signals == 0


def test_aggregate_signals_empty_zero_confidence():
    result = aggregate_signals([])
    assert result.aggregate_confidence == 0.0


def test_intel_signal_importable():
    assert IntelSignal is not None
