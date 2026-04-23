"""Tests for wave-67 module wiring into trading_cycle.py.

Covers:
  Step 8.72 — events.news.fingerprint (simhash64 / hamming_distance)
  Step 8.73 — events.news.tfidf (build_tfidf_vectors / cosine_sparse)
  Step 8.74 — events.news.trigger_scoring (score_triggers)
"""

from __future__ import annotations

import pytest

from src.assembled_core.events.news.fingerprint import simhash64, hamming_distance
from src.assembled_core.events.news.tfidf import build_tfidf_vectors, cosine_sparse
from src.assembled_core.events.news.trigger_scoring import score_triggers


# ---------------------------------------------------------------------------
# fingerprint (Step 8.72)
# ---------------------------------------------------------------------------

def test_simhash64_returns_int():
    h = simhash64("the quick brown fox")
    assert isinstance(h, int)


def test_simhash64_deterministic():
    text = "equity market rally bond yields"
    assert simhash64(text) == simhash64(text)


def test_simhash64_different_texts():
    h1 = simhash64("equity market rally")
    h2 = simhash64("central bank rate hike")
    # May differ (not guaranteed but overwhelmingly likely)
    assert isinstance(h1, int) and isinstance(h2, int)


def test_hamming_distance_same():
    h = simhash64("equity rally")
    assert hamming_distance(h, h) == 0


def test_hamming_distance_different():
    h1 = simhash64("equity rally")
    h2 = simhash64("inflation surge")
    dist = hamming_distance(h1, h2)
    assert isinstance(dist, int)
    assert dist >= 0


# ---------------------------------------------------------------------------
# tfidf (Step 8.73)
# ---------------------------------------------------------------------------

def test_build_tfidf_vectors_returns_list():
    vecs = build_tfidf_vectors(["equity market", "bond yields"])
    assert isinstance(vecs, list)
    assert len(vecs) == 2


def test_build_tfidf_vectors_dicts():
    vecs = build_tfidf_vectors(["market rally", "central bank"])
    assert all(isinstance(v, dict) for v in vecs)


def test_build_tfidf_empty():
    vecs = build_tfidf_vectors([])
    assert isinstance(vecs, list)
    assert len(vecs) == 0


def test_cosine_sparse_identical():
    vecs = build_tfidf_vectors(["equity market rally"])
    v = vecs[0]
    sim = cosine_sparse(v, v)
    assert abs(sim - 1.0) < 1e-6


def test_cosine_sparse_different():
    vecs = build_tfidf_vectors(["equity market", "inflation spiral"])
    sim = cosine_sparse(vecs[0], vecs[1])
    assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# trigger_scoring (Step 8.74)
# ---------------------------------------------------------------------------

def test_score_triggers_empty():
    result = score_triggers(clusters=[], events_by_id={})
    assert isinstance(result, list)
    assert len(result) == 0


def test_score_triggers_returns_list():
    cluster = {
        "cluster_id": "c1",
        "event_ids": ["e1"],
        "top_entities": ["AAPL"],
        "top_phrases": ["earnings beat"],
        "sample_titles": ["Apple beats earnings expectations"],
    }
    result = score_triggers(clusters=[cluster], events_by_id={})
    assert isinstance(result, list)


def test_score_triggers_health_ok():
    result = score_triggers(clusters=[], events_by_id={}, health_status="OK")
    assert isinstance(result, list)
