"""Tests for Phase 2a: _apply_evidence_gate in trading_cycle_v2."""
from __future__ import annotations

import pandas as pd
import pytest

from assembled_core.pipeline.trading_cycle_v2 import _apply_evidence_gate


def _signals(symbols=("AAPL", "MSFT")):
    return pd.DataFrame({
        "symbol": list(symbols),
        "direction": ["LONG"] * len(symbols),
        "score": [1.0] * len(symbols),
    })


def _news_events(records):
    return pd.DataFrame(records)


def test_evidence_gate_disabled_passthrough():
    signals = _signals()
    result, audit = _apply_evidence_gate(signals, None, {"evidence_gate": {"enabled": False}})
    assert len(result) == len(signals)
    assert audit["enabled"] is False


def test_evidence_gate_no_news_events():
    signals = _signals()
    policy = {"evidence_gate": {"enabled": True}}
    result, audit = _apply_evidence_gate(signals, None, policy)
    assert len(result) == len(signals)
    assert audit.get("reason") == "no_news_events"


def test_evidence_gate_empty_news_df():
    signals = _signals()
    policy = {"evidence_gate": {"enabled": True}}
    result, audit = _apply_evidence_gate(signals, pd.DataFrame(), policy)
    assert len(result) == len(signals)


def test_evidence_gate_strong_evidence_passes():
    signals = _signals(["AAPL"])
    # T1 = tier A — strong evidence
    news = _news_events([
        {"symbol": "AAPL", "source_tier": "T1", "source_id": "reuters"},
        {"symbol": "AAPL", "source_tier": "T1", "source_id": "bloomberg"},
    ])
    policy = {"evidence_gate": {"enabled": True, "require_grade": "B"}}
    result, audit = _apply_evidence_gate(signals, news, policy)
    assert len(result) == 1
    assert audit["filtered_count"] == 0


def test_evidence_gate_weak_evidence_filtered():
    signals = _signals(["AAPL"])
    # Only T3 sources — no tier A or independent tier B
    news = _news_events([
        {"symbol": "AAPL", "source_tier": "T3", "source_id": "blog1"},
    ])
    policy = {"evidence_gate": {"enabled": True, "require_grade": "B"}}
    result, audit = _apply_evidence_gate(signals, news, policy)
    assert len(result) == 0
    assert audit["filtered_count"] == 1
    assert "AAPL" in audit["filtered_symbols"]


def test_evidence_gate_non_news_symbol_passes():
    """Symbol not in news_events passes through regardless of evidence gate."""
    signals = _signals(["AAPL", "GOOG"])
    # Only AAPL has weak news coverage
    news = _news_events([
        {"symbol": "AAPL", "source_tier": "T3", "source_id": "blog1"},
    ])
    policy = {"evidence_gate": {"enabled": True, "require_grade": "B"}}
    result, audit = _apply_evidence_gate(signals, news, policy)
    # GOOG passes (not in news events), AAPL filtered
    assert "GOOG" in result["symbol"].values
    assert "AAPL" not in result["symbol"].values


def test_evidence_gate_missing_columns():
    signals = _signals()
    news = pd.DataFrame({"source_tier": ["T1"]})  # no symbol column
    policy = {"evidence_gate": {"enabled": True}}
    result, audit = _apply_evidence_gate(signals, news, policy)
    assert len(result) == len(signals)
    assert audit.get("reason") == "missing_required_columns"


def test_evidence_gate_empty_signals():
    signals = pd.DataFrame(columns=["symbol", "direction", "score"])
    news = _news_events([{"symbol": "AAPL", "source_tier": "T1", "source_id": "r"}])
    policy = {"evidence_gate": {"enabled": True}}
    result, audit = _apply_evidence_gate(signals, news, policy)
    assert result.empty
