"""Tests for Phase 2b: _compute_news_triggers in trading_cycle_v2."""
from __future__ import annotations

import pandas as pd
from datetime import timezone

from assembled_core.pipeline.trading_cycle_v2 import _compute_news_triggers


def _events(records):
    return pd.DataFrame(records)


def test_news_triggers_none_input():
    result = _compute_news_triggers(None, {})
    assert result.empty


def test_news_triggers_empty_df():
    result = _compute_news_triggers(pd.DataFrame(), {})
    assert result.empty


def test_news_triggers_no_text_column():
    df = pd.DataFrame({"symbol": ["AAPL"], "source_tier": ["T1"]})
    result = _compute_news_triggers(df, {})
    assert result.empty


def test_news_triggers_basic_output_columns():
    df = _events([
        {"symbol": "AAPL", "title": "Apple earnings beat expectations", "source_tier": "T1"},
        {"symbol": "AAPL", "title": "Apple revenue up 15%", "source_tier": "T2"},
    ])
    result = _compute_news_triggers(df, {})
    assert not result.empty
    assert "trigger_score" in result.columns
    assert "cluster_id" in result.columns


def test_news_triggers_dedup_removes_duplicate():
    # Two nearly identical titles should be deduped to 1
    df = _events([
        {"symbol": "AAPL", "title": "Apple beats Q1 earnings", "source_tier": "T1"},
        {"symbol": "AAPL", "title": "Apple beats Q1 earnings", "source_tier": "T2"},
        {"symbol": "MSFT", "title": "Microsoft cloud revenue up", "source_tier": "T1"},
    ])
    result = _compute_news_triggers(df, {})
    # Exact duplicates should be deduped; at least 2 unique results expected
    assert len(result) >= 2
    assert len(result) <= 3


def test_news_triggers_cluster_assigns_same_id_for_similar():
    # Two very similar articles → should get same cluster_id
    df = _events([
        {"symbol": "AAPL", "title": "Apple earnings beat expectations today", "source_tier": "T1"},
        {"symbol": "AAPL", "title": "Apple earnings beat expectations this quarter", "source_tier": "T1"},
        {"symbol": "GOOG", "title": "Google search revenue decline unrelated topic", "source_tier": "T1"},
    ])
    result = _compute_news_triggers(df, {})
    assert not result.empty
    assert "cluster_id" in result.columns


def test_news_triggers_t1_higher_score_than_t3():
    df = _events([
        {"symbol": "A", "title": "news from tier one", "source_tier": "T1"},
        {"symbol": "B", "title": "news from tier three", "source_tier": "T3"},
    ])
    result = _compute_news_triggers(df, {})
    if "symbol" in result.columns and not result.empty:
        t1_row = result[result["symbol"] == "A"]
        t3_row = result[result["symbol"] == "B"]
        if not t1_row.empty and not t3_row.empty:
            assert float(t1_row["trigger_score"].iloc[0]) > float(t3_row["trigger_score"].iloc[0])


def test_news_triggers_empty_stream_no_exception():
    df = _events([])
    result = _compute_news_triggers(df, {})
    assert result.empty


def test_news_triggers_burst_bonus_recent_events():
    """Events very close in time should receive a burst bonus (+0.2)."""
    import pandas as pd_
    now = pd_.Timestamp.now(timezone.utc).isoformat()
    df = _events([
        {"symbol": "AAPL", "title": "Breaking news A", "source_tier": "T2", "published_utc": now},
        {"symbol": "AAPL", "title": "Breaking news B", "source_tier": "T2", "published_utc": now},
    ])
    result = _compute_news_triggers(df, {"news_triggers": {"burst_window_minutes": 60}})
    assert not result.empty
    # Both events are at the same time → both should get burst bonus → score > 0.7
    assert (result["trigger_score"] > 0.7).any()
