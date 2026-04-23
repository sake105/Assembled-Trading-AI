"""Tests for wave-83 module wiring into trading_cycle.py.

Covers:
  Step 8.102 — intel.news_source_voting (vote_direction / VoteResult)
  Step 8.103 — intel.news_ticker_velocity (TickerVelocityTracker / TickerSignal)
  Step 8.104 — intel.news_trade_attribution (NewsTradeAttributor)
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.news_source_voting import vote_direction, vote_event_type, VoteResult
from src.assembled_core.intel.news_ticker_velocity import TickerVelocityTracker, TickerSignal
from src.assembled_core.intel.news_trade_attribution import NewsTradeAttributor, NewsLink


# ---------------------------------------------------------------------------
# news_source_voting (Step 8.102)
# ---------------------------------------------------------------------------

def test_vote_direction_empty_returns_vote_result():
    result = vote_direction([])
    assert isinstance(result, VoteResult)


def test_vote_direction_empty_winner_is_str():
    result = vote_direction([])
    assert isinstance(result.winner, str)


def test_vote_direction_empty_margin_zero():
    result = vote_direction([])
    assert result.margin == 0.0


def test_vote_event_type_empty_returns_vote_result():
    result = vote_event_type([])
    assert isinstance(result, VoteResult)


def test_vote_result_importable():
    assert VoteResult is not None


# ---------------------------------------------------------------------------
# news_ticker_velocity (Step 8.103)
# ---------------------------------------------------------------------------

def test_ticker_velocity_tracker_creates():
    tvt = TickerVelocityTracker()
    assert isinstance(tvt, TickerVelocityTracker)


def test_ticker_velocity_tracker_empty_buffers():
    tvt = TickerVelocityTracker()
    assert len(tvt._buffers) == 0


def test_ticker_velocity_tracker_update_empty():
    tvt = TickerVelocityTracker()
    signals = tvt.update([])
    assert isinstance(signals, list)
    assert len(signals) == 0


def test_ticker_signal_importable():
    assert TickerSignal is not None


# ---------------------------------------------------------------------------
# news_trade_attribution (Step 8.104)
# ---------------------------------------------------------------------------

def test_news_trade_attributor_creates():
    nta = NewsTradeAttributor()
    assert isinstance(nta, NewsTradeAttributor)


def test_news_trade_attributor_default_windows():
    nta = NewsTradeAttributor()
    assert nta.pre > 0
    assert nta.post > 0


def test_news_trade_attributor_link_empty_events():
    import pandas as pd
    nta = NewsTradeAttributor()
    result = nta.link_trade_to_events(
        trade={"symbol": "AAPL", "opened_at": "2024-01-15T10:00:00Z"},
        news_events=pd.DataFrame(),
    )
    assert isinstance(result, list)
    assert len(result) == 0


def test_news_link_importable():
    assert NewsLink is not None
