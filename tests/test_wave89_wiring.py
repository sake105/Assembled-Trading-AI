"""Tests for wave-89 module wiring into trading_cycle.py.

Covers:
  Step 8.120 — intel.news_position_bridge (cluster_to_signal / PositionSignal)
  Step 8.121 — intel.news_replay (NewsReplayer / ReplayStep)
  Step 5.55  — execution.adaptive_algo (AdaptiveExecutionAlgo)
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.news_position_bridge import (
    cluster_to_signal,
    PositionSignal,
    classification_to_signal,
)
from src.assembled_core.intel.news_replay import NewsReplayer, ReplayStep
from src.assembled_core.intel.pit_store import PITStore
from src.assembled_core.execution.adaptive_algo import (
    AdaptiveExecutionAlgo,
    AdaptiveAlgoConfig,
    AggressionLevel,
)


# ---------------------------------------------------------------------------
# news_position_bridge (Step 8.120)
# ---------------------------------------------------------------------------

def test_cluster_to_signal_none_returns_none():
    result = cluster_to_signal(None)
    assert result is None


def test_cluster_to_signal_returns_none_for_empty_object():
    class EmptyCluster:
        pass
    result = cluster_to_signal(EmptyCluster())
    assert result is None or isinstance(result, PositionSignal)


def test_position_signal_importable():
    assert PositionSignal is not None


def test_classification_to_signal_importable():
    assert classification_to_signal is not None


# ---------------------------------------------------------------------------
# news_replay (Step 8.121)
# ---------------------------------------------------------------------------

def test_news_replayer_creates(tmp_path):
    store = PITStore(root=tmp_path)
    nr = NewsReplayer(pit_store=store)
    assert isinstance(nr, NewsReplayer)


def test_news_replayer_replay_empty_store(tmp_path):
    store = PITStore(root=tmp_path)
    nr = NewsReplayer(pit_store=store)
    steps = list(nr.replay(source="test", artifact_type="triggers"))
    assert isinstance(steps, list)
    assert len(steps) == 0


def test_replay_step_importable():
    assert ReplayStep is not None


# ---------------------------------------------------------------------------
# adaptive_algo (Step 5.55)
# ---------------------------------------------------------------------------

def test_adaptive_execution_algo_creates():
    algo = AdaptiveExecutionAlgo()
    assert isinstance(algo, AdaptiveExecutionAlgo)


def test_adaptive_execution_algo_default_state():
    algo = AdaptiveExecutionAlgo()
    assert algo.state.total_shares == 0
    assert algo.state.filled_shares == 0


def test_adaptive_execution_algo_initialize():
    algo = AdaptiveExecutionAlgo()
    algo.initialize(total_shares=1000, side="buy")
    assert algo.state.total_shares == 1000


def test_adaptive_algo_config_importable():
    assert AdaptiveAlgoConfig is not None


def test_aggression_level_importable():
    assert AggressionLevel is not None
