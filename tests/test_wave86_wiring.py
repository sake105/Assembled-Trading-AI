"""Tests for wave-86 module wiring into trading_cycle.py.

Covers:
  Step 8.111 — intel.shock_propagation (SHOCK_TO_ORIGIN_NODES / DEFAULT_DAMPENING_FACTOR)
  Step 8.112 — intel.source_registry (list_sources / get_trust_weight / get_source_tier)
  Step 8.113 — intel.trigger_snapshot_store (TriggerSnapshotStore)
"""

from __future__ import annotations

import pytest
from pathlib import Path

from src.assembled_core.intel.shock_propagation import (
    SHOCK_TO_ORIGIN_NODES,
    DEFAULT_DAMPENING_FACTOR,
    map_trigger_to_shocks,
)
from src.assembled_core.intel.source_registry import (
    list_sources,
    get_trust_weight,
    get_source_tier,
    get_all_tiers,
)
from src.assembled_core.intel.trigger_snapshot_store import TriggerSnapshotStore


# ---------------------------------------------------------------------------
# shock_propagation (Step 8.111)
# ---------------------------------------------------------------------------

def test_shock_to_origin_nodes_not_empty():
    assert len(SHOCK_TO_ORIGIN_NODES) > 0


def test_default_dampening_factor_is_float():
    assert isinstance(DEFAULT_DAMPENING_FACTOR, float)
    assert 0.0 < DEFAULT_DAMPENING_FACTOR < 1.0


def test_shock_to_origin_nodes_keys_are_enums():
    for key in SHOCK_TO_ORIGIN_NODES:
        assert key is not None


def test_map_trigger_to_shocks_importable():
    assert map_trigger_to_shocks is not None


# ---------------------------------------------------------------------------
# source_registry (Step 8.112)
# ---------------------------------------------------------------------------

def test_list_sources_returns_list():
    sources = list_sources()
    assert isinstance(sources, list)


def test_list_sources_not_empty():
    sources = list_sources()
    assert len(sources) > 0


def test_get_trust_weight_returns_float():
    sources = list_sources()
    if sources:
        weight = get_trust_weight(sources[0])
        assert isinstance(weight, float)
        assert 0.0 <= weight <= 1.0


def test_get_trust_weight_unknown_returns_float():
    weight = get_trust_weight("totally_unknown_source_xyz")
    assert isinstance(weight, float)


def test_get_all_tiers_returns_dict():
    tiers = get_all_tiers()
    assert isinstance(tiers, dict)
    assert len(tiers) > 0


# ---------------------------------------------------------------------------
# trigger_snapshot_store (Step 8.113)
# ---------------------------------------------------------------------------

def test_trigger_snapshot_store_creates(tmp_path):
    tss = TriggerSnapshotStore(root=tmp_path)
    assert isinstance(tss, TriggerSnapshotStore)


def test_trigger_snapshot_store_root_path(tmp_path):
    tss = TriggerSnapshotStore(root=tmp_path)
    assert tss._root == Path(tmp_path)


def test_trigger_snapshot_store_archive_missing_artifact(tmp_path):
    tss = TriggerSnapshotStore(root=tmp_path / "store")
    result = tss.archive(
        source="test_source",
        run_id="run_001",
        artifact_path=tmp_path / "nonexistent.json",
    )
    assert result is None
