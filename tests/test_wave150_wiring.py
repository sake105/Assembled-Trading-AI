"""Tests for wave-150 module wiring into trading_cycle.py.

Covers:
  Step feat.r — features.registry (FEATURE_REGISTRY / list_all_feature_names)
  Step core.4 — logging_config (setup_logging / generate_run_id)
"""

from __future__ import annotations

import pytest

from src.assembled_core.features.registry import (
    FEATURE_REGISTRY,
    list_all_feature_names,
    validate_registry_unique,
    get_feature_metadata,
)
from src.assembled_core.logging_config import setup_logging, generate_run_id


# ---------------------------------------------------------------------------
# features.registry (Step feat.r)
# ---------------------------------------------------------------------------

def test_feature_registry_importable():
    assert FEATURE_REGISTRY is not None


def test_feature_registry_is_dict():
    assert isinstance(FEATURE_REGISTRY, dict)


def test_feature_registry_has_entries():
    assert len(FEATURE_REGISTRY) > 0


def test_list_all_feature_names():
    names = list_all_feature_names()
    assert isinstance(names, list)
    assert len(names) > 0


def test_validate_registry_unique():
    ok, errors = validate_registry_unique()
    assert isinstance(ok, bool)
    assert isinstance(errors, list)


def test_get_feature_metadata_known():
    names = list_all_feature_names()
    if names:
        meta = get_feature_metadata(names[0])
        assert isinstance(meta, dict)


def test_get_feature_metadata_unknown():
    meta = get_feature_metadata("NONEXISTENT_FEATURE_WAVE150")
    assert meta is None


# ---------------------------------------------------------------------------
# logging_config (Step core.4)
# ---------------------------------------------------------------------------

def test_setup_logging_importable():
    assert setup_logging is not None


def test_generate_run_id_importable():
    assert generate_run_id is not None


def test_generate_run_id_returns_string():
    run_id = generate_run_id(prefix="test")
    assert isinstance(run_id, str)
    assert run_id.startswith("test")
