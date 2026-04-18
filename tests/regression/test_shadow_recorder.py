"""Shadow-mode helper smoke tests (Part D infrastructure)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.assembled_core.ops.shadow_recorder import is_shadow_only, record_shadow


pytestmark = pytest.mark.regression


def test_is_shadow_only_default_true() -> None:
    # Missing policy block -> defaults to True (conservative)
    assert is_shadow_only({}, "correlation_guard") is True


def test_is_shadow_only_explicit_false() -> None:
    policy = {"correlation_guard": {"shadow_only": False}}
    assert is_shadow_only(policy, "correlation_guard") is False


def test_record_shadow_writes_envelope(tmp_path: Path) -> None:
    path = record_shadow(
        "correlation_guard",
        {"deltas": {"AAPL": {"old": 0.3, "new": 0.15}}},
        as_of="2026-04-18",
        meta={"clusters_scaled": 1, "applied": False},
        root=tmp_path,
    )
    assert path is not None and path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["module"] == "correlation_guard"
    assert data["snapshot_date"] == "2026-04-18"
    assert data["payload"]["meta"]["applied"] is False
    assert "AAPL" in data["payload"]["would_apply"]["deltas"]


def test_record_shadow_swallows_errors(tmp_path: Path) -> None:
    # Invalid module id with path separator — shadow_mode raises; helper swallows.
    result = record_shadow(
        "bad/module",
        {"foo": 1},
        root=tmp_path,
    )
    assert result is None
