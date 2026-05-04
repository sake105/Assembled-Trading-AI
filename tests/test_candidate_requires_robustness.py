"""Tests for candidate gate requiring robustness pack (Sprint 12 Final).

These tests verify that candidate status is only allowed when robustness_ok == True.
"""

from __future__ import annotations

from pathlib import Path


import pytest

pytest.importorskip("src.assembled_core.qa.candidate_gate")
from src.assembled_core.qa.candidate_gate import check_candidate_allowed


def test_candidate_allowed_robustness_ok_true():
    """Test that candidate is allowed when robustness_ok == True."""
    candidate_allowed, message = check_candidate_allowed(robustness_ok=True)

    assert candidate_allowed is True
    assert "allowed" in message.lower()


def test_candidate_not_allowed_robustness_ok_false():
    """Test that candidate is not allowed when robustness_ok == False."""
    candidate_allowed, message = check_candidate_allowed(robustness_ok=False)

    assert candidate_allowed is False
    assert "failed" in message.lower() or "not" in message.lower()


def test_candidate_not_allowed_robustness_not_run():
    """Test that candidate is not allowed when robustness pack was not run."""
    candidate_allowed, message = check_candidate_allowed(robustness_ok=None)

    assert candidate_allowed is False
    assert "not run" in message.lower() or "requires" in message.lower()


def test_candidate_allowed_with_path():
    """Test that candidate check works with robustness_pack_path."""
    pack_path = Path("output/robustness_pack_test")
    candidate_allowed, message = check_candidate_allowed(
        robustness_ok=True, robustness_pack_path=pack_path
    )

    assert candidate_allowed is True
    assert str(pack_path) in message or "allowed" in message.lower()


def test_candidate_not_allowed_with_path():
    """Test that candidate check includes path in message when not allowed."""
    pack_path = Path("output/robustness_pack_test")
    candidate_allowed, message = check_candidate_allowed(
        robustness_ok=False, robustness_pack_path=pack_path
    )

    assert candidate_allowed is False
    assert str(pack_path) in message


def test_read_robustness_ok_from_manifest(tmp_path: Path):
    """Test reading robustness_ok from manifest file."""
    from src.assembled_core.qa.candidate_gate import read_robustness_ok_from_manifest

    import json

    # Create manifest with robustness_ok = True
    manifest_path = tmp_path / "run_manifest.json"
    manifest_data = {
        "freq": "1d",
        "robustness_ok": True,
        "robustness_pack_path": "output/robustness_pack_test",
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    robustness_ok = read_robustness_ok_from_manifest(manifest_path)

    assert robustness_ok is True

    # Test with robustness_ok = False
    manifest_data["robustness_ok"] = False
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    robustness_ok = read_robustness_ok_from_manifest(manifest_path)

    assert robustness_ok is False

    # Test with missing robustness_ok
    del manifest_data["robustness_ok"]
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    robustness_ok = read_robustness_ok_from_manifest(manifest_path)

    assert robustness_ok is None

    # Test with non-existent manifest
    non_existent = tmp_path / "non_existent.json"
    robustness_ok = read_robustness_ok_from_manifest(non_existent)

    assert robustness_ok is None
