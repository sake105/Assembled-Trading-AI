"""Tests for candidate gate with reconciliation (Sprint 13).

Tests that reconciliation_ok=False blocks candidate status,
and reconciliation_ok=None allows with warning (backward compatible).
Tests combined gate behavior and report links in messages.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

pytest.importorskip("src.assembled_core.qa.candidate_gate")
from src.assembled_core.qa.candidate_gate import (
    check_candidate_allowed,
    read_reconciliation_ok_from_manifest,
)


def test_candidate_blocked_when_reconciliation_failed():
    """Test that candidate is blocked when reconciliation_ok=False."""
    # Robustness passed, but reconciliation failed
    reconcile_report = "reconcile_report_test/reconcile_2024-01-15.json"
    candidate_allowed, message = check_candidate_allowed(
        robustness_ok=True,
        reconciliation_ok=False,
        reconcile_report_path=reconcile_report,
    )

    assert candidate_allowed is False, (
        "Candidate should be blocked when reconciliation failed"
    )
    assert "Reconciliation failed" in message
    assert reconcile_report in message, "Message should contain report link"
    assert "candidate NOT allowed" in message


def test_candidate_allowed_when_reconciliation_passed():
    """Test that candidate is allowed when reconciliation_ok=True."""
    # Both robustness and reconciliation passed
    candidate_allowed, message = check_candidate_allowed(
        robustness_ok=True,
        reconciliation_ok=True,
    )

    assert candidate_allowed is True, (
        "Candidate should be allowed when reconciliation passed"
    )
    assert "Robustness pack passed" in message
    assert "Reconciliation passed" in message
    assert "candidate allowed" in message


def test_candidate_allowed_when_reconciliation_none_backward_compatible():
    """Test that candidate is allowed when reconciliation_ok=None (backward compatible)."""
    # Robustness passed, reconciliation not run (backward compatible)
    candidate_allowed, message = check_candidate_allowed(
        robustness_ok=True,
        reconciliation_ok=None,
    )

    assert candidate_allowed is True, (
        "Candidate should be allowed when reconciliation_ok=None (backward compatible)"
    )
    assert "Robustness pack passed" in message
    assert "Reconciliation not run" in message
    assert "backward compatible" in message
    assert "candidate allowed" in message


def test_candidate_blocked_when_both_robustness_and_reconciliation_failed():
    """Test that candidate is blocked when both robustness and reconciliation failed."""
    robustness_pack = "robustness_pack_test"
    reconcile_report = "reconcile_report_test/reconcile_2024-01-15.json"
    candidate_allowed, message = check_candidate_allowed(
        robustness_ok=False,
        robustness_pack_path=robustness_pack,
        reconciliation_ok=False,
        reconcile_report_path=reconcile_report,
    )

    assert candidate_allowed is False, (
        "Candidate should be blocked when both gates failed"
    )
    assert "Robustness pack failed" in message
    assert robustness_pack in message, "Message should contain robustness pack link"
    assert "Reconciliation failed" in message
    assert reconcile_report in message, (
        "Message should contain reconciliation report link"
    )
    assert "candidate NOT allowed" in message


def test_candidate_message_includes_report_links_when_available():
    """Test that messages include report links when paths are provided."""
    robustness_pack = "output/robustness_pack_run1"
    reconcile_report = "output/reconcile_report_run1/reconcile_2025-01-15.json"

    # Test with both reports
    candidate_allowed, message = check_candidate_allowed(
        robustness_ok=True,
        robustness_pack_path=robustness_pack,
        reconciliation_ok=True,
        reconcile_report_path=reconcile_report,
    )

    assert candidate_allowed is True
    assert robustness_pack in message, "Message should include robustness pack path"
    assert reconcile_report in message, (
        "Message should include reconciliation report path"
    )

    # Test with only reconciliation report
    candidate_allowed, message = check_candidate_allowed(
        robustness_ok=True,
        reconciliation_ok=False,
        reconcile_report_path=reconcile_report,
    )

    assert candidate_allowed is False
    assert reconcile_report in message, (
        "Message should include reconciliation report path when reconciliation fails"
    )


def test_candidate_combined_gates_all_none_blocks():
    """Test that candidate is blocked when both gates are None (robustness required)."""
    candidate_allowed, message = check_candidate_allowed(
        robustness_ok=None,
        reconciliation_ok=None,
    )

    assert candidate_allowed is False, (
        "Candidate should be blocked when robustness_ok is None (robustness required)"
    )
    assert "Robustness pack not run" in message
    assert "Reconciliation not run" in message
    assert "candidate NOT allowed" in message


def test_candidate_combined_gates_robustness_none_reconciliation_true_blocks():
    """Test that candidate is blocked when robustness is None even if reconciliation is True."""
    candidate_allowed, message = check_candidate_allowed(
        robustness_ok=None,
        reconciliation_ok=True,
    )

    assert candidate_allowed is False, (
        "Candidate should be blocked when robustness_ok is None (robustness required)"
    )
    assert "Robustness pack not run" in message
    assert "candidate NOT allowed" in message


def test_candidate_combined_gates_robustness_true_reconciliation_none():
    """Test that candidate is allowed when robustness is True and reconciliation is None."""
    candidate_allowed, message = check_candidate_allowed(
        robustness_ok=True,
        reconciliation_ok=None,
    )

    assert candidate_allowed is True, (
        "Candidate should be allowed (reconciliation None is backward compatible)"
    )
    assert "Robustness pack passed" in message
    assert "Reconciliation not run" in message
    assert "backward compatible" in message
    assert "candidate allowed" in message


def test_candidate_combined_gates_robustness_false_reconciliation_none():
    """Test that candidate is blocked when robustness is False (even if reconciliation is None)."""
    candidate_allowed, message = check_candidate_allowed(
        robustness_ok=False,
        robustness_pack_path="robustness_pack_test",
        reconciliation_ok=None,
    )

    assert candidate_allowed is False, (
        "Candidate should be blocked when robustness fails (even if reconciliation is None)"
    )
    assert "Robustness pack failed" in message
    assert "Reconciliation not run" in message
    assert "candidate NOT allowed" in message


def test_candidate_combined_gates_robustness_none_reconciliation_false():
    """Test that candidate is blocked when reconciliation is False (even if robustness is None)."""
    candidate_allowed, message = check_candidate_allowed(
        robustness_ok=None,
        reconciliation_ok=False,
        reconcile_report_path="reconcile_report_test/reconcile_2024-01-15.json",
    )

    assert candidate_allowed is False, (
        "Candidate should be blocked when reconciliation fails (even if robustness is None)"
    )
    assert "Robustness pack not run" in message
    assert "Reconciliation failed" in message
    assert "candidate NOT allowed" in message


def test_read_reconciliation_ok_from_manifest_exists(tmp_path: Path):
    """Test reading reconciliation_ok from manifest when field exists."""
    manifest_path = tmp_path / "run_manifest_1d.json"

    manifest = {
        "reconciliation_ok": True,
        "robustness_ok": True,
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    result = read_reconciliation_ok_from_manifest(manifest_path)
    assert result is True


def test_read_reconciliation_ok_from_manifest_missing_field(tmp_path: Path):
    """Test reading reconciliation_ok from manifest when field is missing."""
    manifest_path = tmp_path / "run_manifest_1d.json"

    manifest = {
        "robustness_ok": True,
        # reconciliation_ok missing
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    result = read_reconciliation_ok_from_manifest(manifest_path)
    assert result is None


def test_read_reconciliation_ok_from_manifest_file_not_found(tmp_path: Path):
    """Test reading reconciliation_ok from manifest when file does not exist."""
    manifest_path = tmp_path / "nonexistent_manifest.json"

    result = read_reconciliation_ok_from_manifest(manifest_path)
    assert result is None


def test_read_reconciliation_ok_from_manifest_invalid_json(tmp_path: Path):
    """Test reading reconciliation_ok from manifest when JSON is invalid."""
    manifest_path = tmp_path / "invalid_manifest.json"

    # Write invalid JSON
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("{ invalid json }")

    result = read_reconciliation_ok_from_manifest(manifest_path)
    assert result is None
