"""Tests for diff_classifier: classify a set of edited paths."""

from __future__ import annotations

import sys
from pathlib import Path

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

from hook_utils.diff_classifier import classify_diff  # noqa: E402


def test_only_docs_changed_is_docs_only():
    result = classify_diff(["docs/foo.md", "docs/bar.md"])
    assert result["kind"] == "docs-only"
    assert result["run_full_chain"] is False


def test_only_tests_changed_is_test_only():
    result = classify_diff(["tests/test_a.py", "tests/foo/test_b.py"])
    assert result["kind"] == "test-only"
    assert result["run_full_chain"] is True  # Stage 2+3 still run per spec §4.2.1


def test_mixed_code_and_docs_is_full():
    result = classify_diff(["src/foo.py", "docs/bar.md"])
    assert result["kind"] == "full"
    assert result["run_full_chain"] is True


def test_only_src_is_full():
    result = classify_diff(["src/assembled_core/utils/foo.py"])
    assert result["kind"] == "full"
    assert result["run_full_chain"] is True


def test_workflow_change_is_full():
    result = classify_diff([".github/workflows/ci.yml"])
    assert result["kind"] == "full"
    assert result["run_full_chain"] is True


def test_governance_change_is_full():
    result = classify_diff(["CLAUDE.md"])
    assert result["kind"] == "full"


def test_empty_diff_is_skip():
    result = classify_diff([])
    assert result["kind"] == "skip"
    assert result["run_full_chain"] is False


def test_no_protected_paths_is_skip():
    result = classify_diff(["output/equity.csv", "README.md"])
    assert result["kind"] == "skip"
    assert result["run_full_chain"] is False
