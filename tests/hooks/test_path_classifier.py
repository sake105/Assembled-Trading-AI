"""Tests for the path classifier used by the Stop hook."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

from hook_utils.path_classifier import (  # noqa: E402
    is_protected_path,
    specialists_for_paths,
)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("src/assembled_core/execution/order_router.py", True),
        ("src/assembled_core/data/foo.py", True),
        ("scripts/run_backtest.py", True),
        (".github/workflows/ci.yml", True),
        (".claude/rules/10-core.md", True),
        ("CLAUDE.md", True),
        ("docs/README.md", False),
        ("tests/test_foo.py", False),
        ("output/equity.csv", False),
        ("README.md", False),
        (".claude/agents/foo.md", False),
    ],
)
def test_is_protected_path(path, expected):
    assert is_protected_path(path) is expected


def test_specialists_for_execution_path():
    paths = ["src/assembled_core/execution/order_router.py"]
    specs = specialists_for_paths(paths)
    assert "risk-execution-reviewer" in specs
    assert "test-runner" in specs


def test_specialists_for_risk_pipeline_accounting():
    for sub in ["risk", "pipeline", "accounting", "portfolio", "paper"]:
        specs = specialists_for_paths([f"src/assembled_core/{sub}/x.py"])
        assert "risk-execution-reviewer" in specs, f"missing for {sub}"
        assert "test-runner" in specs


def test_specialists_for_workflow_change():
    specs = specialists_for_paths([".github/workflows/ci.yml"])
    assert "ci-debugger" in specs


def test_specialists_for_governance_change():
    specs = specialists_for_paths(["CLAUDE.md"])
    assert "docs-governance-sync" in specs
    specs2 = specialists_for_paths([".claude/rules/10-core.md"])
    assert "docs-governance-sync" in specs2


def test_specialists_for_pure_utility_code():
    """Plain src/ code without sensitive zone: only test-runner."""
    specs = specialists_for_paths(["src/assembled_core/utils/format.py"])
    assert specs == {"test-runner"}


def test_specialists_for_mixed_paths():
    specs = specialists_for_paths(
        [
            "src/assembled_core/execution/router.py",
            ".github/workflows/ci.yml",
        ]
    )
    assert "risk-execution-reviewer" in specs
    assert "ci-debugger" in specs
    assert "test-runner" in specs


def test_no_protected_paths_returns_empty_specialists():
    specs = specialists_for_paths(["docs/foo.md", "output/x.csv"])
    assert specs == set()
