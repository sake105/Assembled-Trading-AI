"""Smoke test: CI workflow files exist and names are correct.

No YAML parsing, no extra deps. Fast; runs on Windows and Linux.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECTED_WORKFLOWS = [
    ".github/workflows/evidence-pack-ci.yml",
    ".github/workflows/ops-evidence-ci.yml",
    ".github/workflows/accounting-ci.yml",
    ".github/workflows/release-gate-ci.yml",
]

OPTIONAL_WORKFLOWS = [
    ".github/workflows/repo-health.yml",
]


def test_ci_workflows_exist() -> None:
    """Required CI workflow files exist under repo root."""
    for rel_path in EXPECTED_WORKFLOWS:
        path = ROOT / rel_path
        assert path.exists(), f"Missing workflow file: {rel_path}"
        assert path.is_file(), f"Not a file: {rel_path}"


def test_optional_workflows_if_present() -> None:
    """Optional workflow files, when present, are valid files (no fail if missing)."""
    for rel_path in OPTIONAL_WORKFLOWS:
        path = ROOT / rel_path
        if path.exists():
            assert path.is_file(), f"Optional workflow not a file: {rel_path}"
