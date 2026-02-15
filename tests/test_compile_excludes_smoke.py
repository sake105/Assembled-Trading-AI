"""Smoke test: compile exclude list is SSOT and contains expected entries."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_compile_exclude_constants_exist_and_contain_expected() -> None:
    """SSOT: run_checks.py defines compile excludes; assert expected entries exist."""
    repo_root = Path(__file__).resolve().parents[1]
    run_checks_path = repo_root / "scripts" / "dev" / "run_checks.py"
    assert run_checks_path.exists(), f"run_checks.py not found at {run_checks_path}"

    spec = importlib.util.spec_from_file_location("run_checks", run_checks_path)
    assert spec is not None and spec.loader is not None
    run_checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_checks)

    subdirs = getattr(run_checks, "_COMPILE_EXCLUDE_SUBDIRS", None)
    names = getattr(run_checks, "_COMPILE_EXCLUDE_NAMES", None)

    assert subdirs is not None, "_COMPILE_EXCLUDE_SUBDIRS must exist in run_checks.py"
    assert names is not None, "_COMPILE_EXCLUDE_NAMES must exist in run_checks.py"

    assert "scripts/data" in subdirs, "Excludes must contain scripts/data (SSOT)"
    assert "scripts/tools" in subdirs, "Excludes must contain scripts/tools (SSOT)"
    assert "00_seed_demo_data.py" in names, "Excludes must contain 00_seed_demo_data.py (SSOT)"
