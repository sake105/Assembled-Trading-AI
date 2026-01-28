#!/usr/bin/env python3
"""Run code quality checks: py_compile → ruff → pytest.

This script provides a Windows-compatible way to run all checks in sequence.
It uses `python -m` for all tools to avoid PATH issues.

Usage:
    python scripts/dev/run_checks.py [--skip-compile] [--skip-ruff] [--skip-pytest] [--pytest-args ARGS]

Exit codes:
    0: All checks passed
    1: One or more checks failed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_python_executable() -> Path:
    """Find Python executable, preferring venv if available.

    Returns:
        Path to python executable
    """
    # Check for .venv in repo root
    repo_root = Path(__file__).resolve().parents[2]
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return venv_python

    # Fallback to system python
    return Path(sys.executable)


def run_py_compile(python_exe: Path, paths: list[str]) -> tuple[int, str]:
    """Run py_compile on given paths.

    Args:
        python_exe: Path to Python executable
        paths: List of paths to compile

    Returns:
        Tuple of (exit_code, output)
    """
    print("=" * 70)
    print("Step 1: py_compile (syntax check)")
    print("=" * 70)

    cmd = [str(python_exe), "-m", "py_compile"] + paths
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print(f"[OK] py_compile passed for {len(paths)} path(s)")
        else:
            print("[FAIL] py_compile failed:")
            print(result.stdout)
            print(result.stderr)
        return result.returncode, result.stdout + result.stderr
    except Exception as e:
        print(f"[ERROR] py_compile error: {e}")
        return 1, str(e)


def run_ruff(python_exe: Path, paths: list[str]) -> tuple[int, str]:
    """Run ruff check on given paths.

    Args:
        python_exe: Path to Python executable
        paths: List of paths to check

    Returns:
        Tuple of (exit_code, output)
    """
    print("=" * 70)
    print("Step 2: ruff check (linting)")
    print("=" * 70)

    cmd = [str(python_exe), "-m", "ruff", "check"] + paths
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print(f"[OK] ruff check passed for {len(paths)} path(s)")
        else:
            print("[FAIL] ruff check failed:")
            print(result.stdout)
            print(result.stderr)
        return result.returncode, result.stdout + result.stderr
    except Exception as e:
        print(f"[ERROR] ruff check error: {e}")
        return 1, str(e)


def run_pytest(python_exe: Path, pytest_args: list[str]) -> tuple[int, str]:
    """Run pytest with given arguments.

    Args:
        python_exe: Path to Python executable
        pytest_args: Additional pytest arguments

    Returns:
        Tuple of (exit_code, output)
    """
    print("=" * 70)
    print("Step 3: pytest (tests)")
    print("=" * 70)

    cmd = [str(python_exe), "-m", "pytest"] + pytest_args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print("[OK] pytest passed")
        else:
            print("[FAIL] pytest failed:")
            print(result.stdout)
            print(result.stderr)
        return result.returncode, result.stdout + result.stderr
    except Exception as e:
        print(f"[ERROR] pytest error: {e}")
        return 1, str(e)


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Run code quality checks: py_compile → ruff → pytest"
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip py_compile step",
    )
    parser.add_argument(
        "--skip-ruff",
        action="store_true",
        help="Skip ruff check step",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip pytest step",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments to pass to pytest",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=None,
        help="Paths to check with py_compile and ruff (default: src/ tests/). "
        "If set, overrides any preset paths.",
    )
    parser.add_argument(
        "--preset",
        choices=["accounting", "broker_snapshot", "ops_evidence", "evidence_pack"],
        help=(
            "Optional preset for common check bundles "
            "(accounting, broker_snapshot, ops_evidence, or evidence_pack). "
            "Presets are shortcuts; explicit --paths override paths."
        ),
    )

    args = parser.parse_args()

    # Find Python executable
    python_exe = find_python_executable()
    print(f"Using Python: {python_exe}")
    if ".venv" in str(python_exe):
        print("  (using venv)")
    print()

    # Determine paths to check
    repo_root = Path(__file__).resolve().parents[2]

    # Preset-aware path selection; explicit --paths override presets
    if args.paths is not None:
        paths = [str(repo_root / p) for p in args.paths]
    else:
        if args.preset == "accounting":
            preset_paths: list[str] = [
                "src/assembled_core/accounting/",
                "src/assembled_core/pipeline/orchestrator.py",
                "src/assembled_core/qa/candidate_gate.py",
                "scripts/run_eod_pipeline.py",
                "scripts/run_backtest_strategy.py",
                "scripts/run_daily.py",
            ]
        elif args.preset == "broker_snapshot":
            preset_paths = [
                "src/assembled_core/accounting/broker_snapshot.py",
                "src/assembled_core/accounting/broker_snapshot_store.py",
                "src/assembled_core/accounting/broker_snapshot_importer.py",
                "src/assembled_core/accounting/ledger_integration.py",
                "src/assembled_core/qa/backtest_engine.py",
                "scripts/run_backtest_strategy.py",
                "scripts/run_eod_pipeline.py",
                "scripts/import_broker_snapshot.py",
            ]
        elif args.preset == "ops_evidence":
            preset_paths = [
                "src/assembled_core/accounting/evidence_index.py",
                "src/assembled_core/accounting/evidence_pack.py",
                "scripts/export_evidence_pack.py",
            ]
        elif args.preset == "evidence_pack":
            preset_paths = [
                "src/assembled_core/accounting/evidence_index.py",
                "src/assembled_core/accounting/evidence_pack.py",
                "scripts/export_evidence_pack.py",
            ]
        else:
            preset_paths = ["src/", "tests/"]

        paths = [str(repo_root / p) for p in preset_paths]

    # Run checks in sequence
    exit_code = 0
    outputs = []

    # Step 1: py_compile
    if not args.skip_compile:
        code, output = run_py_compile(python_exe, paths)
        exit_code = max(exit_code, code)
        outputs.append(("py_compile", code, output))
        print()

    # Step 2: ruff
    if not args.skip_ruff:
        code, output = run_ruff(python_exe, paths)
        exit_code = max(exit_code, code)
        outputs.append(("ruff", code, output))
        print()

    # Step 3: pytest
    if not args.skip_pytest:
        if args.pytest_args:
            pytest_args = args.pytest_args
        else:
            # Provide useful defaults per preset; fall back to full test suite
            if args.preset == "accounting":
                pytest_args = [
                    "tests/test_reconcile_report_written.py",
                    "tests/test_reconcile_report_csv_broker_meta.py",
                    "tests/test_accounting_report_broker_meta.py",
                    "tests/test_candidate_gate_reconciliation.py",
                    "tests/test_ops_evidence_pack_e2e.py",
                    "-q",
                ]
            elif args.preset == "broker_snapshot":
                pytest_args = [
                    "tests/test_broker_snapshot_smoke.py",
                    "tests/test_broker_snapshot_policy_precedence.py",
                    "tests/test_broker_snapshot_policy_require.py",
                    "tests/test_backtest_write_broker_snapshot_smoke.py",
                    "tests/test_broker_snapshot_importer_smoke.py",
                    "tests/test_broker_snapshot_importer_hardening.py",
                    "tests/test_broker_snapshot_import_e2e_reconciliation.py",
                    "tests/test_import_cli_then_require_reconcile.py",
                    "tests/test_broker_snapshot_namespace_rules.py",
                    "tests/test_ops_evidence_pack_e2e.py",
                    "-q",
                ]
            elif args.preset == "evidence_pack":
                pytest_args = [
                    "tests/test_evidence_index_written.py",
                    "tests/test_evidence_pack_written.py",
                    "tests/test_evidence_pack_deterministic_bytes.py",
                    "tests/test_evidence_pack_manifest_fallback.py",
                    "tests/test_export_evidence_pack_cli_smoke.py",
                    "tests/test_evidence_pack_verify.py",
                    "-q",
                ]
            else:
                pytest_args = ["tests/", "-v"]

        code, output = run_pytest(python_exe, pytest_args)
        exit_code = max(exit_code, code)
        outputs.append(("pytest", code, output))
        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for name, code, _ in outputs:
        status = "[PASS]" if code == 0 else "[FAIL]"
        print(f"{name:15} {status}")

    if exit_code == 0:
        print("\n[OK] All checks passed!")
    else:
        print("\n[FAIL] Some checks failed. See output above for details.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
