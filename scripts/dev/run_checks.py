#!/usr/bin/env python3
"""Run code quality checks: py_compile → ruff → pytest.

This script provides a Windows-compatible way to run all checks in sequence.
It uses `python -m` for all tools to avoid PATH issues.

Usage:
    python scripts/dev/run_checks.py [--skip-compile] [--skip-ruff] [--skip-pytest] [--python-cmd CMD] [--pytest-args ARGS]

Exit codes:
    0: All checks passed
    1: One or more checks failed
"""

from __future__ import annotations

import argparse
import shlex
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


# Documented compile exclusions (scripts/data has intentional syntax; scripts/tools and seed script optional)
# See README or docs: "Compile check excludes scripts/data, scripts/tools, 00_seed_demo_data.py"
_COMPILE_EXCLUDE_SUBDIRS = ("scripts/data", "scripts/tools")
_COMPILE_EXCLUDE_NAMES = ("00_seed_demo_data.py",)


def _compile_with_excludes(repo_root: Path) -> tuple[int, str]:
    """Compile src, tests, scripts excluding documented paths. Uses POSIX-style path checks."""
    import py_compile

    errors: list[str] = []
    for base in ("src", "tests", "scripts"):
        d = repo_root / base
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.py")):
            try:
                rel = p.relative_to(repo_root)
            except ValueError:
                continue
            parts = rel.as_posix().split("/")
            s = "/".join(parts)
            if (
                any(ex in s for ex in _COMPILE_EXCLUDE_SUBDIRS)
                or p.name in _COMPILE_EXCLUDE_NAMES
            ):
                continue
            try:
                py_compile.compile(str(p), doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(f"{p}: {e}")
    if errors:
        msg = "\n".join(errors[:20])
        if len(errors) > 20:
            msg += f"\n... and {len(errors) - 20} more"
        return 1, msg
    return 0, "Compile passed"


def run_py_compile(
    python_cmd: list[str], paths: list[str], repo_root: Path | None = None
) -> tuple[int, str]:
    """Run py_compile on given paths, or compile with documented excludes when repo_root set.

    When repo_root is set and paths equal [src, tests], runs compile with excludes
    (scripts/data, scripts/tools, 00_seed_demo_data.py). Otherwise runs py_compile on paths.
    """
    print("=" * 70)
    print("Step 1: py_compile (syntax check)")
    print("=" * 70)

    norm_paths = [Path(p).resolve().as_posix() for p in paths]
    default_src_tests = [
        (repo_root / "src").resolve().as_posix(),
        (repo_root / "tests").resolve().as_posix(),
    ]
    if repo_root is not None and norm_paths == default_src_tests:
        code, output = _compile_with_excludes(repo_root)
        if code == 0:
            print(
                "[OK] py_compile passed (src, tests, scripts with documented excludes)"
            )
        else:
            print("[FAIL] py_compile failed:")
            print(output)
        return code, output

    cmd = python_cmd + ["-m", "py_compile"] + paths
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


def run_ruff(python_cmd: list[str], paths: list[str]) -> tuple[int, str]:
    """Run ruff check on given paths.

    Args:
        python_cmd: Python command as list (e.g. [\"python\"] or [\"py\", \"-3\"])
        paths: List of paths to check

    Returns:
        Tuple of (exit_code, output)
    """
    print("=" * 70)
    print("Step 2: ruff check (linting)")
    print("=" * 70)

    cmd = python_cmd + ["-m", "ruff", "check"] + paths
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


def run_pytest(python_cmd: list[str], pytest_args: list[str]) -> tuple[int, str]:
    """Run pytest with given arguments.

    Args:
        python_cmd: Python command as list (e.g. [\"python\"] or [\"py\", \"-3\"])
        pytest_args: Additional pytest arguments

    Returns:
        Tuple of (exit_code, output)
    """
    print("=" * 70)
    print("Step 3: pytest (tests)")
    print("=" * 70)

    cmd = python_cmd + ["-m", "pytest"] + pytest_args
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
        choices=[
            "accounting",
            "broker_snapshot",
            "ops_evidence",
            "evidence_pack",
            "release_sprint13",
        ],
        help=(
            "Optional preset for common check bundles "
            "(accounting, broker_snapshot, ops_evidence, evidence_pack, release_sprint13). "
            "Presets are shortcuts; explicit --paths override paths."
        ),
    )
    parser.add_argument(
        "--python-cmd",
        default="",
        metavar="CMD",
        help="Python command for subprocess (e.g. 'python' or 'py -3'). Default: auto-detect. Prefer --python-cmd-list on Windows to avoid quoting.",
    )
    parser.add_argument(
        "--python-cmd-list",
        action="append",
        default=None,
        metavar="ARG",
        help="Python command as separate args (e.g. --python-cmd-list py --python-cmd-list -3). Overrides --python-cmd. No quoting needed.",
    )

    args = parser.parse_args()

    # Python command: --python-cmd-list (preferred) > --python-cmd > auto-detect
    python_cmd_list = getattr(args, "python_cmd_list", None) or []
    if python_cmd_list:
        python_cmd = python_cmd_list
        print(f"Using Python: {' '.join(python_cmd)} (from --python-cmd-list)")
    elif args.python_cmd:
        python_cmd = shlex.split(args.python_cmd)
        print(f"Using Python: {' '.join(python_cmd)} (from --python-cmd)")
    else:
        python_exe = find_python_executable()
        python_cmd = [str(python_exe)]
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
                "scripts/verify_evidence_pack.py",
            ]
        elif args.preset == "evidence_pack":
            preset_paths = [
                "src/assembled_core/accounting/evidence_index.py",
                "src/assembled_core/accounting/evidence_pack.py",
                "scripts/export_evidence_pack.py",
                "scripts/verify_evidence_pack.py",
            ]
        elif args.preset == "release_sprint13":
            preset_paths = [
                "src/assembled_core/accounting/evidence_index.py",
                "src/assembled_core/accounting/evidence_pack.py",
                "scripts/export_evidence_pack.py",
                "scripts/verify_evidence_pack.py",
                "src/assembled_core/pipeline/orchestrator.py",
                "scripts/dev/run_checks.py",
            ]
        else:
            preset_paths = ["src/", "tests/"]

        paths = [str(repo_root / p) for p in preset_paths]

    # Run checks in sequence
    exit_code = 0
    outputs = []

    # Step 1: py_compile
    if not args.skip_compile:
        code, output = run_py_compile(python_cmd, paths, repo_root=repo_root)
        exit_code = max(exit_code, code)
        outputs.append(("py_compile", code, output))
        print()

    # Step 2: ruff
    if not args.skip_ruff:
        code, output = run_ruff(python_cmd, paths)
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
                    "tests/test_verify_evidence_pack_cli_smoke.py",
                    "tests/test_verify_evidence_pack_json_schema_stable.py",
                    "tests/test_evidence_pack_verify.py",
                    "-q",
                ]
            elif args.preset == "ops_evidence":
                # PR-fast: no deterministic_bytes / heavy tests; guard below enforces this
                pytest_args = [
                    # Ops chain: Import -> Require -> Pack -> Verify + CLI smoke (no deterministic-bytes)
                    "tests/test_broker_snapshot_importer_smoke.py",
                    "tests/test_broker_snapshot_importer_hardening.py",
                    "tests/test_broker_snapshot_namespace_rules.py",
                    "tests/test_ops_golden_path_evidence_pack_e2e.py",
                    "tests/test_export_evidence_pack_cli_smoke.py",
                    "tests/test_verify_evidence_pack_cli_smoke.py",
                    "tests/test_evidence_index_written.py",
                    "tests/test_evidence_pack_written.py",
                    "tests/test_ci_workflows_inventory_smoke.py",
                    "-q",
                ]
                # Guard: ops_evidence must stay fast; no deterministic_bytes tests
                _forbidden = ("deterministic_bytes", "test_evidence_pack_deterministic")
                for item in pytest_args:
                    if item.startswith("tests/") and any(f in item for f in _forbidden):
                        _msg = f"ops_evidence preset must not include slow tests (deterministic_bytes): {item!r}"
                        raise ValueError(
                            _msg.encode("ascii", errors="ignore").decode("ascii")
                        )
            elif args.preset == "release_sprint13":
                # Merge safety: CI inventory, verify/export CLI schema and smoke (fail-on-warn, print-pack-path, read_pack_manifest), ops golden path, paths POSIX, docs sanity; no deterministic-bytes
                pytest_args = [
                    "tests/test_ci_workflows_inventory_smoke.py",
                    "tests/test_verify_evidence_pack_cli_smoke.py",
                    "tests/test_export_evidence_pack_cli_smoke.py",
                    "tests/test_evidence_pack_verify.py",
                    "tests/test_verify_evidence_pack_json_schema_stable.py",
                    "tests/test_export_evidence_pack_json_schema_stable.py",
                    "tests/test_ops_golden_path_evidence_pack_e2e.py",
                    "tests/test_ops_archive_ps1_contract_smoke.py",
                    "tests/test_paths_posix_in_outputs_smoke.py",
                    "tests/test_docs_sanity_sprint13.py",
                    "tests/test_docs_links_smoke.py",
                    "tests/test_release_notes_header_smoke.py",
                    # Trading-numeric correctness (CI-002): the block above is
                    # docs/CLI/inventory smoke only — it could stay green while
                    # the trading system is numerically broken. These pin the
                    # money-path math the release actually depends on: transaction
                    # costs, fill model, ledger cash-invariant, position sizing,
                    # position-engine invariants, pre-trade gates, kill switch,
                    # risk controls, reconcile-halt policy, backtest fills. They
                    # import src/pipeline -> config.policy_loader, which needs
                    # PyYAML; the release-gate CI env must therefore install
                    # -r requirements.txt (see release-gate-ci.yml) — the old
                    # ad-hoc minimal list omitted PyYAML so these could not
                    # import there. (numba is optional/guarded, not required.)
                    # Fast + deterministic (111 cases ~3s locally).
                    "tests/test_transaction_costs_commission.py",
                    "tests/test_transaction_costs_slippage.py",
                    "tests/test_transaction_costs_spread.py",
                    "tests/test_fill_model_partial.py",
                    "tests/test_fill_model_costs_consistency.py",
                    "tests/test_ledger_cash_invariant_partial.py",
                    "tests/test_portfolio_position_sizing.py",
                    "tests/test_position_engine_invariants.py",
                    "tests/test_execution_pre_trade_checks.py",
                    "tests/test_execution_kill_switch.py",
                    "tests/test_risk_controls_integration.py",
                    "tests/test_reconcile_halt_policy.py",
                    "tests/test_pipeline_backtest_fills.py",
                    "-q",
                ]
            else:
                pytest_args = ["tests/", "-v"]

        code, output = run_pytest(python_cmd, pytest_args)
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
        print("[OK] All checks passed.")
    else:
        print("[FAIL] Some checks failed. See output above for details.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
