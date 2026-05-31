"""CI-001 regression: the walk-forward release-gate script blocks (non-zero
exit) on an EXECUTION-INTEGRITY failure, even though the E3/E4 statistical
gates stay non-blocking during the grace period.

Before CI-001 ``scripts/release_gate_walk_forward.py`` ran on synthetic data,
never passed ``--enforce`` and grace-returned 0 — so a structurally broken run
(some splits crash and are silently dropped, the survivors averaged) could not
turn the job red. ``run_walk_forward`` already raises when *all* splits fail;
the hole was PARTIAL failure.

CI-001 adds ``_check_execution_integrity`` and a 3-exit-code contract:

    0   report written, execution intact, AND (E3/E4 pass OR grace)
    1   E3/E4 statistical miss AND ``--enforce`` set
    2   EXECUTION INTEGRITY failure — always blocks, regardless of
        ``--enforce`` / grace

These tests pin the helper's reasons and the ``main()`` exit-code matrix.
``build_gate_report`` and ``_synthetic_prices`` are monkeypatched so the test
controls the report without running the real walk-forward backtest.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.release_gate_walk_forward as rg  # noqa: E402


def _report(
    *,
    overall_pass: bool = True,
    n_splits: int = 8,
    n_ok: int = 8,
    n_obs: int = 8,
    e3_value: float = 1.296,
    e4_value: float = 0.445,
) -> dict[str, Any]:
    """A minimal report shaped like ``build_gate_report``'s output."""
    return {
        "walk_forward": {
            "n_splits": n_splits,
            "n_successful_splits": n_ok,
        },
        "deflated_sharpe": {"n_observations": n_obs},
        "gates": {
            "E3_oos_sharpe": {"value": e3_value, "threshold": 0.3},
            "E4_deflated_sharpe": {"value": e4_value, "threshold": 0.5},
        },
        "overall_pass": overall_pass,
    }


# --------------------------------------------------------------------------- #
# _check_execution_integrity — unit
# --------------------------------------------------------------------------- #


def test_integrity_healthy_report_returns_none() -> None:
    assert rg._check_execution_integrity(_report(n_splits=8, n_ok=8, n_obs=8)) is None


def test_integrity_no_splits_flags() -> None:
    reason = rg._check_execution_integrity(_report(n_splits=0, n_ok=0, n_obs=0))
    assert reason is not None
    assert "n_splits=0" in reason


def test_integrity_partial_split_failure_flags() -> None:
    reason = rg._check_execution_integrity(_report(n_splits=8, n_ok=5, n_obs=5))
    assert reason is not None
    assert "5" in reason and "8" in reason


def test_integrity_degenerate_dsr_input_flags() -> None:
    # All splits "succeeded" but DSR input is degenerate → internally
    # inconsistent report = pipeline defect, not a statistical miss.
    reason = rg._check_execution_integrity(_report(n_splits=8, n_ok=8, n_obs=1))
    assert reason is not None
    assert "n_observations=1" in reason


def test_integrity_partial_failure_outranks_degenerate_dsr() -> None:
    # When BOTH a partial split failure and a degenerate DSR input are present,
    # the partial-failure reason must win (it is the more proximate cause and is
    # checked first). Pins the branch ordering inside the helper.
    reason = rg._check_execution_integrity(_report(n_splits=8, n_ok=5, n_obs=1))
    assert reason is not None
    assert "failed to" in reason
    assert "n_observations" not in reason


# --------------------------------------------------------------------------- #
# main() — exit-code matrix
# --------------------------------------------------------------------------- #


def _patch(monkeypatch, report: dict[str, Any]) -> None:
    monkeypatch.setattr(rg, "_synthetic_prices", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(rg, "build_gate_report", lambda *a, **k: report)


def test_main_pass_exits_zero(monkeypatch, tmp_path: Path) -> None:
    _patch(monkeypatch, _report(overall_pass=True))
    assert rg.main(["--out-dir", str(tmp_path)]) == 0


def test_main_statistical_miss_without_enforce_grace_zero(
    monkeypatch, tmp_path: Path
) -> None:
    _patch(monkeypatch, _report(overall_pass=False))
    assert rg.main(["--out-dir", str(tmp_path)]) == 0


def test_main_statistical_miss_with_enforce_exits_one(
    monkeypatch, tmp_path: Path
) -> None:
    _patch(monkeypatch, _report(overall_pass=False))
    assert rg.main(["--out-dir", str(tmp_path), "--enforce"]) == 1


def test_main_integrity_failure_blocks_without_enforce(
    monkeypatch, tmp_path: Path
) -> None:
    # Partial split failure → exit 2 even though --enforce is OFF and the
    # grace period would otherwise return 0.
    _patch(monkeypatch, _report(overall_pass=False, n_splits=8, n_ok=5, n_obs=5))
    assert rg.main(["--out-dir", str(tmp_path)]) == 2


def test_main_integrity_failure_blocks_with_enforce(
    monkeypatch, tmp_path: Path
) -> None:
    # Integrity failure outranks the statistical exit-1 path too.
    _patch(monkeypatch, _report(overall_pass=False, n_splits=8, n_ok=5, n_obs=5))
    assert rg.main(["--out-dir", str(tmp_path), "--enforce"]) == 2
