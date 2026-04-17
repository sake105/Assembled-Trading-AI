"""E0.4 — Reconciliation halt policy unit tests.

These tests pin the threshold logic and halt-flag writer in
``scripts/run_live_paper.py``. They do not exercise the broker-side
sync; that path requires live Alpaca credentials and is covered by the
paper-trading-ci workflow's halt-ack gate end-to-end.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.phase_zero


@pytest.fixture(scope="module")
def runner_module():
    """Load scripts/run_live_paper.py as a module without executing main()."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_live_paper.py"
    spec = importlib.util.spec_from_file_location("run_live_paper_under_test", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_reconcile_policy_defaults(runner_module) -> None:
    policy = runner_module._reconcile_policy({})
    assert policy["halt_on_mismatch"] is True
    assert policy["cash_threshold_usd"] == pytest.approx(100.0)
    assert policy["cash_threshold_bps"] == pytest.approx(10.0)


def test_reconcile_policy_override(runner_module) -> None:
    cfg = {
        "policy": {
            "reconciliation": {
                "halt_on_mismatch": False,
                "cash_threshold_usd": 500.0,
                "cash_threshold_bps": 5.0,
            }
        }
    }
    policy = runner_module._reconcile_policy(cfg)
    assert policy["halt_on_mismatch"] is False
    assert policy["cash_threshold_usd"] == pytest.approx(500.0)
    assert policy["cash_threshold_bps"] == pytest.approx(5.0)


def test_threshold_trips_on_usd_diff(runner_module) -> None:
    policy = {"halt_on_mismatch": True, "cash_threshold_usd": 100.0, "cash_threshold_bps": 10.0}
    tripped, reason = runner_module._mismatch_exceeds_threshold(
        cash_diff=-250.0, broker_equity=1_000_000.0, policy=policy
    )
    assert tripped is True
    assert "cash_diff=$250.00" in reason


def test_threshold_trips_on_bps_only(runner_module) -> None:
    # $50 on $10k equity = 50 bps > 10 bps even though USD threshold OK.
    policy = {"halt_on_mismatch": True, "cash_threshold_usd": 100.0, "cash_threshold_bps": 10.0}
    tripped, reason = runner_module._mismatch_exceeds_threshold(
        cash_diff=50.0, broker_equity=10_000.0, policy=policy
    )
    assert tripped is True
    assert "bps" in reason


def test_threshold_quiet_when_below_both(runner_module) -> None:
    policy = {"halt_on_mismatch": True, "cash_threshold_usd": 100.0, "cash_threshold_bps": 10.0}
    tripped, reason = runner_module._mismatch_exceeds_threshold(
        cash_diff=5.0, broker_equity=1_000_000.0, policy=policy
    )
    assert tripped is False
    assert reason == ""


def test_threshold_survives_zero_equity(runner_module) -> None:
    policy = {"halt_on_mismatch": True, "cash_threshold_usd": 100.0, "cash_threshold_bps": 10.0}
    # No equity signal — USD path alone must still work.
    tripped, _ = runner_module._mismatch_exceeds_threshold(
        cash_diff=150.0, broker_equity=0.0, policy=policy
    )
    assert tripped is True
    tripped_q, _ = runner_module._mismatch_exceeds_threshold(
        cash_diff=50.0, broker_equity=0.0, policy=policy
    )
    assert tripped_q is False


def test_halt_flag_is_written_atomically(runner_module, tmp_path) -> None:
    flag_path = tmp_path / "halt_ack_required.json"
    with patch.object(runner_module, "HALT_FLAG_PATH", flag_path):
        runner_module._write_halt_flag(
            {"reason": "cash_drift", "cash_diff": 412.00, "ledger_cash": 9588.0}
        )
    assert flag_path.exists()
    data = json.loads(flag_path.read_text(encoding="utf-8"))
    assert data["reason"] == "cash_drift"
    assert data["cash_diff"] == pytest.approx(412.00)
    # No stray tmp sibling left behind.
    assert not flag_path.with_suffix(flag_path.suffix + ".tmp").exists()
