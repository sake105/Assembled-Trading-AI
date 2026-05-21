"""F-RX-8 §9.12 (f) regression guard: soft-timeout in scripts/run_live_paper.

Before the Task Scheduler PT30M ExecutionTimeLimit hard-kills the process
(potentially mid-order, leaving stale pending intents like the 2026-05-19
incident), an in-process Timer should write the halt-ack flag and flip a
gate so the main flow exits gracefully at the next checkpoint.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_live_paper.py"


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("rlp_mod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_arm_soft_timeout_fires_and_writes_halt_flag(tmp_path, monkeypatch):
    """Timer fires → halt-ack flag written + in-process gate flipped."""
    mod = _load_module()
    # Reset module state in case a prior test tripped it
    mod._SOFT_TIMEOUT_TRIPPED["flag"] = False
    halt_path = tmp_path / "halt_ack_required.json"
    monkeypatch.setattr(mod, "HALT_FLAG_PATH", halt_path)

    # 0.05s timeout for fast test
    timer = mod._arm_soft_timeout(0.05)
    time.sleep(0.2)  # let the timer fire
    timer.cancel()

    assert mod._SOFT_TIMEOUT_TRIPPED["flag"] is True
    assert halt_path.exists()
    import json as _json

    payload = _json.loads(halt_path.read_text(encoding="utf-8"))
    assert "soft-timeout" in payload.get("reason", "").lower()


def test_check_soft_timeout_no_op_when_not_tripped():
    mod = _load_module()
    mod._SOFT_TIMEOUT_TRIPPED["flag"] = False
    # Must NOT raise SystemExit
    mod._check_soft_timeout("stage_x")


def test_check_soft_timeout_exits_when_tripped():
    mod = _load_module()
    mod._SOFT_TIMEOUT_TRIPPED["flag"] = True
    try:
        with pytest.raises(SystemExit) as excinfo:
            mod._check_soft_timeout("stage_y")
        assert excinfo.value.code == 2
    finally:
        # Reset for other tests
        mod._SOFT_TIMEOUT_TRIPPED["flag"] = False


def test_arm_soft_timeout_cancel_prevents_fire(tmp_path, monkeypatch):
    """Cancelling the timer before it fires must not trip the gate."""
    mod = _load_module()
    mod._SOFT_TIMEOUT_TRIPPED["flag"] = False
    halt_path = tmp_path / "halt_ack_required.json"
    monkeypatch.setattr(mod, "HALT_FLAG_PATH", halt_path)

    timer = mod._arm_soft_timeout(2.0)  # generous timeout
    timer.cancel()
    time.sleep(0.1)

    assert mod._SOFT_TIMEOUT_TRIPPED["flag"] is False
    assert not halt_path.exists()
