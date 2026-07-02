"""Regression guard: config-wired -10% soft drawdown stop in
``scripts/run_live_paper.py:_preflight_checks`` (2026-07-02 pilot relaunch).

Pins the behaviour that Stage-1 review flagged as uncovered:
  - a breach of ``paper_runner.dd_stop_pct`` (fraction of ``start_capital``)
    writes the ack_halt-clearable halt flag and blocks the cycle (return False),
  - equity above the stop does NOT write the flag,
  - a missing ``dd_stop_pct`` falls back to the 0.30 default (back-compat),
  - a broker ``get_account()`` failure is FAIL-CLOSED for the cycle
    (return False) WITHOUT writing the halt flag (self-recovering).

The stop is a SOFT halt (``auto_activate=False``): it must NOT engage the
OPERATOR_KILL_TOKEN-gated persistent kill switch.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_live_paper.py"
_BASELINE = 87874.90  # matches configs/app.yaml paper_runner.start_capital


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("rlp_dd_mod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeAdapter:
    """Minimal broker adapter stub for the preflight drawdown path."""

    def __init__(self, equity: float, raise_account: bool = False):
        self._equity = equity
        self._raise = raise_account

    def get_account(self) -> dict:
        if self._raise:
            raise RuntimeError("simulated broker get_account failure")
        return {"equity": self._equity}

    def get_open_orders(self):  # reached only on the no-breach path
        return []

    def cancel_all_orders(self) -> int:
        return 0


def _prep(mod, monkeypatch, tmp_path):
    """Neutralise the gates BEFORE the drawdown block so the test isolates it."""
    halt_path = tmp_path / "halt_ack_required.json"
    monkeypatch.setattr(mod, "HALT_FLAG_PATH", halt_path)
    import src.assembled_core.execution.kill_switch as ks

    monkeypatch.setattr(ks, "is_kill_switch_engaged", lambda: False)
    # A breach must NOT engage the persistent kill switch (soft-halt contract).
    monkeypatch.setattr(
        ks,
        "activate_kill_switch",
        lambda **_kw: pytest.fail("soft-halt must not engage the kill switch"),
    )
    return halt_path


def test_preflight_dd_breach_writes_halt_flag_and_blocks(tmp_path, monkeypatch):
    mod = _load_module()
    halt_path = _prep(mod, monkeypatch, tmp_path)
    # 79000 < 87874.90 * 0.90 = 79087.41  -> breach at dd_stop_pct=0.10
    adapter = _FakeAdapter(equity=79000.0)
    app_cfg = {"paper_runner": {"start_capital": _BASELINE, "dd_stop_pct": 0.10}}

    assert mod._preflight_checks(adapter, app_cfg) is False
    assert halt_path.exists()
    import json as _json

    payload = _json.loads(halt_path.read_text(encoding="utf-8"))
    assert "drawdown stop" in payload.get("reason", "").lower()
    assert payload.get("source") == "run_live_paper._preflight_checks.drawdown_stop"


def test_preflight_dd_above_stop_writes_no_halt_flag(tmp_path, monkeypatch):
    mod = _load_module()
    halt_path = _prep(mod, monkeypatch, tmp_path)
    # 80000 > 79087.41  -> ~-9.0%, no breach at dd_stop_pct=0.10
    adapter = _FakeAdapter(equity=80000.0)
    app_cfg = {"paper_runner": {"start_capital": _BASELINE, "dd_stop_pct": 0.10}}

    mod._preflight_checks(adapter, app_cfg)
    # The only halt-flag writer inside _preflight_checks is the drawdown branch;
    # no breach -> no flag, regardless of any downstream preflight outcome.
    assert not halt_path.exists()


def test_preflight_dd_failclosed_on_account_error(tmp_path, monkeypatch):
    mod = _load_module()
    halt_path = _prep(mod, monkeypatch, tmp_path)
    adapter = _FakeAdapter(equity=0.0, raise_account=True)
    app_cfg = {"paper_runner": {"start_capital": _BASELINE, "dd_stop_pct": 0.10}}

    # Cannot verify equity -> block THIS cycle (fail-closed) but do NOT halt,
    # so the next scheduled cycle retries once the broker read recovers.
    assert mod._preflight_checks(adapter, app_cfg) is False
    assert not halt_path.exists()


def test_preflight_dd_defaults_to_30pct_when_key_absent(tmp_path, monkeypatch):
    mod = _load_module()
    halt_path = _prep(mod, monkeypatch, tmp_path)
    # dd_stop_pct absent -> default 0.30. equity 79000 vs baseline 100000 = -21%,
    # which is NOT a breach at 30% -> back-compat preserved -> no halt flag.
    adapter = _FakeAdapter(equity=79000.0)
    app_cfg = {"paper_runner": {"start_capital": 100000.0}}

    mod._preflight_checks(adapter, app_cfg)
    assert not halt_path.exists()


def test_preflight_dd_breach_with_failed_halt_write_still_blocks(
    tmp_path, monkeypatch, caplog
):
    """F-senior-1 guard: a CONFIRMED breach whose halt-flag write FAILS must
    still block the cycle (return False) and be reported as an un-persisted
    halt (CRITICAL), NOT misclassified as the transient self-recovering skip.
    """
    import logging

    mod = _load_module()
    _prep(mod, monkeypatch, tmp_path)

    def _boom(_payload):
        raise OSError("simulated disk-full on halt-flag write")

    monkeypatch.setattr(mod, "_write_halt_flag", _boom)
    adapter = _FakeAdapter(equity=79000.0)  # breach at dd_stop_pct=0.10
    app_cfg = {"paper_runner": {"start_capital": _BASELINE, "dd_stop_pct": 0.10}}

    with caplog.at_level(logging.CRITICAL):
        assert mod._preflight_checks(adapter, app_cfg) is False

    text = caplog.text
    assert "halt-flag write" in text and "FAILED" in text
    # Must NOT be downgraded to the transient/self-recovering evaluation path.
    assert "self-recovering" not in text
