"""Operator-token access control on deactivate_kill_switch (Paket 4b / GO_LIVE C2).

deactivate_kill_switch() requires OPERATOR_KILL_TOKEN in ENV and the caller
to supply a matching token.  activate_kill_switch() has no such requirement
(low-threshold: emergency stop must never be gated).
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def _isolated_ks(monkeypatch, tmp_path):
    """Run kill-switch tests against isolated temp files."""
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_AUDIT", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_SENTINEL", str(tmp_path / ".sentinel"))
    monkeypatch.delenv("ASSEMBLED_KILL_SWITCH", raising=False)


# ---------------------------------------------------------------------------
# Case 1 — deactivate without token → denied
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_deactivate_denied_when_no_token_supplied(_isolated_ks, monkeypatch):
    """operator_token=None while ENV is set → PermissionError; switch stays engaged."""
    monkeypatch.setenv("OPERATOR_KILL_TOKEN", "secret-token")
    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
        is_kill_switch_engaged,
    )

    activate_kill_switch(reason="setup", actor="t")
    with pytest.raises(PermissionError):
        deactivate_kill_switch(reason="no-token", actor="t")  # operator_token omitted
    assert is_kill_switch_engaged(), (
        "switch must stay engaged after rejected deactivate"
    )


# ---------------------------------------------------------------------------
# Case 2 — deactivate with wrong token → denied
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_deactivate_denied_with_wrong_token(_isolated_ks, monkeypatch):
    """Wrong token → PermissionError; switch stays engaged."""
    monkeypatch.setenv("OPERATOR_KILL_TOKEN", "correct-token")
    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
        is_kill_switch_engaged,
    )

    activate_kill_switch(reason="setup", actor="t")
    with pytest.raises(PermissionError):
        deactivate_kill_switch(reason="bad", actor="t", operator_token="wrong-token")
    assert is_kill_switch_engaged(), (
        "switch must stay engaged after rejected deactivate"
    )


# ---------------------------------------------------------------------------
# Case 3 — deactivate with correct token → success
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_deactivate_succeeds_with_correct_token(_isolated_ks, monkeypatch):
    """Correct token → deactivation proceeds; switch not engaged."""
    _TOKEN = "correct-token"
    monkeypatch.setenv("OPERATOR_KILL_TOKEN", _TOKEN)
    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
        is_kill_switch_engaged,
    )

    activate_kill_switch(reason="setup", actor="t")
    deactivate_kill_switch(reason="authorized", actor="t", operator_token=_TOKEN)
    assert not is_kill_switch_engaged()


# ---------------------------------------------------------------------------
# Case 4 — activate requires no token
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_activate_requires_no_token(_isolated_ks, monkeypatch):
    """activate_kill_switch needs no operator token — emergency stop must always work."""
    monkeypatch.delenv("OPERATOR_KILL_TOKEN", raising=False)
    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        is_kill_switch_engaged,
    )

    activate_kill_switch(reason="no-token-needed", actor="t")
    assert is_kill_switch_engaged()


# ---------------------------------------------------------------------------
# Case 5 — OPERATOR_KILL_TOKEN not set in ENV → fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_deactivate_fail_closed_when_env_not_set(_isolated_ks, monkeypatch):
    """If OPERATOR_KILL_TOKEN is absent from ENV, deactivation is always denied."""
    monkeypatch.delenv("OPERATOR_KILL_TOKEN", raising=False)
    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
    )

    activate_kill_switch(reason="setup", actor="t")
    with pytest.raises(PermissionError, match="OPERATOR_KILL_TOKEN"):
        deactivate_kill_switch(reason="fail-closed", actor="t", operator_token="any")


# ---------------------------------------------------------------------------
# Case 6 — rejected attempts appear in audit log with correct action
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_deactivate_reject_written_to_audit_log(_isolated_ks, monkeypatch, tmp_path):
    """Both rejection types produce a REJECT_DEACTIVATE audit entry; chain stays valid."""
    monkeypatch.setenv("OPERATOR_KILL_TOKEN", "secret")
    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
        verify_audit_chain,
    )

    activate_kill_switch(reason="setup", actor="t")

    # Wrong token → reject
    with pytest.raises(PermissionError):
        deactivate_kill_switch(reason="bad", actor="t", operator_token="wrong")

    audit_path = tmp_path / "audit.jsonl"
    lines = [
        json.loads(ln)
        for ln in audit_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    actions = [rec["action"] for rec in lines]
    assert "REJECT_DEACTIVATE" in actions

    # Chain integrity must be preserved even after rejected attempts
    ok, n = verify_audit_chain(audit_path)
    assert ok is True
    assert n >= 2  # ACTIVATE + REJECT_DEACTIVATE


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_deactivate_denied_with_empty_string_token(_isolated_ks, monkeypatch):
    """operator_token='' (empty string) is rejected even when ENV is set."""
    monkeypatch.setenv("OPERATOR_KILL_TOKEN", "real-token")
    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
        is_kill_switch_engaged,
    )

    activate_kill_switch(reason="setup", actor="t")
    with pytest.raises(PermissionError):
        deactivate_kill_switch(reason="empty", actor="t", operator_token="")
    assert is_kill_switch_engaged()


@pytest.mark.fast
def test_deactivate_fail_closed_operator_token_none_env_absent(
    _isolated_ks, monkeypatch
):
    """operator_token=None + ENV absent → PermissionError (fail-closed, default call signature)."""
    monkeypatch.delenv("OPERATOR_KILL_TOKEN", raising=False)
    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
    )

    activate_kill_switch(reason="setup", actor="t")
    with pytest.raises(PermissionError, match="OPERATOR_KILL_TOKEN"):
        deactivate_kill_switch(
            reason="default-call", actor="t"
        )  # operator_token defaults to None


@pytest.mark.fast
def test_env_absent_path_also_writes_reject_audit(_isolated_ks, monkeypatch, tmp_path):
    """ENV-absent rejection (not just wrong-token) writes REJECT_DEACTIVATE + chain stays valid."""
    monkeypatch.delenv("OPERATOR_KILL_TOKEN", raising=False)
    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
        verify_audit_chain,
    )

    activate_kill_switch(reason="setup", actor="t")
    with pytest.raises(PermissionError):
        deactivate_kill_switch(reason="env-absent", actor="t", operator_token="any")

    audit_path = tmp_path / "audit.jsonl"
    lines = [
        json.loads(ln)
        for ln in audit_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    actions = [rec["action"] for rec in lines]
    assert "REJECT_DEACTIVATE" in actions

    ok, _ = verify_audit_chain(audit_path)
    assert ok is True
