"""Unit tests for ops._paper_runner_gates.apply_reconcile_block_gate.

The reconcile-block gate is a SAFETY pre-trade seam (FU-1 / next-cycle
blocking). It reads the durable ``reconcile_latest.json`` artifact
(schema ``run.reconcile.v1``) and, when ARMED, refuses to let the next
paper/live cycle trade unless it can positively prove the last reconcile
passed. Default-off must be a pure pass-through; ARMED must be
fail-closed on FAIL / unverified / missing / unreadable.

These tests DISCRIMINATE: a wrong default (armed-when-disabled), a
fail-open on a missing/unreadable artifact, or a not-blocked on FAIL
each flips an assertion below.

Mirrors the _StubCtx + tmp_path style of tests/test_wave20_wiring.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.assembled_core.ops._paper_runner_gates import (
    ReconcileDecision,
    apply_reconcile_block_gate,
)


@dataclass
class _StubCtx:
    reconcile_gate_state: dict[str, Any] | None = None


def _write_artifact(
    root: Path, status: str, *, generated_utc: str | None = None
) -> Path:
    """Write output/reconcile_latest.json under root with the given status."""
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reconcile_latest.json"
    report: dict[str, Any] = {
        "schema_version": "run.reconcile.v1",
        "generated_utc": generated_utc or datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# (a) disabled -> pure pass-through regardless of artifact contents
# ---------------------------------------------------------------------------


def test_disabled_is_passthrough_even_with_fail_artifact(tmp_path: Path) -> None:
    # A FAIL artifact is present, but the gate is OFF -> must NOT block and
    # must NOT touch ctx (byte-identical no-op).
    _write_artifact(tmp_path, "FAIL")
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(
        ctx, paper_cfg={"reconcile_block": {"enabled": False}}, root=tmp_path
    )
    assert isinstance(out, ReconcileDecision)
    assert out.blocked is False
    assert out.reason == ""
    assert ctx.reconcile_gate_state is None  # untouched


def test_disabled_when_key_absent_is_passthrough(tmp_path: Path) -> None:
    # No reconcile_block config at all -> default off -> pass-through.
    _write_artifact(tmp_path, "FAIL")
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(ctx, paper_cfg={}, root=tmp_path)
    assert out.blocked is False
    assert ctx.reconcile_gate_state is None


# ---------------------------------------------------------------------------
# (b) armed + FAIL -> blocked reconcile_fail
# ---------------------------------------------------------------------------


def test_armed_fail_blocks(tmp_path: Path) -> None:
    _write_artifact(tmp_path, "FAIL")
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(
        ctx, paper_cfg={"reconcile_block": {"enabled": True}}, root=tmp_path
    )
    assert out.blocked is True
    assert out.reason == "reconcile_fail"
    assert out.status == "FAIL"
    assert ctx.reconcile_gate_state is not None
    assert ctx.reconcile_gate_state["blocked"] is True


def test_armed_fail_blocks_regardless_of_block_on(tmp_path: Path) -> None:
    # FAIL must ALWAYS block when armed, even if block_on omits "fail".
    _write_artifact(tmp_path, "FAIL")
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(
        ctx,
        paper_cfg={"reconcile_block": {"enabled": True, "block_on": ["unverified"]}},
        root=tmp_path,
    )
    assert out.blocked is True
    assert out.reason == "reconcile_fail"


# ---------------------------------------------------------------------------
# (c) armed + OK -> not blocked
# ---------------------------------------------------------------------------


def test_armed_ok_not_blocked(tmp_path: Path) -> None:
    _write_artifact(tmp_path, "OK")
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(
        ctx, paper_cfg={"reconcile_block": {"enabled": True}}, root=tmp_path
    )
    assert out.blocked is False
    assert out.reason == ""
    assert out.status == "OK"
    assert ctx.reconcile_gate_state is not None
    assert ctx.reconcile_gate_state["blocked"] is False


# ---------------------------------------------------------------------------
# (d) armed + artifact missing -> fail-closed reconcile_unverified
# ---------------------------------------------------------------------------


def test_armed_missing_artifact_fail_closed(tmp_path: Path) -> None:
    # No artifact written at all.
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(
        ctx, paper_cfg={"reconcile_block": {"enabled": True}}, root=tmp_path
    )
    assert out.blocked is True
    assert out.reason == "reconcile_unverified"


# ---------------------------------------------------------------------------
# (e) armed + malformed JSON -> fail-closed reconcile_unverified
# ---------------------------------------------------------------------------


def test_armed_malformed_json_fail_closed(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reconcile_latest.json").write_text("{not valid json", encoding="utf-8")
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(
        ctx, paper_cfg={"reconcile_block": {"enabled": True}}, root=tmp_path
    )
    assert out.blocked is True
    assert out.reason == "reconcile_unverified"


def test_armed_no_status_field_fail_closed(tmp_path: Path) -> None:
    # Well-formed JSON but missing the status field -> cannot prove pass.
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reconcile_latest.json").write_text(
        json.dumps({"schema_version": "run.reconcile.v1"}), encoding="utf-8"
    )
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(
        ctx, paper_cfg={"reconcile_block": {"enabled": True}}, root=tmp_path
    )
    assert out.blocked is True
    assert out.reason == "reconcile_unverified"


def _write_raw_artifact(root: Path, report: dict[str, Any]) -> Path:
    """Write output/reconcile_latest.json verbatim (no status coercion)."""
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reconcile_latest.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def test_armed_empty_status_string_fail_closed(tmp_path: Path) -> None:
    # status present but EMPTY ("") — a partial-write / schema-drift artifact.
    # Pre-fix: the ``"status" not in report`` guard passes (key IS present),
    # status normalizes to "" -> falls to the "other" branch -> not blocked
    # under default block_on=["fail"] (FAIL-OPEN). Post-fix: fail-closed.
    _write_raw_artifact(tmp_path, {"schema_version": "run.reconcile.v1", "status": ""})
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(
        ctx, paper_cfg={"reconcile_block": {"enabled": True}}, root=tmp_path
    )
    assert out.blocked is True
    assert out.reason == "reconcile_unverified"
    # Contract: status is str | None — the unverified path must NOT leak the raw
    # artifact dict into the status field / audit log (F-auditor-1).
    assert out.status is None
    assert ctx.reconcile_gate_state is not None
    assert ctx.reconcile_gate_state["blocked"] is True


def test_armed_null_status_fail_closed(tmp_path: Path) -> None:
    # status present but JSON null -> report.get("status") is None.
    # Pre-fix: key present, ``None or ""`` -> "" -> "other" branch -> not
    # blocked (FAIL-OPEN). Post-fix: fail-closed reconcile_unverified.
    _write_raw_artifact(
        tmp_path, {"schema_version": "run.reconcile.v1", "status": None}
    )
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(
        ctx, paper_cfg={"reconcile_block": {"enabled": True}}, root=tmp_path
    )
    assert out.blocked is True
    assert out.reason == "reconcile_unverified"
    assert ctx.reconcile_gate_state is not None
    assert ctx.reconcile_gate_state["blocked"] is True


def test_armed_whitespace_status_fail_closed(tmp_path: Path) -> None:
    # status present but whitespace-only ("   ") -> .strip() empties it.
    # Pre-fix: key present, normalizes to "" -> "other" branch -> not blocked
    # (FAIL-OPEN). Post-fix: fail-closed reconcile_unverified.
    _write_raw_artifact(
        tmp_path, {"schema_version": "run.reconcile.v1", "status": "   "}
    )
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(
        ctx, paper_cfg={"reconcile_block": {"enabled": True}}, root=tmp_path
    )
    assert out.blocked is True
    assert out.reason == "reconcile_unverified"
    assert ctx.reconcile_gate_state is not None
    assert ctx.reconcile_gate_state["blocked"] is True


# ---------------------------------------------------------------------------
# (f) block_on ["fail","unverified"] honored vs default ["fail"]
# ---------------------------------------------------------------------------


def test_unverified_status_blocked_only_when_in_block_on(tmp_path: Path) -> None:
    # status "WARN" is an "other/unverified" value.
    _write_artifact(tmp_path, "WARN")

    # default block_on=["fail"] -> WARN does NOT block.
    ctx_default = _StubCtx()
    out_default = apply_reconcile_block_gate(
        ctx_default,
        paper_cfg={"reconcile_block": {"enabled": True}},
        root=tmp_path,
    )
    assert out_default.blocked is False
    assert out_default.reason == ""
    assert out_default.status == "WARN"

    # block_on includes "unverified" -> WARN BLOCKS with reconcile_other.
    ctx_strict = _StubCtx()
    out_strict = apply_reconcile_block_gate(
        ctx_strict,
        paper_cfg={
            "reconcile_block": {"enabled": True, "block_on": ["fail", "unverified"]}
        },
        root=tmp_path,
    )
    assert out_strict.blocked is True
    assert out_strict.reason == "reconcile_other"


# ---------------------------------------------------------------------------
# (g) block_if_stale_hours freshness guard
# ---------------------------------------------------------------------------


def test_stale_ok_blocked(tmp_path: Path) -> None:
    # OK but generated 48h ago, threshold 36h -> blocked reconcile_stale.
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    _write_artifact(tmp_path, "OK", generated_utc=old)
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(
        ctx,
        paper_cfg={"reconcile_block": {"enabled": True, "block_if_stale_hours": 36}},
        root=tmp_path,
    )
    assert out.blocked is True
    assert out.reason == "reconcile_stale"


def test_fresh_ok_not_blocked_with_stale_guard(tmp_path: Path) -> None:
    # OK generated 1h ago, threshold 36h -> not blocked.
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _write_artifact(tmp_path, "OK", generated_utc=recent)
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(
        ctx,
        paper_cfg={"reconcile_block": {"enabled": True, "block_if_stale_hours": 36}},
        root=tmp_path,
    )
    assert out.blocked is False
    assert out.reason == ""


def test_stale_guard_disabled_by_default_keeps_old_ok(tmp_path: Path) -> None:
    # Without block_if_stale_hours, an old OK is still NOT blocked.
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    _write_artifact(tmp_path, "OK", generated_utc=old)
    ctx = _StubCtx()
    out = apply_reconcile_block_gate(
        ctx, paper_cfg={"reconcile_block": {"enabled": True}}, root=tmp_path
    )
    assert out.blocked is False
    assert out.reason == ""
