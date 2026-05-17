"""Tests for Evidence Engine — action gate."""

from __future__ import annotations
from datetime import datetime, timezone
import pytest
from src.assembled_core.events.evidence_engine import (
    EvidenceGrade,
    check_evidence_grade_gate,
)
from src.assembled_core.events.crisis_alpha.gates import (
    check_evidence_grade_gate_from_ctx,
)
from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext

_NOW = datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.fast
@pytest.mark.fast
class TestActionGate:
    def test_grade_a_allows_active_with_b_required(self):
        ok, reason = check_evidence_grade_gate(EvidenceGrade.A, require_for_active="B")
        assert ok is True
        assert "OK" in reason

    def test_grade_b_allows_active_with_b_required(self):
        ok, reason = check_evidence_grade_gate(EvidenceGrade.B, require_for_active="B")
        assert ok is True

    def test_grade_c_blocks_active_with_b_required(self):
        ok, reason = check_evidence_grade_gate(EvidenceGrade.C, require_for_active="B")
        assert ok is False
        assert "BLOCKED" in reason

    def test_grade_d_blocks_active_with_b_required(self):
        ok, reason = check_evidence_grade_gate(EvidenceGrade.D, require_for_active="B")
        assert ok is False

    def test_grade_c_allows_active_with_c_required(self):
        ok, reason = check_evidence_grade_gate(EvidenceGrade.C, require_for_active="C")
        assert ok is True

    def test_grade_a_required_blocks_b(self):
        ok, reason = check_evidence_grade_gate(EvidenceGrade.B, require_for_active="A")
        assert ok is False


@pytest.mark.fast
@pytest.mark.fast
class TestEvidenceGradeGateFromCtx:
    def _make_ctx(self, evidence_grade=None):
        ctx = CrisisAlphaContext(
            timestamp_utc=_NOW,
            health_ok=True,
            social_only=False,
            news_trigger_items=[{"severity": 2}],
            geo_sources=3,
            market_stress_ok=True,
            geo_score=2,
        )
        if evidence_grade is not None:
            ctx.evidence_grade = evidence_grade
        return ctx

    def test_no_grade_denies(self):
        # T2.5: missing evidence grade must deny, not pass (default-deny)
        ctx = self._make_ctx(evidence_grade=None)
        ok, reason = check_evidence_grade_gate_from_ctx(ctx)
        assert ok is False
        assert "DENIED" in reason

    def test_grade_a_passes(self):
        ctx = self._make_ctx(evidence_grade="A")
        ok, reason = check_evidence_grade_gate_from_ctx(ctx)
        assert ok is True

    def test_grade_b_passes_with_b_required(self):
        ctx = self._make_ctx(evidence_grade="B")
        ok, reason = check_evidence_grade_gate_from_ctx(ctx, require_for_active="B")
        assert ok is True

    def test_grade_c_blocks_with_b_required(self):
        ctx = self._make_ctx(evidence_grade="C")
        ok, reason = check_evidence_grade_gate_from_ctx(ctx, require_for_active="B")
        assert ok is False

    def test_grade_d_blocks(self):
        ctx = self._make_ctx(evidence_grade="D")
        ok, reason = check_evidence_grade_gate_from_ctx(ctx, require_for_active="B")
        assert ok is False

    def test_unknown_grade_denies(self):
        # T2.5: unknown evidence grade must deny, not pass (default-deny)
        ctx = self._make_ctx(evidence_grade="X")
        ok, reason = check_evidence_grade_gate_from_ctx(ctx)
        assert ok is False
        assert "DENIED" in reason
