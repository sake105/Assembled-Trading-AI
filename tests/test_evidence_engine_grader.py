"""Tests for Evidence Engine — grader, grades, misinfo risk."""

from __future__ import annotations
import pytest
from src.assembled_core.events.evidence_engine import (
    EvidenceGrade,
    grade_evidence,
    compute_misinfo_risk,
)


@pytest.mark.fast
@pytest.mark.fast
class TestEvidenceGrade:
    def test_grade_a_allows_active(self):
        assert EvidenceGrade.A.allows_active() is True

    def test_grade_b_allows_active(self):
        assert EvidenceGrade.B.allows_active() is True

    def test_grade_c_does_not_allow_active(self):
        assert EvidenceGrade.C.allows_active() is False

    def test_grade_d_does_not_allow_active(self):
        assert EvidenceGrade.D.allows_active() is False

    def test_grade_a_allows_watch(self):
        assert EvidenceGrade.A.allows_watch() is True

    def test_grade_d_does_not_allow_watch(self):
        assert EvidenceGrade.D.allows_watch() is False


@pytest.mark.fast
@pytest.mark.fast
class TestEvidenceGrader:
    def make_evidence(self, tier_a=0, tier_b_ind=0, tier_b=0, ok=None):
        if ok is None:
            ok = tier_a >= 1 or tier_b_ind >= 2
        return {
            "tierA_count": tier_a,
            "tierB_count": tier_b,
            "tierB_independent_count": tier_b_ind,
            "evidence_ok": ok,
        }

    def test_grade_a_two_tier_a_sources(self):
        ev = self.make_evidence(tier_a=2)
        grade = grade_evidence(ev)
        assert grade == EvidenceGrade.A

    def test_grade_a_three_tier_b_independent(self):
        ev = self.make_evidence(tier_b_ind=3, tier_b=3, ok=True)
        grade = grade_evidence(ev)
        assert grade == EvidenceGrade.A

    def test_grade_a_blocked_by_high_misinfo(self):
        ev = self.make_evidence(tier_a=2)
        grade = grade_evidence(ev, misinfo_risk_score=0.75)
        assert grade != EvidenceGrade.A  # high misinfo blocks A

    def test_grade_b_one_tier_a(self):
        ev = self.make_evidence(tier_a=1, ok=True)
        grade = grade_evidence(ev)
        assert grade == EvidenceGrade.B

    def test_grade_b_two_tier_b_independent(self):
        ev = self.make_evidence(tier_b_ind=2, tier_b=2, ok=True)
        grade = grade_evidence(ev)
        assert grade == EvidenceGrade.B

    def test_grade_b_blocked_by_extreme_misinfo(self):
        ev = self.make_evidence(tier_a=1, ok=True)
        grade = grade_evidence(ev, misinfo_risk_score=0.95)
        assert grade != EvidenceGrade.B  # extreme misinfo blocks B

    def test_grade_c_one_tier_b_domain(self):
        ev = self.make_evidence(tier_b_ind=1, tier_b=1, ok=False)
        grade = grade_evidence(ev)
        assert grade == EvidenceGrade.C

    def test_grade_d_no_evidence(self):
        ev = self.make_evidence(tier_a=0, tier_b_ind=0, ok=False)
        grade = grade_evidence(ev)
        assert grade == EvidenceGrade.D

    def test_grade_d_at_grade_c_boundary(self):
        # No tier_a, no tier_b_ind => D
        ev = {
            "tierA_count": 0,
            "tierB_count": 0,
            "tierB_independent_count": 0,
            "evidence_ok": False,
        }
        grade = grade_evidence(ev)
        assert grade == EvidenceGrade.D


@pytest.mark.fast
@pytest.mark.fast
class TestMisinfoRisk:
    def make_evidence(self, tier_a=1, tier_b_ind=2, tier_b=2):
        return {
            "tierA_count": tier_a,
            "tierB_count": tier_b,
            "tierB_independent_count": tier_b_ind,
            "evidence_ok": True,
        }

    def test_no_risk_strong_evidence(self):
        ev = self.make_evidence(tier_a=2, tier_b_ind=3, tier_b=3)
        score = compute_misinfo_risk(ev, social_only=False)
        assert score < 0.30

    def test_social_only_high_risk(self):
        ev = self.make_evidence(tier_a=0, tier_b_ind=0, tier_b=0)
        score = compute_misinfo_risk(ev, social_only=True)
        assert score >= 0.70

    def test_no_tier_a_raises_risk(self):
        ev = self.make_evidence(tier_a=0, tier_b_ind=2, tier_b=2)
        score = compute_misinfo_risk(ev, social_only=False)
        assert score > 0.0

    def test_burst_adds_risk(self):
        ev = self.make_evidence(tier_a=1)
        score_normal = compute_misinfo_risk(
            ev, event_count=5, burst_window_minutes=30.0
        )
        score_burst = compute_misinfo_risk(ev, event_count=5, burst_window_minutes=2.0)
        assert score_burst > score_normal

    def test_single_domain_adds_risk(self):
        ev_single = self.make_evidence(tier_a=0, tier_b_ind=1, tier_b=5)
        ev_multi = self.make_evidence(tier_a=0, tier_b_ind=4, tier_b=5)
        score_single = compute_misinfo_risk(ev_single)
        score_multi = compute_misinfo_risk(ev_multi)
        assert score_single > score_multi

    def test_clamped_at_one(self):
        ev = self.make_evidence(tier_a=0, tier_b_ind=0, tier_b=0)
        score = compute_misinfo_risk(
            ev, social_only=True, event_count=10, burst_window_minutes=1.0
        )
        assert score <= 1.0

    def test_score_is_float(self):
        ev = self.make_evidence()
        score = compute_misinfo_risk(ev)
        assert isinstance(score, float)
