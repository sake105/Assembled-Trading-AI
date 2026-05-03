"""Tests for NewsDecay."""

from __future__ import annotations


import pytest

from src.assembled_core.intel.news_decay import DecayProfile, NewsDecay


@pytest.mark.phase12
class TestNewsDecay:
    def test_t0_is_full_impact(self):
        d = NewsDecay()
        assert d.impact_remaining("sanctions", 0) == 1.0
        assert d.impact_remaining("sanctions", -5) == 1.0

    def test_exponential_halves_at_half_life(self):
        d = NewsDecay()
        prof = d.profile("sanctions")
        assert prof.kind == "exponential"
        frac = d.impact_remaining("sanctions", prof.parameter_min)
        assert frac == pytest.approx(0.5, abs=1e-6)

    def test_exponential_quarters_at_double_half_life(self):
        d = NewsDecay()
        prof = d.profile("sanctions")
        frac = d.impact_remaining("sanctions", 2 * prof.parameter_min)
        assert frac == pytest.approx(0.25, abs=1e-6)

    def test_linear_zero_at_lifetime(self):
        d = NewsDecay()
        prof = d.profile("market_stress")
        assert prof.kind == "linear"
        assert d.impact_remaining("market_stress", prof.parameter_min) == 0.0

    def test_linear_halfway(self):
        d = NewsDecay()
        prof = d.profile("market_stress")
        frac = d.impact_remaining("market_stress", prof.parameter_min / 2)
        assert frac == pytest.approx(0.5, abs=1e-6)

    def test_unknown_event_uses_default(self):
        d = NewsDecay()
        prof = d.profile("something_exotic")
        # At one half-life the fraction must be exactly 0.5 — exercise the
        # current default instead of a hard-coded number that drifts with
        # decay-table tuning.
        frac = d.impact_remaining("something_exotic", prof.parameter_min)
        assert frac == pytest.approx(0.5, abs=1e-6)

    def test_overrides(self):
        d = NewsDecay(overrides={"earnings": DecayProfile("linear", 30)})
        assert d.profile("earnings").kind == "linear"
        assert d.impact_remaining("earnings", 15) == pytest.approx(0.5)

    def test_scale_bps(self):
        d = NewsDecay()
        # 100 bps after 1 half-life of sanctions → 50 bps
        prof = d.profile("sanctions")
        scaled = d.scale_bps("sanctions", 100.0, prof.parameter_min)
        assert scaled == pytest.approx(50.0, abs=1e-3)

    def test_very_old_event_nearly_zero(self):
        d = NewsDecay()
        # Compute adaptively: 25 half-lives always lands at ~3e-8.
        prof = d.profile("sanctions")
        frac = d.impact_remaining("sanctions", prof.parameter_min * 25)
        assert frac < 1e-6
