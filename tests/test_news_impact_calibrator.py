"""Tests for ImpactCalibrator."""

from __future__ import annotations

import pytest

from src.assembled_core.intel.news_impact_calibrator import ImpactCalibrator


@pytest.mark.phase12
class TestImpactCalibrator:
    def test_empty_report(self):
        cal = ImpactCalibrator()
        assert cal.report() == {}

    def test_sparse_samples_filtered(self):
        cal = ImpactCalibrator(min_samples_for_report=5)
        cal.observe("sanctions", -50.0, -80.0)
        assert cal.report() == {}

    def test_include_sparse(self):
        cal = ImpactCalibrator(min_samples_for_report=5)
        cal.observe("sanctions", -50.0, -80.0)
        r = cal.report(include_sparse=True)
        assert "sanctions" in r
        assert r["sanctions"].n == 1

    def test_running_mean_matches(self):
        cal = ImpactCalibrator(min_samples_for_report=1)
        samples = [(-50.0, -100.0), (-40.0, -60.0), (-30.0, -10.0)]
        for p, r in samples:
            cal.observe("war", p, r)
        entry = cal.report()["war"]
        assert entry.n == 3
        assert entry.mean_pred_bps == pytest.approx(-40.0, abs=1e-2)
        assert entry.mean_realised_bps == pytest.approx(
            sum(r for _, r in samples) / 3, abs=1e-2
        )

    def test_bias_positive_when_underestimating_bearish_magnitude(self):
        # realised worse than predicted → mean_real < mean_pred → bias negative
        cal = ImpactCalibrator(min_samples_for_report=1)
        for _ in range(5):
            cal.observe("sanctions", -50.0, -100.0)
        entry = cal.report()["sanctions"]
        assert entry.bias_bps < 0  # we over-predicted (pred=-50, real=-100 means real more negative)

    def test_recommend_prior_adjustment_gated(self):
        cal = ImpactCalibrator(min_samples_for_report=5)
        cal.observe("rate_cut", 20.0, 50.0)
        assert cal.recommend_prior_adjustment("rate_cut") == 0.0

    def test_recommend_prior_adjustment_capped(self):
        cal = ImpactCalibrator(min_samples_for_report=1)
        for _ in range(10):
            cal.observe("super_spike", 10.0, 10_000.0)
        assert cal.recommend_prior_adjustment("super_spike") == 200.0

    def test_save_and_load_roundtrip(self, tmp_path):
        cal = ImpactCalibrator(min_samples_for_report=1)
        for _ in range(3):
            cal.observe("tariff", -20.0, -35.0)
        p = tmp_path / "cal.json"
        cal.save(p)

        cal2 = ImpactCalibrator(min_samples_for_report=1)
        cal2.load(p)
        r = cal2.report()
        assert r["tariff"].n == 3
        assert r["tariff"].mean_realised_bps == pytest.approx(-35.0)
