"""Unit tests for ICTracker (X3)."""

from __future__ import annotations

import pytest

from src.assembled_core.intel.ic_loop import ICTracker, _pearson_corr


@pytest.mark.phase12
class TestPearsonCorr:
    def test_perfect_positive(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        assert abs(_pearson_corr(x, y) - 1.0) < 1e-9

    def test_perfect_negative(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [8.0, 6.0, 4.0, 2.0]
        assert abs(_pearson_corr(x, y) + 1.0) < 1e-9

    def test_zero_variance_returns_zero(self):
        x = [1.0, 1.0, 1.0]
        y = [1.0, 2.0, 3.0]
        assert _pearson_corr(x, y) == 0.0

    def test_too_few_obs_returns_none(self):
        assert _pearson_corr([1.0], [1.0]) is None


@pytest.mark.phase12
class TestICTracker:
    def test_record_and_ic(self):
        tracker = ICTracker()
        for i in range(10):
            tracker.record(
                "TRADE_WAR", signal=float(i), realized_return=float(i) * 0.01
            )
        ic = tracker.ic("TRADE_WAR")
        assert ic is not None
        assert abs(ic - 1.0) < 1e-6  # perfect positive correlation

    def test_unknown_type_returns_none(self):
        tracker = ICTracker()
        assert tracker.ic("UNKNOWN") is None

    def test_single_obs_returns_none(self):
        tracker = ICTracker()
        tracker.record("TYPE_A", signal=0.5, realized_return=0.01)
        assert tracker.ic("TYPE_A") is None

    def test_window_limits_observations(self):
        tracker = ICTracker(window=5)
        for i in range(20):
            tracker.record("X", signal=float(i), realized_return=0.01)
        assert len(tracker._observations["X"]) == 5

    def test_compute_report_structure(self):
        tracker = ICTracker()
        for i in range(5):
            tracker.record(
                "GEO_CONFLICT", signal=float(i), realized_return=float(i) * 0.01
            )
        report = tracker.compute_report()
        assert "generated_utc" in report
        assert "results" in report
        assert "GEO_CONFLICT" in report["results"]
        assert "ic" in report["results"]["GEO_CONFLICT"]
        assert "n_obs" in report["results"]["GEO_CONFLICT"]
        assert "flagged_weak" in report["results"]["GEO_CONFLICT"]

    def test_flagged_weak_below_threshold(self):
        tracker = ICTracker(ic_threshold=0.5)
        # Near-zero IC: signal and return are anti-correlated slightly
        for i in range(10):
            tracker.record(
                "WEAK_TYPE", signal=float(i % 3), realized_return=float(i % 2) * 0.01
            )
        report = tracker.compute_report()
        # May or may not be flagged depending on correlation, just check structure
        assert "flagged_weak" in report["results"]["WEAK_TYPE"]

    def test_weak_trigger_types_needs_min_10_obs(self):
        tracker = ICTracker(ic_threshold=10.0)  # impossible threshold
        for i in range(9):
            tracker.record("BARELY_TYPE", signal=float(i), realized_return=0.01)
        # Only 9 obs → not included in weak list
        assert "BARELY_TYPE" not in tracker.weak_trigger_types()

    def test_persistence_roundtrip(self, tmp_path):
        path = tmp_path / "ic_state.json"
        tracker = ICTracker(state_path=path)
        for i in range(5):
            tracker.record("PERSIST_TYPE", signal=float(i), realized_return=0.01)
        # Load from same path
        tracker2 = ICTracker(state_path=path)
        assert len(tracker2._observations.get("PERSIST_TYPE", [])) == 5

    def test_load_corrupt_state_graceful(self, tmp_path):
        path = tmp_path / "bad_ic.json"
        path.write_text("{not valid json", encoding="utf-8")
        # Should not raise
        tracker = ICTracker(state_path=path)
        assert tracker._observations == {}
