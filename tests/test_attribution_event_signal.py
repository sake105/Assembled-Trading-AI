"""Tests for T6.4 compute_event_signal_attribution and T6.6 SourceUptimeTracker latency."""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.accounting.attribution import compute_event_signal_attribution
from src.assembled_core.intel.health_monitor import SourceUptimeTracker


@pytest.mark.fast
class TestEventSignalAttribution:
    def _make_fills(self, buckets: list[str], cost_bps: float = 5.0) -> pd.DataFrame:
        rows = []
        for i, b in enumerate(buckets):
            rows.append(
                {
                    "symbol": f"SYM{i}",
                    "event_signal_bucket": b,
                    "fill_qty": 100.0,
                    "fill_price": 50.0,
                    "total_cost_bps": cost_bps,
                }
            )
        return pd.DataFrame(rows)

    def test_basic_aggregation(self):
        fills = self._make_fills(["news_geo", "news_geo", "disclosure_form4"])
        result = compute_event_signal_attribution(fills)
        assert "event_signal_bucket" in result.columns
        assert set(result["event_signal_bucket"]) == {"news_geo", "disclosure_form4"}
        news_row = result[result["event_signal_bucket"] == "news_geo"].iloc[0]
        assert news_row["n_fills"] == 2

    def test_missing_column_returns_empty(self):
        fills = pd.DataFrame([{"symbol": "AAPL", "fill_qty": 100, "fill_price": 50}])
        result = compute_event_signal_attribution(fills)
        assert result.empty

    def test_empty_fills(self):
        result = compute_event_signal_attribution(pd.DataFrame())
        assert result.empty

    def test_cost_bps_aggregated(self):
        fills = self._make_fills(["bucket_a", "bucket_a"], cost_bps=10.0)
        result = compute_event_signal_attribution(fills)
        row = result[result["event_signal_bucket"] == "bucket_a"].iloc[0]
        assert row["total_cost_bps"] > 0

    def test_custom_bucket_column(self):
        fills = pd.DataFrame(
            [
                {
                    "symbol": "X",
                    "my_bucket": "geo_conflict",
                    "fill_qty": 50.0,
                    "fill_price": 100.0,
                    "total_cost_bps": 3.0,
                }
            ]
        )
        result = compute_event_signal_attribution(fills, bucket_column="my_bucket")
        assert not result.empty
        assert result.iloc[0]["event_signal_bucket"] == "geo_conflict"


@pytest.mark.fast
class TestSourceUptimeTrackerLatency:
    def test_latency_history_tracked(self):
        tracker = SourceUptimeTracker()
        for ms in [10.0, 20.0, 30.0]:
            tracker.record("src_a", success=True, latency_ms=ms)
        s = tracker._sources["src_a"]
        assert s["latency_history"] == [10.0, 20.0, 30.0]

    def test_p95_latency(self):
        tracker = SourceUptimeTracker()
        for ms in range(1, 101):
            tracker.record("src_b", success=True, latency_ms=float(ms))
        p95 = tracker.p95_latency_ms("src_b")
        assert p95 is not None
        assert 90.0 <= p95 <= 100.0

    def test_p95_no_data_returns_none(self):
        tracker = SourceUptimeTracker()
        assert tracker.p95_latency_ms("missing") is None

    def test_queue_depth_tracked(self):
        tracker = SourceUptimeTracker()
        tracker.record("src_c", success=True, queue_depth=42)
        assert tracker._sources["src_c"]["queue_depth"] == 42

    def test_snapshot_includes_p95(self):
        tracker = SourceUptimeTracker()
        tracker.record("src_d", success=True, latency_ms=55.0)
        snap = tracker.snapshot()
        assert "p95_latency_ms" in snap["src_d"]

    def test_latency_history_window_capped(self):
        tracker = SourceUptimeTracker(window=5)
        for ms in range(20):
            tracker.record("src_e", success=True, latency_ms=float(ms))
        assert len(tracker._sources["src_e"]["latency_history"]) == 5
