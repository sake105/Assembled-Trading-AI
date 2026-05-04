"""Tests for HistogramSnapshot + slippage_histogram + render_prometheus_text histogram path."""

from __future__ import annotations

import math


from assembled_core.ops.metrics_exporter import (
    export_metrics,
    render_prometheus_text,
    slippage_histogram,
)


def test_slippage_histogram_bucket_counts():
    obs = [-15, -3, 0, 1, 8, 25]
    h = slippage_histogram(obs)
    assert h.count == 6
    assert math.isclose(h.sum, sum(obs))
    # -15 falls in bucket -10 (first b >= -15 is -10)
    assert h.buckets[-10] >= 1
    # 25 falls in bucket 50
    assert h.buckets[50] >= 1
    # +Inf bucket must equal total count (cumulative)
    assert h.buckets[float("inf")] == 6


def test_slippage_histogram_empty():
    h = slippage_histogram([])
    assert h.count == 0
    assert h.sum == 0.0
    assert h.buckets[float("inf")] == 0


def test_slippage_histogram_all_extreme_positive():
    h = slippage_histogram([500, 1000])
    # All values > 200, so only +Inf bucket gets them
    assert h.buckets[200] == 0
    assert h.buckets[float("inf")] == 2


def test_slippage_histogram_all_extreme_negative():
    h = slippage_histogram([-500, -300])
    # All values <= -200
    assert h.buckets[-200] == 2
    assert h.buckets[float("inf")] == 2


def test_render_prometheus_text_histogram_lines():
    h = slippage_histogram([-15, 0, 5])
    text = render_prometheus_text(
        metrics={},
        histograms={"trading_slippage_bps": h},
        labels={"strategy": "test"},
    )
    assert "# TYPE trading_slippage_bps histogram" in text
    assert "trading_slippage_bps_bucket" in text
    assert 'le="+Inf"' in text
    assert "trading_slippage_bps_sum" in text
    assert "trading_slippage_bps_count" in text


def test_render_prometheus_text_histogram_sum_count_values():
    obs = [1.0, 2.0, 3.0]
    h = slippage_histogram(obs)
    text = render_prometheus_text(metrics={}, histograms={"slippage": h})
    assert "slippage_sum" in text
    assert "slippage_count" in text
    # sum should be 6.0
    assert "6.0" in text
    # count should be 3
    assert " 3" in text


def test_render_prometheus_text_histogram_invalid_name_skipped(caplog):
    h = slippage_histogram([0])
    import logging

    with caplog.at_level(logging.WARNING):
        text = render_prometheus_text(metrics={}, histograms={"123-bad-name": h})
    assert "123-bad-name" not in text
    assert "skipping invalid histogram name" in caplog.text


def test_render_prometheus_text_mixed_metrics_and_histograms():
    h = slippage_histogram([0])
    text = render_prometheus_text(
        metrics={"assembled_orders_total": 42},
        histograms={"assembled_slippage_bps": h},
        labels={"env": "test"},
    )
    assert "assembled_orders_total" in text
    assert "assembled_slippage_bps_bucket" in text


def test_export_metrics_accepts_histograms(tmp_path):
    h = slippage_histogram([1, 2, 3])
    result = export_metrics(
        {"assembled_orders_total": 5},
        histograms={"assembled_slippage_bps": h},
        path=tmp_path / "test.prom",
    )
    assert result["push"]["status"] == "skipped"
    prom_text = (tmp_path / "test.prom").read_text()
    assert "assembled_slippage_bps_bucket" in prom_text
    assert "assembled_orders_total" in prom_text


def test_histogram_inf_bucket_last_in_output():
    h = slippage_histogram([0])
    text = render_prometheus_text(metrics={}, histograms={"s": h})
    lines = [ln for ln in text.splitlines() if "s_bucket" in ln]
    assert lines[-1].startswith('s_bucket{le="+Inf"}')
