"""Tests for ops/metrics_exporter.py (Sprint 4 / Plan C15)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.ops.metrics_exporter import (  # noqa: E402
    export_metrics,
    push_to_gateway,
    render_prometheus_text,
    write_metrics_file,
)


def test_render_basic_metric() -> None:
    text = render_prometheus_text({"assembled_invested_pct": 0.82})
    assert "assembled_invested_pct 0.82" in text
    assert text.startswith("# Exported ")


def test_render_with_labels() -> None:
    text = render_prometheus_text(
        {"assembled_orders_total": 17},
        labels={"strategy": "multifactor_v1", "env": "paper"},
    )
    # label order not guaranteed in dict, so check substrings
    assert 'strategy="multifactor_v1"' in text
    assert 'env="paper"' in text
    assert "assembled_orders_total" in text
    assert "17.0" in text


def test_render_skips_invalid_metric_name() -> None:
    text = render_prometheus_text(
        {"good_metric": 1.0, "bad metric with space": 2.0}
    )
    assert "good_metric" in text
    assert "bad metric" not in text


def test_render_skips_non_numeric_value() -> None:
    text = render_prometheus_text({"good": 1.0, "bad": "not_a_number"})  # type: ignore[dict-item]
    assert "good 1.0" in text
    assert "bad " not in text


def test_render_escapes_label_value_quotes() -> None:
    text = render_prometheus_text(
        {"m": 1.0},
        labels={"k": 'quote"inside'},
    )
    assert 'k="quote\\"inside"' in text


def test_write_metrics_file(tmp_path: Path) -> None:
    p = tmp_path / "m.prom"
    text = "# hi\nfoo 1\n"
    result = write_metrics_file(text, path=p)
    assert result == p
    assert p.read_text(encoding="utf-8") == text


def test_export_metrics_no_gateway(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ASSEMBLED_PROM_PUSHGATEWAY_URL", raising=False)
    out = export_metrics(
        {"assembled_invested_pct": 0.82},
        labels={"strategy": "multifactor_v1"},
        path=tmp_path / "m.prom",
    )
    assert out["push"]["status"] == "skipped"
    assert out["metrics_count"] >= 1
    content = (tmp_path / "m.prom").read_text(encoding="utf-8")
    assert "assembled_invested_pct" in content


def test_export_metrics_with_gateway(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ASSEMBLED_PROM_PUSHGATEWAY_URL", "http://example.invalid:9091")

    captured: dict = {}

    class _Resp:
        status_code = 202

    class _FakeRequests:
        def post(self, url, data, headers, timeout):
            captured["url"] = url
            captured["data"] = data
            captured["headers"] = headers
            captured["timeout"] = timeout
            return _Resp()

    monkeypatch.setitem(sys.modules, "requests", _FakeRequests())

    out = export_metrics(
        {"assembled_invested_pct": 0.75},
        path=tmp_path / "m.prom",
        job="unit_test_job",
    )
    assert out["push"]["status"] == "sent"
    assert out["push"]["http_status"] == 202
    assert "unit_test_job" in captured["url"]
    assert b"assembled_invested_pct" in captured["data"]


def test_push_to_gateway_handles_post_exception(monkeypatch) -> None:
    class _FakeRequests:
        def post(self, *a, **kw):
            raise RuntimeError("network down")

    monkeypatch.setitem(sys.modules, "requests", _FakeRequests())
    result = push_to_gateway("foo 1\n", "http://example.invalid:9091")
    assert result["status"] == "error"
    assert "post_failed" in result["reason"]


def test_push_to_gateway_non_2xx(monkeypatch) -> None:
    class _Resp:
        status_code = 500

    class _FakeRequests:
        def post(self, *a, **kw):
            return _Resp()

    monkeypatch.setitem(sys.modules, "requests", _FakeRequests())
    result = push_to_gateway("foo 1\n", "http://example.invalid:9091")
    assert result["status"] == "error"
    assert result["http_status"] == 500
