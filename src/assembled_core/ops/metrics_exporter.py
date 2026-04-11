"""Metrics exporter — Prometheus text format + optional push-gateway.

Sprint 4 / Plan C15. A tiny metrics exporter that writes Prometheus
text-format output to a file (which a scraper or sidecar can ship)
or pushes it to a Prometheus Pushgateway URL if one is configured.

Design constraints:

* no mandatory dependency on ``prometheus_client`` — the text format
  is small enough to emit by hand
* stateless across runs: every call accepts a full snapshot of
  ``{metric_name: value}`` plus optional labels; history is not tracked
* degrades gracefully: missing URL → file-only, network error → logged
  but never raised
* Prometheus-compatible naming is enforced by a simple regex

Typical use from the runner:

    snapshot = {
        "assembled_invested_pct": 0.82,
        "assembled_drawdown_pct": -0.03,
        "assembled_orders_generated_total": 17,
    }
    export_metrics(snapshot, labels={"strategy": "multifactor_v1"})

"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_METRIC_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_DEFAULT_PATH = Path("output") / "metrics" / "assembled.prom"


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_labels(labels: dict[str, Any] | None) -> str:
    if not labels:
        return ""
    parts: list[str] = []
    for k, v in labels.items():
        if not _LABEL_NAME_RE.match(str(k)):
            continue
        parts.append(f'{k}="{_escape_label_value(str(v))}"')
    if not parts:
        return ""
    return "{" + ",".join(parts) + "}"


def render_prometheus_text(
    metrics: dict[str, float | int],
    *,
    labels: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> str:
    """Render a metrics snapshot in Prometheus text format.

    Invalid metric names are dropped with a warning rather than raising,
    so a typo in one metric can't break the whole push.
    """
    ts = now or datetime.now(timezone.utc)
    header = f"# Exported {ts.isoformat()}\n"
    label_block = _format_labels(labels)

    lines: list[str] = [header]
    for raw_name, raw_value in metrics.items():
        name = str(raw_name)
        if not _METRIC_NAME_RE.match(name):
            logger.warning("[metrics_exporter] skipping invalid name %r", name)
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            logger.warning("[metrics_exporter] skipping non-numeric %s=%r", name, raw_value)
            continue
        lines.append(f"{name}{label_block} {value}")
    return "\n".join(lines) + "\n"


def write_metrics_file(
    text: str,
    path: str | Path | None = None,
) -> Path:
    """Write the rendered text to ``path``. Overwrites."""
    p = Path(path or _DEFAULT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def push_to_gateway(
    text: str,
    gateway_url: str,
    job: str = "assembled_trading",
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """POST text-format metrics to a Prometheus Pushgateway.

    Returns a result dict with ``status`` and either ``http_status`` or
    ``reason``. Never raises.
    """
    try:
        import requests
    except Exception as exc:  # noqa: BLE001
        logger.warning("[metrics_exporter] requests not available: %s", exc)
        return {"status": "error", "reason": "requests_missing"}

    url = gateway_url.rstrip("/") + f"/metrics/job/{job}"
    try:
        resp = requests.post(
            url,
            data=text.encode("utf-8"),
            headers={"Content-Type": "text/plain; version=0.0.4"},
            timeout=timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[metrics_exporter] push failed: %s", exc)
        return {"status": "error", "reason": f"post_failed:{exc!r}"}

    if 200 <= resp.status_code < 300:
        return {"status": "sent", "http_status": resp.status_code}
    return {"status": "error", "http_status": resp.status_code}


def export_metrics(
    metrics: dict[str, float | int],
    *,
    labels: dict[str, Any] | None = None,
    path: str | Path | None = None,
    gateway_url: str | None = None,
    gateway_url_env: str = "ASSEMBLED_PROM_PUSHGATEWAY_URL",
    job: str = "assembled_trading",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Render + write + optionally push metrics. Returns a result dict.

    Gateway URL resolution:
    1. explicit ``gateway_url`` argument
    2. ``os.environ[gateway_url_env]``
    3. None → file-only export
    """
    text = render_prometheus_text(metrics, labels=labels, now=now)
    written = write_metrics_file(text, path=path)

    url = gateway_url or os.environ.get(gateway_url_env, "").strip() or None
    push_result: dict[str, Any]
    if url:
        push_result = push_to_gateway(text, url, job=job)
    else:
        push_result = {"status": "skipped", "reason": "no_gateway_url"}

    return {
        "file": str(written),
        "bytes": len(text),
        "metrics_count": text.count("\n") - 1,  # minus header
        "push": push_result,
    }


__all__ = [
    "render_prometheus_text",
    "write_metrics_file",
    "push_to_gateway",
    "export_metrics",
]
