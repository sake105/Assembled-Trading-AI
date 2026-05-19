"""Audit trail for trading cycle decisions — Item 102.

Appends one JSON line per decision to ``output/audit/trading_decisions.jsonl``.
Designed to be safe to call even if the caller has an error:

- All exceptions are caught and logged (never re-raised).
- The output file is opened in append mode; writes are atomic at the OS level
  for lines < PIPE_BUF (4096 bytes on Linux) — sufficient for our records.
- If the output directory does not exist it is created automatically.

Usage::

    from src.assembled_core.ops.audit_trail import log_trade_decision

    log_trade_decision(
        symbol="AAPL",
        signal_score=0.87,
        sizing_cap_hit=False,
        edcl_trigger=True,
        order_type="market_buy",
        reasoning={"top_factors": ["momentum", "news_sentiment"], "conviction": 0.87},
    )
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Output path — relative to repo root. Override via AUDIT_TRAIL_PATH env var.
# parents[3] from src/assembled_core/ops/audit_trail.py = repo root.
# (parents[4] was off-by-one; the audit trail was being written to the repo's
# parent directory whenever AUDIT_TRAIL_PATH was unset. Fixed in 2026-05-19
# external-data audit, same pattern as rss_fetcher.py.)
_DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3] / "output" / "audit" / "trading_decisions.jsonl"
)


def _get_output_path() -> Path:
    override = os.environ.get("AUDIT_TRAIL_PATH", "").strip()
    return Path(override) if override else _DEFAULT_OUTPUT


def log_trade_decision(
    symbol: str,
    signal_score: float,
    sizing_cap_hit: bool,
    edcl_trigger: bool,
    order_type: str,
    reasoning: dict[str, Any] | None = None,
    *,
    as_of: str | None = None,
    run_id: str | None = None,
) -> None:
    """Append one trading decision record to the append-only audit trail.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        signal_score: Composite signal / conviction score [0.0, 1.0].
        sizing_cap_hit: True if position sizing was capped by a risk rule.
        edcl_trigger: True if EDCL event-driven sizing was active.
        order_type: Free-form string describing the order (e.g. "market_buy",
            "market_sell", "no_trade_conviction_too_low").
        reasoning: Optional dict with additional context (top factors, conviction
            breakdown, regime, etc.).  Must be JSON-serialisable.
        as_of: ISO-8601 date/timestamp for the decision.  Defaults to UTC now.
        run_id: Optional run identifier for linking decisions to a specific run.

    Returns:
        None.  Never raises.
    """
    try:
        output_path = _get_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        record: dict[str, Any] = {
            "ts": as_of or datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "signal_score": float(signal_score),
            "sizing_cap_hit": bool(sizing_cap_hit),
            "edcl_trigger": bool(edcl_trigger),
            "order_type": order_type,
        }
        if run_id is not None:
            record["run_id"] = run_id
        if reasoning:
            record["reasoning"] = reasoning

        line = json.dumps(record, default=str)
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    except Exception as exc:  # noqa: BLE001
        logger.warning("[audit_trail] Failed to log decision for %s: %s", symbol, exc)


def read_decisions(
    date_str: str | None = None,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read decisions from the audit trail, optionally filtered by date prefix.

    Args:
        date_str: Optional date prefix (e.g. "2026-05-07") to filter records.
        path: Override output path (defaults to standard location).

    Returns:
        List of decision records (dicts).  Empty list if file not found.
    """
    output_path = path or _get_output_path()
    if not output_path.exists():
        return []

    records: list[dict[str, Any]] = []
    try:
        with output_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if date_str and not rec.get("ts", "").startswith(date_str):
                        continue
                    records.append(rec)
                except json.JSONDecodeError:
                    pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("[audit_trail] Failed to read decisions: %s", exc)

    return records


__all__ = ["log_trade_decision", "read_decisions"]
