"""ops/decision_log.py — Per-cycle decision reasoning log (Backlog Item 103).

Writes a machine-readable JSON-Lines log that records *why* the system placed
each trade: top factors, conviction score, trigger IDs, sizing rationale.

Usage in the trading cycle::

    from src.assembled_core.ops.decision_log import DecisionLogger
    dlog = DecisionLogger(log_dir="output/decisions")
    dlog.record(
        cycle_date="2026-05-08",
        symbol="NVDA",
        side="buy",
        conviction=0.85,
        top_factors=[("momentum_12m_excl_1m", 0.82), ("insider_cluster", 0.71)],
        edcl_trigger_ids=["ENERGY_SUPPLY_RISK"],
        sizing_notes="ATR stop 0.08, conformal_factor 0.92",
    )
    dlog.flush()  # writes pending entries to disk

The log is append-only, one JSON object per line, timestamped UTC.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_LOG_DIR = Path("output/decisions")
_LOG_FILENAME_FMT = "decision_log_{date}.jsonl"


class DecisionLogger:
    """Append-only decision reasoning logger.

    Thread-safety: single-process only (no locks). For multi-process setups
    use a different log_dir per worker and merge offline.
    """

    def __init__(
        self,
        log_dir: str | Path = _DEFAULT_LOG_DIR,
        *,
        auto_flush: bool = True,
        max_pending: int = 100,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._auto_flush = auto_flush
        self._max_pending = max_pending
        self._pending: list[dict[str, Any]] = []
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        cycle_date: str,
        symbol: str,
        side: str,
        conviction: float | None = None,
        top_factors: list[tuple[str, float]] | None = None,
        edcl_trigger_ids: list[str] | None = None,
        sizing_notes: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record a single trading decision.

        Args:
            cycle_date:       Trading date (YYYY-MM-DD).
            symbol:           Ticker symbol.
            side:             "buy" | "sell" | "short" | "cover".
            conviction:       EDCL conviction score [0, 1], or None if not applicable.
            top_factors:      Up to 5 (factor_name, factor_score) pairs, descending.
            edcl_trigger_ids: EDCL trigger type names that fired.
            sizing_notes:     Human-readable sizing rationale (ATR, conformal factor, etc.).
            extra:            Arbitrary additional fields.
        """
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "cycle_date": cycle_date,
            "symbol": symbol,
            "side": side,
        }
        if conviction is not None:
            entry["conviction"] = round(float(conviction), 4)
        if top_factors:
            entry["top_factors"] = [
                {"factor": name, "score": round(float(score), 4)}
                for name, score in top_factors[:5]
            ]
        if edcl_trigger_ids:
            entry["edcl_triggers"] = list(edcl_trigger_ids)
        if sizing_notes:
            entry["sizing_notes"] = sizing_notes
        if extra:
            entry.update(extra)

        self._pending.append(entry)
        if self._auto_flush and len(self._pending) >= self._max_pending:
            self.flush()

    def flush(self) -> int:
        """Write all pending entries to disk, grouped by cycle_date.

        Returns:
            Number of entries written.
        """
        if not self._pending:
            return 0

        # Group by cycle_date so entries land in the right dated file.
        by_date: dict[str, list[dict]] = {}
        for entry in self._pending:
            raw = entry.get(
                "cycle_date", datetime.now(timezone.utc).strftime("%Y-%m-%d")
            )
            date_key = raw.replace("-", "")
            by_date.setdefault(date_key, []).append(entry)

        n = 0
        for date_str, entries in by_date.items():
            log_path = self._log_dir / _LOG_FILENAME_FMT.format(date=date_str)
            try:
                with log_path.open("a", encoding="utf-8") as fh:
                    for entry in entries:
                        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n += len(entries)
                log.debug(
                    "[DecisionLog] Wrote %d entries to %s", len(entries), log_path
                )
            except OSError as exc:
                log.error("[DecisionLog] Failed to write log %s: %s", log_path, exc)
        self._pending.clear()
        return n

    def __del__(self) -> None:
        if self._pending:
            self.flush()

    # ------------------------------------------------------------------
    # Read helpers (forensics)
    # ------------------------------------------------------------------

    def read_log(self, date_str: str) -> list[dict[str, Any]]:
        """Read all entries for a given date (YYYYMMDD)."""
        log_path = self._log_dir / _LOG_FILENAME_FMT.format(date=date_str)
        if not log_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        log.warning(
                            "[DecisionLog] Skipped malformed line in %s", log_path
                        )
        return entries

    def query(
        self,
        *,
        date_str: str,
        symbol: str | None = None,
        min_conviction: float | None = None,
    ) -> list[dict[str, Any]]:
        """Filter log entries by optional criteria."""
        entries = self.read_log(date_str)
        if symbol is not None:
            entries = [e for e in entries if e.get("symbol") == symbol]
        if min_conviction is not None:
            entries = [
                e for e in entries if float(e.get("conviction", 0)) >= min_conviction
            ]
        return entries


# ---------------------------------------------------------------------------
# Module-level convenience singleton (optional)
# ---------------------------------------------------------------------------

_default_logger: DecisionLogger | None = None


def get_default_logger(log_dir: str | Path = _DEFAULT_LOG_DIR) -> DecisionLogger:
    """Return or create the module-level default DecisionLogger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = DecisionLogger(log_dir=log_dir)
    return _default_logger


__all__ = ["DecisionLogger", "get_default_logger"]
