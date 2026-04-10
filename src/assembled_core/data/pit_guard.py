"""Point-in-Time (PIT) safety guard.

Ensures that no feature or signal computation uses data that was not yet
available at the ``as_of`` timestamp.  Two modes:

* **assert mode** (default in tests / debug): raises ``PITViolationError``
  immediately when future data is detected.
* **log mode** (production): logs a WARNING and optionally truncates the
  offending rows so the pipeline can continue safely.

Usage::

    from src.assembled_core.data.pit_guard import PITGuard, PITViolationError

    guard = PITGuard(as_of=pd.Timestamp("2024-06-15", tz="UTC"))
    guard.validate(df, timestamp_col="timestamp")        # raises if future rows
    clean = guard.truncate(df, timestamp_col="timestamp") # returns filtered df
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PIT audit log (append-only JSONL for warn-mode usage tracking)
# ---------------------------------------------------------------------------
_DEFAULT_PIT_AUDIT_LOG = Path("output/ops/pit_guard_audit.jsonl")


def _pit_audit_path() -> Path:
    override = os.environ.get("ASSEMBLED_PIT_AUDIT_LOG", "")
    return Path(override) if override else _DEFAULT_PIT_AUDIT_LOG


def _append_pit_audit(event: dict) -> None:
    """Append a JSON-lines entry to the PIT audit log."""
    p = _pit_audit_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    event["ts"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    except Exception as exc:
        logger.error("[PITGuard] Failed to write audit log %s: %s", p, exc)


class PITViolationError(Exception):
    """Raised when data violates point-in-time constraints."""


class PITGuard:
    """Validates that data does not contain rows from the future.

    Args:
        as_of: The reference timestamp. All data must be <= this value.
        mode: ``"assert"`` to raise on violation, ``"warn"`` to log and continue.
    """

    def __init__(
        self,
        as_of: pd.Timestamp,
        mode: str = "assert",
    ) -> None:
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        self.as_of = as_of
        self.mode = mode

        # Audit: log every warn-mode instantiation so ops can track usage
        if mode == "warn":
            _append_pit_audit({
                "action": "INIT_WARN_MODE",
                "as_of": str(as_of),
            })

    # ------------------------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        context: str = "",
    ) -> bool:
        """Check that all rows in *df* have timestamps <= ``as_of``.

        Args:
            df: DataFrame to check.
            timestamp_col: Column containing timestamps.
            context: Optional label for error messages (e.g. module name).

        Returns:
            True if no violation, False if violation detected (warn mode).

        Raises:
            PITViolationError: In assert mode when future data is found.
        """
        if df.empty or timestamp_col not in df.columns:
            return True

        ts = pd.to_datetime(df[timestamp_col], utc=True)
        future_mask = ts > self.as_of
        n_future = int(future_mask.sum())

        if n_future == 0:
            return True

        max_ts = ts[future_mask].max()
        msg = (
            f"PIT violation{f' ({context})' if context else ''}: "
            f"{n_future} rows have timestamp > as_of={self.as_of} "
            f"(latest: {max_ts})"
        )

        if self.mode == "assert":
            raise PITViolationError(msg)

        # Audit: record every warn-mode violation for ops traceability
        _append_pit_audit({
            "action": "WARN_VIOLATION",
            "as_of": str(self.as_of),
            "context": context,
            "n_future_rows": n_future,
            "latest_future_ts": str(max_ts),
        })
        logger.warning(msg)
        return False

    # ------------------------------------------------------------------

    def truncate(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        context: str = "",
    ) -> pd.DataFrame:
        """Return *df* with rows after ``as_of`` removed.

        Always logs if rows are dropped.  Does NOT raise in any mode.
        """
        if df.empty or timestamp_col not in df.columns:
            return df

        ts = pd.to_datetime(df[timestamp_col], utc=True)
        keep_mask = ts <= self.as_of
        n_dropped = int((~keep_mask).sum())

        if n_dropped > 0:
            logger.warning(
                "PIT truncate%s: dropped %d future rows (as_of=%s)",
                f" ({context})" if context else "",
                n_dropped,
                self.as_of,
            )

        return df.loc[keep_mask].copy()
