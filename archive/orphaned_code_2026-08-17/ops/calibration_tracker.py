"""Model calibration tracker (Item 63).

Tracks ML model probability calibration over time using the Brier Score.
Detects calibration drift and flags when the model is over- or under-confident
so the EDCL threshold (0.85) remains meaningful.

Usage::

    from src.assembled_core.ops.calibration_tracker import CalibrationTracker

    tracker = CalibrationTracker()
    tracker.record(predicted_prob=0.78, actual_outcome=1, as_of="2026-05-08")
    score = tracker.brier_score(window_days=30)

Brier score interpretation:
  * Perfect calibration: 0.0
  * Random model:        0.25
  * Inverted model:      1.0
  A score > 0.25 means the model is worse than random.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_STORE = Path("output") / "calibration_log.jsonl"


class CalibrationTracker:
    """Append-only log of (predicted_prob, actual_outcome) pairs.

    Args:
        store_path: JSONL file path for persistence. Created on first write.
        model_id:   Identifier tag stored with each record (default "default").
    """

    def __init__(
        self,
        store_path: Path | str | None = None,
        model_id: str = "default",
    ) -> None:
        self._path = Path(store_path) if store_path is not None else _DEFAULT_STORE
        self._model_id = model_id
        self._buffer: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ record
    def record(
        self,
        predicted_prob: float,
        actual_outcome: int,
        as_of: str | date | None = None,
    ) -> None:
        """Append one prediction/outcome pair.

        Args:
            predicted_prob: Model probability in [0, 1].
            actual_outcome: Binary label — 1 (correct direction) or 0 (wrong).
            as_of:          Trading date (YYYY-MM-DD string or date). Defaults to today.
        """
        if as_of is None:
            as_of_str = datetime.now(tz=timezone.utc).date().isoformat()
        elif isinstance(as_of, str):
            as_of_str = as_of[:10]
        else:
            as_of_str = as_of.isoformat()

        p = float(np.clip(predicted_prob, 0.0, 1.0))
        o = int(actual_outcome)
        self._buffer.append(
            {"as_of": as_of_str, "model_id": self._model_id, "prob": p, "outcome": o}
        )

    # ------------------------------------------------------------------ flush
    def flush(self) -> int:
        """Write buffered records to the JSONL store."""
        if not self._buffer:
            return 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            for rec in self._buffer:
                f.write(json.dumps(rec) + "\n")
        n = len(self._buffer)
        self._buffer.clear()
        return n

    # ------------------------------------------------------------------ load
    def _load(self, window_days: int | None = None) -> list[dict[str, Any]]:
        """Load records from store, optionally filtering to last *window_days*."""
        if not self._path.exists():
            return []
        cutoff: str | None = None
        if window_days is not None:
            cutoff = (
                datetime.now(tz=timezone.utc).date() - timedelta(days=window_days)
            ).isoformat()
        records: list[dict[str, Any]] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("model_id") != self._model_id:
                    continue
                if cutoff and rec.get("as_of", "") < cutoff:
                    continue
                records.append(rec)
        return records

    # ------------------------------------------------------------------ brier
    def brier_score(self, window_days: int = 30) -> float | None:
        """Compute the Brier Score over the last *window_days*.

        Returns:
            Brier score in [0, 1], or None if fewer than 2 records are available.
        """
        records = self._load(window_days=window_days)
        if len(records) < 2:
            return None
        probs = np.array([r["prob"] for r in records], dtype=float)
        outcomes = np.array([r["outcome"] for r in records], dtype=float)
        return float(np.mean((probs - outcomes) ** 2))

    # ------------------------------------------------------------------ drift
    def is_drift_detected(
        self,
        window_days: int = 30,
        threshold: float = 0.20,
    ) -> bool:
        """Return True if the Brier Score exceeds *threshold* (drift detected).

        Args:
            window_days: Rolling window for Brier score calculation.
            threshold:   Alert if Brier score > threshold (default 0.20).
                         Baseline for a random model is 0.25; 0.20 is a warning level.
        """
        score = self.brier_score(window_days=window_days)
        if score is None:
            return False
        drifted = score > threshold
        if drifted:
            log.warning(
                "[calibration] model_id=%s Brier score %.4f exceeds threshold %.4f — "
                "calibration drift detected over last %d days",
                self._model_id,
                score,
                threshold,
                window_days,
            )
        return drifted

    # ------------------------------------------------------------------ summary
    def summary(self, window_days: int = 30) -> dict[str, Any]:
        """Return a summary dict for reporting."""
        records = self._load(window_days=window_days)
        score = self.brier_score(window_days=window_days)
        return {
            "model_id": self._model_id,
            "window_days": window_days,
            "n_records": len(records),
            "brier_score": score,
            "drift_detected": self.is_drift_detected(window_days=window_days),
        }


__all__ = ["CalibrationTracker"]
