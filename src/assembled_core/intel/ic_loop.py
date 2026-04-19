"""IC Feedback Loop (X3) — measures Information Coefficient per trigger type.

Computes rolling IC between GeoTrigger signals and realized price returns.
Weak trigger types (IC < threshold) can be flagged for deactivation.

Usage:
    tracker = ICTracker("output/intel/ic_loop.json")
    tracker.record(trigger_type="TRADE_WAR", signal=0.8, realized_return=0.02)
    report = tracker.compute_report()
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_IC_THRESHOLD = 0.05   # IC below this → flagged as weak
_DEFAULT_WINDOW = 60           # rolling window (observations)


class ICTracker:
    """Tracks IC per trigger type and computes rolling IC reports.

    State is persisted as JSON after each update.
    """

    def __init__(
        self,
        state_path: str | Path | None = None,
        *,
        window: int = _DEFAULT_WINDOW,
        ic_threshold: float = _DEFAULT_IC_THRESHOLD,
    ) -> None:
        self._path = Path(state_path) if state_path else None
        self._window = window
        self._threshold = ic_threshold
        # {trigger_type: [(signal, realized_return), ...]}
        self._observations: dict[str, list[tuple[float, float]]] = {}
        if self._path and self._path.exists():
            self._load()

    def record(
        self,
        trigger_type: str,
        signal: float,
        realized_return: float,
    ) -> None:
        """Record a (signal, realized_return) pair for a trigger type."""
        bucket = self._observations.setdefault(trigger_type, [])
        bucket.append((float(signal), float(realized_return)))
        # Keep only last `window` observations
        if len(bucket) > self._window:
            self._observations[trigger_type] = bucket[-self._window:]
        if self._path:
            self._save()

    def ic(self, trigger_type: str) -> float | None:
        """Compute Pearson IC for a trigger type. Returns None if < 2 observations."""
        obs = self._observations.get(trigger_type, [])
        if len(obs) < 2:
            return None
        signals = [x[0] for x in obs]
        returns = [x[1] for x in obs]
        return _pearson_corr(signals, returns)

    def compute_report(self) -> dict[str, Any]:
        """Compute IC report for all tracked trigger types."""
        now = datetime.now(tz=timezone.utc).isoformat()
        results = {}
        for ttype, obs in self._observations.items():
            ic_val = self.ic(ttype)
            results[ttype] = {
                "ic": round(ic_val, 4) if ic_val is not None else None,
                "n_obs": len(obs),
                "flagged_weak": ic_val is not None and ic_val < self._threshold,
            }
            if results[ttype]["flagged_weak"]:
                logger.warning(
                    "[WARN] IC-Loop: trigger_type=%s IC=%.4f < threshold=%.4f — flagged weak",
                    ttype, ic_val, self._threshold,
                )
        return {"generated_utc": now, "window": self._window, "results": results}

    def weak_trigger_types(self) -> list[str]:
        """Return trigger types with IC below threshold (minimum 10 observations)."""
        weak = []
        for ttype, obs in self._observations.items():
            if len(obs) < 10:
                continue
            ic_val = self.ic(ttype)
            if ic_val is not None and ic_val < self._threshold:
                weak.append(ttype)
        return weak

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "window": self._window,
            "ic_threshold": self._threshold,
            "observations": {
                k: [[s, r] for s, r in v]
                for k, v in self._observations.items()
            },
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._window = int(data.get("window", self._window))
            self._threshold = float(data.get("ic_threshold", self._threshold))
            raw = data.get("observations", {})
            self._observations = {
                k: [(float(s), float(r)) for s, r in v]
                for k, v in raw.items()
            }
        except Exception as exc:
            logger.warning("[WARN] IC-Loop: failed to load state from %s: %s", self._path, exc)


def _pearson_corr(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient without numpy dependency."""
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return cov / (sx * sy)
