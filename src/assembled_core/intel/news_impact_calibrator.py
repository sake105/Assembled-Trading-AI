"""Skeleton calibrator for news impact priors.

Given observed returns following labelled news events, update per-event-type
priors (mean impact in bps, dispersion) and recommend adjustments to the
default priors used by `news_impact.estimate_impact`.

Status: skeleton. Learning math is intentionally simple (running mean +
variance) — not a full Bayesian calibration. Intended as a data-collection
entry point so a later replacement can be dropped in without touching call
sites.

Usage:
    cal = ImpactCalibrator()
    cal.observe("sanctions", pred_bps=-50.0, realised_bps=-82.0)
    ...
    report = cal.report()
    for event_type, stat in report.items():
        print(event_type, stat.n, stat.mean_realised_bps, stat.bias_bps)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class _RunningStat:
    n: int = 0
    mean_pred: float = 0.0
    mean_real: float = 0.0
    m2_real: float = 0.0   # Welford's M2 for realised variance
    abs_err_sum: float = 0.0

    def update(self, pred: float, real: float) -> None:
        self.n += 1
        # running mean for pred
        self.mean_pred += (pred - self.mean_pred) / self.n
        # Welford for realised
        delta = real - self.mean_real
        self.mean_real += delta / self.n
        delta2 = real - self.mean_real
        self.m2_real += delta * delta2
        self.abs_err_sum += abs(pred - real)

    def variance(self) -> float:
        return self.m2_real / self.n if self.n > 1 else 0.0

    def stddev(self) -> float:
        return math.sqrt(self.variance())

    def mae(self) -> float:
        return self.abs_err_sum / self.n if self.n > 0 else 0.0

    def bias(self) -> float:
        """Positive = we underestimate (actual is more extreme)."""
        return self.mean_real - self.mean_pred


@dataclass
class CalibrationEntry:
    event_type: str
    n: int
    mean_pred_bps: float
    mean_realised_bps: float
    stddev_realised_bps: float
    mae_bps: float
    bias_bps: float


class ImpactCalibrator:
    """Collects realised-vs-predicted impact samples per event_type."""

    def __init__(self, min_samples_for_report: int = 5) -> None:
        self._stats: dict[str, _RunningStat] = {}
        self._min_n = max(1, min_samples_for_report)

    def observe(self, event_type: str, pred_bps: float, realised_bps: float) -> None:
        key = (event_type or "").lower().strip() or "unknown"
        stat = self._stats.setdefault(key, _RunningStat())
        stat.update(float(pred_bps), float(realised_bps))

    def report(self, include_sparse: bool = False) -> dict[str, CalibrationEntry]:
        out: dict[str, CalibrationEntry] = {}
        for et, st in self._stats.items():
            if st.n < self._min_n and not include_sparse:
                continue
            out[et] = CalibrationEntry(
                event_type=et,
                n=st.n,
                mean_pred_bps=round(st.mean_pred, 3),
                mean_realised_bps=round(st.mean_real, 3),
                stddev_realised_bps=round(st.stddev(), 3),
                mae_bps=round(st.mae(), 3),
                bias_bps=round(st.bias(), 3),
            )
        return out

    def recommend_prior_adjustment(self, event_type: str) -> float:
        """Return an additive bps adjustment to apply to the default prior.

        Returns 0.0 until enough samples exist. Capped at ±200 bps to prevent
        a noisy sample from steering the prior aggressively.
        """
        st = self._stats.get((event_type or "").lower().strip())
        if st is None or st.n < self._min_n:
            return 0.0
        return max(-200.0, min(200.0, st.bias()))

    def save(self, path: str | Path) -> None:
        data = {
            et: {
                "n": st.n,
                "mean_pred": st.mean_pred,
                "mean_real": st.mean_real,
                "m2_real": st.m2_real,
                "abs_err_sum": st.abs_err_sum,
            }
            for et, st in self._stats.items()
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for et, raw in data.items():
                st = _RunningStat()
                st.n = int(raw.get("n", 0))
                st.mean_pred = float(raw.get("mean_pred", 0.0))
                st.mean_real = float(raw.get("mean_real", 0.0))
                st.m2_real = float(raw.get("m2_real", 0.0))
                st.abs_err_sum = float(raw.get("abs_err_sum", 0.0))
                self._stats[et] = st
        except Exception as exc:
            logger.warning("[WARN] ImpactCalibrator.load: %s", exc)


__all__ = ["ImpactCalibrator", "CalibrationEntry"]
