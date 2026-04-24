"""Signal Decay Tracking — misst IC-Halbwertszeit von Signalen.

Ein Signal ist nicht stabil über Zeit: IC nimmt ab, wenn Markt-Teilnehmer
den Edge arbitrieren oder sich die zugrunde liegende Dynamik ändert.

Tracking-Metriken:
- IC pro Horizon (1d/5d/20d)
- Halbwertszeit der IC (exponentieller Decay-Fit)
- Rolling-IC-Trend (falling/stable/rising)

Ergänzt `feature_importance_tracker.py` (Round 2): FI misst Feature-Wichtigkeit,
Decay misst Signal-Prädiktionsqualität direkt.

PIT-Invariante: Nur realisierte historische Returns werden genutzt.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DecaySnapshot:
    """Ein Snapshot der Signal-Performance zu einem Zeitpunkt."""

    as_of: str
    signal_ic: dict[str, dict[str, float]]
    """{signal_name: {horizon_1d: ic, horizon_5d: ic, horizon_20d: ic}}"""
    n_samples: int = 0


@dataclass
class DecayReport:
    """Analyse-Output pro Signal."""

    signal_name: str
    current_ic: dict[str, float]
    """Aktuell gemessene IC je Horizon."""
    historical_ic: dict[str, list[float]]
    """Zeitreihe IC je Horizon."""
    halflife_days: dict[str, float]
    """Exponentielle Halbwertszeit der IC pro Horizon. inf = keine Decay."""
    trend: dict[str, str]
    """'falling' / 'stable' / 'rising' pro Horizon."""


class SignalDecayTracker:
    """Persistenter Tracker für Signal-Decay über Zeit."""

    def __init__(
        self,
        state_path: Path | None = None,
        history_window: int = 12,
        horizons: list[int] | None = None,
    ) -> None:
        self.state_path = state_path or Path("output/ml/signal_decay_history.json")
        self.history_window = history_window
        self.horizons = horizons or [1, 5, 20]
        self._snapshots: list[DecaySnapshot] = self._load()

    def _load(self) -> list[DecaySnapshot]:
        if not self.state_path.exists():
            return []
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return [DecaySnapshot(**s) for s in data.get("snapshots", [])]
        except Exception as exc:
            logger.warning("[SignalDecay] Load failed: %s", exc)
            return []

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        trimmed = self._snapshots[-self.history_window:]
        data = {
            "snapshots": [
                {
                    "as_of": s.as_of,
                    "signal_ic": s.signal_ic,
                    "n_samples": s.n_samples,
                }
                for s in trimmed
            ]
        }
        self.state_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    def record_snapshot(
        self,
        predictions_by_signal: dict[str, pd.Series],
        realized_returns_by_horizon: dict[int, pd.Series],
        as_of: str | pd.Timestamp | None = None,
    ) -> DecaySnapshot:
        """Misst IC pro Signal × Horizon und persistiert.

        Args:
            predictions_by_signal: {signal_name: pd.Series mit Predictions}
            realized_returns_by_horizon: {horizon_days: pd.Series mit realisierten Returns}
            as_of: Timestamp (default: now)
        """
        if as_of is None:
            as_of = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        elif isinstance(as_of, pd.Timestamp):
            as_of = as_of.strftime("%Y-%m-%d")

        signal_ic: dict[str, dict[str, float]] = {}
        n_samples = 0

        for sig_name, preds in predictions_by_signal.items():
            per_horizon: dict[str, float] = {}
            for h, rets in realized_returns_by_horizon.items():
                common = preds.index.intersection(rets.index)
                if len(common) < 20:
                    per_horizon[f"horizon_{h}d"] = 0.0
                    continue
                p = preds.loc[common].values
                r = rets.loc[common].values
                if np.std(p) < 1e-9 or np.std(r) < 1e-9:
                    per_horizon[f"horizon_{h}d"] = 0.0
                    continue
                ic = np.corrcoef(p, r)[0, 1]
                per_horizon[f"horizon_{h}d"] = float(ic) if not np.isnan(ic) else 0.0
                n_samples = max(n_samples, len(common))
            signal_ic[sig_name] = per_horizon

        snap = DecaySnapshot(as_of=as_of, signal_ic=signal_ic, n_samples=n_samples)
        self._snapshots.append(snap)
        self._save()
        logger.info(
            "[SignalDecay] Snapshot %s: %d Signale × %d Horizonte (n=%d)",
            as_of, len(signal_ic), len(self.horizons), n_samples,
        )
        return snap

    def get_report(self, signal_name: str) -> DecayReport | None:
        """Report pro Signal mit Halbwertszeit + Trend."""
        history: dict[str, list[float]] = {}
        for s in self._snapshots:
            ic_dict = s.signal_ic.get(signal_name, {})
            for h in self.horizons:
                key = f"horizon_{h}d"
                history.setdefault(key, []).append(ic_dict.get(key, 0.0))

        if not history or all(len(v) == 0 for v in history.values()):
            return None

        current = {k: v[-1] if v else 0.0 for k, v in history.items()}
        halflife = {k: self._estimate_halflife(v) for k, v in history.items()}
        trend = {k: self._trend(v) for k, v in history.items()}

        return DecayReport(
            signal_name=signal_name,
            current_ic=current,
            historical_ic=history,
            halflife_days=halflife,
            trend=trend,
        )

    def _estimate_halflife(self, ic_values: list[float]) -> float:
        """Exponential-Fit auf |IC|-Reihe → Halbwertszeit in Snapshots.

        Wenn IC stabil oder steigt → inf.
        """
        if len(ic_values) < 3:
            return float("inf")
        abs_ic = np.abs(np.array(ic_values))
        if abs_ic.max() < 1e-6:
            return float("inf")
        # Log-linear fit: log(|IC|) = -λ·t + c
        t = np.arange(len(abs_ic))
        log_ic = np.log(np.maximum(abs_ic, 1e-9))
        try:
            slope, _ = np.polyfit(t, log_ic, 1)
            if slope >= -1e-6:
                return float("inf")
            halflife = float(np.log(2) / abs(slope))
            return halflife
        except Exception:
            return float("inf")

    def _trend(self, ic_values: list[float], threshold: float = 0.005) -> str:
        if len(ic_values) < 3:
            return "unknown"
        recent = np.array(ic_values[-3:])
        mean_diff = float(np.mean(np.diff(np.abs(recent))))
        if mean_diff < -threshold:
            return "falling"
        if mean_diff > threshold:
            return "rising"
        return "stable"

    def degraded_signals(
        self,
        decay_threshold_pct: float = 50.0,
        min_snapshots: int = 3,
        horizon_key: str = "horizon_5d",
    ) -> list[str]:
        """Signale deren IC in den letzten N Snapshots > decay_threshold_pct gefallen ist."""
        if len(self._snapshots) < min_snapshots:
            return []

        degraded: list[str] = []
        first_snap = self._snapshots[-min_snapshots]
        last_snap = self._snapshots[-1]

        for sig in last_snap.signal_ic:
            first_ic = first_snap.signal_ic.get(sig, {}).get(horizon_key, 0.0)
            last_ic = last_snap.signal_ic.get(sig, {}).get(horizon_key, 0.0)
            if abs(first_ic) < 1e-6:
                continue
            decline_pct = (abs(first_ic) - abs(last_ic)) / abs(first_ic) * 100.0
            if decline_pct > decay_threshold_pct:
                degraded.append(sig)
        return degraded

    def summary(self) -> dict:
        if not self._snapshots:
            return {"n_snapshots": 0}
        last = self._snapshots[-1]
        return {
            "n_snapshots": len(self._snapshots),
            "last_as_of": last.as_of,
            "signals_tracked": list(last.signal_ic.keys()),
            "n_signals": len(last.signal_ic),
        }

    def history_length(self) -> int:
        return len(self._snapshots)


__all__ = [
    "DecaySnapshot",
    "DecayReport",
    "SignalDecayTracker",
]
