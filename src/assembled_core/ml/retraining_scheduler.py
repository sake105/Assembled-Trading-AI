from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalDetail:
    """Individual retraining signal result."""

    name: str
    fired: bool
    reason: str
    value: float


@dataclass
class RetrainingRecommendation:
    """Result of a retraining evaluation."""

    checked_at: str
    signals_fired: int
    decision: str
    auto_deploy: bool
    notes: str
    signal_details: list[SignalDetail] = field(default_factory=list)


class RetrainingScheduler:
    """Evaluate whether an ML model should be retrained based on staleness and performance signals."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._cfg: dict = {"auto_deploy": False}
        if config_path is not None:
            try:
                import yaml
                from pathlib import Path

                raw = (
                    yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
                )
                merged = {**self._cfg, **raw}
                merged["auto_deploy"] = False  # never allow auto-deploy from config
                self._cfg = merged
            except Exception as exc:
                logger.debug("[WARN] RetrainingScheduler config load failed: %s", exc)

    def adapt_hyperparameters_via_bandit(
        self, state_path: Optional[str] = None
    ) -> None:
        """No-op stub: bandit-based HPO adaptation placeholder."""
        logger.debug("[SKIP] adapt_hyperparameters_via_bandit: not yet implemented")

    def evaluate(
        self,
        model_last_trained_date: Optional[date] = None,
        ic_series: Optional[pd.Series] = None,
        equity_since_retrain: Optional[pd.Series] = None,
        regime_series: Optional[pd.Series] = None,
    ) -> RetrainingRecommendation:
        """Evaluate retraining signals and return a recommendation."""
        checked_at = datetime.now(timezone.utc).isoformat()
        details: list[SignalDetail] = []

        # Signal 1: staleness
        days_since = None
        if model_last_trained_date is not None:
            today = datetime.now(timezone.utc).date()
            days_since = (today - model_last_trained_date).days
        s1_fired = days_since is not None and days_since >= 30
        details.append(
            SignalDetail(
                name="days_since_retrain",
                fired=s1_fired,
                reason=(
                    f"days_since_retrain={days_since}"
                    if days_since is not None
                    else "no training date"
                ),
                value=float(days_since) if days_since is not None else -1.0,
            )
        )

        # Signal 2: IC decay
        ic_mean = None
        if ic_series is not None and len(ic_series) >= 20:
            ic_mean = float(ic_series.iloc[-20:].mean())
        s2_fired = ic_mean is not None and ic_mean < 0.02
        details.append(
            SignalDetail(
                name="ic_decay",
                fired=s2_fired,
                reason=(
                    f"ic_mean_last20={ic_mean:.4f}"
                    if ic_mean is not None
                    else "insufficient ic data"
                ),
                value=float(ic_mean) if ic_mean is not None else float("nan"),
            )
        )

        # Signal 3: equity MDD
        mdd = None
        if equity_since_retrain is not None and len(equity_since_retrain) >= 2:
            eq = equity_since_retrain.dropna()
            if len(eq) >= 2:
                roll_max = eq.cummax()
                drawdowns = (eq - roll_max) / roll_max.replace(0, float("nan"))
                mdd = float(drawdowns.min())
        s3_fired = mdd is not None and mdd < -0.15
        details.append(
            SignalDetail(
                name="equity_mdd",
                fired=s3_fired,
                reason=f"mdd={mdd:.4f}" if mdd is not None else "no equity data",
                value=float(mdd) if mdd is not None else float("nan"),
            )
        )

        # Signal 4: regime shift
        n_regimes = None
        if regime_series is not None and len(regime_series) >= 10:
            n_regimes = int(regime_series.iloc[-10:].nunique())
        s4_fired = n_regimes is not None and n_regimes > 2
        details.append(
            SignalDetail(
                name="regime_shift",
                fired=s4_fired,
                reason=(
                    f"distinct_regimes_last10={n_regimes}"
                    if n_regimes is not None
                    else "no regime data"
                ),
                value=float(n_regimes) if n_regimes is not None else float("nan"),
            )
        )

        signals_fired = sum(d.fired for d in details)

        if signals_fired >= 2:
            decision = "RETRAIN"
        elif signals_fired == 1:
            decision = "MONITOR"
        else:
            decision = "HOLD"

        fired_names = [d.name for d in details if d.fired]
        notes = (
            f"Fired: {fired_names}"
            if fired_names
            else "No signals fired — model appears healthy."
        )

        logger.debug(
            "[OK] RetrainingScheduler: decision=%s signals_fired=%d",
            decision,
            signals_fired,
        )

        return RetrainingRecommendation(
            checked_at=checked_at,
            signals_fired=signals_fired,
            decision=decision,
            auto_deploy=False,
            notes=notes,
            signal_details=details,
        )
