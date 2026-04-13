"""Retraining Scheduler for Assembled-Trading-AI (Phase 8 autonomous improvement).

Aggregates 5 independent retrain signals and produces a RetrainingRecommendation.
NEVER auto-deploys — auto_deploy=False is the hard default and must not be overridden.

Signals:
  1. Calendar   : model age > max_model_age_days (default 90)
  2. IC Degradation : rolling IC < ic_threshold for > ic_bad_day_window days
  3. Feature Drift  : KS-stat or PSI > feature_drift_threshold (default 0.2)
  4. Regime Change  : HMM state transition detected
  5. Perf Drift     : drawdown > drawdown_threshold (default 8%) since last retrain

Decision matrix:
  0 signals -> no_retrain
  1 signal  -> log_and_monitor
  2 signals -> recommend
  3+ signals -> urgent

Log prefix: [RETRAIN-SCHED]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_PREFIX = "[RETRAIN-SCHED]"

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "self_learning.yaml"


def _load_thresholds(config_path: Path | None = None) -> dict:
    """Load threshold overrides from self_learning.yaml if available."""
    path = config_path or _DEFAULT_CONFIG_PATH
    defaults: dict = {
        "max_model_age_days": 90,
        "ic_threshold": -0.02,
        "ic_bad_day_window": 5,
        "feature_drift_threshold": 0.2,
        "drawdown_threshold": 0.08,
        "retrain_cooldown_days": 30,
        "max_retrain_per_quarter": 4,
        "auto_deploy": False,
    }
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="ascii", errors="replace") as fh:
            raw = yaml.safe_load(fh) or {}
        sl = raw.get("self_learning", {})
        rt = sl.get("retraining", {})
        fb = sl.get("feedback_loop", {})
        gr = sl.get("guardrails", {})

        defaults["max_model_age_days"] = rt.get("max_model_age_days", defaults["max_model_age_days"])
        defaults["auto_deploy"] = rt.get("auto_deploy", defaults["auto_deploy"])
        defaults["ic_threshold"] = fb.get("ic_degradation_threshold", defaults["ic_threshold"])
        defaults["retrain_cooldown_days"] = fb.get("retrain_cooldown_days", defaults["retrain_cooldown_days"])
        defaults["max_retrain_per_quarter"] = fb.get("max_retrain_per_quarter", defaults["max_retrain_per_quarter"])
        defaults["max_drawdown"] = gr.get("max_drawdown", 0.20)
    except Exception as exc:
        logger.debug("%s could not load config from %s: %s", _PREFIX, path, exc)
    return defaults


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RetainSignalDetail:
    """Detail for one evaluated signal."""

    name: str
    fired: bool
    reason: str
    value: float | None = None


@dataclass
class RetrainingRecommendation:
    """Output of RetrainingScheduler.evaluate().

    NEVER auto-deploys — decision field is recommendation only.
    """

    checked_at: str
    """ISO-8601 UTC timestamp."""

    signals_fired: int
    """Number of signals that fired (0-5)."""

    decision: str
    """One of: no_retrain | log_and_monitor | recommend | urgent"""

    signal_details: list[RetainSignalDetail] = field(default_factory=list)

    auto_deploy: bool = False
    """Always False — human must approve any model swap."""

    notes: list[str] = field(default_factory=list)
    """Additional context for the decision."""


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class RetrainingScheduler:
    """Evaluates 5 retrain signals and emits a RetrainingRecommendation.

    Usage::

        scheduler = RetrainingScheduler()
        rec = scheduler.evaluate(
            model_last_trained_date=date(2025, 1, 1),
            ic_series=pd.Series([...]),
            train_df=train_df,
            recent_df=recent_df,
            feature_cols=[...],
            regime_series=pd.Series([...]),
            equity_since_retrain=pd.Series([...]),
        )
        if rec.decision in ("recommend", "urgent"):
            # Human reviews — never auto-deploy
            ...
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._cfg = _load_thresholds(config_path)
        # Hard-enforce auto_deploy=False regardless of config
        self._cfg["auto_deploy"] = False
        logger.debug("%s initialized with thresholds: %s", _PREFIX, self._cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model_last_trained_date: date | None = None,
        ic_series: pd.Series | None = None,
        train_df: pd.DataFrame | None = None,
        recent_df: pd.DataFrame | None = None,
        feature_cols: list[str] | None = None,
        regime_series: pd.Series | None = None,
        equity_since_retrain: pd.Series | None = None,
    ) -> RetrainingRecommendation:
        """Evaluate all 5 signals and return a RetrainingRecommendation.

        All arguments are optional — missing data causes that signal to be skipped
        rather than raise an error.
        """
        now_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        details: list[RetainSignalDetail] = []
        notes: list[str] = []

        # Signal 1: Calendar age
        details.append(self._check_calendar(model_last_trained_date))

        # Signal 2: IC degradation
        details.append(self._check_ic_degradation(ic_series))

        # Signal 3: Feature drift (KS / PSI)
        details.append(self._check_feature_drift(train_df, recent_df, feature_cols))

        # Signal 4: Regime change
        details.append(self._check_regime_change(regime_series))

        # Signal 5: Performance drift (drawdown)
        details.append(self._check_perf_drift(equity_since_retrain))

        fired = sum(1 for d in details if d.fired)

        decision = self._decision_matrix(fired)

        if decision != "no_retrain":
            fired_names = [d.name for d in details if d.fired]
            logger.info(
                "%s signals=%d decision=%s fired=%s",
                _PREFIX, fired, decision, fired_names,
            )
        else:
            logger.debug("%s signals=0 decision=no_retrain", _PREFIX)

        return RetrainingRecommendation(
            checked_at=now_utc,
            signals_fired=fired,
            decision=decision,
            signal_details=details,
            auto_deploy=False,  # HARD DEFAULT — never override
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Signal implementations
    # ------------------------------------------------------------------

    def _check_calendar(self, last_trained: date | None) -> RetainSignalDetail:
        """Signal 1: model age > max_model_age_days."""
        threshold = int(self._cfg.get("max_model_age_days", 90))
        if last_trained is None:
            return RetainSignalDetail(
                name="calendar_age",
                fired=False,
                reason="last_trained_date not provided — signal skipped",
            )
        try:
            age_days = (date.today() - last_trained).days
            fired = age_days > threshold
            return RetainSignalDetail(
                name="calendar_age",
                fired=fired,
                reason=(
                    f"model age {age_days}d > threshold {threshold}d"
                    if fired
                    else f"model age {age_days}d <= threshold {threshold}d"
                ),
                value=float(age_days),
            )
        except Exception as exc:
            logger.debug("%s calendar signal error: %s", _PREFIX, exc)
            return RetainSignalDetail(
                name="calendar_age",
                fired=False,
                reason=f"error computing age: {exc}",
            )

    def _check_ic_degradation(self, ic_series: pd.Series | None) -> RetainSignalDetail:
        """Signal 2: rolling IC < threshold for > ic_bad_day_window consecutive days."""
        threshold = float(self._cfg.get("ic_threshold", -0.02))
        window = int(self._cfg.get("ic_bad_day_window", 5))
        if ic_series is None or len(ic_series) == 0:
            return RetainSignalDetail(
                name="ic_degradation",
                fired=False,
                reason="ic_series not provided — signal skipped",
            )
        try:
            bad_days = (ic_series < threshold).astype(int)
            # Count consecutive bad days at the tail
            tail = bad_days.iloc[-window:] if len(bad_days) >= window else bad_days
            consecutive = int(tail.sum())
            fired = consecutive >= window
            recent_ic = float(ic_series.iloc[-window:].mean()) if len(ic_series) >= window else float(ic_series.mean())
            return RetainSignalDetail(
                name="ic_degradation",
                fired=fired,
                reason=(
                    f"{consecutive} of last {window} days IC < {threshold:.4f} (mean={recent_ic:.4f})"
                    if fired
                    else f"IC degradation not sustained (consecutive_bad={consecutive}/{window})"
                ),
                value=recent_ic,
            )
        except Exception as exc:
            logger.debug("%s IC degradation signal error: %s", _PREFIX, exc)
            return RetainSignalDetail(
                name="ic_degradation",
                fired=False,
                reason=f"error computing IC degradation: {exc}",
            )

    def _check_feature_drift(
        self,
        train_df: pd.DataFrame | None,
        recent_df: pd.DataFrame | None,
        feature_cols: list[str] | None,
    ) -> RetainSignalDetail:
        """Signal 3: KS-test or PSI > feature_drift_threshold (default 0.2)."""
        drift_threshold = float(self._cfg.get("feature_drift_threshold", 0.2))
        if train_df is None or recent_df is None or not feature_cols:
            return RetainSignalDetail(
                name="feature_drift",
                fired=False,
                reason="train_df/recent_df/feature_cols not provided — signal skipped",
            )
        try:
            from src.assembled_core.ml.model_monitoring import detect_feature_drift  # type: ignore

            result = detect_feature_drift(
                train_df=train_df,
                recent_df=recent_df,
                feature_cols=feature_cols,
                p_value_threshold=0.01,
            )
            drift_score = float(result.get("drift_score", 0.0))
            fired = drift_score > drift_threshold
            return RetainSignalDetail(
                name="feature_drift",
                fired=fired,
                reason=(
                    f"drift_score={drift_score:.4f} > threshold={drift_threshold}"
                    if fired
                    else f"drift_score={drift_score:.4f} <= threshold={drift_threshold}"
                ),
                value=drift_score,
            )
        except ImportError:
            # Fallback: simple KS-based check
            return self._check_feature_drift_ks(train_df, recent_df, feature_cols, drift_threshold)
        except Exception as exc:
            logger.debug("%s feature drift signal error: %s", _PREFIX, exc)
            return RetainSignalDetail(
                name="feature_drift",
                fired=False,
                reason=f"error computing feature drift: {exc}",
            )

    def _check_feature_drift_ks(
        self,
        train_df: pd.DataFrame,
        recent_df: pd.DataFrame,
        feature_cols: list[str],
        drift_threshold: float,
    ) -> RetainSignalDetail:
        """KS-based drift fallback (used when model_monitoring unavailable)."""
        try:
            from scipy.stats import ks_2samp  # type: ignore

            drifted = 0
            tested = 0
            for col in feature_cols:
                if col not in train_df.columns or col not in recent_df.columns:
                    continue
                a = train_df[col].dropna().values
                b = recent_df[col].dropna().values
                if len(a) < 30 or len(b) < 10:
                    continue
                tested += 1
                stat, _ = ks_2samp(a, b)
                if stat > drift_threshold:
                    drifted += 1
            if tested == 0:
                return RetainSignalDetail(
                    name="feature_drift",
                    fired=False,
                    reason="no testable features",
                )
            drift_score = drifted / tested
            fired = drift_score > drift_threshold
            return RetainSignalDetail(
                name="feature_drift",
                fired=fired,
                reason=f"KS drift_score={drift_score:.4f} (drifted={drifted}/{tested})",
                value=drift_score,
            )
        except ImportError:
            return RetainSignalDetail(
                name="feature_drift",
                fired=False,
                reason="scipy not available — feature drift skipped",
            )

    def _check_regime_change(self, regime_series: pd.Series | None) -> RetainSignalDetail:
        """Signal 4: HMM state transition detected in regime_series tail."""
        if regime_series is None or len(regime_series) < 2:
            return RetainSignalDetail(
                name="regime_change",
                fired=False,
                reason="regime_series not provided or too short — signal skipped",
            )
        try:
            tail = regime_series.dropna()
            if len(tail) < 2:
                return RetainSignalDetail(
                    name="regime_change",
                    fired=False,
                    reason="insufficient regime data after dropna",
                )
            # Detect transition: last state differs from penultimate state
            recent_state = tail.iloc[-1]
            prev_state = tail.iloc[-2]
            fired = recent_state != prev_state
            return RetainSignalDetail(
                name="regime_change",
                fired=fired,
                reason=(
                    f"regime transition detected: {prev_state} -> {recent_state}"
                    if fired
                    else f"regime stable: {recent_state}"
                ),
                value=1.0 if fired else 0.0,
            )
        except Exception as exc:
            logger.debug("%s regime change signal error: %s", _PREFIX, exc)
            return RetainSignalDetail(
                name="regime_change",
                fired=False,
                reason=f"error computing regime change: {exc}",
            )

    def _check_perf_drift(self, equity_since_retrain: pd.Series | None) -> RetainSignalDetail:
        """Signal 5: max drawdown > drawdown_threshold since last retrain."""
        threshold = float(self._cfg.get("drawdown_threshold", 0.08))
        if equity_since_retrain is None or len(equity_since_retrain) < 2:
            return RetainSignalDetail(
                name="perf_drift",
                fired=False,
                reason="equity_since_retrain not provided or too short — signal skipped",
            )
        try:
            eq = equity_since_retrain.dropna()
            if len(eq) < 2:
                return RetainSignalDetail(
                    name="perf_drift",
                    fired=False,
                    reason="insufficient equity data after dropna",
                )
            rolling_max = eq.cummax()
            drawdown = ((eq - rolling_max) / rolling_max.clip(lower=1e-9)).min()
            max_dd = float(abs(drawdown))
            fired = max_dd > threshold
            return RetainSignalDetail(
                name="perf_drift",
                fired=fired,
                reason=(
                    f"max_drawdown={max_dd:.4f} > threshold={threshold}"
                    if fired
                    else f"max_drawdown={max_dd:.4f} <= threshold={threshold}"
                ),
                value=max_dd,
            )
        except Exception as exc:
            logger.debug("%s perf drift signal error: %s", _PREFIX, exc)
            return RetainSignalDetail(
                name="perf_drift",
                fired=False,
                reason=f"error computing perf drift: {exc}",
            )

    # ------------------------------------------------------------------
    # Decision matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _decision_matrix(n_signals: int) -> str:
        """Map number of fired signals to decision string."""
        if n_signals == 0:
            return "no_retrain"
        if n_signals == 1:
            return "log_and_monitor"
        if n_signals == 2:
            return "recommend"
        return "urgent"


__all__ = [
    "RetrainingScheduler",
    "RetrainingRecommendation",
    "RetainSignalDetail",
]
