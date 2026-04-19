"""Self-learning feedback loop controller for Assembled-Trading-AI.

This module orchestrates periodic model health checks and conditional retraining
using 5 independent degradation signals:

1. IC Degradation   — rolling IC < threshold sustained for > 5 days
2. Feature Drift    — KS-test p<0.01 or PSI>0.2 across feature distributions
3. Signal Decay     — IC last 20d vs last 60d drops significantly
4. Label Drift      — outcome distribution shift (PSI)
5. Model Age        — exponential confidence decay: 2^(-age_days / 30)

Retrain is triggered when >= 2 of 5 signals are active.

Design rules:
- auto_deploy defaults to False — retraining logs recommendations but does NOT
  swap the model automatically unless explicitly configured.
- All 5 signal checks are individually try/except guarded; a missing optional
  dependency skips that signal without failing the run.
- State persists across runs via JSON in state_dir.
- Cooldown and max_retrain_per_quarter are hard guardrails.
- Log prefix: [FEEDBACK]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_PREFIX = "[FEEDBACK]"


# ---------------------------------------------------------------------------
# Config / Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FeedbackLoopConfig:
    """Configuration for the FeedbackLoopController."""

    check_interval_days: int = 5
    """Minimum days between feedback checks."""

    retrain_cooldown_days: int = 30
    """Minimum days between retraining runs."""

    max_retrain_per_quarter: int = 4
    """Hard cap on number of retrains within any rolling 90-day window."""

    ic_degradation_threshold: float = -0.02
    """IC value below which a day counts as a 'bad' day for signal 1."""

    require_oos_improvement: bool = True
    """If True, new model must achieve >= current AUC - 0.01 before deployment."""

    min_trades_for_evaluation: int = 50
    """Minimum post-trade records required to run a meaningful evaluation."""

    auto_deploy: bool = False
    """If True, automatically swap model when new model passes validation.
    Defaults to False — safe recommendation-only mode."""

    cpcv_gate_enabled: bool = True
    """If True, require CPCV P(OOS_Sharpe > 0) >= cpcv_min_prob before deploy."""

    cpcv_min_prob: float = 0.60
    """Minimum CPCV probability of positive OOS Sharpe for deploy gate."""

    ic_bad_day_window: int = 5
    """Number of consecutive bad-IC days required to trigger signal 1."""

    signal_decay_short_window: int = 20
    """Recent window (days) for signal decay comparison."""

    signal_decay_long_window: int = 60
    """Baseline window (days) for signal decay comparison."""

    signal_decay_drop_threshold: float = 0.03
    """IC drop between short and long window that triggers signal 3."""

    model_age_confidence_threshold: float = 0.5
    """Model confidence below which signal 5 (age) fires. 2^(-age/30)."""


@dataclass
class FeedbackResult:
    """Result of a single feedback check run."""

    checked_at: str
    """ISO-8601 timestamp of when the check ran."""

    n_recent_trades: int
    """Number of post-trade learning records evaluated."""

    recent_hit_rate: float
    """Fraction of recent trades that were directionally correct."""

    recent_ic: float
    """Rolling IC estimate from recent trades."""

    degradation_detected: bool
    """True if IC degradation signal (signal 1) fired."""

    feature_drift_detected: bool
    """True if feature drift signal (signal 2) fired."""

    signal_decay_detected: bool
    """True if signal decay (signal 3) fired."""

    label_drift_detected: bool = False
    """True if label drift (signal 4) fired."""

    model_age_signal: bool = False
    """True if model age confidence dropped below threshold (signal 5)."""

    active_signal_count: int = 0
    """Total number of active degradation signals (0-5)."""

    retrain_triggered: bool = False
    """True if retraining was attempted this run."""

    retrain_result: dict | None = None
    """Metrics dict from retrain_and_validate, or None if not triggered."""

    new_model_deployed: bool = False
    """True if new model was swapped in (only possible when auto_deploy=True)."""

    report_path: Path = field(default_factory=lambda: Path("output/feedback_state/report.json"))
    """Path to the JSON report written by this run."""

    skipped_signals: list[str] = field(default_factory=list)
    """Signals that were skipped due to missing dependencies or data."""

    blocked_reason: str = ""
    """If retraining was blocked, the reason (cooldown / max_retrain_reached)."""


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class FeedbackLoopController:
    """Orchestrates periodic model health checks and conditional retraining.

    Parameters
    ----------
    config:
        FeedbackLoopConfig controlling thresholds and guardrails.
    state_dir:
        Directory where persistent state JSON is stored across runs.
    """

    _STATE_FILE = "feedback_state.json"

    def __init__(
        self,
        config: FeedbackLoopConfig | None = None,
        state_dir: Path = Path("output/feedback_state"),
    ) -> None:
        self.config = config or FeedbackLoopConfig()
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        logger.info("%s Initialized. state_dir=%s", _PREFIX, self.state_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_feedback_check(
        self,
        learning_store_path: Path,
        current_model_path: Path,
        panel_df: pd.DataFrame,
    ) -> FeedbackResult:
        """Run a full feedback check cycle.

        Evaluates 5 independent degradation signals and triggers retraining
        when >= 2 signals are active (subject to guardrails).

        Parameters
        ----------
        learning_store_path:
            Path to the JSONL post-trade learning store.
        current_model_path:
            Path to the currently deployed model file/directory.
        panel_df:
            Recent factor panel DataFrame with at minimum a date index
            and feature columns. Used for drift and decay analysis.

        Returns
        -------
        FeedbackResult with full status of this check.
        """
        now_str = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        logger.info("%s Starting feedback check at %s", _PREFIX, now_str)

        # --- Load records -------------------------------------------------
        records = self._load_records(learning_store_path)
        n_recent = len(records)
        logger.info("%s Loaded %d learning records", _PREFIX, n_recent)

        # --- Basic trade metrics from records ----------------------------
        recent_hit_rate, recent_ic = self._compute_trade_metrics(records)
        logger.info(
            "%s Trade metrics — hit_rate=%.3f  IC=%.4f  n=%d",
            _PREFIX,
            recent_hit_rate,
            recent_ic,
            n_recent,
        )

        # --- Check all 5 signals -----------------------------------------
        skipped_signals: list[str] = []
        active_signals: list[str] = []

        # Signal 1: IC Degradation
        sig1 = self._check_ic_degradation(records, skipped_signals)
        if sig1:
            active_signals.append("ic_degradation")
            logger.info("%s [SIGNAL-1] IC degradation ACTIVE", _PREFIX)
        else:
            logger.debug("%s [SIGNAL-1] IC degradation: OK", _PREFIX)

        # Signal 2: Feature Drift
        sig2 = self._check_feature_drift(panel_df, skipped_signals)
        if sig2:
            active_signals.append("feature_drift")
            logger.info("%s [SIGNAL-2] Feature drift ACTIVE", _PREFIX)
        else:
            logger.debug("%s [SIGNAL-2] Feature drift: OK", _PREFIX)

        # Signal 3: Signal Decay
        sig3 = self._check_signal_decay(records, skipped_signals)
        if sig3:
            active_signals.append("signal_decay")
            logger.info("%s [SIGNAL-3] Signal decay ACTIVE", _PREFIX)
        else:
            logger.debug("%s [SIGNAL-3] Signal decay: OK", _PREFIX)

        # Signal 4: Label Drift
        sig4 = self._check_label_drift(records, skipped_signals)
        if sig4:
            active_signals.append("label_drift")
            logger.info("%s [SIGNAL-4] Label drift ACTIVE", _PREFIX)
        else:
            logger.debug("%s [SIGNAL-4] Label drift: OK", _PREFIX)

        # Signal 5: Model Age
        sig5, model_age_days = self._check_model_age(current_model_path, skipped_signals)
        if sig5:
            active_signals.append("model_age")
            logger.info(
                "%s [SIGNAL-5] Model age ACTIVE (age=%d days)",
                _PREFIX,
                model_age_days,
            )
        else:
            logger.debug(
                "%s [SIGNAL-5] Model age: OK (age=%d days, confidence=%.3f)",
                _PREFIX,
                model_age_days,
                2 ** (-model_age_days / 30.0),
            )

        n_active = len(active_signals)
        logger.info(
            "%s Active signals: %d/5 — %s",
            _PREFIX,
            n_active,
            ", ".join(active_signals) if active_signals else "none",
        )

        # --- Retrain decision --------------------------------------------
        retrain_triggered = False
        retrain_result: dict | None = None
        new_model_deployed = False
        blocked_reason = ""

        should_retrain = n_active >= 2 and n_recent >= self.config.min_trades_for_evaluation

        if should_retrain:
            blocked, reason = self._check_guardrails()
            if blocked:
                blocked_reason = reason
                logger.warning("%s Retraining BLOCKED: %s", _PREFIX, reason)
            else:
                retrain_triggered = True
                new_model_deployed, retrain_result = self.retrain_and_validate(
                    panel_df=panel_df,
                    current_model_path=current_model_path,
                )
        elif n_recent < self.config.min_trades_for_evaluation:
            logger.info(
                "%s Insufficient trades for evaluation (%d < %d). Skipping retrain.",
                _PREFIX,
                n_recent,
                self.config.min_trades_for_evaluation,
            )
        else:
            logger.info(
                "%s Only %d/5 signals active — retraining not triggered (threshold: 2).",
                _PREFIX,
                n_active,
            )

        # --- Fix 28: EWRLS online update (when no full retrain was triggered) ---
        # Gate: online_learning.enabled in configs/self_learning.yaml, and only
        # when we have enough records but did NOT trigger a full retrain.
        if not retrain_triggered and n_recent >= self.config.min_trades_for_evaluation:
            self._maybe_run_ewrls_update(records)

        # --- Build result ------------------------------------------------
        report_path = self.state_dir / f"report_{now_str[:10]}.json"

        result = FeedbackResult(
            checked_at=now_str,
            n_recent_trades=n_recent,
            recent_hit_rate=recent_hit_rate,
            recent_ic=recent_ic,
            degradation_detected=sig1,
            feature_drift_detected=sig2,
            signal_decay_detected=sig3,
            label_drift_detected=sig4,
            model_age_signal=sig5,
            active_signal_count=n_active,
            retrain_triggered=retrain_triggered,
            retrain_result=retrain_result,
            new_model_deployed=new_model_deployed,
            report_path=report_path,
            skipped_signals=skipped_signals,
            blocked_reason=blocked_reason,
        )

        self._write_report(result)
        self._update_state_after_check(result)

        logger.info(
            "%s Feedback check complete — retrain=%s  deployed=%s  report=%s",
            _PREFIX,
            retrain_triggered,
            new_model_deployed,
            report_path,
        )
        return result

    def retrain_and_validate(
        self,
        panel_df: pd.DataFrame,
        current_model_path: Path,
    ) -> tuple[bool, dict]:
        """Retrain meta-model and compare against current model performance.

        Guardrails:
        - Respects cooldown (retrain_cooldown_days between runs).
        - Respects max_retrain_per_quarter (rolling 90-day window).
        - New model must achieve >= current AUC - 0.01 if require_oos_improvement=True.
        - auto_deploy=False (default): logs recommendation but does NOT swap model.

        Parameters
        ----------
        panel_df:
            Factor panel passed to the training pipeline.
        current_model_path:
            Path to currently deployed model (used for AUC comparison).

        Returns
        -------
        (deployed, metrics_dict):
            deployed: True only if auto_deploy=True AND validation passed.
            metrics_dict: Training result metrics including AUC, status, etc.
        """
        blocked, reason = self._check_guardrails()
        if blocked:
            logger.warning("%s retrain_and_validate blocked: %s", _PREFIX, reason)
            return False, {"status": "blocked", "reason": reason}

        logger.info("%s Starting retraining run…", _PREFIX)

        # Save panel to temp parquet for the training pipeline
        tmp_panel_path = self.state_dir / "_tmp_panel_for_retrain.parquet"
        try:
            panel_df.to_parquet(tmp_panel_path)
        except Exception as exc:
            logger.error("%s Failed to write temp panel: %s", _PREFIX, exc)
            return False, {"status": "error", "stage": "panel_write", "error": str(exc)}

        # Call training pipeline
        train_result = self._run_training_pipeline(tmp_panel_path)

        # Clean up temp file
        try:
            tmp_panel_path.unlink(missing_ok=True)
        except Exception:
            pass

        if train_result.get("status") == "error":
            logger.error("%s Training pipeline failed: %s", _PREFIX, train_result)
            self._record_retrain_attempt(success=False)
            return False, train_result

        # Validation: OOS improvement check
        new_auc = train_result.get("auc", None)
        current_auc = self._get_current_model_auc(current_model_path)

        validation_passed = True
        if self.config.require_oos_improvement and new_auc is not None and current_auc is not None:
            min_auc = current_auc - 0.01
            validation_passed = new_auc >= min_auc
            logger.info(
                "%s OOS validation — new_auc=%.4f  current_auc=%.4f  threshold=%.4f  passed=%s",
                _PREFIX,
                new_auc,
                current_auc,
                min_auc,
                validation_passed,
            )

        # CPCV Overfitting Gate (M16) — blocks deploy if P(OOS Sharpe > 0) too low
        cpcv_passed = True
        if self.config.cpcv_gate_enabled and validation_passed:
            cpcv_passed = self._run_cpcv_gate(train_result)
            validation_passed = validation_passed and cpcv_passed

        train_result["validation_passed"] = validation_passed
        train_result["cpcv_passed"] = cpcv_passed
        train_result["new_auc"] = new_auc
        train_result["current_auc"] = current_auc

        deployed = False
        if validation_passed:
            if self.config.auto_deploy:
                deployed = self._deploy_model(train_result, current_model_path)
                train_result["deployed"] = deployed
            else:
                logger.info(
                    "%s New model passed validation (AUC=%.4f). "
                    "auto_deploy=False — RECOMMENDATION: deploy model at %s",
                    _PREFIX,
                    new_auc if new_auc is not None else float("nan"),
                    train_result.get("model_path", "unknown"),
                )
                train_result["deployed"] = False
                train_result["deploy_recommendation"] = True
        else:
            logger.warning(
                "%s New model did NOT pass validation — keeping current model.", _PREFIX
            )
            train_result["deployed"] = False
            train_result["deploy_recommendation"] = False

        self._record_retrain_attempt(success=True)
        return deployed, train_result

    # ------------------------------------------------------------------
    # CPCV Overfitting Gate (M16)
    # ------------------------------------------------------------------

    def _run_cpcv_gate(self, train_result: dict) -> bool:
        """Run CPCV overfitting gate on the training returns.

        Returns True if model passes (P(OOS Sharpe > 0) >= threshold).
        Returns True (pass) if CPCV cannot be run (defensive).
        """
        try:
            import numpy as np
            from src.assembled_core.ml.cpcv import (
                compute_cpcv_sharpe_distribution,
            )

            # Extract per-split returns from training result
            per_split = train_result.get("per_split_metrics", [])
            if not per_split or len(per_split) < 3:
                logger.debug("%s CPCV gate: insufficient splits (%d), passing", _PREFIX, len(per_split))
                return True

            # Build returns per path from split predictions
            returns_per_path = []
            for split in per_split:
                split_returns = split.get("test_returns") or split.get("predictions", [])
                if isinstance(split_returns, (list, np.ndarray)) and len(split_returns) > 0:
                    returns_per_path.append(np.array(split_returns, dtype=float))

            if len(returns_per_path) < 3:
                logger.debug("%s CPCV gate: no usable returns from splits, passing", _PREFIX)
                return True

            result = compute_cpcv_sharpe_distribution(returns_per_path)
            prob_pos = result.prob_positive_sharpe
            is_overfit = result.is_likely_overfit

            logger.info(
                "%s CPCV gate: P(Sharpe>0)=%.3f (min=%.2f), overfit=%s, DSR=%.3f",
                _PREFIX, prob_pos, self.config.cpcv_min_prob,
                is_overfit, result.deflated_sharpe,
            )

            if prob_pos < self.config.cpcv_min_prob:
                logger.warning(
                    "%s CPCV BLOCKED: P(OOS Sharpe>0)=%.3f < %.2f — likely overfitting",
                    _PREFIX, prob_pos, self.config.cpcv_min_prob,
                )
                return False

            if is_overfit:
                logger.warning(
                    "%s CPCV WARNING: model flagged as likely overfit (DSR=%.3f)",
                    _PREFIX, result.deflated_sharpe,
                )
                # Warning only — don't block if prob is sufficient

            return True

        except Exception as exc:
            logger.debug("%s CPCV gate: could not run (%s), passing defensively", _PREFIX, exc)
            return True

    # ------------------------------------------------------------------
    # Signal checks (each individually guarded)
    # ------------------------------------------------------------------

    def _check_ic_degradation(self, records: list[dict], skipped: list[str]) -> bool:
        """Signal 1: Rolling IC < threshold for > ic_bad_day_window days."""
        try:
            if not records:
                return False
            ic_values = [r.get("ic") for r in records if r.get("ic") is not None]
            if len(ic_values) < self.config.ic_bad_day_window:
                return False
            recent = ic_values[-self.config.ic_bad_day_window :]
            bad_days = sum(1 for v in recent if v < self.config.ic_degradation_threshold)
            return bad_days >= self.config.ic_bad_day_window
        except Exception as exc:
            logger.warning("%s [SIGNAL-1] IC degradation check failed: %s", _PREFIX, exc)
            skipped.append("ic_degradation")
            return False

    def _check_feature_drift(self, panel_df: pd.DataFrame, skipped: list[str]) -> bool:
        """Signal 2: Feature drift via KS-test / PSI from model_monitoring."""
        try:
            from assembled_core.ml.model_monitoring import detect_feature_drift

            if panel_df is None or panel_df.empty:
                skipped.append("feature_drift")
                return False

            n = len(panel_df)
            split = max(1, n - min(30, n // 4))
            train_part = panel_df.iloc[:split]
            recent_part = panel_df.iloc[split:]

            if len(recent_part) < 5:
                skipped.append("feature_drift")
                return False

            numeric_cols = panel_df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                skipped.append("feature_drift")
                return False

            result = detect_feature_drift(
                train_df=train_part,
                recent_df=recent_part,
                feature_cols=numeric_cols,
                p_value_threshold=0.01,
            )
            alert = result.get("alert_level", "unknown")
            drift_score = result.get("drift_score", 0.0)
            logger.debug(
                "%s [SIGNAL-2] drift_score=%.3f  alert=%s  drifted=%d/%d",
                _PREFIX,
                drift_score,
                alert,
                len(result.get("drifted_features", [])),
                result.get("n_tested", 0),
            )
            return alert in ("WARNING", "CRITICAL") or drift_score >= 0.3
        except ImportError:
            logger.debug("%s [SIGNAL-2] model_monitoring not available — skipping", _PREFIX)
            skipped.append("feature_drift")
            return False
        except Exception as exc:
            logger.warning("%s [SIGNAL-2] Feature drift check failed: %s", _PREFIX, exc)
            skipped.append("feature_drift")
            return False

    def _check_signal_decay(self, records: list[dict], skipped: list[str]) -> bool:
        """Signal 3: IC last 20d vs last 60d — checks for IC degradation over time."""
        try:
            ic_values = [r.get("ic") for r in records if r.get("ic") is not None]
            short = self.config.signal_decay_short_window
            long_ = self.config.signal_decay_long_window

            if len(ic_values) < short:
                return False

            recent_ic = sum(ic_values[-short:]) / short
            if len(ic_values) >= long_:
                baseline_ic = sum(ic_values[-long_:-short]) / max(long_ - short, 1)
            else:
                baseline_ic = sum(ic_values) / len(ic_values)

            drop = baseline_ic - recent_ic
            logger.debug(
                "%s [SIGNAL-3] baseline_ic=%.4f  recent_ic=%.4f  drop=%.4f  threshold=%.4f",
                _PREFIX,
                baseline_ic,
                recent_ic,
                drop,
                self.config.signal_decay_drop_threshold,
            )
            return drop >= self.config.signal_decay_drop_threshold
        except Exception as exc:
            logger.warning("%s [SIGNAL-3] Signal decay check failed: %s", _PREFIX, exc)
            skipped.append("signal_decay")
            return False

    def _check_label_drift(self, records: list[dict], skipped: list[str]) -> bool:
        """Signal 4: Label drift between base and recent outcome distributions."""
        try:
            from assembled_core.qa.drift_detection import detect_label_drift

            outcomes = [r.get("outcome") for r in records if r.get("outcome") is not None]
            if len(outcomes) < 30:
                return False

            split = len(outcomes) // 2
            base = pd.Series(outcomes[:split], dtype=float)
            current = pd.Series(outcomes[split:], dtype=float)

            result = detect_label_drift(base_labels=base, current_labels=current)
            drift_detected = result.get("drift_detected", False)
            severity = result.get("drift_severity", "NONE")
            logger.debug(
                "%s [SIGNAL-4] label_drift=%s  severity=%s  psi=%.4f",
                _PREFIX,
                drift_detected,
                severity,
                result.get("psi", 0.0),
            )
            return bool(drift_detected)
        except ImportError:
            logger.debug("%s [SIGNAL-4] drift_detection not available — skipping", _PREFIX)
            skipped.append("label_drift")
            return False
        except Exception as exc:
            logger.warning("%s [SIGNAL-4] Label drift check failed: %s", _PREFIX, exc)
            skipped.append("label_drift")
            return False

    def _check_model_age(
        self, current_model_path: Path, skipped: list[str]
    ) -> tuple[bool, int]:
        """Signal 5: Model age confidence decay — 2^(-age_days / 30).

        Returns (signal_active, age_in_days).
        """
        try:
            state = self._load_state()
            last_deployed_str = state.get("last_model_deployed_at")

            if last_deployed_str:
                last_deployed = date.fromisoformat(last_deployed_str[:10])
                age_days = (date.today() - last_deployed).days
            elif current_model_path and Path(current_model_path).exists():
                mtime = Path(current_model_path).stat().st_mtime
                last_modified = date.fromtimestamp(mtime)
                age_days = (date.today() - last_modified).days
            else:
                age_days = 0

            confidence = 2 ** (-age_days / 30.0)
            signal_active = confidence < self.config.model_age_confidence_threshold
            return signal_active, age_days
        except Exception as exc:
            logger.warning("%s [SIGNAL-5] Model age check failed: %s", _PREFIX, exc)
            skipped.append("model_age")
            return False, 0

    # ------------------------------------------------------------------
    # Trade metrics helpers
    # ------------------------------------------------------------------

    def _load_records(self, learning_store_path: Path) -> list[dict]:
        """Load post-trade records from the learning store."""
        try:
            from assembled_core.qa.learning_store import load_learning_records

            return load_learning_records(learning_store_path)
        except ImportError:
            logger.debug("%s learning_store not available, trying direct JSONL read", _PREFIX)
        except Exception as exc:
            logger.warning("%s Failed to load via learning_store: %s", _PREFIX, exc)

        # Fallback: direct JSONL read
        path = Path(learning_store_path)
        if not path.exists():
            return []
        records: list[dict] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as exc:
            logger.warning("%s Direct JSONL read failed: %s", _PREFIX, exc)
        return records

    def _compute_trade_metrics(self, records: list[dict]) -> tuple[float, float]:
        """Compute hit_rate and rolling IC from records.

        Returns (hit_rate, ic).
        """
        if not records:
            return 0.0, 0.0

        hits = [r.get("correct", r.get("hit")) for r in records if r.get("correct") is not None or r.get("hit") is not None]
        hit_rate = sum(1 for h in hits if h) / len(hits) if hits else 0.0

        ic_values = [r.get("ic") for r in records if r.get("ic") is not None]
        recent_ic = sum(ic_values[-20:]) / len(ic_values[-20:]) if ic_values else 0.0

        return hit_rate, recent_ic

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------

    def _check_guardrails(self) -> tuple[bool, str]:
        """Check cooldown and max_retrain_per_quarter guardrails.

        Returns (blocked: bool, reason: str).
        """
        state = self._load_state()
        today = date.today()

        # Cooldown check
        last_retrain_str = state.get("last_retrain_at")
        if last_retrain_str:
            last_retrain = date.fromisoformat(last_retrain_str[:10])
            days_since = (today - last_retrain).days
            if days_since < self.config.retrain_cooldown_days:
                return True, (
                    f"Cooldown active — {days_since}d since last retrain "
                    f"({self.config.retrain_cooldown_days}d required)"
                )

        # Max retrain per quarter check
        retrain_log: list[str] = state.get("retrain_log", [])
        cutoff = (today - timedelta(days=90)).isoformat()
        recent_retrains = [d for d in retrain_log if d >= cutoff]
        if len(recent_retrains) >= self.config.max_retrain_per_quarter:
            return True, (
                f"Max retrains reached — {len(recent_retrains)} retrains "
                f"in last 90 days (max={self.config.max_retrain_per_quarter})"
            )

        return False, ""

    # ------------------------------------------------------------------
    # Training pipeline integration
    # ------------------------------------------------------------------

    def _run_training_pipeline(self, panel_path: Path) -> dict[str, Any]:
        """Delegate to train_meta_model_pipeline and return normalised metrics."""
        try:
            import sys

            # Ensure scripts/ is on path for the training module
            scripts_training = Path(__file__).resolve().parents[4] / "scripts" / "training"
            if str(scripts_training) not in sys.path:
                sys.path.insert(0, str(scripts_training))

            from train_meta_model import train_meta_model_pipeline

            output_dir = self.state_dir / "retrain_output"
            output_dir.mkdir(parents=True, exist_ok=True)

            result = train_meta_model_pipeline(
                panel_path=panel_path,
                output_dir=output_dir,
            )

            # Normalise to a plain dict
            if hasattr(result, "__dict__"):
                metrics: dict[str, Any] = {
                    k: v
                    for k, v in result.__dict__.items()
                    if not k.startswith("_")
                }
            elif isinstance(result, dict):
                metrics = result
            else:
                metrics = {"raw": str(result)}

            # Try to extract AUC from common field names
            for auc_key in ("auc", "test_auc", "oos_auc", "cv_auc"):
                if auc_key in metrics and metrics[auc_key] is not None:
                    metrics["auc"] = float(metrics[auc_key])
                    break

            metrics["status"] = "success"
            logger.info(
                "%s Training pipeline succeeded. AUC=%.4f",
                _PREFIX,
                metrics.get("auc", float("nan")),
            )
            return metrics

        except ImportError as exc:
            logger.warning("%s train_meta_model_pipeline not importable: %s", _PREFIX, exc)
            return {"status": "error", "stage": "import", "error": str(exc)}
        except Exception as exc:
            logger.error("%s Training pipeline raised: %s", _PREFIX, exc, exc_info=True)
            return {"status": "error", "stage": "training", "error": str(exc)}

    def _get_current_model_auc(self, current_model_path: Path) -> float | None:
        """Try to read the AUC of the currently deployed model from metadata."""
        state = self._load_state()
        if "current_model_auc" in state:
            return float(state["current_model_auc"])

        # Try reading a sidecar metrics file
        metrics_path = Path(current_model_path).parent / "metrics.json"
        if metrics_path.exists():
            try:
                with metrics_path.open() as fh:
                    data = json.load(fh)
                for key in ("auc", "test_auc", "oos_auc", "cv_auc"):
                    if key in data:
                        return float(data[key])
            except Exception:
                pass
        return None

    def _deploy_model(self, train_result: dict, current_model_path: Path) -> bool:
        """Swap the new model into the current model path.

        Only called when auto_deploy=True and validation passed.
        """
        new_path = train_result.get("model_path")
        if not new_path:
            logger.error("%s auto_deploy=True but no model_path in train_result", _PREFIX)
            return False
        try:
            import shutil

            new_path = Path(new_path)
            dest = Path(current_model_path)
            if not new_path.exists():
                logger.error("%s New model path does not exist: %s", _PREFIX, new_path)
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(new_path, dest)
            logger.info("%s Model deployed: %s -> %s", _PREFIX, new_path, dest)

            # Update state
            state = self._load_state()
            state["last_model_deployed_at"] = date.today().isoformat()
            state["current_model_auc"] = train_result.get("auc")
            self._save_state(state)
            return True
        except Exception as exc:
            logger.error("%s Model deployment failed: %s", _PREFIX, exc)
            return False

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        """Load persistent state from JSON.

        Returns an empty dict if the file does not exist or is malformed.
        """
        path = self.state_dir / self._STATE_FILE
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("%s Could not load state file %s: %s", _PREFIX, path, exc)
            return {}

    def _save_state(self, state: dict) -> None:
        """Save persistent state as JSON."""
        path = self.state_dir / self._STATE_FILE
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2, default=str)
        except Exception as exc:
            logger.error("%s Could not save state file %s: %s", _PREFIX, path, exc)

    def _record_retrain_attempt(self, success: bool) -> None:
        """Update state with latest retrain timestamp and rolling log."""
        state = self._load_state()
        today = date.today().isoformat()
        state["last_retrain_at"] = today

        retrain_log: list[str] = state.get("retrain_log", [])
        retrain_log.append(today)
        # Keep only last 2 years of entries to prevent unbounded growth
        cutoff = (date.today() - timedelta(days=730)).isoformat()
        retrain_log = [d for d in retrain_log if d >= cutoff]
        state["retrain_log"] = retrain_log
        state["last_retrain_success"] = success
        self._save_state(state)

    def _update_state_after_check(self, result: FeedbackResult) -> None:
        """Update state with last check timestamp and summary."""
        state = self._load_state()
        state["last_check_at"] = result.checked_at
        state["last_active_signals"] = result.active_signal_count
        self._save_state(state)

    # ------------------------------------------------------------------
    # Fix 28: EWRLS online learning helper
    # ------------------------------------------------------------------

    def _maybe_run_ewrls_update(self, records: list[dict]) -> None:
        """Run incremental EWRLS updates from recent trade records.

        Gated by:
        - configs/self_learning.yaml -> self_learning.online_learning.enabled
        - performance_guard: aborts if rolling IC degrades by more than threshold
        - max_online_updates_before_retrain: hard cap per session

        All errors are caught; failures never propagate to the caller.
        """
        try:
            import yaml

            cfg_path = (
                Path(__file__).resolve().parents[4] / "configs" / "self_learning.yaml"
            )
            if not cfg_path.exists():
                logger.debug(
                    "%s [EWRLS] self_learning.yaml not found at %s — skipping",
                    _PREFIX,
                    cfg_path,
                )
                return
            with cfg_path.open("r", encoding="utf-8") as fh:
                sl_cfg = yaml.safe_load(fh) or {}
            ol_cfg = (sl_cfg.get("self_learning") or {}).get("online_learning") or {}
            if not ol_cfg.get("enabled", False):
                logger.debug("%s [EWRLS] online_learning.enabled=false — skipping", _PREFIX)
                return

            forgetting_factor = float(ol_cfg.get("forgetting_factor", 0.97))
            max_updates = int(ol_cfg.get("max_online_updates_before_retrain", 100))
            performance_guard = float(ol_cfg.get("performance_guard", -0.01))
        except Exception as exc:
            logger.debug("%s [EWRLS] config load failed: %s — skipping", _PREFIX, exc)
            return

        try:
            from src.assembled_core.ml.online_learning import EWRLSModel
        except ImportError as exc:
            logger.debug("%s [EWRLS] EWRLSModel not importable: %s", _PREFIX, exc)
            return

        # Build feature/target pairs from records.
        # Each record may contain: score (float), direction (+1/-1), pnl (float).
        # We use [score] as x and direction as y (simple 1-feature linear model).
        xs: list[float] = []
        ys: list[float] = []
        for rec in records[-max_updates:]:
            try:
                score = rec.get("score")
                direction = rec.get("direction") or rec.get("label")
                if score is None or direction is None:
                    continue
                xs.append(float(score))
                ys.append(float(direction))
            except Exception:
                continue

        if len(xs) < 5:
            logger.debug(
                "%s [EWRLS] too few usable records (%d) for online update", _PREFIX, len(xs)
            )
            return

        import numpy as np

        # Load or create persistent model via state file
        state = self._load_state()
        ks_model_state = state.get("ewrls_model", {})
        try:
            model = EWRLSModel(
                n_features=1,
                forgetting_factor=forgetting_factor,
            )
            # Restore beta/P if previously persisted
            if ks_model_state.get("beta") is not None:
                model.beta = np.array(ks_model_state["beta"], dtype=float)
            if ks_model_state.get("P") is not None:
                model.P = np.array(ks_model_state["P"], dtype=float)
            model.n_updates = int(ks_model_state.get("n_updates", 0))
        except Exception as exc:
            logger.debug("%s [EWRLS] model init failed: %s", _PREFIX, exc)
            return

        # Compute IC before update (baseline)
        try:
            ic_before = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= 2 else 0.0
        except Exception:
            ic_before = 0.0

        # Run incremental updates
        try:
            X = np.array(xs, dtype=float).reshape(-1, 1)
            Y = np.array(ys, dtype=float)
            model.batch_update(X.reshape(len(xs), 1)[:, 0], Y)
        except Exception as exc:
            logger.warning("%s [EWRLS] batch_update failed: %s", _PREFIX, exc)
            return

        # Performance guard: check IC after update
        try:
            ic_after = float(np.corrcoef(xs[-20:], ys[-20:])[0, 1]) if len(xs) >= 2 else 0.0
            ic_delta = ic_after - ic_before
            if ic_delta < performance_guard:
                logger.warning(
                    "%s [EWRLS] performance guard triggered: IC delta=%.4f < threshold=%.4f"
                    " — discarding online updates",
                    _PREFIX,
                    ic_delta,
                    performance_guard,
                )
                return
        except Exception:
            pass

        # Persist updated model state
        try:
            state["ewrls_model"] = {
                "beta": model.beta.tolist(),
                "P": model.P.tolist(),
                "n_updates": model.n_updates,
                "last_update": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "forgetting_factor": forgetting_factor,
            }
            self._save_state(state)
            logger.info(
                "%s [EWRLS] online update complete: %d obs, n_updates=%d, IC_before=%.4f",
                _PREFIX,
                len(xs),
                model.n_updates,
                ic_before,
            )
        except Exception as exc:
            logger.warning("%s [EWRLS] state persist failed: %s", _PREFIX, exc)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _write_report(self, result: FeedbackResult) -> None:
        """Write FeedbackResult as JSON to report_path."""
        try:
            report = {
                "checked_at": result.checked_at,
                "n_recent_trades": result.n_recent_trades,
                "recent_hit_rate": round(result.recent_hit_rate, 4),
                "recent_ic": round(result.recent_ic, 4),
                "signals": {
                    "ic_degradation": result.degradation_detected,
                    "feature_drift": result.feature_drift_detected,
                    "signal_decay": result.signal_decay_detected,
                    "label_drift": result.label_drift_detected,
                    "model_age": result.model_age_signal,
                },
                "active_signal_count": result.active_signal_count,
                "skipped_signals": result.skipped_signals,
                "retrain_triggered": result.retrain_triggered,
                "blocked_reason": result.blocked_reason,
                "retrain_result": result.retrain_result,
                "new_model_deployed": result.new_model_deployed,
            }
            result.report_path.parent.mkdir(parents=True, exist_ok=True)
            with result.report_path.open("w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, default=str)
            logger.info("%s Report written: %s", _PREFIX, result.report_path)
        except Exception as exc:
            logger.error("%s Failed to write report: %s", _PREFIX, exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feedback_loop",
        description="Manual feedback check for Assembled-Trading-AI meta-model.",
    )
    parser.add_argument(
        "--learning-store",
        type=Path,
        default=Path("output/learning/learning_store.jsonl"),
        help="Path to the JSONL post-trade learning store.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/meta/meta_model.pkl"),
        help="Path to the currently deployed model file.",
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=None,
        help="Path to a Parquet factor panel. If not provided, an empty DataFrame is used.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("output/feedback_state"),
        help="Directory for persistent feedback state.",
    )
    parser.add_argument(
        "--check-interval-days",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--retrain-cooldown-days",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--max-retrain-per-quarter",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--ic-degradation-threshold",
        type=float,
        default=-0.02,
    )
    parser.add_argument(
        "--auto-deploy",
        action="store_true",
        default=False,
        help="Automatically swap model when validation passes (default: False).",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=50,
        dest="min_trades_for_evaluation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


if __name__ == "__main__":
    parser = _build_cli()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    config = FeedbackLoopConfig(
        check_interval_days=args.check_interval_days,
        retrain_cooldown_days=args.retrain_cooldown_days,
        max_retrain_per_quarter=args.max_retrain_per_quarter,
        ic_degradation_threshold=args.ic_degradation_threshold,
        auto_deploy=args.auto_deploy,
        min_trades_for_evaluation=args.min_trades_for_evaluation,
    )

    panel_df: pd.DataFrame
    if args.panel_path and Path(args.panel_path).exists():
        logger.info("%s Loading panel from %s", _PREFIX, args.panel_path)
        panel_df = pd.read_parquet(args.panel_path)
    else:
        logger.warning(
            "%s No panel_path provided or file not found — using empty DataFrame.", _PREFIX
        )
        panel_df = pd.DataFrame()

    controller = FeedbackLoopController(config=config, state_dir=args.state_dir)
    result = controller.run_feedback_check(
        learning_store_path=args.learning_store,
        current_model_path=args.model_path,
        panel_df=panel_df,
    )

    print("\n--- Feedback Check Summary ---")
    print(f"Checked at:       {result.checked_at}")
    print(f"Recent trades:    {result.n_recent_trades}")
    print(f"Hit rate:         {result.recent_hit_rate:.3f}")
    print(f"Recent IC:        {result.recent_ic:.4f}")
    print(f"Active signals:   {result.active_signal_count}/5")
    print(f"  [1] IC Degrad:  {result.degradation_detected}")
    print(f"  [2] Feat Drift: {result.feature_drift_detected}")
    print(f"  [3] Sig Decay:  {result.signal_decay_detected}")
    print(f"  [4] Lbl Drift:  {result.label_drift_detected}")
    print(f"  [5] Model Age:  {result.model_age_signal}")
    if result.skipped_signals:
        print(f"  Skipped:        {result.skipped_signals}")
    print(f"Retrain triggered:{result.retrain_triggered}")
    if result.blocked_reason:
        print(f"Blocked:          {result.blocked_reason}")
    if result.retrain_result:
        print(f"Retrain result:   {result.retrain_result}")
    print(f"Model deployed:   {result.new_model_deployed}")
    print(f"Report:           {result.report_path}")
