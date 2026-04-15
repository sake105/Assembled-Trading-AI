"""Self-Healing + Autonomous Operations (M35).

Implements automated recovery and escalation:
1. Data Pipeline Recovery: cascading data source fallback
2. Model Degradation -> Auto-Retrain trigger
3. Risk Escalation Ladder:
   - Feature Drift -> Model Retrain
   - IC Degradation -> Position Sizing Reduction
   - Drawdown > 10% -> Capital Reduction
   - Drawdown > 15% -> Strategy Switch to defensive
   - Drawdown > 20% -> Kill Switch

All actions are logged and auditable. No action is taken silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EscalationLevel(str, Enum):
    """Risk escalation levels."""
    NORMAL = "normal"
    WATCH = "watch"  # feature drift detected
    REDUCE = "reduce"  # IC degradation
    DEFENSIVE = "defensive"  # moderate drawdown
    CRITICAL = "critical"  # severe drawdown
    KILL = "kill"  # kill switch threshold


@dataclass
class EscalationState:
    """Current escalation state."""
    level: EscalationLevel
    trigger_reason: str
    triggered_at: str
    actions_taken: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class HealingAction:
    """Recorded self-healing action."""
    timestamp: str
    action_type: str
    description: str
    success: bool
    details: dict[str, Any] = field(default_factory=dict)


class DataSourceCascade:
    """Cascading data source fallback.

    If primary source fails, try secondary, then tertiary, etc.
    """

    def __init__(self) -> None:
        self._sources: list[tuple[str, Callable]] = []
        self._history: list[HealingAction] = []

    def register_source(self, name: str, fetch_fn: Callable) -> None:
        """Register a data source in priority order.

        Args:
            name: Source name (e.g., "yahoo", "polygon", "alpha_vantage").
            fetch_fn: Callable that fetches data. Should raise on failure.
        """
        self._sources.append((name, fetch_fn))

    def fetch(self, **kwargs) -> tuple[Any, str]:
        """Try sources in cascade order.

        Returns:
            Tuple of (data, source_name) from first successful source.

        Raises:
            RuntimeError: If all sources fail.
        """
        errors = []
        for name, fn in self._sources:
            try:
                data = fn(**kwargs)
                if data is not None:
                    self._history.append(HealingAction(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        action_type="data_fetch",
                        description=f"Fetched from {name}",
                        success=True,
                    ))
                    return data, name
            except Exception as e:
                errors.append((name, str(e)))
                self._history.append(HealingAction(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    action_type="data_fetch_fallback",
                    description=f"{name} failed: {e}",
                    success=False,
                ))
                logger.warning("[SelfHeal] %s failed: %s — trying next source", name, e)

        raise RuntimeError(f"All data sources failed: {errors}")

    @property
    def history(self) -> list[HealingAction]:
        return list(self._history)


class RiskEscalationLadder:
    """Risk escalation ladder with automatic responses.

    Monitors portfolio metrics and escalates through levels:
    1. Feature Drift → Model Retrain
    2. IC Degradation → Position Sizing Reduction
    3. Drawdown > threshold_1 → Capital Reduction
    4. Drawdown > threshold_2 → Strategy Switch
    5. Drawdown > threshold_3 → Kill Switch
    """

    def __init__(
        self,
        dd_reduce: float = 0.10,
        dd_defensive: float = 0.15,
        dd_kill: float = 0.20,
        ic_degradation_threshold: float = 0.0,
        sizing_reduction_factor: float = 0.5,
    ) -> None:
        self.dd_reduce = dd_reduce
        self.dd_defensive = dd_defensive
        self.dd_kill = dd_kill
        self.ic_degradation_threshold = ic_degradation_threshold
        self.sizing_reduction_factor = sizing_reduction_factor

        self._state = EscalationState(
            level=EscalationLevel.NORMAL,
            trigger_reason="initialized",
            triggered_at=datetime.now(timezone.utc).isoformat(),
        )
        self._history: list[EscalationState] = []

    def evaluate(
        self,
        current_drawdown: float,
        current_ic: float | None = None,
        feature_drift_score: float | None = None,
    ) -> EscalationState:
        """Evaluate current metrics and determine escalation level.

        Args:
            current_drawdown: Current drawdown as negative fraction (e.g., -0.12).
            current_ic: Current information coefficient.
            feature_drift_score: Feature drift score (0 = no drift, >1 = significant).

        Returns:
            Updated EscalationState.
        """
        dd = abs(current_drawdown)
        actions = []

        # Determine level based on worst condition
        if dd >= self.dd_kill:
            level = EscalationLevel.KILL
            reason = f"Drawdown {dd:.1%} >= kill threshold {self.dd_kill:.1%}"
            actions.append("KILL_SWITCH_ACTIVATED")
        elif dd >= self.dd_defensive:
            level = EscalationLevel.CRITICAL
            reason = f"Drawdown {dd:.1%} >= defensive threshold {self.dd_defensive:.1%}"
            actions.append("SWITCH_TO_DEFENSIVE_STRATEGY")
            actions.append(f"REDUCE_CAPITAL_BY_{int(self.sizing_reduction_factor*100)}%")
        elif dd >= self.dd_reduce:
            level = EscalationLevel.REDUCE
            reason = f"Drawdown {dd:.1%} >= reduce threshold {self.dd_reduce:.1%}"
            actions.append(f"REDUCE_SIZING_BY_{int(self.sizing_reduction_factor*100)}%")
        elif current_ic is not None and current_ic < self.ic_degradation_threshold:
            level = EscalationLevel.REDUCE
            reason = f"IC {current_ic:.4f} < threshold {self.ic_degradation_threshold}"
            actions.append("REDUCE_POSITION_SIZING")
            actions.append("TRIGGER_MODEL_RETRAIN")
        elif feature_drift_score is not None and feature_drift_score > 1.0:
            level = EscalationLevel.WATCH
            reason = f"Feature drift {feature_drift_score:.2f} > 1.0"
            actions.append("TRIGGER_MODEL_RETRAIN")
        else:
            level = EscalationLevel.NORMAL
            reason = "All metrics within bounds"

        new_state = EscalationState(
            level=level,
            trigger_reason=reason,
            triggered_at=datetime.now(timezone.utc).isoformat(),
            actions_taken=actions,
            metrics={
                "drawdown": round(current_drawdown, 6),
                "ic": round(current_ic, 6) if current_ic is not None else None,
                "feature_drift": round(feature_drift_score, 4) if feature_drift_score is not None else None,
            },
        )

        # Log escalation changes
        if level != self._state.level:
            logger.warning(
                "[RiskEscalation] %s -> %s: %s",
                self._state.level.value, level.value, reason,
            )
            self._history.append(new_state)

        self._state = new_state
        return new_state

    @property
    def current_state(self) -> EscalationState:
        return self._state

    @property
    def history(self) -> list[EscalationState]:
        return list(self._history)

    def get_sizing_multiplier(self) -> float:
        """Get position sizing multiplier based on current escalation.

        Returns:
            Float between 0 and 1. 1.0 = full size.
        """
        level = self._state.level
        if level == EscalationLevel.KILL:
            return 0.0
        elif level == EscalationLevel.CRITICAL:
            return self.sizing_reduction_factor * 0.5
        elif level == EscalationLevel.REDUCE:
            return self.sizing_reduction_factor
        elif level == EscalationLevel.WATCH:
            return 0.8
        return 1.0


__all__ = [
    "EscalationLevel",
    "EscalationState",
    "HealingAction",
    "DataSourceCascade",
    "RiskEscalationLadder",
]
