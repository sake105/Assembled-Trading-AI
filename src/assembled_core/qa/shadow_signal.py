"""Shadow-mode validation and canary deployment helpers.

From 32_VALIDIERUNG.md §32.6 / §32.7.

Shadow-mode: signals run alongside live trading for ≥60 days without
affecting positions.  Rolling IC / hit-rate / simulated Sharpe are
tracked to decide promotion.

Canary-deployment: graduated size ramp (0% → 10% → 33% → 100%) with
automatic rollback on unexpected drawdown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ShadowSignal
# ---------------------------------------------------------------------------

CANARY_SCHEDULE: list[tuple[int, int | None, float]] = [
    (0, 5, 0.0),  # days 1-5: shadow only
    (6, 20, 0.10),  # days 6-20: 10 % if Sharpe_15d > 0.5
    (21, 45, 0.33),  # days 21-45: 33 % if Sharpe > 0.5 + DD < 1.5× training
    (46, None, 1.0),  # day 46+: full size
]


@dataclass
class ShadowSignal:
    """Wraps a signal handler, runs it in shadow (no orders) until promoted.

    Args:
        name: Unique signal name.
        handler: Callable that receives context and returns a dict with
                 at least {'score': float, 'side': int, 'return_next': float}.
        live: If False (default), signal runs but no orders are placed.
    """

    name: str
    handler: Callable[[Any], dict[str, Any]]
    live: bool = False
    shadow_trades: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, context: Any) -> dict[str, Any]:
        """Run the handler, log the result, and optionally suppress ordering.

        Returns:
            Signal result dict with 'shadow' and 'signal_name' keys added.
        """
        try:
            result = self.handler(context)
        except Exception as exc:
            logger.warning("ShadowSignal %s handler error: %s", self.name, exc)
            result = {"score": 0.0, "side": 0, "error": str(exc)}

        result["shadow"] = not self.live
        result["signal_name"] = self.name
        result["emitted_at"] = datetime.now(tz=timezone.utc).isoformat()

        if not self.live:
            self.shadow_trades.append(result)

        logger.debug(
            "[%s shadow=%s] score=%.3f",
            self.name,
            not self.live,
            result.get("score", 0.0),
        )
        return result

    def rolling_ic(self, window: int = 60) -> float:
        """Spearman IC (score vs. next-period return) over last *window* trades."""
        recent = self.shadow_trades[-window:]
        if len(recent) < 5:
            return float("nan")
        scores = [t.get("score", 0.0) for t in recent]
        returns = [t.get("return_next", 0.0) for t in recent]
        try:
            from scipy.stats import spearmanr

            ic, _ = spearmanr(scores, returns)
            return float(ic)
        except ImportError:
            # manual rank correlation
            n = len(scores)
            rank_s = np.argsort(np.argsort(scores))
            rank_r = np.argsort(np.argsort(returns))
            d_sq = ((rank_s - rank_r) ** 2).sum()
            return float(1.0 - 6.0 * d_sq / max(n * (n**2 - 1), 1))

    def should_promote(
        self, ic_threshold: float = 0.03, sharpe_threshold: float = 0.5
    ) -> bool:
        """Return True if shadow IC and simulated Sharpe meet promotion criteria."""
        ic = self.rolling_ic(60)
        if np.isnan(ic) or ic < ic_threshold:
            return False
        returns = [
            t.get("return_next", 0.0) * np.sign(t.get("score", 0.0))
            for t in self.shadow_trades[-60:]
        ]
        if len(returns) < 10:
            return False
        sharpe = np.mean(returns) / max(np.std(returns), 1e-9) * np.sqrt(252)
        return float(sharpe) >= sharpe_threshold


# ---------------------------------------------------------------------------
# Canary deployment helpers
# ---------------------------------------------------------------------------


def canary_size(
    days_since_live: int,
    sharpe_15d: float,
    drawdown_ratio: float,
) -> float:
    """Return the target size fraction for a canary-deployed signal.

    Args:
        days_since_live: Calendar days since the signal went live (canary phase).
        sharpe_15d: Rolling 15-day annualised Sharpe of the signal.
        drawdown_ratio: Current drawdown / training-period 95th-percentile drawdown.

    Returns:
        Target size in [0, 1].
    """
    for start, end, target in CANARY_SCHEDULE:
        in_range = end is None or days_since_live <= end
        if days_since_live >= start and in_range:
            if target > 0.0:
                if sharpe_15d < 0.5 or drawdown_ratio > 1.5:
                    return 0.0  # pause
            return target
    return 1.0


def auto_rollback(drawdown_observed: float, drawdown_95q_simulated: float) -> bool:
    """Return True if the observed drawdown exceeds 2× the simulated 95th-pct.

    Caller is responsible for actually pausing the signal.
    """
    if drawdown_95q_simulated <= 0:
        return False
    return drawdown_observed > 2.0 * drawdown_95q_simulated


# ---------------------------------------------------------------------------
# Walk-forward drift detection
# ---------------------------------------------------------------------------


def detect_wf_drift(
    oos_sharpes: list[float],
    window: int = 5,
    alarm_threshold: float = -1.0,
) -> str:
    """Detect performance degradation across walk-forward folds.

    Args:
        oos_sharpes: OOS Sharpe ratios per fold in chronological order.
        window: Number of recent folds to compare against history.
        alarm_threshold: Z-score below which drift is signaled.

    Returns:
        "DRIFT" or "OK".
    """
    if len(oos_sharpes) <= window:
        return "OK"
    recent = np.mean(oos_sharpes[-window:])
    historical = np.array(oos_sharpes[:-window])
    mu, sigma = float(np.mean(historical)), float(np.std(historical))
    if sigma < 1e-9:
        return "OK"
    z = (recent - mu) / sigma
    return "DRIFT" if z < alarm_threshold else "OK"


__all__ = [
    "ShadowSignal",
    "canary_size",
    "auto_rollback",
    "detect_wf_drift",
    "CANARY_SCHEDULE",
]
