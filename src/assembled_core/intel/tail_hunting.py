"""EDCL Phase G — Tail-Hunting Execution Layer.

Reads pre-positioned trade plans from configs/tail_hunting_v1.yaml and matches
them against the active TriggerBasket. When a plan's trigger types overlap with
fired basket triggers AND conviction >= activation_conviction, the plan is
returned as an active TailHuntSignal for the pipeline to execute.

All plans are disabled by default (enabled: false in YAML). Enable only after
30-day paper-run validation per the EDCL activation sequence in decisions.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parents[3] / "configs" / "tail_hunting_v1.yaml"


@dataclass
class TailHuntSignal:
    """A matched and activated tail-hunt trade plan."""

    event_name: str
    direction: str  # "long" or "short"
    primary_assets: list[str]
    hedge_assets: list[str]
    max_position_size: float
    activation_conviction: float
    current_conviction: float
    description: str = ""
    matched_triggers: list[str] = field(default_factory=list)

    def size_fraction(self) -> float:
        """Linear scale from activation_conviction → 1.0 maps to 0 → max_position_size."""
        denom = (
            1.0 - self.activation_conviction
            if self.activation_conviction < 1.0
            else 1.0
        )
        scale = max(
            0.0,
            min(1.0, (self.current_conviction - self.activation_conviction) / denom),
        )
        return self.max_position_size * scale

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_name": self.event_name,
            "direction": self.direction,
            "primary_assets": self.primary_assets,
            "hedge_assets": self.hedge_assets,
            "max_position_size": self.max_position_size,
            "size_fraction": self.size_fraction(),
            "activation_conviction": self.activation_conviction,
            "current_conviction": self.current_conviction,
            "matched_triggers": self.matched_triggers,
            "description": self.description,
        }


def load_tail_plans(config_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load and parse tail_hunting_v1.yaml. Returns {event_name: plan_dict}.

    Returns empty dict on any error (graceful degradation).
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("tail_events", {})
    except FileNotFoundError:
        log.debug("tail_hunting config not found at %s — Phase G disabled", path)
        return {}
    except Exception as exc:
        log.warning("tail_hunting config load failed (%s): %s", path, exc)
        return {}


def match_tail_plans(
    basket: Any,
    conviction: float,
    config_path: str | Path | None = None,
) -> list[TailHuntSignal]:
    """Match active TriggerBasket against tail-hunt plans.

    Args:
        basket: TriggerBasket from build_trigger_basket() (Phase B).
        conviction: EDCL conviction score from compute_conviction_score() (Phase C).
        config_path: Override path for tail_hunting_v1.yaml.

    Returns:
        List of activated TailHuntSignal objects. Empty if no plans match or
        all matching plans are disabled.
    """
    if basket is None or not basket.is_active():
        return []

    plans = load_tail_plans(config_path)
    if not plans:
        return []

    # Build set of fired trigger type names (upper-cased for case-insensitive match)
    fired_names = {ttype.name.upper() for ttype, _ in (basket.fired_triggers or [])}

    signals: list[TailHuntSignal] = []
    for event_name, plan in plans.items():
        if not plan.get("enabled", False):
            continue

        required_triggers = [t.upper() for t in (plan.get("triggers") or [])]
        if not required_triggers:
            continue

        matched = [t for t in required_triggers if t in fired_names]
        if not matched:
            continue

        activation_threshold = float(plan.get("activation_conviction", 0.70))
        if conviction < activation_threshold:
            log.debug(
                "[TAIL-G] %s: conviction %.3f < threshold %.3f — skip",
                event_name,
                conviction,
                activation_threshold,
            )
            continue

        sig = TailHuntSignal(
            event_name=event_name,
            direction=str(plan.get("direction", "long")).lower(),
            primary_assets=list(plan.get("primary_assets") or []),
            hedge_assets=list(plan.get("hedge_assets") or []),
            max_position_size=float(plan.get("max_position_size", 0.15)),
            activation_conviction=activation_threshold,
            current_conviction=conviction,
            description=str(plan.get("description", "")),
            matched_triggers=matched,
        )
        signals.append(sig)
        log.info(
            "[TAIL-G] %s ACTIVATED: conviction=%.3f direction=%s primary=%s size=%.2f",
            event_name,
            conviction,
            sig.direction,
            sig.primary_assets,
            sig.size_fraction(),
        )

    return signals


def tail_signals_to_targets(
    signals: list[TailHuntSignal],
    existing_targets: "dict[str, float] | None" = None,
) -> dict[str, float]:
    """Convert TailHuntSignals to position weight targets.

    Long plans: add weight to primary_assets, subtract from hedge_assets.
    Short plans: subtract weight from primary_assets, add to hedge_assets.

    Args:
        signals: Activated TailHuntSignal list from match_tail_plans().
        existing_targets: Current target positions to overlay on. Defaults to {}.

    Returns:
        Updated position targets dict {ticker: weight}.
    """
    targets: dict[str, float] = dict(existing_targets or {})

    for sig in signals:
        frac = sig.size_fraction()
        if frac <= 0:
            continue

        per_asset = frac / max(len(sig.primary_assets), 1)
        per_hedge = frac / max(len(sig.hedge_assets), 1) if sig.hedge_assets else 0.0

        if sig.direction == "long":
            for ticker in sig.primary_assets:
                targets[ticker] = targets.get(ticker, 0.0) + per_asset
            for ticker in sig.hedge_assets:
                targets[ticker] = targets.get(ticker, 0.0) - per_hedge
        else:  # short
            for ticker in sig.primary_assets:
                targets[ticker] = targets.get(ticker, 0.0) - per_asset
            for ticker in sig.hedge_assets:
                targets[ticker] = targets.get(ticker, 0.0) + per_hedge

    return targets


__all__ = [
    "TailHuntSignal",
    "load_tail_plans",
    "match_tail_plans",
    "tail_signals_to_targets",
]
