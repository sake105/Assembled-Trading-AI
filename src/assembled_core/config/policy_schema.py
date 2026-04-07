"""Pydantic schema for policy.yaml validation (MEDIUM-6.3).

Validates the structure and key fields of the trading policy configuration.
Validation is soft by default — warnings are logged for unknown or missing
fields, but the raw dict is always returned so the pipeline is not blocked.

Usage::

    from assembled_core.config.policy_schema import validate_policy
    from assembled_core.config.policy_loader import load_policy

    raw = load_policy()
    policy, warnings = validate_policy(raw)
    # policy is the original raw dict (unchanged)
    # warnings is a list of validation issue strings (empty if all ok)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_policy(policy: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate a loaded policy dict against expected structure.

    Does NOT raise on failure — returns the original dict plus a list of
    warning strings. Call this after load_policy() for soft validation.

    Args:
        policy: Raw policy dict from load_policy().

    Returns:
        Tuple of (policy dict unchanged, list of warning strings).
        Empty warnings list means validation passed.
    """
    warnings: list[str] = []

    if not policy:
        warnings.append("policy is empty or could not be loaded")
        return policy, warnings

    # --- policy_version ---
    if "policy_version" not in policy:
        warnings.append("missing field: policy_version")

    # --- scope ---
    scope = policy.get("scope") or {}
    if policy.get("scope") is None:
        warnings.append("missing section: scope")
    else:
        if scope.get("leverage_allowed") is True:
            warnings.append("scope.leverage_allowed=true — verify this is intentional")
        if scope.get("shorts_allowed") is True:
            warnings.append("scope.shorts_allowed=true — verify this is intentional")

    # --- risk_limits ---
    rl = policy.get("risk_limits") or {}
    if not rl:
        warnings.append("missing section: risk_limits")
    else:
        dd = rl.get("max_drawdown") or {}
        kill = dd.get("kill")
        hard = dd.get("hard")
        soft = dd.get("soft")
        if kill is None:
            warnings.append(
                "risk_limits.max_drawdown.kill not set — kill switch threshold unknown"
            )
        elif not (0 < kill <= 1.0):
            warnings.append(f"risk_limits.max_drawdown.kill={kill} out of range (0, 1]")
        if hard and soft and kill:
            if not (soft < hard < kill):
                warnings.append(
                    f"risk_limits.max_drawdown thresholds out of order: "
                    f"soft={soft} hard={hard} kill={kill} (expected soft < hard < kill)"
                )
        if rl.get("max_position_weight") is None:
            warnings.append("risk_limits.max_position_weight not set")
        elif not (0 < rl["max_position_weight"] <= 1.0):
            warnings.append(
                f"risk_limits.max_position_weight={rl['max_position_weight']} out of range (0, 1]"
            )

    # --- georisk_overlay ---
    geo = policy.get("georisk_overlay") or {}
    if geo.get("enabled") and not geo.get("mapping"):
        warnings.append("georisk_overlay.enabled=true but no state mapping defined")

    # --- market_stress ---
    ms = policy.get("market_stress") or {}
    if ms.get("enabled") and not ms.get("metrics"):
        warnings.append("market_stress.enabled=true but no metrics defined")

    # --- execution_policy ---
    ep = policy.get("execution_policy") or {}
    if ep.get("mode_default") not in (None, "paper", "live", "simulation"):
        warnings.append(
            f"execution_policy.mode_default='{ep['mode_default']}' "
            "is not one of: paper, live, simulation"
        )

    if warnings:
        for w in warnings:
            logger.warning("POLICY_VALIDATION: %s", w)
    else:
        logger.debug("POLICY_VALIDATION: all checks passed")

    return policy, warnings


# ---------------------------------------------------------------------------
# Cross-Field Policy Consistency Checks (Plan 11.1)
# ---------------------------------------------------------------------------


def validate_policy_consistency(policy: dict[str, Any]) -> list[str]:
    """Run cross-field consistency checks on policy.

    Checks:
    - max_short <= max_gross
    - max_weight <= max_gross
    - target_vol < max_vol (if both set)
    - kill > hard > soft drawdown thresholds
    - max_positions > 0 if enabled

    Args:
        policy: Raw policy dict.

    Returns:
        List of consistency violation strings (empty = all consistent).
    """
    violations: list[str] = []

    rl = policy.get("risk_limits") or {}
    scope = policy.get("scope") or {}

    # Exposure consistency
    max_gross = rl.get("max_gross_exposure")
    max_short = rl.get("max_short_gross")
    max_weight = rl.get("max_position_weight")

    if max_short is not None and max_gross is not None:
        if max_short > max_gross:
            violations.append(
                f"max_short_gross ({max_short}) > max_gross_exposure ({max_gross})"
            )

    if max_weight is not None and max_gross is not None:
        if max_weight > max_gross:
            violations.append(
                f"max_position_weight ({max_weight}) > max_gross_exposure ({max_gross})"
            )

    # Volatility consistency
    target_vol = rl.get("target_volatility")
    max_vol = rl.get("max_volatility")
    if target_vol is not None and max_vol is not None:
        if target_vol >= max_vol:
            violations.append(
                f"target_volatility ({target_vol}) >= max_volatility ({max_vol})"
            )

    # Drawdown ordering
    dd = rl.get("max_drawdown") or {}
    kill = dd.get("kill")
    hard = dd.get("hard")
    soft = dd.get("soft")
    if kill is not None and hard is not None and soft is not None:
        if not (soft < hard < kill):
            violations.append(
                f"Drawdown thresholds out of order: soft={soft}, hard={hard}, kill={kill}"
            )

    # Leverage check
    if not scope.get("leverage_allowed", False):
        if max_gross is not None and max_gross > 1.0:
            violations.append(
                f"max_gross_exposure ({max_gross}) > 1.0 but leverage_allowed=false"
            )

    # Max positions
    max_pos = rl.get("max_positions")
    if max_pos is not None and max_pos <= 0:
        violations.append(f"max_positions ({max_pos}) must be > 0")

    if violations:
        for v in violations:
            logger.warning("POLICY_CONSISTENCY: %s", v)

    return violations
