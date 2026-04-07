"""Feature Flag Audit (Plan 11.6).

Validate that each feature builder respects its flag in policy config.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Known feature flags and their controlling config keys
FEATURE_FLAGS: dict[str, str] = {
    "ta_features": "features.ta.enabled",
    "macro_features": "features.macro.enabled",
    "earnings_features": "features.earnings.enabled",
    "insider_features": "features.insider.enabled",
    "congress_features": "features.congress.enabled",
    "options_features": "features.options.enabled",
    "disclosure_features": "features.disclosure.enabled",
    "satellite_features": "features.satellite.enabled",
    "intel_features": "features.intel.enabled",
    "geopolitical_features": "features.geopolitical.enabled",
}


def audit_feature_flags(
    policy: dict,
    active_features: list[str] | None = None,
) -> dict:
    """Audit feature flags against policy configuration.

    Args:
        policy: Policy configuration dict.
        active_features: List of currently active feature names.

    Returns:
        Dict with enabled, disabled, missing, and violations.
    """
    enabled = []
    disabled = []
    missing = []

    features_config = policy.get("features", {})

    for flag_name, config_path in FEATURE_FLAGS.items():
        parts = config_path.split(".")
        value = policy
        found = True
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                found = False
                break

        if not found:
            missing.append(flag_name)
        elif value:
            enabled.append(flag_name)
        else:
            disabled.append(flag_name)

    # Check for violations: feature active but flag disabled
    violations = []
    if active_features:
        for feat in active_features:
            if feat in disabled:
                violations.append(f"{feat} is active but flag is disabled")

    return {
        "enabled": enabled,
        "disabled": disabled,
        "missing": missing,
        "violations": violations,
        "n_total": len(FEATURE_FLAGS),
    }


__all__ = ["FEATURE_FLAGS", "audit_feature_flags"]
