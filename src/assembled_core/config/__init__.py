"""Configuration package for Assembled Trading AI.

This package provides:
- Core path/frequency constants (OUTPUT_DIR, SUPPORTED_FREQS)
- `settings.py`: New Pydantic Settings-based configuration (environment modes, paths)
"""

from __future__ import annotations

import logging
from pathlib import Path

_logger = logging.getLogger(__name__)

# --- Core constants (previously in config/config.py) ---
_BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = _BASE_DIR / "output"
SUPPORTED_FREQS = ("1d", "5min")


def get_output_path(*parts: str) -> Path:
    """Get a path within the output directory."""
    return OUTPUT_DIR.joinpath(*parts)


def get_base_dir() -> Path:
    """Get the repository root directory."""
    return _BASE_DIR


# Base __all__ — always available (hard dependencies).
__all__ = [
    "OUTPUT_DIR",
    "SUPPORTED_FREQS",
    "get_base_dir",
    "get_output_path",
]

# Import new settings (pydantic_settings may not be available in all environments)
try:
    from src.assembled_core.config.settings import (
        Environment,
        RuntimeProfile,
        Settings,
        get_runtime_profile,
        get_settings,
        reset_settings,
    )

    __all__.extend(
        [
            "Environment",
            "RuntimeProfile",
            "Settings",
            "get_runtime_profile",
            "get_settings",
            "reset_settings",
        ]
    )
except ImportError:
    # pydantic_settings not installed — settings features unavailable
    _logger.warning(
        "[Config] pydantic_settings not installed — settings features unavailable"
    )

# Import factor bundles (optional, to avoid circular imports)
try:
    from src.assembled_core.config.factor_bundles import (
        FactorBundleConfig,
        FactorBundleOptions,
        FactorConfig,
        list_available_factor_bundles,
        load_factor_bundle,
    )

    __all__.extend(
        [
            "FactorBundleConfig",
            "FactorConfig",
            "FactorBundleOptions",
            "load_factor_bundle",
            "list_available_factor_bundles",
        ]
    )
except ImportError:
    # Factor bundles module may not be available in all contexts
    _logger.warning("[Config] factor_bundles module not available")

# Import config models (strict validation)
from src.assembled_core.config.models import (  # noqa: F401
    FeatureConfig,
    GateConfig,
    GateThresholdConfig,
    RiskConfig,
    SignalConfig,
    ensure_feature_config,
    ensure_gate_config,
    ensure_risk_config,
    ensure_signal_config,
)

# Wired 2026-04-22: previously orphan secrets_loader
from src.assembled_core.logging_config import (  # noqa: F401
    JSONFormatter,
    configure_json_logging,
)

__all__.extend(
    [
        # Config models
        "FeatureConfig",
        "SignalConfig",
        "RiskConfig",
        "GateConfig",
        "GateThresholdConfig",
        # Helper functions
        "ensure_feature_config",
        "ensure_signal_config",
        "ensure_risk_config",
        "ensure_gate_config",
    ]
)
