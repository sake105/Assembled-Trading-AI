"""Configuration package for Assembled Trading AI.

This package provides:
- `config.py`: Legacy configuration (OUTPUT_DIR, SUPPORTED_FREQS)
- `settings.py`: New Pydantic Settings-based configuration (environment modes, paths)
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

# Import legacy config for backward compatibility
from src.assembled_core.config.config import (
    OUTPUT_DIR,
    SUPPORTED_FREQS,
    get_base_dir,
    get_output_path,
)

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
    __all__.extend([
        "Environment",
        "RuntimeProfile",
        "Settings",
        "get_runtime_profile",
        "get_settings",
        "reset_settings",
    ])
except ImportError:
    # pydantic_settings not installed — settings features unavailable
    _logger.warning("[Config] pydantic_settings not installed — settings features unavailable")

# Import factor bundles (optional, to avoid circular imports)
try:
    from src.assembled_core.config.factor_bundles import (
        FactorBundleConfig,
        FactorConfig,
        FactorBundleOptions,
        load_factor_bundle,
        list_available_factor_bundles,
    )
    __all__.extend([
        "FactorBundleConfig",
        "FactorConfig",
        "FactorBundleOptions",
        "load_factor_bundle",
        "list_available_factor_bundles",
    ])
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
from src.assembled_core.config.secrets_loader import (  # noqa: F401
    get_secret,
    is_secret_set,
    load_env_file,
)
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
        # Secrets loader
        "load_env_file",
        "get_secret",
        "is_secret_set",
    ]
)
