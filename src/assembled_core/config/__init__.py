"""Configuration package for Assembled Trading AI.

This package provides:
- `config.py`: Legacy configuration (OUTPUT_DIR, SUPPORTED_FREQS)
- `settings.py`: New Pydantic Settings-based configuration (environment modes, paths)
"""
from __future__ import annotations

# Import legacy config for backward compatibility
from src.assembled_core.config.config import OUTPUT_DIR, SUPPORTED_FREQS, get_base_dir, get_output_path

# Import new settings
from src.assembled_core.config.settings import (
    Environment,
    RuntimeProfile,
    Settings,
    get_runtime_profile,
    get_settings,
    reset_settings,
)

__all__ = [
    # Legacy exports (for backward compatibility)
    "OUTPUT_DIR",
    "SUPPORTED_FREQS",
    "get_base_dir",
    "get_output_path",
    # New settings exports
    "Environment",
    "RuntimeProfile",
    "Settings",
    "get_runtime_profile",
    "get_settings",
    "reset_settings",
    # Factor bundles exports
    "FactorBundleConfig",
    "FactorConfig",
    "FactorBundleOptions",
    "load_factor_bundle",
    "list_available_factor_bundles",
]

# Import factor bundles (optional, to avoid circular imports)
try:
    from src.assembled_core.config.factor_bundles import (
        FactorBundleConfig,
        FactorConfig,
        FactorBundleOptions,
        load_factor_bundle,
        list_available_factor_bundles,
    )
except ImportError:
    # Factor bundles module may not be available in all contexts
    pass

