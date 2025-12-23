"""Factor Bundle Configuration Module.

This module provides utilities for loading and managing factor bundle configurations
from YAML files. Factor bundles define combinations of factors with weights and
processing options for use in factor-based strategies.

Example:
    from src.assembled_core.config.factor_bundles import load_factor_bundle, list_available_factor_bundles

    # Load a bundle
    bundle = load_factor_bundle("config/factor_bundles/macro_world_etfs_core_bundle.yaml")

    # List all available bundles
    bundles = list_available_factor_bundles()
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import sys

# Add project root to path if needed
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class FactorConfig:
    """Configuration for a single factor in a bundle.

    Attributes:
        name: Factor name (must match column names in factor DataFrames).
               Can be any factor column name, including ML alpha factors
               (e.g., "ml_alpha_ridge_20d", "factor_mom", "returns_12m").
        weight: Weight in the bundle (should sum to 1.0 across all factors)
        direction: "positive" = higher values are better, "negative" = lower values are better
    """

    name: str
    weight: float
    direction: Literal["positive", "negative"]


@dataclass
class FactorBundleOptions:
    """Processing options for factor bundles.

    Attributes:
        winsorize: Whether to winsorize factor values (clip extreme values)
        winsorize_limits: Quantile limits for winsorization (e.g., [0.01, 0.99])
        zscore: Whether to z-score normalize factors (mean=0, std=1)
        neutralize_by: Optional field to neutralize factors against (e.g., "sector")
    """

    winsorize: bool = True
    winsorize_limits: list[float] = field(default_factory=lambda: [0.01, 0.99])
    zscore: bool = True
    neutralize_by: str | None = None


@dataclass
class FactorBundleConfig:
    """Configuration for a factor bundle.

    A factor bundle defines a combination of factors with weights and processing options
    for use in factor-based strategies.

    Attributes:
        universe: Universe identifier (e.g., "macro_world_etfs", "universe_ai_tech")
        factor_set: Factor set identifier (e.g., "core", "core+alt_full")
        horizon_days: Forward return horizon in days
        factors: List of factor configurations with weights and directions
        options: Processing options for the bundle
    """

    universe: str
    factor_set: str
    horizon_days: int
    factors: list[FactorConfig]
    options: FactorBundleOptions

    def __post_init__(self) -> None:
        """Validate bundle configuration after initialization."""
        # Check that weights sum to approximately 1.0
        total_weight = sum(f.weight for f in self.factors)
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(
                f"Factor weights must sum to 1.0, but sum to {total_weight:.4f}. "
                f"Bundle: {self.universe}, factor_set: {self.factor_set}"
            )

        # Check that all factors have valid directions
        for factor in self.factors:
            if factor.direction not in ("positive", "negative"):
                raise ValueError(
                    f"Factor direction must be 'positive' or 'negative', "
                    f"got '{factor.direction}' for factor '{factor.name}'"
                )

        # Check winsorize limits
        if self.options.winsorize:
            if len(self.options.winsorize_limits) != 2:
                raise ValueError(
                    f"winsorize_limits must have 2 elements, "
                    f"got {len(self.options.winsorize_limits)}"
                )
            lower, upper = self.options.winsorize_limits
            if not (0.0 <= lower < upper <= 1.0):
                raise ValueError(
                    f"winsorize_limits must be in [0, 1] with lower < upper, "
                    f"got [{lower}, {upper}]"
                )


def load_factor_bundle(path: str | Path) -> FactorBundleConfig:
    """Load a factor bundle configuration from a YAML file.

    Args:
        path: Path to YAML file (relative to project root or absolute)

    Returns:
        FactorBundleConfig with parsed configuration

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the YAML is invalid or configuration is invalid
        yaml.YAMLError: If the YAML cannot be parsed
    """
    path = Path(path)

    # If relative path, try relative to project root
    if not path.is_absolute():
        root_path = ROOT / path
        if root_path.exists():
            path = root_path
        elif path.exists():
            pass  # Use as-is if it exists
        else:
            raise FileNotFoundError(f"Factor bundle file not found: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Factor bundle file not found: {path}")

    # Load YAML
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file {path}: {e}") from e

    # Validate required fields
    required_fields = ["universe", "factor_set", "horizon_days", "factors"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(
            f"Missing required fields in bundle config: {', '.join(missing_fields)}. "
            f"File: {path}"
        )

    # Parse factors
    factors = []
    for factor_data in data["factors"]:
        if "name" not in factor_data:
            raise ValueError(f"Factor missing 'name' field in {path}")
        if "weight" not in factor_data:
            raise ValueError(
                f"Factor '{factor_data['name']}' missing 'weight' field in {path}"
            )
        if "direction" not in factor_data:
            raise ValueError(
                f"Factor '{factor_data['name']}' missing 'direction' field in {path}"
            )

        factors.append(
            FactorConfig(
                name=factor_data["name"],
                weight=float(factor_data["weight"]),
                direction=factor_data["direction"],
            )
        )

    # Parse options (with defaults)
    options_data = data.get("options", {})
    options = FactorBundleOptions(
        winsorize=options_data.get("winsorize", True),
        winsorize_limits=options_data.get("winsorize_limits", [0.01, 0.99]),
        zscore=options_data.get("zscore", True),
        neutralize_by=options_data.get("neutralize_by"),
    )

    # Create bundle config
    bundle = FactorBundleConfig(
        universe=data["universe"],
        factor_set=data["factor_set"],
        horizon_days=int(data["horizon_days"]),
        factors=factors,
        options=options,
    )

    return bundle


def list_available_factor_bundles(
    root_dir: str | Path = "config/factor_bundles",
) -> list[Path]:
    """List all available factor bundle YAML files.

    Args:
        root_dir: Root directory to search for bundle files (default: "config/factor_bundles")
                  Can be relative to project root or absolute

    Returns:
        List of Path objects to YAML files, sorted by name
    """
    root_path = Path(root_dir)

    # If relative path, try relative to project root
    if not root_path.is_absolute():
        root_path = ROOT / root_path

    if not root_path.exists():
        return []

    # Find all YAML files
    bundle_files = sorted(root_path.glob("*.yaml")) + sorted(root_path.glob("*.yml"))

    return bundle_files
