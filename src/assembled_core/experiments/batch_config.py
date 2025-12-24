"""Batch configuration for systematic backtest runs.

This module provides a stable contract for batch backtest configurations with:
- Individual run specifications (RunSpec)
- Grid search expansion (parameter sweeps)
- Deterministic ordering and seed support
- Output structure and tagging

Example YAML config:
    batch_name: "ai_tech_core_vs_mlalpha"
    description: "Compare AI/Tech core bundle vs. ML-alpha bundles"
    output_root: "output/batch_backtests"
    run_tag: "experiment_2025"
    seed: 42
    max_workers: 4
    fail_fast: false

    base_args:
      freq: "1d"
      data_source: "local"
      strategy: "multifactor_long_short"
      rebalance_freq: "M"
      max_gross_exposure: 1.0
      start_capital: 100000.0

    runs:
      - id: "core_2015_2020"
        bundle_path: "config/factor_bundles/ai_tech_core_bundle.yaml"
        start_date: "2015-01-01"
        end_date: "2020-12-31"

    # Optional: grid search
    grid:
      max_gross_exposure: [0.6, 0.8, 1.0, 1.2]
      commission_bps: [0.0, 0.5, 1.0]
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class RunSpec(BaseModel):
    """Specification for a single backtest run.

    Attributes:
        id: Unique identifier for the run (used in output paths)
        bundle_path: Path to factor bundle YAML file
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        tags: Optional list of tags for experiment tracking
        overrides: Optional dict of parameter overrides (merged with base_args)
    """

    id: str = Field(..., description="Unique run identifier")
    bundle_path: Path = Field(..., description="Path to factor bundle YAML file")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    tags: list[str] = Field(default_factory=list, description="Tags for experiment tracking")
    overrides: dict[str, Any] = Field(
        default_factory=dict, description="Parameter overrides (merged with base_args)"
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate run ID: alphanumeric, underscore, hyphen only."""
        if not v:
            raise ValueError("run id must not be empty")
        if not all(c.isalnum() or c in ("_", "-") for c in v):
            raise ValueError(f"run id must contain only alphanumeric, underscore, hyphen: {v}")
        return v

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date format: YYYY-MM-DD."""
        parts = v.split("-")
        if len(parts) != 3:
            raise ValueError(f"date must be in YYYY-MM-DD format: {v}")
        try:
            int(parts[0])  # year
            int(parts[1])  # month
            int(parts[2])  # day
        except ValueError as e:
            raise ValueError(f"date must be in YYYY-MM-DD format: {v}") from e
        return v

    @field_validator("bundle_path", mode="before")
    @classmethod
    def validate_bundle_path(cls, v: Any) -> Path:
        """Convert bundle_path to Path."""
        if isinstance(v, Path):
            return v
        if isinstance(v, str):
            return Path(v)
        raise ValueError(f"bundle_path must be str or Path, got {type(v)}")


class BatchConfig(BaseModel):
    """Batch configuration for multiple backtest runs.

    Attributes:
        batch_name: Unique name for the batch (used in output paths)
        description: Human-readable description
        output_root: Base output directory
        run_tag: Optional tag for all runs (appended to run IDs)
        seed: Optional random seed for reproducibility
        max_workers: Maximum number of parallel workers (default: 1)
        fail_fast: If True, abort batch on first failure (default: False)
        base_args: Base arguments for all runs (merged with run overrides)
        runs: List of individual run specifications
        grid: Optional grid search specification (expanded to runs)
    """

    batch_name: str = Field(..., description="Unique batch name")
    description: str = Field(..., description="Human-readable description")
    output_root: Path = Field(..., description="Base output directory")
    run_tag: str | None = Field(None, description="Optional tag for all runs")
    seed: int | None = Field(None, description="Optional random seed")
    max_workers: int = Field(1, ge=1, description="Maximum parallel workers")
    fail_fast: bool = Field(False, description="Abort on first failure")
    base_args: dict[str, Any] = Field(
        default_factory=dict, description="Base arguments for all runs"
    )
    runs: list[RunSpec] = Field(default_factory=list, description="Individual run specifications")
    grid: dict[str, list[Any]] | None = Field(
        None, description="Optional grid search (expanded to runs)"
    )

    @field_validator("batch_name")
    @classmethod
    def validate_batch_name(cls, v: str) -> str:
        """Validate batch name: alphanumeric, underscore, hyphen only."""
        if not v:
            raise ValueError("batch_name must not be empty")
        if not all(c.isalnum() or c in ("_", "-") for c in v):
            raise ValueError(
                f"batch_name must contain only alphanumeric, underscore, hyphen: {v}"
            )
        return v

    @field_validator("output_root", mode="before")
    @classmethod
    def validate_output_root(cls, v: Any) -> Path:
        """Convert output_root to Path."""
        if isinstance(v, Path):
            return v
        if isinstance(v, str):
            return Path(v)
        raise ValueError(f"output_root must be str or Path, got {type(v)}")

    def expand_runs(self) -> list[RunSpec]:
        """Expand grid search to individual runs.

        If grid is specified, generates all combinations and creates RunSpec instances.
        If no grid, returns runs as-is.

        Returns:
            List of RunSpec instances (expanded from grid if applicable)

        Raises:
            ValueError: If both runs and grid are specified, or if grid is empty
        """
        if self.grid is None:
            return self.runs.copy()

        if self.runs:
            raise ValueError(
                "Cannot specify both 'runs' and 'grid' in batch config. "
                "Use 'grid' for parameter sweeps or 'runs' for individual runs."
            )

        if not self.grid:
            raise ValueError("grid must not be empty if specified")

        # Generate all combinations
        keys = list(self.grid.keys())
        values = list(self.grid.values())
        combinations = list(itertools.product(*values))

        expanded_runs: list[RunSpec] = []

        for idx, combo in enumerate(combinations):
            # Build run ID from grid values (sanitize values for valid IDs)
            def sanitize_value(v: Any) -> str:
                """Sanitize value for use in run ID (alphanumeric, underscore, hyphen only)."""
                s = str(v).replace(".", "_").replace("-", "_")
                # Remove any remaining invalid characters
                return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in s)

            run_id_parts = [f"{k}_{sanitize_value(v)}" for k, v in zip(keys, combo)]
            run_id = "_".join(run_id_parts)

            # Add run_tag if specified
            if self.run_tag:
                run_id = f"{self.run_tag}_{run_id}"

            # Build overrides dict from grid combination
            grid_overrides = dict(zip(keys, combo))

            # Merge with base_args (grid values take precedence)
            merged_args = {**self.base_args, **grid_overrides}

            # Extract required fields for RunSpec (must be in base_args)
            bundle_path_raw = merged_args.get("bundle_path")
            start_date = merged_args.get("start_date")
            end_date = merged_args.get("end_date")

            if not bundle_path_raw or not start_date or not end_date:
                raise ValueError(
                    f"grid expansion: base_args must contain bundle_path, start_date, end_date. "
                    f"Got base_args keys: {list(self.base_args.keys())}"
                )

            bundle_path = Path(bundle_path_raw)

            # Remove required fields from overrides (they're not overrides, they're required)
            overrides = {k: v for k, v in merged_args.items() if k not in ("bundle_path", "start_date", "end_date")}

            run_spec = RunSpec(
                id=run_id,
                bundle_path=bundle_path,
                start_date=start_date,
                end_date=end_date,
                tags=["grid_search"] + (self.run_tag.split("_") if self.run_tag else []),
                overrides=overrides,
            )

            expanded_runs.append(run_spec)

        # Sort for deterministic ordering
        expanded_runs.sort(key=lambda r: r.id)

        logger.info(
            f"Expanded grid search: {len(combinations)} combinations from {len(keys)} parameters"
        )

        return expanded_runs


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file with optional PyYAML dependency.

    Args:
        path: Path to YAML file

    Returns:
        Parsed dictionary

    Raises:
        RuntimeError: If PyYAML is not installed
        FileNotFoundError: If file does not exist
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. "
            "Install via 'pip install pyyaml' or use JSON config instead."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed dictionary

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If JSON is invalid
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_batch_config(path: Path) -> BatchConfig:
    """Load and validate batch configuration from YAML/JSON.

    This function provides clear error messages for common validation issues:
    - Missing required fields
    - Invalid date formats
    - Invalid run IDs or batch names
    - Grid expansion errors

    Args:
        path: Path to YAML or JSON config file

    Returns:
        Validated BatchConfig instance

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config is invalid (with descriptive error message)
        RuntimeError: If YAML is requested but PyYAML is not installed
    """
    if not path.exists():
        raise FileNotFoundError(f"Batch config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        raw = _load_yaml(path)
    elif suffix == ".json":
        raw = _load_json(path)
    else:
        raise ValueError(
            f"Unsupported config file extension: {path.suffix}. "
            "Supported: .yaml, .yml, .json"
        )

    if not isinstance(raw, dict):
        raise ValueError(
            f"Batch config root must be a mapping/object, got {type(raw).__name__}"
        )

    # Resolve relative paths relative to config file directory
    config_dir = path.parent.resolve()

    # Normalize paths in raw dict
    if "output_root" in raw and isinstance(raw["output_root"], str):
        output_root_raw = raw["output_root"]
        if not Path(output_root_raw).is_absolute():
            raw["output_root"] = str(config_dir / output_root_raw)
        else:
            raw["output_root"] = output_root_raw

    # Normalize bundle_path in runs
    if "runs" in raw and isinstance(raw["runs"], list):
        for run in raw["runs"]:
            if isinstance(run, dict) and "bundle_path" in run:
                bundle_path_raw = run["bundle_path"]
                if isinstance(bundle_path_raw, str) and not Path(bundle_path_raw).is_absolute():
                    run["bundle_path"] = str(config_dir / bundle_path_raw)
                elif isinstance(bundle_path_raw, str):
                    run["bundle_path"] = bundle_path_raw

    # Normalize bundle_path in base_args (for grid search)
    if "base_args" in raw and isinstance(raw["base_args"], dict):
        if "bundle_path" in raw["base_args"]:
            bundle_path_raw = raw["base_args"]["bundle_path"]
            if isinstance(bundle_path_raw, str) and not Path(bundle_path_raw).is_absolute():
                raw["base_args"]["bundle_path"] = str(config_dir / bundle_path_raw)
            elif isinstance(bundle_path_raw, str):
                raw["base_args"]["bundle_path"] = bundle_path_raw

    try:
        config = BatchConfig.model_validate(raw)
    except Exception as e:
        # Provide more context in error message
        raise ValueError(
            f"Failed to validate batch config from {path}: {e}\n"
            f"Config keys: {list(raw.keys())}"
        ) from e

    # Validate that runs or grid is specified
    if not config.runs and not config.grid:
        raise ValueError("Batch config must specify either 'runs' or 'grid'")

    # Validate that base_args contains required fields for grid
    if config.grid:
        required_base_args = {"bundle_path", "start_date", "end_date"}
        missing = required_base_args - set(config.base_args.keys())
        if missing:
            raise ValueError(
                f"Grid search requires base_args to contain: {required_base_args}. "
                f"Missing: {missing}"
            )

    return config

