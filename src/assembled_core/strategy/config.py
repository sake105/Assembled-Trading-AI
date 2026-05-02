"""Strategy configuration schema with Pydantic validation.

From 39_HYPERPARAMETER_GOVERNANCE.md §5.2.

Validates strategy YAML configs (composite weights, thresholds, risk params).
Works with or without PyYAML installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class CompositeWeights(BaseModel):
    mtf: float = Field(default=0.15, ge=0, le=1)
    classical_ta: float = Field(default=0.20, ge=0, le=1)
    microstructure: float = Field(default=0.10, ge=0, le=1)
    volume_profile: float = Field(default=0.10, ge=0, le=1)
    chart_pattern: float = Field(default=0.05, ge=0, le=1)
    vol_surface: float = Field(default=0.10, ge=0, le=1)
    breadth: float = Field(default=0.15, ge=0, le=1)
    seasonality: float = Field(default=0.05, ge=0, le=1)
    news: float = Field(default=0.10, ge=0, le=1)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "CompositeWeights":
        total = sum(self.model_dump().values())
        if not 0.98 <= total <= 1.02:
            raise ValueError(f"Composite weights must sum to ~1.0, got {total:.4f}")
        return self


class Thresholds(BaseModel):
    buy: float = Field(default=0.5, gt=0, lt=1)
    sell: float = Field(default=-0.5, gt=-1, lt=0)


class RiskParams(BaseModel):
    max_position_pct_of_equity: float = Field(default=0.05, gt=0, le=0.25)
    max_daily_loss_pct: float = Field(default=0.02, gt=0, le=0.10)
    kill_switch_loss_pct: float = Field(default=0.06, gt=0, le=0.20)


class StrategyConfig(BaseModel):
    strategy_id: str
    description: str = ""
    created: str = ""
    author: str = ""

    composite_weights: CompositeWeights = Field(default_factory=CompositeWeights)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    regime_multipliers: dict[str, float] = Field(
        default_factory=lambda: {"bull": 1.0, "bear": 0.7, "neutral": 0.9}
    )
    news: dict[str, Any] = Field(default_factory=dict)
    risk: RiskParams = Field(default_factory=RiskParams)
    model_versions: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StrategyConfig":
        """Load and validate a strategy config YAML file."""
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"[StrategyConfig] Malformed YAML in {path}: {exc}") from exc
        except ImportError:
            import json as _json
            with open(path, encoding="utf-8") as f:
                data = _json.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyConfig":
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


_ACTIVE_CONFIG_DIR = Path("config/strategies")


def load_strategy_config(path: Path | str) -> StrategyConfig:
    """Load a StrategyConfig from a YAML or JSON file path.

    Convenience wrapper for StrategyConfig.from_yaml() — preferred over
    calling from_yaml() directly so callers don't need to import the class.
    """
    return StrategyConfig.from_yaml(path)


def load_active_config(environment: str = "paper") -> StrategyConfig:
    """Load the active strategy config for the given environment.

    Looks for ``config/strategies/<environment>.yaml`` (or .json fallback).
    Falls back to a default StrategyConfig if the file does not exist.

    Args:
        environment: 'paper', 'live', or 'backtest'.

    Returns:
        Validated StrategyConfig.
    """
    for suffix in (".yaml", ".yml", ".json"):
        candidate = _ACTIVE_CONFIG_DIR / f"{environment}{suffix}"
        if candidate.exists():
            logger.info("Loading active config: %s", candidate)
            return StrategyConfig.from_yaml(candidate)

    logger.warning(
        "No active config found for environment '%s' in %s — using defaults",
        environment,
        _ACTIVE_CONFIG_DIR,
    )
    return StrategyConfig(strategy_id=f"default_{environment}")


__all__ = [
    "CompositeWeights",
    "Thresholds",
    "RiskParams",
    "StrategyConfig",
    "load_strategy_config",
    "load_active_config",
]
