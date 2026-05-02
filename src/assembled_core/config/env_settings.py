"""Multi-environment settings with type-safe config and risk-limit validation.

From 36_MULTI_ENVIRONMENT_SETUP.md §2.

This module adds DEV/STAGING/PROD environment separation on top of the
existing settings.py (which stays untouched). Use get_env_settings() for
new code that needs strict environment isolation.

Environments:
  dev      — offline/mock; never calls live Alpaca
  staging  — Alpaca Paper; acceptance testing ≥7 days before promote
  prod     — Alpaca Live; real money

ATA_ENVIRONMENT env-var selects the environment (default: dev).
Per-environment .env files live in config/env/.env.{dev,staging,prod}.
"""
from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Env(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class TradingMode(str, Enum):
    MOCK = "mock"      # no external API calls; fixture responses
    PAPER = "paper"    # Alpaca Paper
    LIVE = "live"      # Alpaca Live (real money)


# ---------------------------------------------------------------------------
# Nested settings blocks
# ---------------------------------------------------------------------------

class AlpacaConfig(BaseSettings):
    """Alpaca credentials. Optional in MOCK mode."""
    api_key: SecretStr = Field(default=SecretStr(""))
    secret_key: SecretStr = Field(default=SecretStr(""))
    base_url: str = Field(default="https://paper-api.alpaca.markets/")
    data_url: str = Field(default="https://data.alpaca.markets/")

    model_config = SettingsConfigDict(env_prefix="ATA_ALPACA_", extra="ignore")


class RiskLimits(BaseSettings):
    """Hard trading limits. Must be explicit per environment.

    kill_switch_loss must exceed max_daily_loss (enforced by validator).
    """
    max_position_usd: float = Field(default=1_000.0, gt=0)
    max_daily_loss_usd: float = Field(default=500.0, gt=0)
    max_open_positions: int = Field(default=10, ge=1, le=100)
    kill_switch_loss_usd: float = Field(default=1_500.0, gt=0)

    @field_validator("kill_switch_loss_usd", mode="before")
    @classmethod
    def _coerce_float(cls, v: Any) -> float:
        return float(v)

    @model_validator(mode="after")
    def kill_switch_exceeds_daily_loss(self) -> "RiskLimits":
        if self.kill_switch_loss_usd <= self.max_daily_loss_usd:
            raise ValueError(
                f"kill_switch_loss_usd ({self.kill_switch_loss_usd}) must be "
                f"> max_daily_loss_usd ({self.max_daily_loss_usd})"
            )
        return self

    model_config = SettingsConfigDict(env_prefix="ATA_RISK_", extra="ignore")


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------

class EnvSettings(BaseSettings):
    """Environment-aware settings. Load via get_env_settings().

    Required env-vars (can be in per-env .env file):
      ATA_ENVIRONMENT  = dev | staging | prod
      ATA_TRADING_MODE = mock | paper | live
      ATA_RISK_MAX_POSITION_USD, ATA_RISK_MAX_DAILY_LOSS_USD, ...

    Optional:
      ATA_LOG_LEVEL            (default: INFO)
      ATA_ENABLE_NEWS_FEATURES (default: true)
      ATA_ENABLE_SHADOW_MODE   (default: false)
    """
    environment: Env = Field(default=Env.DEV)
    trading_mode: TradingMode = Field(default=TradingMode.MOCK)
    log_level: str = Field(default="INFO")
    enable_news_features: bool = Field(default=True)
    enable_shadow_mode: bool = Field(default=False)

    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    risk: RiskLimits = Field(default_factory=RiskLimits)

    model_config = SettingsConfigDict(
        env_prefix="ATA_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_mode_matches_env(self) -> "EnvSettings":
        if self.environment == Env.DEV and self.trading_mode == TradingMode.LIVE:
            raise ValueError("DEV environment must not use trading_mode=live")
        if self.environment == Env.PROD and self.trading_mode == TradingMode.MOCK:
            raise ValueError("PROD environment must not use trading_mode=mock")
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ENV_DIR = _REPO_ROOT / "config" / "env"


@lru_cache(maxsize=1)
def get_env_settings(env_override: str | None = None) -> EnvSettings:
    """Load settings for the active environment.

    Resolution order:
    1. `env_override` argument (for tests)
    2. ATA_ENVIRONMENT env-var
    3. Default: "dev"

    Looks for config/env/.env.{env} file. If the file does not exist,
    settings are loaded from process environment only (useful in CI).
    """
    env_name = (env_override or os.environ.get("ATA_ENVIRONMENT", "dev")).strip().lower()

    if env_name not in {e.value for e in Env}:
        raise RuntimeError(
            f"Unknown ATA_ENVIRONMENT={env_name!r}. Allowed: dev, staging, prod."
        )

    env_file = _ENV_DIR / f".env.{env_name}"
    kwargs: dict[str, Any] = {"ATA_ENVIRONMENT": env_name}

    if env_file.exists():
        return EnvSettings(_env_file=str(env_file), **kwargs)
    return EnvSettings(**kwargs)


def clear_env_settings_cache() -> None:
    """Invalidate the settings cache (useful in tests)."""
    get_env_settings.cache_clear()
