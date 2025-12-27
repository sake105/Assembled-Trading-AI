"""Central settings configuration using Pydantic Settings.

This module provides a centralized configuration system with environment-based
modes (BACKTEST, PAPER, DEV) and configurable paths for data, output, and logs.

Usage:
    >>> from src.assembled_core.config.settings import get_settings
    >>> settings = get_settings()
    >>> print(settings.output_dir)
    >>> print(settings.environment)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Environment modes for the trading system."""

    BACKTEST = "BACKTEST"  # Offline backtesting mode
    PAPER = "PAPER"  # Paper trading mode (simulated execution)
    DEV = "DEV"  # Development mode
    # LIVE = "LIVE"  # Future: Live trading mode (not yet implemented)


class RuntimeProfile(str, Enum):
    """Runtime profile for CLI commands and scripts.

    Determines the execution mode and behavior of commands.
    Can be set via environment variable ASSEMBLED_RUNTIME_PROFILE or CLI arguments.
    """

    BACKTEST = "BACKTEST"  # Offline backtesting mode (no live execution)
    PAPER = "PAPER"  # Paper trading mode (simulated execution via Paper-Trading-API)
    DEV = "DEV"  # Development mode (default for most commands)


class Settings(BaseSettings):
    """Central settings for the trading system.

    Settings can be overridden via environment variables (uppercase, with underscores).
    Example: ASSEMBLED_ENVIRONMENT=BACKTEST, ASSEMBLED_OUTPUT_DIR=/custom/path

    Attributes:
        environment: Current environment mode (BACKTEST, PAPER, DEV)
        base_dir: Repository root directory (auto-detected)
        data_dir: Directory for input data (raw prices, events, etc.)
        output_dir: Directory for pipeline outputs (orders, reports, equity curves)
        logs_dir: Directory for log files
        watchlist_file: Path to watchlist file (default: watchlist.txt in repo root)
        supported_freqs: Tuple of supported trading frequencies
    """

    # Environment mode
    environment: Environment = Field(
        default=Environment.DEV, description="Environment mode: BACKTEST, PAPER, or DEV"
    )

    # Base directory (auto-detected from file location)
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__)
        .resolve()
        .parents[
            3
        ],  # config/settings.py -> config -> assembled_core -> src -> repo root
        description="Repository root directory",
    )

    # Directory paths
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "data",
        description="Directory for input data (raw prices, events, etc.)",
    )

    output_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "output",
        description="Directory for pipeline outputs (orders, reports, equity curves)",
    )

    logs_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "logs",
        description="Directory for log files",
    )

    models_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "models",
        description="Directory for trained models (meta-models, etc.)",
    )

    experiments_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "experiments",
        description="Directory for experiment tracking runs",
    )

    # File paths
    watchlist_file: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "watchlist.txt",
        description="Path to watchlist file (default: watchlist.txt in repo root)",
    )

    # Sample data paths
    sample_data_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "sample",
        description="Directory for sample data files",
    )

    sample_events_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
        / "data"
        / "sample"
        / "events",
        description="Directory for sample event data (insider, shipping, etc.)",
    )

    sample_eod_file: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
        / "data"
        / "sample"
        / "eod_sample.parquet",
        description="Default sample EOD price file",
    )

    # Trading configuration
    supported_freqs: tuple[str, ...] = Field(
        default=("1d", "5min"), description="Supported trading frequencies"
    )

    # Default values for backtests
    default_start_capital: float = Field(
        default=10000.0, description="Default starting capital for backtests"
    )

    # Data source configuration
    data_source: Literal["local", "yahoo", "finnhub", "twelve_data"] = Field(
        default="local",
        description="Data source type: 'local' (Parquet files), 'yahoo' (Yahoo Finance API), "
        "'finnhub' (Finnhub API), or 'twelve_data' (Twelve Data API)",
    )

    # API Keys for data providers
    finnhub_api_key: str | None = Field(
        default=None,
        description="Finnhub API key (required for finnhub data source). "
        "Set via ASSEMBLED_FINNHUB_API_KEY environment variable.",
    )

    twelve_data_api_key: str | None = Field(
        default=None,
        description="Twelve Data API key (required for twelve_data data source). "
        "Set via ASSEMBLED_TWELVE_DATA_API_KEY environment variable.",
    )

    # Local data root for Alt-Daten snapshots (optional)
    # If set, Parquet files are loaded from <local_data_root>/<freq>/<symbol>.parquet
    # If None, uses default behavior (output/aggregates/{freq}.parquet)
    local_data_root: Path | None = Field(
        default=None,
        description="Root directory for Alt-Daten snapshots. "
        "If set, loads from <local_data_root>/<freq>/<symbol>.parquet. "
        "If None, uses default behavior (output/aggregates/{freq}.parquet). "
        "Override via ASSEMBLED_LOCAL_DATA_ROOT environment variable.",
    )

    # Default universe (used if no universe file provided)
    default_universe: list[str] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL"],
        description="Default list of symbols to use if no universe file is provided",
    )

    # Performance optimization flags
    use_numba: bool = Field(
        default=False,
        description="Enable Numba JIT acceleration for backtest loops (default: False). "
        "Set via ASSEMBLED_USE_NUMBA environment variable (true/1/yes to enable). "
        "Falls back gracefully if Numba is not installed.",
    )

    model_config = SettingsConfigDict(
        env_prefix="ASSEMBLED_",  # Environment variables: ASSEMBLED_ENVIRONMENT, ASSEMBLED_OUTPUT_DIR, etc.
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )

    def __init__(self, **kwargs):
        """Initialize settings with proper path resolution."""
        # If base_dir is provided, resolve other paths relative to it
        if "base_dir" in kwargs:
            base = Path(kwargs["base_dir"]).resolve()
            if "data_dir" not in kwargs:
                kwargs["data_dir"] = base / "data"
            if "output_dir" not in kwargs:
                kwargs["output_dir"] = base / "output"
            if "logs_dir" not in kwargs:
                kwargs["logs_dir"] = base / "logs"
            if "watchlist_file" not in kwargs:
                kwargs["watchlist_file"] = base / "watchlist.txt"
            if "sample_data_dir" not in kwargs:
                kwargs["sample_data_dir"] = base / "data" / "sample"
            if "sample_events_dir" not in kwargs:
                kwargs["sample_events_dir"] = base / "data" / "sample" / "events"
            if "sample_eod_file" not in kwargs:
                kwargs["sample_eod_file"] = (
                    base / "data" / "sample" / "eod_sample.parquet"
                )
            if "models_dir" not in kwargs:
                kwargs["models_dir"] = base / "models"
            if "experiments_dir" not in kwargs:
                kwargs["experiments_dir"] = base / "experiments"

        # Handle local_data_root (can be string from env var)
        if "local_data_root" in kwargs and kwargs["local_data_root"] is not None:
            if isinstance(kwargs["local_data_root"], str):
                kwargs["local_data_root"] = Path(kwargs["local_data_root"]).resolve()

        super().__init__(**kwargs)

    def model_post_init(self, __context) -> None:
        """Post-initialization: ensure directories exist and resolve paths."""
        # Resolve all paths to absolute paths
        self.base_dir = self.base_dir.resolve()
        self.data_dir = self.data_dir.resolve()
        self.output_dir = self.output_dir.resolve()
        self.logs_dir = self.logs_dir.resolve()
        self.watchlist_file = self.watchlist_file.resolve()
        self.sample_data_dir = self.sample_data_dir.resolve()
        self.sample_events_dir = self.sample_events_dir.resolve()
        self.sample_eod_file = self.sample_eod_file.resolve()
        self.models_dir = self.models_dir.resolve()
        self.experiments_dir = self.experiments_dir.resolve()

        # Resolve local_data_root if set
        if self.local_data_root is not None:
            self.local_data_root = Path(self.local_data_root).resolve()

        # Create directories if they don't exist (except base_dir and local_data_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance (singleton pattern)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance (singleton).

    Returns:
        Settings instance (cached after first call)
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing).

    This allows tests to override settings by creating a new instance.
    """
    global _settings
    _settings = None


def get_runtime_profile(
    profile: str | RuntimeProfile | None = None,
    env_var: str = "ASSEMBLED_RUNTIME_PROFILE",
) -> RuntimeProfile:
    """Get the current runtime profile from CLI arguments or environment variable.

    Priority order:
    1. Explicit `profile` argument (from CLI)
    2. Environment variable `ASSEMBLED_RUNTIME_PROFILE`
    3. Default: `RuntimeProfile.DEV`

    Args:
        profile: Explicit profile value (from CLI argument, can be string or RuntimeProfile)
        env_var: Environment variable name to check (default: ASSEMBLED_RUNTIME_PROFILE)

    Returns:
        RuntimeProfile enum value

    Examples:
        >>> # From CLI argument
        >>> get_runtime_profile(profile="BACKTEST")
        RuntimeProfile.BACKTEST

        >>> # From environment variable
        >>> import os
        >>> os.environ["ASSEMBLED_RUNTIME_PROFILE"] = "PAPER"
        >>> get_runtime_profile()
        RuntimeProfile.PAPER

        >>> # Default
        >>> get_runtime_profile()
        RuntimeProfile.DEV
    """
    import os

    # Priority 1: Explicit profile argument (from CLI)
    if profile is not None:
        if isinstance(profile, RuntimeProfile):
            return profile
        # Try to convert string to RuntimeProfile
        profile_str = str(profile).upper()
        try:
            return RuntimeProfile(profile_str)
        except ValueError:
            # Invalid profile, fall through to default
            pass

    # Priority 2: Environment variable
    env_value = os.environ.get(env_var, "").strip().upper()
    if env_value:
        try:
            return RuntimeProfile(env_value)
        except ValueError:
            # Invalid value in env var, fall through to default
            pass

    # Priority 3: Default
    return RuntimeProfile.DEV
