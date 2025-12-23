# src/assembled_core/ema_config.py
"""EMA (Exponential Moving Average) configuration for trading strategies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmaConfig:
    """EMA crossover strategy parameters.

    Attributes:
        fast: Fast EMA period (shorter-term trend)
        slow: Slow EMA period (longer-term trend)
    """

    fast: int
    slow: int

    def __post_init__(self) -> None:
        """Validate EMA configuration."""
        if self.fast <= 0:
            raise ValueError(f"fast must be > 0, got {self.fast}")
        if self.slow <= 0:
            raise ValueError(f"slow must be > 0, got {self.slow}")
        if self.fast >= self.slow:
            raise ValueError(f"fast ({self.fast}) must be < slow ({self.slow})")


# Default EMA configurations per frequency
# Based on current working pipeline values
DEFAULT_EMA_1D = EmaConfig(fast=20, slow=60)
DEFAULT_EMA_5MIN = EmaConfig(fast=20, slow=60)

# Alternative configurations for experimentation (5min)
# These can be easily switched by changing DEFAULT_EMA_5MIN above
ALTERNATIVE_EMA_5MIN_FAST = EmaConfig(fast=10, slow=30)


def get_default_ema_config(freq: str) -> EmaConfig:
    """Return the default EMA configuration for a given frequency.

    Args:
        freq: Frequency string ("1d" or "5min")

    Returns:
        EmaConfig instance with default parameters for the frequency.

    Raises:
        ValueError: If freq is not supported.
    """
    if freq == "1d":
        return DEFAULT_EMA_1D
    elif freq == "5min":
        return DEFAULT_EMA_5MIN
    else:
        raise ValueError(f"Unsupported frequency: {freq}. Use '1d' or '5min'.")
