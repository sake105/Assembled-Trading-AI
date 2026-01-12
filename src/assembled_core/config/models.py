"""Pydantic models for strict configuration validation.

This module provides Pydantic BaseModel classes for all configuration dictionaries
used throughout the trading pipeline. All models use extra="forbid" to ensure
unknown keys are rejected at validation time.

All models are backward-compatible: they accept dict | BaseModel | None and
validate/convert to the appropriate model instance.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================================
# Feature Configuration
# ============================================================================


class FeatureConfig(BaseModel):
    """Configuration for technical analysis feature computation.

    Attributes:
        ma_windows: Tuple of moving average window sizes (default: (20, 50, 200))
        atr_window: ATR (Average True Range) window size (default: 14)
        rsi_window: RSI (Relative Strength Index) window size (default: 14)
        include_rsi: Whether to include RSI computation (default: True)
    """

    ma_windows: tuple[int, ...] = Field(
        default=(20, 50, 200),
        description="Moving average window sizes",
    )
    atr_window: int = Field(default=14, ge=1, description="ATR window size")
    rsi_window: int = Field(default=14, ge=1, description="RSI window size")
    include_rsi: bool = Field(default=True, description="Include RSI computation")

    @field_validator("ma_windows")
    @classmethod
    def validate_ma_windows(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        """Validate that all MA windows are positive."""
        if not v:
            raise ValueError("ma_windows cannot be empty")
        if any(w <= 0 for w in v):
            raise ValueError("All MA windows must be > 0")
        return v

    model_config = ConfigDict(extra="forbid", frozen=False)


# ============================================================================
# Signal Configuration
# ============================================================================


class SignalConfig(BaseModel):
    """Configuration for signal generation.

    This is a flexible config that can contain strategy-specific parameters.
    Common keys:
        - ma_fast: Fast moving average period (for trend strategies)
        - ma_slow: Slow moving average period (for trend strategies)
        - threshold: Signal threshold (for threshold-based strategies)
        - etc.

    Note: This model allows arbitrary additional fields to support different
    strategy types. However, we still validate that known fields have correct types.
    """

    # Common signal parameters (optional, as different strategies use different params)
    ma_fast: int | None = Field(default=None, ge=1, description="Fast MA period")
    ma_slow: int | None = Field(default=None, ge=1, description="Slow MA period")

    @field_validator("ma_slow")
    @classmethod
    def validate_ma_slow(cls, v: int | None, info: Any) -> int | None:
        """Validate that ma_slow > ma_fast if both are provided."""
        if v is not None and "ma_fast" in info.data:
            ma_fast = info.data["ma_fast"]
            if ma_fast is not None and v <= ma_fast:
                raise ValueError(f"ma_slow ({v}) must be > ma_fast ({ma_fast})")
        return v

    model_config = ConfigDict(extra="allow", frozen=False)  # Allow extra for strategy-specific params


# ============================================================================
# Risk Configuration
# ============================================================================


class RiskConfig(BaseModel):
    """Configuration for risk controls.

    This config can contain various risk-related parameters. Currently,
    most risk controls use PreTradeConfig (separate dataclass), but this
    model provides a place for additional risk parameters.

    Attributes:
        enable_kill_switch: Enable kill switch check (default: True)
        enable_pre_trade_checks: Enable pre-trade checks (default: True)
        Other strategy-specific risk parameters can be added here.
    """

    enable_kill_switch: bool = Field(default=True, description="Enable kill switch")
    enable_pre_trade_checks: bool = Field(
        default=True, description="Enable pre-trade checks"
    )

    model_config = ConfigDict(extra="allow", frozen=False)  # Allow extra for future risk params


# ============================================================================
# QA Gate Configuration
# ============================================================================


class GateThresholdConfig(BaseModel):
    """Configuration for a single QA gate threshold.

    Attributes:
        min: Minimum value for OK status (optional)
        max: Maximum value for OK status (optional)
        warning: Warning threshold (optional)
    """

    min: float | None = Field(default=None, description="Minimum threshold")
    max: float | None = Field(default=None, description="Maximum threshold")
    warning: float | None = Field(default=None, description="Warning threshold")

    model_config = ConfigDict(extra="forbid", frozen=False)


class GateConfig(BaseModel):
    """Configuration for QA gates evaluation.

    Attributes:
        sharpe: Sharpe ratio gate thresholds (default: min=1.0, warning=0.5)
        max_drawdown: Max drawdown gate thresholds (default: max=-20.0, warning=-15.0)
        turnover: Turnover gate thresholds (default: max=50.0, warning=30.0)
        cagr: CAGR gate thresholds (default: min=0.05, warning=0.0)
        volatility: Volatility gate thresholds (default: max=0.30, warning=0.25)
        hit_rate: Hit rate gate thresholds (default: min=0.50, warning=0.40)
        profit_factor: Profit factor gate thresholds (default: min=1.5, warning=1.2)
    """

    sharpe: GateThresholdConfig = Field(
        default_factory=lambda: GateThresholdConfig(min=1.0, warning=0.5),
        description="Sharpe ratio gate",
    )
    max_drawdown: GateThresholdConfig = Field(
        default_factory=lambda: GateThresholdConfig(max=-20.0, warning=-15.0),
        description="Max drawdown gate",
    )
    turnover: GateThresholdConfig = Field(
        default_factory=lambda: GateThresholdConfig(max=50.0, warning=30.0),
        description="Turnover gate",
    )
    cagr: GateThresholdConfig = Field(
        default_factory=lambda: GateThresholdConfig(min=0.05, warning=0.0),
        description="CAGR gate",
    )
    volatility: GateThresholdConfig = Field(
        default_factory=lambda: GateThresholdConfig(max=0.30, warning=0.25),
        description="Volatility gate",
    )
    hit_rate: GateThresholdConfig = Field(
        default_factory=lambda: GateThresholdConfig(min=0.50, warning=0.40),
        description="Hit rate gate",
    )
    profit_factor: GateThresholdConfig = Field(
        default_factory=lambda: GateThresholdConfig(min=1.5, warning=1.2),
        description="Profit factor gate",
    )

    model_config = ConfigDict(extra="forbid", frozen=False)

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Convert to dict format expected by evaluate_all_gates."""
        result: dict[str, dict[str, float]] = {}
        for gate_name in [
            "sharpe",
            "max_drawdown",
            "turnover",
            "cagr",
            "volatility",
            "hit_rate",
            "profit_factor",
        ]:
            gate_config = getattr(self, gate_name)
            gate_dict: dict[str, float] = {}
            if gate_config.min is not None:
                gate_dict["min"] = gate_config.min
            if gate_config.max is not None:
                gate_dict["max"] = gate_config.max
            if gate_config.warning is not None:
                gate_dict["warning"] = gate_config.warning
            if gate_dict:
                result[gate_name] = gate_dict
        return result


# ============================================================================
# Helper Functions for Backward Compatibility
# ============================================================================


def ensure_feature_config(
    config: dict[str, Any] | FeatureConfig | None,
) -> FeatureConfig | None:
    """Ensure feature config is a FeatureConfig instance or None.

    Args:
        config: Dict, FeatureConfig, or None

    Returns:
        FeatureConfig instance or None

    Raises:
        ValidationError: If config dict has invalid values or unknown keys
    """
    if config is None:
        return None
    if isinstance(config, FeatureConfig):
        return config
    return FeatureConfig(**config)


def ensure_signal_config(
    config: dict[str, Any] | SignalConfig | None,
) -> SignalConfig:
    """Ensure signal config is a SignalConfig instance.

    Args:
        config: Dict, SignalConfig, or None (defaults to empty SignalConfig)

    Returns:
        SignalConfig instance (never None)

    Raises:
        ValidationError: If config dict has invalid values or unknown keys
    """
    if config is None:
        return SignalConfig()
    if isinstance(config, SignalConfig):
        return config
    return SignalConfig(**config)


def ensure_risk_config(
    config: dict[str, Any] | RiskConfig | None,
) -> RiskConfig:
    """Ensure risk config is a RiskConfig instance.

    Args:
        config: Dict, RiskConfig, or None (defaults to empty RiskConfig)

    Returns:
        RiskConfig instance (never None)

    Raises:
        ValidationError: If config dict has invalid values or unknown keys
    """
    if config is None:
        return RiskConfig()
    if isinstance(config, RiskConfig):
        return config
    return RiskConfig(**config)


def ensure_gate_config(
    config: dict[str, dict[str, float]] | GateConfig | None,
) -> GateConfig:
    """Ensure gate config is a GateConfig instance.

    Args:
        config: Dict, GateConfig, or None (defaults to GateConfig with defaults)

    Returns:
        GateConfig instance (never None)

    Raises:
        ValidationError: If config dict has invalid structure or unknown keys
    """
    if config is None:
        return GateConfig()
    if isinstance(config, GateConfig):
        return config
    # Convert dict format to GateConfig
    # Expected format: {"sharpe": {"min": 1.0, "warning": 0.5}, ...}
    gate_config_dict: dict[str, Any] = {}
    for gate_name in [
        "sharpe",
        "max_drawdown",
        "turnover",
        "cagr",
        "volatility",
        "hit_rate",
        "profit_factor",
    ]:
        if gate_name in config:
            gate_config_dict[gate_name] = GateThresholdConfig(**config[gate_name])
    return GateConfig(**gate_config_dict)
