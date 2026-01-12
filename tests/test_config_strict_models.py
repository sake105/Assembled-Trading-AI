"""Tests for strict configuration models (Pydantic BaseModels).

This module tests that all configuration dictionaries are properly validated
using Pydantic models with extra="forbid" to reject unknown keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.models import (
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


# ============================================================================
# FeatureConfig Tests
# ============================================================================


def test_feature_config_defaults() -> None:
    """Test that FeatureConfig has correct defaults."""
    config = FeatureConfig()
    assert config.ma_windows == (20, 50, 200)
    assert config.atr_window == 14
    assert config.rsi_window == 14
    assert config.include_rsi is True


def test_feature_config_custom_values() -> None:
    """Test that FeatureConfig accepts custom values."""
    config = FeatureConfig(
        ma_windows=(10, 30, 100),
        atr_window=20,
        rsi_window=21,
        include_rsi=False,
    )
    assert config.ma_windows == (10, 30, 100)
    assert config.atr_window == 20
    assert config.rsi_window == 21
    assert config.include_rsi is False


def test_feature_config_unknown_key_forbidden() -> None:
    """Test that FeatureConfig rejects unknown keys (extra='forbid')."""
    with pytest.raises(ValidationError) as exc_info:
        FeatureConfig(unknown_key="value")
    assert "unknown_key" in str(exc_info.value).lower() or "extra" in str(exc_info.value).lower()


def test_feature_config_ma_windows_validation() -> None:
    """Test that FeatureConfig validates MA windows are positive."""
    with pytest.raises(ValidationError):
        FeatureConfig(ma_windows=(-1, 20, 50))
    with pytest.raises(ValidationError):
        FeatureConfig(ma_windows=(0, 20, 50))
    with pytest.raises(ValidationError):
        FeatureConfig(ma_windows=())


def test_feature_config_atr_window_validation() -> None:
    """Test that FeatureConfig validates ATR window >= 1."""
    with pytest.raises(ValidationError):
        FeatureConfig(atr_window=0)
    with pytest.raises(ValidationError):
        FeatureConfig(atr_window=-1)


def test_ensure_feature_config_dict() -> None:
    """Test that ensure_feature_config accepts dict and converts to FeatureConfig."""
    config_dict = {"ma_windows": (10, 30), "atr_window": 20}
    config = ensure_feature_config(config_dict)
    assert isinstance(config, FeatureConfig)
    assert config.ma_windows == (10, 30)
    assert config.atr_window == 20
    assert config.rsi_window == 14  # Default


def test_ensure_feature_config_none() -> None:
    """Test that ensure_feature_config returns None for None input."""
    assert ensure_feature_config(None) is None


def test_ensure_feature_config_already_model() -> None:
    """Test that ensure_feature_config returns same instance if already FeatureConfig."""
    config = FeatureConfig(ma_windows=(10, 30))
    result = ensure_feature_config(config)
    assert result is config


# ============================================================================
# SignalConfig Tests
# ============================================================================


def test_signal_config_defaults() -> None:
    """Test that SignalConfig has correct defaults."""
    config = SignalConfig()
    assert config.ma_fast is None
    assert config.ma_slow is None


def test_signal_config_ma_validation() -> None:
    """Test that SignalConfig validates ma_slow > ma_fast."""
    # Valid: ma_slow > ma_fast
    config = SignalConfig(ma_fast=20, ma_slow=50)
    assert config.ma_fast == 20
    assert config.ma_slow == 50

    # Invalid: ma_slow <= ma_fast
    with pytest.raises(ValidationError):
        SignalConfig(ma_fast=50, ma_slow=20)
    with pytest.raises(ValidationError):
        SignalConfig(ma_fast=50, ma_slow=50)


def test_signal_config_extra_keys_allowed() -> None:
    """Test that SignalConfig allows extra keys (for strategy-specific params)."""
    # SignalConfig uses extra="allow" to support different strategy types
    config = SignalConfig(ma_fast=20, ma_slow=50, threshold=0.5, custom_param="value")
    assert config.ma_fast == 20
    assert config.ma_slow == 50
    assert hasattr(config, "threshold")
    assert hasattr(config, "custom_param")


def test_ensure_signal_config_dict() -> None:
    """Test that ensure_signal_config accepts dict and converts to SignalConfig."""
    config_dict = {"ma_fast": 20, "ma_slow": 50}
    config = ensure_signal_config(config_dict)
    assert isinstance(config, SignalConfig)
    assert config.ma_fast == 20
    assert config.ma_slow == 50


def test_ensure_signal_config_none() -> None:
    """Test that ensure_signal_config returns empty SignalConfig for None."""
    config = ensure_signal_config(None)
    assert isinstance(config, SignalConfig)
    assert config.ma_fast is None


# ============================================================================
# RiskConfig Tests
# ============================================================================


def test_risk_config_defaults() -> None:
    """Test that RiskConfig has correct defaults."""
    config = RiskConfig()
    assert config.enable_kill_switch is True
    assert config.enable_pre_trade_checks is True


def test_risk_config_custom_values() -> None:
    """Test that RiskConfig accepts custom values."""
    config = RiskConfig(enable_kill_switch=False, enable_pre_trade_checks=False)
    assert config.enable_kill_switch is False
    assert config.enable_pre_trade_checks is False


def test_risk_config_extra_keys_allowed() -> None:
    """Test that RiskConfig allows extra keys (for future risk params)."""
    config = RiskConfig(custom_risk_param=0.5)
    assert hasattr(config, "custom_risk_param")


def test_ensure_risk_config_dict() -> None:
    """Test that ensure_risk_config accepts dict and converts to RiskConfig."""
    config_dict = {"enable_kill_switch": False}
    config = ensure_risk_config(config_dict)
    assert isinstance(config, RiskConfig)
    assert config.enable_kill_switch is False
    assert config.enable_pre_trade_checks is True  # Default


def test_ensure_risk_config_none() -> None:
    """Test that ensure_risk_config returns empty RiskConfig for None."""
    config = ensure_risk_config(None)
    assert isinstance(config, RiskConfig)
    assert config.enable_kill_switch is True  # Default


# ============================================================================
# GateConfig Tests
# ============================================================================


def test_gate_config_defaults() -> None:
    """Test that GateConfig has correct defaults."""
    config = GateConfig()
    assert config.sharpe.min == 1.0
    assert config.sharpe.warning == 0.5
    assert config.max_drawdown.max == -20.0
    assert config.max_drawdown.warning == -15.0
    assert config.turnover.max == 50.0
    assert config.turnover.warning == 30.0


def test_gate_config_custom_values() -> None:
    """Test that GateConfig accepts custom values."""
    config = GateConfig(
        sharpe=GateThresholdConfig(min=2.0, warning=1.0),
        max_drawdown=GateThresholdConfig(max=-30.0, warning=-25.0),
    )
    assert config.sharpe.min == 2.0
    assert config.sharpe.warning == 1.0
    assert config.max_drawdown.max == -30.0
    assert config.max_drawdown.warning == -25.0


def test_gate_config_unknown_key_forbidden() -> None:
    """Test that GateConfig rejects unknown keys (extra='forbid')."""
    with pytest.raises(ValidationError):
        GateConfig(unknown_gate="value")


def test_gate_config_to_dict() -> None:
    """Test that GateConfig.to_dict() returns expected format."""
    config = GateConfig()
    result = config.to_dict()
    assert isinstance(result, dict)
    assert "sharpe" in result
    assert result["sharpe"]["min"] == 1.0
    assert result["sharpe"]["warning"] == 0.5
    assert "max_drawdown" in result
    assert result["max_drawdown"]["max"] == -20.0


def test_ensure_gate_config_dict() -> None:
    """Test that ensure_gate_config accepts dict and converts to GateConfig."""
    config_dict = {
        "sharpe": {"min": 2.0, "warning": 1.0},
        "max_drawdown": {"max": -30.0, "warning": -25.0},
    }
    config = ensure_gate_config(config_dict)
    assert isinstance(config, GateConfig)
    assert config.sharpe.min == 2.0
    assert config.sharpe.warning == 1.0
    assert config.max_drawdown.max == -30.0


def test_ensure_gate_config_none() -> None:
    """Test that ensure_gate_config returns default GateConfig for None."""
    config = ensure_gate_config(None)
    assert isinstance(config, GateConfig)
    assert config.sharpe.min == 1.0  # Default


# ============================================================================
# Integration Tests
# ============================================================================


def test_trading_context_backward_compatibility() -> None:
    """Test that TradingContext accepts dict configs (backward compatibility)."""
    from src.assembled_core.pipeline.trading_cycle import TradingContext
    import pandas as pd

    # Create minimal prices DataFrame
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })

    # Test with dict configs (backward compatible)
    ctx = TradingContext(
        prices=prices,
        feature_config={"ma_windows": (10, 30), "atr_window": 20},
        signal_config={"ma_fast": 20, "ma_slow": 50},
        risk_config={"enable_kill_switch": False},
    )

    # Configs should be validated and converted internally
    assert ctx.feature_config is not None
    assert ctx.signal_config is not None
    assert ctx.risk_config is not None


def test_trading_context_with_models() -> None:
    """Test that TradingContext accepts Pydantic model configs."""
    from src.assembled_core.pipeline.trading_cycle import TradingContext
    import pandas as pd

    # Create minimal prices DataFrame
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })

    # Test with Pydantic model configs
    ctx = TradingContext(
        prices=prices,
        feature_config=FeatureConfig(ma_windows=(10, 30), atr_window=20),
        signal_config=SignalConfig(ma_fast=20, ma_slow=50),
        risk_config=RiskConfig(enable_kill_switch=False),
    )

    # Configs should be used directly
    assert isinstance(ctx.feature_config, FeatureConfig)
    assert isinstance(ctx.signal_config, SignalConfig)
    assert isinstance(ctx.risk_config, RiskConfig)
