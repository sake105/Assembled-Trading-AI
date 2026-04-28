"""Tests for src/assembled_core/config/env_settings.py."""
from __future__ import annotations

import pytest

from assembled_core.config.env_settings import (
    Env,
    EnvSettings,
    RiskLimits,
    TradingMode,
    clear_env_settings_cache,
    get_env_settings,
)


@pytest.fixture(autouse=True)
def clear_cache():
    yield
    clear_env_settings_cache()


def _set_dev_env(monkeypatch, mode: str = "mock"):
    """Set minimal env vars for a valid dev environment."""
    monkeypatch.setenv("ATA_ENVIRONMENT", "dev")
    monkeypatch.setenv("ATA_TRADING_MODE", mode)
    monkeypatch.setenv("ATA_RISK_MAX_POSITION_USD", "1000")
    monkeypatch.setenv("ATA_RISK_MAX_DAILY_LOSS_USD", "200")
    monkeypatch.setenv("ATA_RISK_MAX_OPEN_POSITIONS", "5")
    monkeypatch.setenv("ATA_RISK_KILL_SWITCH_LOSS_USD", "500")


# ---------------------------------------------------------------------------
# RiskLimits validator
# ---------------------------------------------------------------------------

class TestRiskLimits:
    def test_valid_config(self):
        r = RiskLimits(
            max_position_usd=1000,
            max_daily_loss_usd=500,
            max_open_positions=10,
            kill_switch_loss_usd=2000,
        )
        assert r.kill_switch_loss_usd == 2000

    def test_kill_switch_must_exceed_daily_loss(self):
        with pytest.raises(Exception):
            RiskLimits(
                max_position_usd=1000,
                max_daily_loss_usd=500,
                max_open_positions=10,
                kill_switch_loss_usd=500,   # equal → invalid
            )

    def test_kill_switch_below_daily_loss_fails(self):
        with pytest.raises(Exception):
            RiskLimits(
                max_position_usd=1000,
                max_daily_loss_usd=500,
                max_open_positions=10,
                kill_switch_loss_usd=300,   # less → invalid
            )

    def test_zero_position_fails(self):
        with pytest.raises(Exception):
            RiskLimits(max_position_usd=0)


# ---------------------------------------------------------------------------
# EnvSettings mode validation
# ---------------------------------------------------------------------------

class TestEnvSettings:
    def test_dev_mock_valid(self, monkeypatch):
        _set_dev_env(monkeypatch, mode="mock")
        s = EnvSettings()
        assert s.environment == Env.DEV
        assert s.trading_mode == TradingMode.MOCK

    def test_dev_paper_valid(self, monkeypatch):
        _set_dev_env(monkeypatch, mode="paper")
        s = EnvSettings()
        assert s.trading_mode == TradingMode.PAPER

    def test_dev_live_rejected(self, monkeypatch):
        _set_dev_env(monkeypatch, mode="live")
        with pytest.raises(Exception):
            EnvSettings()

    def test_prod_mock_rejected(self, monkeypatch):
        monkeypatch.setenv("ATA_ENVIRONMENT", "prod")
        monkeypatch.setenv("ATA_TRADING_MODE", "mock")
        monkeypatch.setenv("ATA_RISK_MAX_POSITION_USD", "10000")
        monkeypatch.setenv("ATA_RISK_MAX_DAILY_LOSS_USD", "1000")
        monkeypatch.setenv("ATA_RISK_MAX_OPEN_POSITIONS", "30")
        monkeypatch.setenv("ATA_RISK_KILL_SWITCH_LOSS_USD", "5000")
        with pytest.raises(Exception):
            EnvSettings()

    def test_staging_paper_valid(self, monkeypatch):
        monkeypatch.setenv("ATA_ENVIRONMENT", "staging")
        monkeypatch.setenv("ATA_TRADING_MODE", "paper")
        monkeypatch.setenv("ATA_RISK_MAX_POSITION_USD", "5000")
        monkeypatch.setenv("ATA_RISK_MAX_DAILY_LOSS_USD", "500")
        monkeypatch.setenv("ATA_RISK_MAX_OPEN_POSITIONS", "20")
        monkeypatch.setenv("ATA_RISK_KILL_SWITCH_LOSS_USD", "2000")
        s = EnvSettings()
        assert s.environment == Env.STAGING
        assert s.trading_mode == TradingMode.PAPER

    def test_risk_limits_accessible(self, monkeypatch):
        _set_dev_env(monkeypatch)
        s = EnvSettings()
        assert s.risk.max_position_usd == 1000.0
        assert s.risk.kill_switch_loss_usd == 500.0

    def test_log_level_default(self, monkeypatch):
        _set_dev_env(monkeypatch)
        s = EnvSettings()
        assert s.log_level == "INFO"

    def test_shadow_mode_default_false(self, monkeypatch):
        _set_dev_env(monkeypatch)
        s = EnvSettings()
        assert s.enable_shadow_mode is False

    def test_alpaca_defaults_present(self, monkeypatch):
        _set_dev_env(monkeypatch)
        s = EnvSettings()
        assert s.alpaca is not None


# ---------------------------------------------------------------------------
# get_env_settings
# ---------------------------------------------------------------------------

class TestGetEnvSettings:
    def test_default_env_is_dev(self, monkeypatch):
        monkeypatch.delenv("ATA_ENVIRONMENT", raising=False)
        s = get_env_settings()
        assert s.environment == Env.DEV

    def test_override_parameter(self, monkeypatch):
        _set_dev_env(monkeypatch, mode="mock")
        s = get_env_settings(env_override="dev")
        assert s.environment == Env.DEV

    def test_unknown_env_raises(self):
        with pytest.raises(RuntimeError, match="Unknown"):
            get_env_settings(env_override="production")

    def test_cache_cleared(self, monkeypatch):
        _set_dev_env(monkeypatch, mode="mock")
        s1 = get_env_settings(env_override="dev")
        clear_env_settings_cache()
        s2 = get_env_settings(env_override="dev")
        assert s1.environment == s2.environment

    def test_prod_live_valid(self, monkeypatch):
        monkeypatch.setenv("ATA_ENVIRONMENT", "prod")
        monkeypatch.setenv("ATA_TRADING_MODE", "live")
        monkeypatch.setenv("ATA_RISK_MAX_POSITION_USD", "10000")
        monkeypatch.setenv("ATA_RISK_MAX_DAILY_LOSS_USD", "1000")
        monkeypatch.setenv("ATA_RISK_MAX_OPEN_POSITIONS", "30")
        monkeypatch.setenv("ATA_RISK_KILL_SWITCH_LOSS_USD", "5000")
        s = get_env_settings(env_override="prod")
        assert s.environment == Env.PROD
        assert s.trading_mode == TradingMode.LIVE
