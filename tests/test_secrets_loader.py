"""Tests for assembled_core/config/secrets_loader.py (spec 36)."""

from __future__ import annotations

import pytest

from src.assembled_core.config.secrets_loader import get_secret, store_secret


class TestGetSecretEnvFallback:
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("ATA_STAGING_ALPACA_API_KEY", "test-key-123")
        # Force keyring ImportError path
        import sys

        monkeypatch.setitem(sys.modules, "keyring", None)
        val = get_secret("staging", "alpaca_api_key")
        assert val == "test-key-123"

    def test_missing_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("ATA_STAGING_ALPACA_API_KEY", raising=False)
        import sys

        monkeypatch.setitem(sys.modules, "keyring", None)
        with pytest.raises(RuntimeError, match="keyring not installed"):
            get_secret("staging", "alpaca_api_key")

    def test_env_var_name_upper(self, monkeypatch):
        monkeypatch.setenv("ATA_PROD_DB_PASSWORD", "secret")
        import sys

        monkeypatch.setitem(sys.modules, "keyring", None)
        val = get_secret("prod", "db_password")
        assert val == "secret"


class TestStoreSecretNoKeyring:
    def test_raises_without_keyring(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "keyring", None)
        with pytest.raises(RuntimeError, match="keyring package not installed"):
            store_secret("dev", "test_key", value="my-value")
