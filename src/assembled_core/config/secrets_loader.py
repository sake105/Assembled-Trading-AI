"""Keyring-based secret storage for multi-environment credentials.

From 36_MULTI_ENVIRONMENT_SETUP.md §5.2.

Secrets are stored once (interactively) and retrieved at runtime from
the OS keychain (macOS Keychain, Windows Credential Manager, Linux Secret Service).
Falls back to environment variables if keyring is not installed.

Usage:
    # Store once:
    store_secret("staging", "alpaca_api_key")

    # Retrieve at runtime:
    api_key = get_secret("staging", "alpaca_api_key")

    # Or via environment variable fallback (CI):
    # export ATA_STAGING_ALPACA_API_KEY=xxx
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_SERVICE = "assembled-trading-ai"


def _env_var_name(env: str, key: str) -> str:
    return f"ATA_{env.upper()}_{key.upper()}"


def store_secret(env: str, key: str, value: str | None = None) -> None:
    """Store a secret in the OS keychain for the given environment and key.

    If value is None, prompts interactively via getpass (suitable for one-time
    CLI setup). In automated contexts, pass value directly.

    Args:
        env: Environment name ('dev', 'staging', 'prod', 'paper').
        key: Credential key (e.g. 'alpaca_api_key').
        value: Secret value. If None, prompts via getpass.
    """
    if value is None:
        import getpass
        value = getpass.getpass(f"{env.upper()} {key}: ")

    try:
        import keyring  # type: ignore[import]
        keyring.set_password(_SERVICE, f"{env}:{key}", value)
        logger.info("Stored %s:%s in keychain (%s)", env, key, _SERVICE)
    except ImportError:
        logger.warning("keyring not installed — cannot persist secret for %s:%s", env, key)
        raise RuntimeError(
            "keyring package not installed. Install with: pip install keyring"
        )


def get_secret(env: str, key: str) -> str:
    """Retrieve a secret from the OS keychain.

    Falls back to the environment variable ATA_<ENV>_<KEY> if keyring is
    not available (useful for CI environments where keyring cannot be used).

    Args:
        env: Environment name.
        key: Credential key.

    Returns:
        Secret string value.

    Raises:
        RuntimeError if the secret is not found in either keychain or env vars.
    """
    env_var = _env_var_name(env, key)
    env_fallback = os.environ.get(env_var)

    try:
        import keyring  # type: ignore[import]
        value = keyring.get_password(_SERVICE, f"{env}:{key}")
        if value is not None:
            return value
        if env_fallback is not None:
            logger.debug("Keychain miss for %s:%s — using env var %s", env, key, env_var)
            return env_fallback
        raise RuntimeError(
            f"Secret {env}:{key} not found in keychain ({_SERVICE}) or env var {env_var}"
        )
    except ImportError:
        if env_fallback is not None:
            logger.debug("keyring not available — using env var %s for %s:%s", env_var, env, key)
            return env_fallback
        raise RuntimeError(
            f"keyring not installed and env var {env_var} not set. "
            f"Either install keyring or set {env_var}."
        )


__all__ = ["store_secret", "get_secret"]
