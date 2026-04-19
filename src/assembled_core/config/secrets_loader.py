"""Secret-key loader — reads env vars, optionally from .env file.

Usage:
    from src.assembled_core.config.secrets_loader import get_secret, load_env_file
    key = get_secret("NEWSAPI_KEY")   # raises if missing and required=True

Security (CLAUDE.md §20):
- Never log secret values, only presence/absence
- .env is gitignored; configs/secrets/ contains templates only
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ENV_FILE = _REPO_ROOT / ".env"

_loaded = False


def load_env_file(path: Path | None = None) -> int:
    """Load key=value pairs from .env file into os.environ (if not already set).

    Returns number of new variables loaded.
    """
    global _loaded
    env_path = path or _ENV_FILE
    if not env_path.exists():
        return 0
    loaded = 0
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded += 1
    if loaded:
        logger.debug("[OK] secrets_loader: loaded %d vars from %s", loaded, env_path)
    _loaded = True
    return loaded


def get_secret(name: str, *, required: bool = False, default: str | None = None) -> str | None:
    """Return the value of env var `name`, loading .env first if not yet done.

    Args:
        name: Environment variable name (e.g. "NEWSAPI_KEY").
        required: If True, raise ValueError when variable is absent.
        default: Fallback when not required and absent.

    Returns:
        The secret value, or `default` if absent and not required.

    Raises:
        ValueError: If `required=True` and the variable is not set.
    """
    if not _loaded:
        load_env_file()
    value = os.environ.get(name)
    if value:
        logger.debug("[OK] secret '%s' is set", name)
        return value
    if required:
        raise ValueError(
            f"Required secret '{name}' is not set. "
            f"Set it in .env or as an environment variable. "
            f"See configs/secrets/README.md."
        )
    logger.debug("[SKIP] secret '%s' not set — returning default", name)
    return default


def is_secret_set(name: str) -> bool:
    """Return True if the secret is available (non-empty)."""
    return bool(get_secret(name))
