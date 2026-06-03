"""Feature flag system for environment-specific rollout control.

From 36_MULTI_ENVIRONMENT_SETUP.md §8.

Flags follow a 4-stage lifecycle:
  off → shadow → canary → on

Golden rule: new features in prod always start as 'shadow', then 'canary', then 'on'.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, fields
from typing import Literal

logger = logging.getLogger(__name__)

FlagState = Literal["off", "shadow", "canary", "on"]


@dataclass
class FeatureFlags:
    """Environment-controlled feature flags.

    Each flag controls one feature's activation state:
      off    — not executed
      shadow — executed in parallel but result ignored (for comparison)
      canary — active for a subset of tickers (10% by default)
      on     — fully active
    """

    news_sentiment_v2: FlagState = "off"
    regime_ml_model: FlagState = "shadow"
    news_topic_clustering: FlagState = "canary"
    trend_baseline: FlagState = "on"

    def _flag_state(self, flag_name: str) -> str:
        """Return the state for a known flag, or fail loud on an unknown name.

        Silently defaulting an unknown/typo flag to 'off' (inactive) used to hide
        wiring bugs: a misspelled flag name would simply never activate, with no
        signal. Resolving against the dataclass fields makes typos fail fast.

        Raises:
            ValueError: If ``flag_name`` is not a declared feature flag field.
        """
        valid = {f.name for f in fields(self)}
        if flag_name not in valid:
            raise ValueError(f"Unknown feature flag: {flag_name}")
        return getattr(self, flag_name)

    def is_active(self, flag_name: str, ticker: str = "") -> bool:
        """Return True if the flag should produce a result for this ticker.

        'on' always returns True.
        'canary' returns True for ~10% of tickers (stable hash-based selection).
        'shadow' and 'off' return False (caller handles shadow logging separately).

        Raises:
            ValueError: If ``flag_name`` is not a declared feature flag field.
        """
        state = self._flag_state(flag_name)
        if state == "on":
            return True
        if state == "canary":
            # Deterministic across process restarts: the built-in hash() is salted
            # per-process (unless PYTHONHASHSEED is pinned), which would make the
            # "stable 10% canary subset" silently change on every restart.
            return (
                bool(ticker)
                and int(hashlib.md5(ticker.encode()).hexdigest(), 16) % 10 == 0
            )
        return False

    def is_shadow(self, flag_name: str) -> bool:
        """Return True if this flag is in shadow mode (run but ignore result).

        Raises:
            ValueError: If ``flag_name`` is not a declared feature flag field.
        """
        return self._flag_state(flag_name) == "shadow"


def load_flags() -> FeatureFlags:
    """Load feature flags based on the current environment setting.

    Reads from Settings.environment. Falls back to conservative defaults
    (shadow/off) if Settings cannot be loaded.
    """
    try:
        from assembled_core.config.settings import get_settings

        settings = get_settings()
        env = (
            settings.environment.value
            if hasattr(settings.environment, "value")
            else str(settings.environment)
        )
    except Exception as exc:
        logger.warning(
            "load_flags: could not read settings (%s) — using prod defaults", exc
        )
        env = "prod"

    if env in ("dev", "development"):
        return FeatureFlags(
            news_sentiment_v2="on",
            regime_ml_model="on",
            news_topic_clustering="on",
            trend_baseline="on",
        )
    elif env in ("staging", "paper"):
        return FeatureFlags(
            news_sentiment_v2="shadow",
            regime_ml_model="shadow",
            news_topic_clustering="canary",
            trend_baseline="on",
        )
    else:
        return FeatureFlags(
            news_sentiment_v2="off",
            regime_ml_model="off",
            news_topic_clustering="shadow",
            trend_baseline="on",
        )


def emit_startup_banner() -> None:
    """Log a clearly visible environment/mode banner at startup.

    Prevents accidentally treating dev as prod. Uses different banner characters
    per environment so the environment is immediately visible in logs.
    """
    try:
        from assembled_core.config.settings import get_settings

        settings = get_settings()
        env = (
            settings.environment.value
            if hasattr(settings.environment, "value")
            else str(settings.environment)
        )
        mode_attr = getattr(settings, "trading_mode", None)
        mode = (
            mode_attr.value
            if hasattr(mode_attr, "value")
            else str(mode_attr)
            if mode_attr
            else "unknown"
        )
    except Exception:
        env = "unknown"
        mode = "unknown"

    banner_char = {"dev": "_", "development": "_", "staging": "-", "paper": "-"}.get(
        env, "!"
    )
    width = 60
    lines = [
        banner_char * width,
        "  Assembled-Trading-AI",
        f"  Environment: {env.upper()}",
        f"  Trading Mode: {mode.upper()}",
        banner_char * width,
    ]
    for line in lines:
        logger.info(line)


__all__ = ["FeatureFlags", "FlagState", "load_flags", "emit_startup_banner"]
