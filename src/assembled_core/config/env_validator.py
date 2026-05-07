"""ENV validation — call validate_env() at the start of any critical script.

Usage
-----
from assembled_core.config.env_validator import validate_env

# Validate all required vars (raises RuntimeError if any are missing):
validate_env()

# Validate only a specific subset:
validate_env(required={"ALPACA_API_KEY": "Alpaca API key"})
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required vars — absence raises RuntimeError at startup.
# ---------------------------------------------------------------------------
REQUIRED_VARS: dict[str, str] = {
    "ALPACA_API_KEY": "Alpaca trading API key (paper or live account)",
    "ALPACA_API_SECRET": "Alpaca trading API secret",
}

# ---------------------------------------------------------------------------
# Optional vars — absence logs a warning; pipeline degrades gracefully.
# ---------------------------------------------------------------------------
OPTIONAL_VARS: dict[str, str] = {
    # Data sources
    "ASSEMBLED_FINNHUB_API_KEY": "Finnhub news/earnings data (degrades to zero-fill if missing)",
    "FINNHUB_API_KEY": "Finnhub key (alternate name used in some scripts)",
    "ALPHAVANTAGE_KEY": "Alpha Vantage daily OHLCV (degrades to zero-fill if missing)",
    "ALPHAVANTAGE_API_KEY": "Alpha Vantage key (alternate name used in some scripts)",
    "ASSEMBLED_TWELVE_DATA_API_KEY": "Twelve Data price feed fallback",
    "POLYGON_API_KEY": "Polygon.io price and news data",
    "NEWSAPI_KEY": "NewsAPI.org news pipeline",
    "FRED_API_KEY": "FRED macro data (GDP, CPI, VIX, GPR)",
    "NOAA_CDO_TOKEN": "NOAA Climate Data Online — weather/climate features",
    "ANTHROPIC_API_KEY": "Anthropic Claude — news_rag summarization",
    # Monitoring / alerting
    "SENTRY_DSN": "Sentry error tracking",
    "DISCORD_WEBHOOK_URL": "Discord trade/alert notifications",
    "DISCORD_WEBHOOK": "Discord webhook (alternate name)",
    "TELEGRAM_BOT_TOKEN": "Telegram alert bot token",
    "TELEGRAM_CHAT_ID": "Telegram target chat/channel ID",
    "ALERT_EMAIL_TO": "Alert email recipient",
    "SMTP_HOST": "SMTP server host for email alerts",
    "SMTP_USER": "SMTP username",
    "SMTP_PASS": "SMTP password",
    "MLFLOW_TRACKING_URI": "MLflow tracking server URI",
    # Infrastructure
    "QUESTDB_HOST": "QuestDB host (tick store)",
    "QUESTDB_PORT": "QuestDB port",
    "QUESTDB_USER": "QuestDB username",
    "QUESTDB_PASS": "QuestDB password",
    "QUESTDB_DB": "QuestDB database name",
    "REDIS_URL": "Redis URL — EventBus side-channel and cache",
    # System / runtime
    "ATA_ENVIRONMENT": "Runtime environment: paper | live | backtest | research",
    "ENVIRONMENT": "Alternative environment flag (settings.py)",
    "ASSEMBLED_RUNTIME_PROFILE": "Runtime profile: development | production | ci",
    "ASSEMBLED_POLICY_PATH": "Path to policy.yaml override",
    "ASSEMBLED_LOCAL_DATA_ROOT": "Local data root override",
    "FEATURE_STORE_PATH": "Feature store path override",
    "ASSEMBLED_KILL_SWITCH": "Kill-switch flag — set to 1 to halt all trading",
    "ASSEMBLED_RISK_STATE_PERSISTENCE_MODE": "Risk state persistence: memory | file | redis",
    "ASSEMBLED_RUN_ID": "Unique run ID for reproducibility tracking",
    "ASSEMBLED_STRICT_PIT_CHECKS": "Strict PIT data checks: 1 | 0",
    "PAPER": "Paper mode flag: 1 | 0",
    "PDT_RULE_ACTIVE": "PDT rule enforcement: 1 | 0",
    "AS_CORE_STRICT_QTY": "Strict quantity enforcement in execution: 1 | 0",
    "PYTHONHASHSEED": "Fixed hash seed for reproducible runs",
}


def validate_env(
    required: dict[str, str] | None = None,
    warn_missing_optional: bool = True,
) -> None:
    """Validate required ENV vars. Raises RuntimeError if any are missing.

    Parameters
    ----------
    required:
        Override the default REQUIRED_VARS dict. Keys are variable names,
        values are human-readable descriptions used in error messages.
        If None, the module-level REQUIRED_VARS are used.
    warn_missing_optional:
        If True (default), log a warning for each missing optional var.

    Raises
    ------
    RuntimeError
        If one or more required variables are absent from the environment.
        All missing vars are collected before raising — not one at a time.
    """
    vars_to_check = required if required is not None else REQUIRED_VARS

    missing: list[str] = []
    for var, description in vars_to_check.items():
        if not os.environ.get(var):
            missing.append(f"  {var!r}: {description}")

    if missing:
        lines = "\n".join(missing)
        raise RuntimeError(
            f"Missing required environment variable(s):\n{lines}\n\n"
            "Copy .env.example to .env and fill in the values, "
            "or export them in your shell before running."
        )

    if warn_missing_optional:
        for var, description in OPTIONAL_VARS.items():
            if not os.environ.get(var):
                logger.warning(
                    "[env] Optional ENV var not set: %s — %s", var, description
                )
