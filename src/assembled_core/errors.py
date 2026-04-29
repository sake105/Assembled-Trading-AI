"""Assembled Trading AI — central error classification.

Every exception in the system falls into one of three severity categories:

* **Recoverable** — transient failures that can be retried (network timeout,
  temporary file lock, rate limit).  Callers should retry with backoff.
* **Degradable** — non-critical failures where the pipeline can continue in
  a degraded state (missing optional data source, stale feature).  Callers
  should log a warning and proceed with reduced quality.
* **Fatal** — invariant violations or safety-critical failures that must stop
  processing immediately (PIT violation, kill switch breach, ledger corruption).

Usage::

    from src.assembled_core.errors import (
        RecoverableError,
        DegradableError,
        FatalTradingError,
        DataFeedError,
        PriceLookupError,
        RiskLimitBreached,
        ReconciliationError,
    )
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Base categories
# ---------------------------------------------------------------------------

class AssembledError(Exception):
    """Base class for all Assembled Trading AI errors."""


class RecoverableError(AssembledError):
    """Transient error — caller should retry with backoff.

    Examples: network timeout, API rate limit, temporary file lock.
    """


class DegradableError(AssembledError):
    """Non-critical error — pipeline can continue in degraded mode.

    Examples: optional data source unavailable, stale feature value,
    non-critical enrichment failure.
    """


class FatalTradingError(AssembledError):
    """Safety-critical error — processing must stop immediately.

    Examples: PIT violation in assert mode, kill switch breach,
    ledger integrity failure, risk limit breach.
    """


# ---------------------------------------------------------------------------
# Domain-specific errors (Recoverable)
# ---------------------------------------------------------------------------

class DataFeedError(RecoverableError):
    """Data feed temporarily unavailable or returned invalid data."""


class PriceLookupError(RecoverableError):
    """Price lookup failed for a symbol — may be transient."""

    def __init__(self, symbol: str, details: str = "") -> None:
        self.symbol = symbol
        self.details = details
        msg = f"Price lookup failed for {symbol}"
        if details:
            msg += f": {details}"
        super().__init__(msg)


class BrokerConnectionError(RecoverableError):
    """Broker API connection failed — retry with backoff."""


class UniverseLookupError(RecoverableError):
    """PIT universe lookup returned no members for the given as_of timestamp.

    Raised by ``get_universe_members_pit`` when a strict PIT query finds no
    active members — typically means the universe history file is missing,
    the as_of is outside the stored range, or the universe is genuinely empty
    at that point in time. Callers must decide whether to fail the run or
    fall back to a safer default.
    """

    def __init__(self, universe_name: str, as_of: str, details: str = "") -> None:
        self.universe_name = universe_name
        self.as_of = as_of
        self.details = details
        msg = f"No universe members for '{universe_name}' at {as_of}"
        if details:
            msg += f": {details}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Domain-specific errors (Degradable)
# ---------------------------------------------------------------------------

class StaleDataWarning(DegradableError):
    """Data is stale but pipeline can proceed with reduced quality."""

    def __init__(self, source: str, age_days: int, threshold_days: int) -> None:
        self.source = source
        self.age_days = age_days
        self.threshold_days = threshold_days
        super().__init__(
            f"Stale data from {source}: {age_days} days old (threshold: {threshold_days})"
        )


class OptionalSourceUnavailable(DegradableError):
    """An optional data source is unavailable — proceed without it."""


# ---------------------------------------------------------------------------
# Domain-specific errors (Fatal)
# ---------------------------------------------------------------------------

class RiskLimitBreached(FatalTradingError):
    """A hard risk limit has been breached — all trading must stop."""

    def __init__(self, limit_name: str, value: float, threshold: float) -> None:
        self.limit_name = limit_name
        self.value = value
        self.threshold = threshold
        super().__init__(
            f"Risk limit breached: {limit_name}={value:.4f} (threshold={threshold:.4f})"
        )


class ReconciliationError(FatalTradingError):
    """Ledger reconciliation failed — positions or cash do not match."""


class KillSwitchActive(FatalTradingError):
    """Kill switch is engaged — no orders may be submitted."""


class PITViolation(FatalTradingError):
    """Point-in-time constraint violated — future data detected."""
