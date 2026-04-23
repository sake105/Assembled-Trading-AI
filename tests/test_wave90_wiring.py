"""Tests for wave-90 module wiring into trading_cycle.py.

Covers:
  Step 5.56 — execution.api_resilience (RetryPolicy / RateLimiter)
  Step 5.57 — execution.broker_adapter (BrokerOrder / BrokerPosition)
  Step 5.58 — execution.broker_execution (BrokerExecutionResult)
"""

from __future__ import annotations

import pytest

from src.assembled_core.execution.api_resilience import (
    RetryPolicy,
    DEFAULT_RETRY_POLICY,
    RateLimiter,
    is_retryable_http_status,
)
from src.assembled_core.execution.broker_adapter import BrokerOrder, BrokerPosition, BrokerAdapter
from src.assembled_core.execution.broker_execution import BrokerExecutionResult


# ---------------------------------------------------------------------------
# api_resilience (Step 5.56)
# ---------------------------------------------------------------------------

def test_retry_policy_creates():
    rp = RetryPolicy()
    assert isinstance(rp, RetryPolicy)


def test_default_retry_policy_exists():
    assert DEFAULT_RETRY_POLICY is not None
    assert DEFAULT_RETRY_POLICY.max_retries > 0


def test_rate_limiter_creates():
    rl = RateLimiter(max_requests=100)
    assert isinstance(rl, RateLimiter)


def test_is_retryable_http_status_429():
    assert is_retryable_http_status(429) is True


def test_is_retryable_http_status_200():
    assert is_retryable_http_status(200) is False


# ---------------------------------------------------------------------------
# broker_adapter (Step 5.57)
# ---------------------------------------------------------------------------

def test_broker_order_importable():
    assert BrokerOrder is not None


def test_broker_position_importable():
    assert BrokerPosition is not None


def test_broker_adapter_is_abstract():
    import abc
    assert issubclass(BrokerAdapter, abc.ABC)


# ---------------------------------------------------------------------------
# broker_execution (Step 5.58)
# ---------------------------------------------------------------------------

def test_broker_execution_result_creates():
    ber = BrokerExecutionResult()
    assert isinstance(ber, BrokerExecutionResult)


def test_broker_execution_result_empty():
    ber = BrokerExecutionResult()
    assert len(ber.submitted) == 0
    assert len(ber.filled) == 0
    assert len(ber.errors) == 0


def test_broker_execution_result_dry_run_default():
    ber = BrokerExecutionResult()
    assert ber.dry_run is False


def test_broker_execution_result_fields():
    ber = BrokerExecutionResult()
    assert hasattr(ber, "rejected")
    assert hasattr(ber, "timed_out")
    assert hasattr(ber, "fills_for_ledger")
