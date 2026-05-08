"""Concurrency test for the drawdown damper in multifactor_v2.

Tests that module-level state (_DD_DAMPER / _DD_LOCK) is consistent
when 8 threads call update_drawdown_damper() simultaneously.

Item 14 — backlog.
"""

from __future__ import annotations

import datetime
import math
import threading
from typing import Any


from src.assembled_core.strategies.multifactor_v2 import (
    reset_dd_damper,
    update_drawdown_damper,
)

# Access the module-level state dict directly for post-test assertions.
import src.assembled_core.strategies.multifactor_v2 as _mfv2

_N_THREADS = 8


def _run_update(equity: float, results: list[Any], idx: int) -> None:
    """Thread worker — calls update_drawdown_damper and stores result."""
    try:
        activated = update_drawdown_damper(equity)
        results[idx] = ("ok", activated)
    except Exception as exc:  # noqa: BLE001
        results[idx] = ("error", exc)


class TestDDDamperConcurrency:
    def setup_method(self) -> None:
        """Reset damper to a clean state before each test."""
        reset_dd_damper()

    def teardown_method(self) -> None:
        """Reset damper after each test to avoid cross-test contamination."""
        reset_dd_damper()

    def test_no_crash_under_concurrent_updates(self) -> None:
        """8 threads calling update_drawdown_damper simultaneously must not crash."""
        results: list[Any] = [None] * _N_THREADS
        threads = [
            threading.Thread(
                target=_run_update,
                args=(1.0, results, i),
                daemon=True,
            )
            for i in range(_N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # All threads must have returned without error
        for i, r in enumerate(results):
            assert r is not None, f"Thread {i} never finished (join timeout?)"
            status, val = r
            assert status == "ok", f"Thread {i} raised exception: {val}"
            assert isinstance(val, bool), f"Thread {i} returned non-bool: {val!r}"

    def test_state_valid_after_concurrent_updates(self) -> None:
        """State dict fields must be valid (no None/NaN) after 8 concurrent updates."""
        results: list[Any] = [None] * _N_THREADS
        # Mix of equity values including values that dip below peak
        equity_values = [1.0, 0.98, 1.02, 0.95, 1.05, 0.90, 1.10, 0.85]
        threads = [
            threading.Thread(
                target=_run_update,
                args=(equity_values[i], results, i),
                daemon=True,
            )
            for i in range(_N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        state = _mfv2._DD_DAMPER

        # peak_equity must be a finite positive float
        assert isinstance(state["peak_equity"], float), "peak_equity must be float"
        assert math.isfinite(state["peak_equity"]), "peak_equity must be finite"
        assert state["peak_equity"] > 0, "peak_equity must be positive"

        # current_equity must be a finite float
        assert isinstance(
            state["current_equity"], float
        ), "current_equity must be float"
        assert math.isfinite(state["current_equity"]), "current_equity must be finite"

        # damper_active must be bool
        assert isinstance(state["damper_active"], bool), "damper_active must be bool"

        # damper_until is either None or a date
        if state["damper_until"] is not None:
            assert isinstance(
                state["damper_until"], datetime.date
            ), "damper_until must be datetime.date or None"

        # All threads must have returned ok
        errors = [(i, r) for i, r in enumerate(results) if r is None or r[0] != "ok"]
        assert not errors, f"Thread errors: {errors}"

    def test_peak_equity_monotonically_non_decreasing(self) -> None:
        """Peak equity must be >= the maximum of all submitted equity values.

        Because peak is updated under the lock with max(), peak must be >=
        the highest equity value submitted across all threads.
        """
        reset_dd_damper()
        equity_values = [0.80, 0.90, 1.00, 1.10, 1.20, 1.05, 0.95, 1.15]
        expected_peak = max(equity_values)

        results: list[Any] = [None] * _N_THREADS
        threads = [
            threading.Thread(
                target=_run_update,
                args=(equity_values[i], results, i),
                daemon=True,
            )
            for i in range(_N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        state = _mfv2._DD_DAMPER
        assert (
            state["peak_equity"] >= expected_peak
        ), f"peak_equity {state['peak_equity']} < expected {expected_peak}"

    def test_reset_clears_state(self) -> None:
        """reset_dd_damper must restore the initial state regardless of prior calls."""
        # Drive peak equity up
        update_drawdown_damper(5.0)
        update_drawdown_damper(0.01)  # large drawdown — may activate damper

        reset_dd_damper()

        state = _mfv2._DD_DAMPER
        assert state["peak_equity"] == 1.0
        assert state["current_equity"] == 1.0
        assert state["damper_active"] is False
        assert state["damper_until"] is None
