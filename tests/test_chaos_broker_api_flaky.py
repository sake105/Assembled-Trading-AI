"""Chaos test: flaky broker API (Plan C21).

``submit_orders_to_broker`` must isolate one failing order from the
rest: a 5xx on order N cannot block order N+1. This test installs a
fake broker adapter that raises on every second submission and
verifies that the loop still processes every order and returns the
expected pattern of results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.broker_adapter import (  # noqa: E402
    BrokerAdapter,
    BrokerOrder,
    BrokerPosition,
)
from src.assembled_core.execution.broker_execution import (  # noqa: E402
    submit_orders_to_broker,
)


class FlakyAdapter(BrokerAdapter):
    """Fake adapter that raises on every other submission.

    Simulates a broker API that is intermittently returning 5xx.
    The exact exception class does not matter — broker_execution
    wraps the call in try/except Exception.
    """

    def __init__(self, fail_every: int = 2) -> None:
        self.call_count = 0
        self.fail_every = fail_every
        self.submitted: list[dict] = []

    def submit_market_order(
        self, symbol: str, qty: float, side: str, **kwargs
    ) -> BrokerOrder:
        self.call_count += 1
        if self.call_count % self.fail_every == 0:
            raise RuntimeError(f"simulated 503 on call {self.call_count}")
        self.submitted.append({"symbol": symbol, "qty": qty, "side": side})
        return BrokerOrder(
            order_id=f"ord_{self.call_count}",
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="market",
            status="accepted",
        )

    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        *,
        time_in_force: str = "day",
        comment: str = "",
    ) -> BrokerOrder:
        raise NotImplementedError

    def submit_stop_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        *,
        limit_price: float | None = None,
        time_in_force: str = "day",
        comment: str = "",
    ) -> BrokerOrder:
        raise NotImplementedError

    def get_order_status(self, order_id: str) -> BrokerOrder:
        raise NotImplementedError

    def get_positions(self) -> list[BrokerPosition]:
        return []

    def get_open_orders(self) -> list[BrokerOrder]:
        return []

    def cancel_all_orders(self) -> int:
        return 0

    def get_account(self) -> dict:
        return {"cash": 10_000.0, "equity": 10_000.0}

    def health_check(self) -> bool:
        return True

    @property
    def is_paper(self) -> bool:
        return True


def _orders_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        [{"symbol": f"SYM{i}", "side": "buy", "qty": 1.0 + i} for i in range(n)]
    )


def test_flaky_broker_isolates_failures(tmp_path: Path, monkeypatch) -> None:
    """With fail_every=2, the adapter raises on calls 2, 4, 6, ...
    Expected: every odd-indexed order is a BrokerOrder, every
    even-indexed order is None. No exception escapes the loop.
    """
    # Ensure the kill switch is not engaged for this test.
    from src.assembled_core.execution import kill_switch

    monkeypatch.setattr(kill_switch, "is_kill_switch_engaged", lambda: False)

    adapter = FlakyAdapter(fail_every=2)
    orders = _orders_df(6)
    intent_path = tmp_path / "intent_store.jsonl"

    results, intent_keys = submit_orders_to_broker(
        adapter,
        orders,
        intent_store_path=str(intent_path),
    )

    assert len(results) == 6
    # Three successes (calls 1, 3, 5), three failures (calls 2, 4, 6).
    successes = [r for r in results if r is not None]
    failures = [r for r in results if r is None]
    assert len(successes) == 3
    assert len(failures) == 3
    # The failures should alternate with the successes.
    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None
    assert results[3] is None


def test_all_failing_broker_returns_all_none(tmp_path: Path, monkeypatch) -> None:
    from src.assembled_core.execution import kill_switch

    monkeypatch.setattr(kill_switch, "is_kill_switch_engaged", lambda: False)

    adapter = FlakyAdapter(fail_every=1)  # every call fails
    orders = _orders_df(4)

    results, _ = submit_orders_to_broker(
        adapter,
        orders,
        intent_store_path=str(tmp_path / "intent.jsonl"),
    )
    assert len(results) == 4
    assert all(r is None for r in results)
    # And critically: no exception escaped.


def test_kill_switch_blocks_broker_calls(tmp_path: Path, monkeypatch) -> None:
    """When the kill switch is engaged, submit_orders_to_broker must
    not touch the adapter at all. This is the safety invariant that
    protects against 'the broker is flaky AND we want to place new
    orders' scenarios during an incident.
    """
    from src.assembled_core.execution import kill_switch

    monkeypatch.setattr(kill_switch, "is_kill_switch_engaged", lambda: True)

    adapter = FlakyAdapter(fail_every=1)
    orders = _orders_df(3)

    results, _ = submit_orders_to_broker(
        adapter,
        orders,
        intent_store_path=str(tmp_path / "intent.jsonl"),
    )

    assert len(results) == 3
    assert all(r is None for r in results)
    assert adapter.call_count == 0
