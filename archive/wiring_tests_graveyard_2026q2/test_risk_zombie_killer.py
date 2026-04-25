"""Tests for zombie killer (time stop) — M6-T05.

Covers:
- _parse_utc: valid, invalid, missing timezone
- check_zombie_position: hold limit, gain threshold, no price data, unparseable ts
- get_zombie_positions: disabled policy, mixed positions, policy config
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.phase12

from src.assembled_core.risk.zombie_killer import (
    check_zombie_position,
    get_zombie_positions,
)

NOW = datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)


def _pos(
    symbol: str = "GLD",
    held_hours: float = 100.0,
    entry_price: float | None = 200.0,
    current_price: float | None = 200.0,
    side: str = "long",
) -> dict:
    entry_ts = (NOW - timedelta(hours=held_hours)).isoformat()
    p: dict = {"symbol": symbol, "entry_ts": entry_ts, "side": side}
    if entry_price is not None:
        p["entry_price"] = entry_price
    if current_price is not None:
        p["current_price"] = current_price
    return p


# ---------------------------------------------------------------------------
# check_zombie_position
# ---------------------------------------------------------------------------


class TestCheckZombiePosition:
    def test_within_hold_limit_not_zombie(self):
        # Held 50h < 5*24=120h limit
        pos = _pos(held_hours=50.0, current_price=200.0)
        is_zombie, reason = check_zombie_position(
            pos, NOW, max_hold_days=5.0, min_gain_pct=0.005
        )
        assert is_zombie is False
        assert reason == ""

    def test_past_hold_limit_no_gain_is_zombie(self):
        # Held 200h > 120h, flat price (0% gain < 0.5%)
        pos = _pos(held_hours=200.0, entry_price=100.0, current_price=100.0)
        is_zombie, reason = check_zombie_position(
            pos, NOW, max_hold_days=5.0, min_gain_pct=0.005
        )
        assert is_zombie is True
        assert "zombie_killer" in reason
        assert "GLD" in reason

    def test_past_hold_limit_sufficient_gain_not_zombie(self):
        # Held 200h, but gained 2% > 0.5%
        pos = _pos(held_hours=200.0, entry_price=100.0, current_price=102.0)
        is_zombie, reason = check_zombie_position(
            pos, NOW, max_hold_days=5.0, min_gain_pct=0.005
        )
        assert is_zombie is False

    def test_past_hold_limit_small_gain_is_zombie(self):
        # Held 200h, gained 0.2% < 0.5%
        pos = _pos(held_hours=200.0, entry_price=100.0, current_price=100.2)
        is_zombie, reason = check_zombie_position(
            pos, NOW, max_hold_days=5.0, min_gain_pct=0.005
        )
        assert is_zombie is True

    def test_no_price_data_past_hold_limit_is_zombie(self):
        # No entry_price/current_price → conservative flag
        pos = _pos(held_hours=200.0, entry_price=None, current_price=None)
        is_zombie, reason = check_zombie_position(
            pos, NOW, max_hold_days=5.0, min_gain_pct=0.005
        )
        assert is_zombie is True
        assert "no price data" in reason

    def test_no_price_data_within_hold_limit_not_zombie(self):
        pos = _pos(held_hours=50.0, entry_price=None, current_price=None)
        is_zombie, reason = check_zombie_position(
            pos, NOW, max_hold_days=5.0, min_gain_pct=0.005
        )
        assert is_zombie is False

    def test_unparseable_entry_ts_safe_default_not_zombie(self):
        pos = {
            "symbol": "X",
            "entry_ts": "not-a-date",
            "entry_price": 100.0,
            "current_price": 100.0,
        }
        is_zombie, reason = check_zombie_position(
            pos, NOW, max_hold_days=1.0, min_gain_pct=0.0
        )
        assert is_zombie is False

    def test_missing_entry_ts_safe_default_not_zombie(self):
        pos = {"symbol": "X"}
        is_zombie, reason = check_zombie_position(
            pos, NOW, max_hold_days=1.0, min_gain_pct=0.0
        )
        assert is_zombie is False

    def test_short_position_gain_correct(self):
        # Short: entry=100, current=97 → gain = 100/97 - 1 ≈ 3.09%
        pos = _pos(
            held_hours=200.0, entry_price=100.0, current_price=97.0, side="short"
        )
        is_zombie, _ = check_zombie_position(
            pos, NOW, max_hold_days=5.0, min_gain_pct=0.005
        )
        assert is_zombie is False  # 3% gain > 0.5%

    def test_short_position_no_gain_is_zombie(self):
        # Short: entry=100, current=100 → 0% gain
        pos = _pos(
            held_hours=200.0, entry_price=100.0, current_price=100.0, side="short"
        )
        is_zombie, _ = check_zombie_position(
            pos, NOW, max_hold_days=5.0, min_gain_pct=0.005
        )
        assert is_zombie is True

    def test_custom_hold_days_and_gain(self):
        # 2 day limit, 1% min gain
        pos = _pos(
            held_hours=55.0, entry_price=100.0, current_price=100.5
        )  # held ~2.3 days, 0.5% gain
        is_zombie, _ = check_zombie_position(
            pos, NOW, max_hold_days=2.0, min_gain_pct=0.01
        )
        # 0.5% gain < 1% min → zombie
        assert is_zombie is True

    def test_reason_contains_symbol_and_hours(self):
        pos = _pos(
            symbol="TLT", held_hours=200.0, entry_price=100.0, current_price=100.0
        )
        _, reason = check_zombie_position(
            pos, NOW, max_hold_days=5.0, min_gain_pct=0.005
        )
        assert "TLT" in reason
        assert "200" in reason or "200.0" in reason


# ---------------------------------------------------------------------------
# get_zombie_positions
# ---------------------------------------------------------------------------


class TestGetZombiePositions:
    def _policy(self, enabled=True, max_hold_days=5.0, min_gain_pct=0.005) -> dict:
        return {
            "zombie_killer": {
                "enabled": enabled,
                "max_hold_days": max_hold_days,
                "min_gain_pct": min_gain_pct,
            }
        }

    def test_disabled_returns_empty(self):
        positions = [_pos(held_hours=500.0, current_price=100.0)]
        result = get_zombie_positions(positions, NOW, self._policy(enabled=False))
        assert result == []

    def test_empty_positions_returns_empty(self):
        result = get_zombie_positions([], NOW, self._policy())
        assert result == []

    def test_none_positions_returns_empty(self):
        result = get_zombie_positions(None, NOW, self._policy())  # type: ignore[arg-type]
        assert result == []

    def test_mixed_positions_only_zombies_returned(self):
        positions = [
            _pos(
                symbol="GLD", held_hours=200.0, entry_price=100.0, current_price=100.0
            ),  # zombie
            _pos(
                symbol="TLT", held_hours=50.0, entry_price=100.0, current_price=100.0
            ),  # not zombie (recent)
            _pos(
                symbol="SHY", held_hours=200.0, entry_price=100.0, current_price=102.0
            ),  # not zombie (gain)
        ]
        result = get_zombie_positions(positions, NOW, self._policy())
        symbols = {pos["symbol"] for pos, _ in result}
        assert "GLD" in symbols
        assert "TLT" not in symbols
        assert "SHY" not in symbols

    def test_all_zombies(self):
        positions = [
            _pos(symbol="A", held_hours=200.0, entry_price=100.0, current_price=100.0),
            _pos(symbol="B", held_hours=250.0, entry_price=100.0, current_price=99.0),
        ]
        result = get_zombie_positions(positions, NOW, self._policy())
        assert len(result) == 2

    def test_policy_config_respected(self):
        # Custom: max_hold_days=1 → 50h > 24h → zombie
        positions = [_pos(held_hours=50.0, entry_price=100.0, current_price=100.0)]
        result = get_zombie_positions(positions, NOW, self._policy(max_hold_days=1.0))
        assert len(result) == 1

    def test_returns_position_and_reason_tuples(self):
        positions = [_pos(symbol="VIXY", held_hours=200.0, current_price=100.0)]
        result = get_zombie_positions(positions, NOW, self._policy())
        pos, reason = result[0]
        assert pos["symbol"] == "VIXY"
        assert isinstance(reason, str)
        assert len(reason) > 0
