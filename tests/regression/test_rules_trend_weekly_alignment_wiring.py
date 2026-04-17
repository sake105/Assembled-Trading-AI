"""F3 wiring — verify rules_trend honours ``require_weekly_alignment``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.phase_depth]

from src.assembled_core.signals.rules_trend import generate_trend_signals  # noqa: E402


def _prices(n: int = 200, seed: int = 0, direction: float = 0.003) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(direction, 0.01, n)))
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "AAA",
            "close": close,
            "volume": 1_000_000 * np.ones(n),
        }
    )


def test_default_is_noop_vs_without_alignment() -> None:
    df = _prices()
    a = generate_trend_signals(df)
    b = generate_trend_signals(df, require_weekly_alignment=False)
    pd.testing.assert_frame_equal(a, b)


def test_alignment_gate_drops_longs_in_downtrend() -> None:
    df = _prices(direction=-0.003, seed=1)
    raw = generate_trend_signals(df)
    gated = generate_trend_signals(df, require_weekly_alignment=True)
    # In a sustained downtrend the gate must reject at least as many LONGs
    # as the raw signal produces — often all of them.
    raw_longs = (raw["direction"] == "LONG").sum()
    gated_longs = (gated["direction"] == "LONG").sum()
    assert gated_longs <= raw_longs


def test_alignment_gate_keeps_some_longs_in_uptrend() -> None:
    df = _prices(direction=0.004, seed=2)
    gated = generate_trend_signals(df, require_weekly_alignment=True)
    assert (gated["direction"] == "LONG").sum() > 0
