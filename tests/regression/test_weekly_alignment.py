"""F3 — Weekly-alignment filter regression pins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.phase_depth]

from src.assembled_core.features.weekly_alignment import (  # noqa: E402
    WeeklyAlignmentConfig,
    add_weekly_alignment,
)


def _frame(n_days: int = 180, seed: int = 0, direction: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    prices = 100.0 * np.exp(np.cumsum(rng.normal(direction, 0.01, n_days)))
    return pd.DataFrame(
        {"close": prices, "daily_trend": np.sign(np.diff(prices, prepend=prices[0]))},
        index=idx,
    )


def test_adds_expected_columns() -> None:
    df = _frame()
    out = add_weekly_alignment(df)
    assert "weekly_ema_slope" in out.columns
    assert "weekly_alignment_ok" in out.columns
    assert len(out) == len(df)


def test_agreement_in_uptrend() -> None:
    df = _frame(direction=0.003, seed=1)
    out = add_weekly_alignment(df)
    tail = out.iloc[-40:]
    # In a persistent uptrend, weekly slope should mostly agree with positive
    # daily trend → many True alignments near the end.
    pos_agree = ((tail["daily_trend"] > 0) & (tail["weekly_alignment_ok"])).sum()
    pos_count = (tail["daily_trend"] > 0).sum()
    assert pos_count > 0
    assert pos_agree / pos_count > 0.7


def test_disagreement_suppresses_alignment() -> None:
    df = _frame(direction=-0.003, seed=2)
    out = add_weekly_alignment(df)
    tail = out.iloc[-40:]
    # Long daily-trend bars in a downtrend week should NOT clear the gate.
    conflict = ((tail["daily_trend"] > 0) & (tail["weekly_ema_slope"] < 0))
    assert (tail.loc[conflict, "weekly_alignment_ok"] == False).all()  # noqa: E712


def test_missing_column_raises() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2023-01-02", periods=2))
    with pytest.raises(ValueError, match="daily_trend"):
        add_weekly_alignment(df)


def test_non_datetime_index_raises() -> None:
    df = pd.DataFrame({"close": [1.0], "daily_trend": [1]})
    with pytest.raises(ValueError, match="datetime-indexed"):
        add_weekly_alignment(df)


def test_per_symbol_isolation() -> None:
    df1 = _frame(direction=0.005, seed=3).assign(symbol="A")
    df2 = _frame(direction=-0.005, seed=4).assign(symbol="B")
    panel = pd.concat([df1, df2])
    out = add_weekly_alignment(panel)
    a_tail = out[out["symbol"] == "A"].iloc[-20:]
    b_tail = out[out["symbol"] == "B"].iloc[-20:]
    # Slopes must diverge in sign between the two symbols.
    assert a_tail["weekly_ema_slope"].mean() > b_tail["weekly_ema_slope"].mean()


def test_config_override_affects_slope() -> None:
    df = _frame(direction=0.002, seed=5)
    fast = add_weekly_alignment(df, config=WeeklyAlignmentConfig(ema_span=3))
    slow = add_weekly_alignment(df, config=WeeklyAlignmentConfig(ema_span=40))
    # A much slower EMA must be less reactive than a 3-span EMA.
    assert fast["weekly_ema_slope"].abs().mean() >= slow["weekly_ema_slope"].abs().mean()
