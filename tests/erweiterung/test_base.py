"""Tests for erweiterung._base."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from erweiterung import _base


def test_stable_hash_deterministic():
    h1 = _base.stable_hash("foo", 1, 2.5)
    h2 = _base.stable_hash("foo", 1, 2.5)
    assert h1 == h2
    assert len(h1) == 16


def test_safe_div_scalar():
    assert _base.safe_div(10, 2) == 5
    assert _base.safe_div(10, 0) == 0
    assert _base.safe_div(10, 0, default=-1) == -1
    # NaN numerator falls back to default (since out is not finite)
    assert _base.safe_div(np.nan, 1) == 0.0
    assert _base.safe_div(np.nan, 1, default=-99) == -99


def test_safe_div_series():
    a = pd.Series([1, 2, 3, 4])
    b = pd.Series([2, 0, 2, 0])
    out = _base.safe_div(a, b, default=-1)
    assert out.iloc[0] == 0.5
    assert out.iloc[1] == -1
    assert out.iloc[2] == 1.5
    assert out.iloc[3] == -1


def test_zscore():
    s = pd.Series([1, 2, 3, 4, 5])
    z = _base.zscore(s)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1) < 1e-9


def test_zscore_robust_zero_mad():
    s = pd.Series([1, 1, 1, 1, 1])
    z = _base.zscore(s, robust=True)
    assert (z == 0).all()


def test_winsorize():
    s = pd.Series(list(range(100)))
    w = _base.winsorize(s, 0.05, 0.95)
    assert w.min() >= s.quantile(0.05)
    assert w.max() <= s.quantile(0.95)


def test_to_utc_date():
    d1 = _base.to_utc_date("2024-01-15")
    assert d1.tzinfo is not None
    assert d1.day == 15
    d2 = _base.to_utc_date(pd.Timestamp("2024-01-15", tz="US/Eastern"))
    assert d2.tzinfo is not None


def test_rate_limited_decorator():
    calls = []

    @_base.rate_limited(min_interval_s=0.05)
    def f():
        calls.append(time.monotonic())

    f()
    f()
    f()
    diffs = np.diff(calls)
    assert (diffs >= 0.04).all()


def test_retry_with_backoff_success():
    attempts = {"n": 0}

    @_base.retry_with_backoff(max_attempts=3, base_delay=0.01)
    def f():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("fail")
        return "ok"

    assert f() == "ok"
    assert attempts["n"] == 2


def test_retry_with_backoff_giveup():
    attempts = {"n": 0}

    @_base.retry_with_backoff(max_attempts=2, base_delay=0.01)
    def f():
        attempts["n"] += 1
        raise ValueError("always fails")

    with pytest.raises(ValueError):
        f()
    assert attempts["n"] == 2


def test_fetch_result_dataclass_immutable():
    fr = _base.FetchResult(
        df=pd.DataFrame({"a": [1]}),
        source="test",
        as_of=pd.Timestamp.utcnow(),
        rows=1,
    )
    with pytest.raises(Exception):
        fr.source = "modified"  # type: ignore[misc]
