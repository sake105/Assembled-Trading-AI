"""B5 — Pin the feature-cache PIT filter.

The plan's B5 fix: feature cache (factor store) must refuse to leak rows whose
``timestamp > as_of`` even if the cache was written with future rows present.
The mechanism already exists in ``data/factor_store.load_factors`` (``as_of``
filter) and in ``features/factor_store_integration.build_or_load_factors``
(the filter is re-applied after a cache hit). This test pins both layers so
a future refactor cannot silently remove the filter.

Test shape (mirrors plan's B5 / D2 spec):

- Write a panel covering 2024-01 through 2024-12.
- Read with ``as_of=2024-06-15``. Max timestamp returned must be ≤ 2024-06-15.
- Read without ``as_of``. Full range must be returned (no over-filtering).
- Source-level pin: ``load_factors`` must contain the ``as_of`` comparison.
- Integration-level pin: ``build_or_load_factors`` must re-apply the filter
  on cache hit (defense in depth against partial-cache edge cases).

This is a *regression pin*, not a feature build — the cache already works.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.data.factor_store import (
    compute_universe_key,
    load_factors,
    store_factors,
)
from src.assembled_core.features.factor_store_integration import (
    build_or_load_factors,
)

pytestmark = pytest.mark.phase_speed


def _make_panel_with_future_rows() -> pd.DataFrame:
    """A 2024 panel of one symbol with monthly bars Jan→Dec."""
    timestamps = pd.date_range(
        start="2024-01-15", end="2024-12-15", freq="MS", tz="UTC"
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["AAPL"] * len(timestamps),
            "ta_ma_20_v1": [100.0 + i for i in range(len(timestamps))],
        }
    )


def test_load_factors_filters_rows_past_as_of(tmp_path: Path) -> None:
    universe = compute_universe_key(symbols=["AAPL"])
    df = _make_panel_with_future_rows()

    store_factors(
        df=df,
        factor_group="core_ta_test",
        freq="1d",
        universe_key=universe,
        mode="overwrite",
        factors_root=tmp_path,
    )

    cutoff = pd.Timestamp("2024-06-15", tz="UTC")
    loaded = load_factors(
        factor_group="core_ta_test",
        freq="1d",
        universe_key=universe,
        as_of=cutoff,
        factors_root=tmp_path,
    )
    assert loaded is not None, "store was populated but load returned None"
    assert not loaded.empty
    assert loaded["timestamp"].max() <= cutoff, (
        f"PIT leak: load_factors returned timestamps past as_of={cutoff}: "
        f"max={loaded['timestamp'].max()}"
    )


def test_load_factors_returns_full_range_without_as_of(tmp_path: Path) -> None:
    universe = compute_universe_key(symbols=["AAPL"])
    df = _make_panel_with_future_rows()

    store_factors(
        df=df,
        factor_group="core_ta_test2",
        freq="1d",
        universe_key=universe,
        mode="overwrite",
        factors_root=tmp_path,
    )

    loaded = load_factors(
        factor_group="core_ta_test2",
        freq="1d",
        universe_key=universe,
        factors_root=tmp_path,
    )
    assert loaded is not None
    assert len(loaded) == len(df), (
        "without as_of the loader must return the full cached range"
    )


def test_build_or_load_factors_reapplies_as_of_on_cache_hit(
    tmp_path: Path,
) -> None:
    """Defense in depth: even if ``load_factors`` returns the full range
    (e.g. because end_date was masked when as_of was passed), the integration
    layer must re-apply the cutoff before returning."""
    universe = compute_universe_key(symbols=["AAPL"])
    df = _make_panel_with_future_rows()

    store_factors(
        df=df,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe,
        mode="overwrite",
        factors_root=tmp_path,
    )

    prices = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "symbol": df["symbol"],
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000.0,
        }
    )

    cutoff = pd.Timestamp("2024-06-15", tz="UTC")
    out = build_or_load_factors(
        prices=prices,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe,
        as_of=cutoff,
        factors_root=tmp_path,
    )
    assert not out.empty
    assert out["timestamp"].max() <= cutoff, (
        f"PIT leak at integration layer: max={out['timestamp'].max()} > as_of={cutoff}"
    )


def test_warm_cache_skips_builder_fn(tmp_path: Path) -> None:
    """Cache-hit path must avoid calling the (expensive) builder. The plan's
    B5 speedup target (5-10× warm) only holds if the builder is actually
    skipped on subsequent reads. If a future refactor makes the cache read
    always fall through to ``builder_fn``, this pin catches it."""
    universe = compute_universe_key(symbols=["AAPL"])
    df = _make_panel_with_future_rows()

    store_factors(
        df=df,
        factor_group="core_ta_skip",
        freq="1d",
        universe_key=universe,
        mode="overwrite",
        factors_root=tmp_path,
    )

    prices = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "symbol": df["symbol"],
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000.0,
        }
    )

    call_count = {"n": 0}

    def _tattling_builder(prices_in: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        call_count["n"] += 1
        raise AssertionError(
            "builder_fn was invoked even though the cache is populated — "
            "warm-path regression, B5 speed target broken"
        )

    out = build_or_load_factors(
        prices=prices,
        factor_group="core_ta_skip",
        freq="1d",
        universe_key=universe,
        factors_root=tmp_path,
        builder_fn=_tattling_builder,
    )
    assert not out.empty
    assert call_count["n"] == 0


def test_load_factors_source_contains_as_of_filter() -> None:
    src = inspect.getsource(load_factors)
    assert "as_of" in src, "as_of kwarg was removed from load_factors"
    assert 'df["timestamp"] <= _to_ts(as_of)' in src, (
        "B5 regression: load_factors no longer applies the as_of filter. "
        "Cached future rows can now leak into historical reads."
    )


def test_build_or_load_factors_source_reapplies_as_of() -> None:
    src = inspect.getsource(build_or_load_factors)
    # Defense-in-depth line: the integration layer re-applies the filter on
    # cache hit. Removing it re-opens a PIT leak if ``end_date`` ever masks
    # the store-level cutoff.
    assert "cached_factors[\n                        cached_factors[\"timestamp\"] <= as_of" in src or (
        "cached_factors" in src and '["timestamp"] <= as_of' in src
    ), (
        "B5 regression: build_or_load_factors no longer re-applies the as_of "
        "filter on cache hit — relying solely on the store layer is a single "
        "point of failure."
    )
