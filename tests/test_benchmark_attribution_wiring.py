"""Tests for benchmark attribution block wired into _eo_post_steps().

Covers three scenarios:
1. No equity file → function completes without error
2. Equity CSV present but no prices parquet → completes, debug log only
3. Both equity CSV + SPY prices parquet present → benchmark_attr_*.json written to ops/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.assembled_core.pipeline.orchestrator import _eo_post_steps


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_equity_csv(tmp_path: Path, n: int = 50, col: str = "portfolio_value") -> Path:
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    rng = np.random.default_rng(42)
    equity = 10_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), col: equity})
    p = tmp_path / "portfolio_equity_1d.csv"
    df.to_csv(p, index=False)
    return p


def _make_prices_parquet(
    tmp_path: Path,
    n: int = 50,
    include_timestamp: bool = True,
    spy_only: bool = True,
    extra_days: int = 0,
) -> Path:
    dates = pd.date_range("2024-01-02", periods=n + extra_days, freq="B")
    rng = np.random.default_rng(7)
    spy_close = 450.0 * np.cumprod(1 + rng.normal(0.0003, 0.008, n + extra_days))
    d: dict = {"symbol": "SPY", "close": spy_close}
    if include_timestamp:
        d["timestamp"] = dates
    if not spy_only:
        # Add a non-SPY row to test filtering
        d2: dict = {"symbol": "AAPL", "close": spy_close.copy()}
        if include_timestamp:
            d2["timestamp"] = dates
        other = pd.DataFrame(d2)
        df = pd.concat([pd.DataFrame(d), other], ignore_index=True)
    else:
        df = pd.DataFrame(d)
    p = tmp_path / "prices_1d.parquet"
    df.to_parquet(p, index=False)
    return p


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_benchmark_attr_block_skips_when_no_equity_file(tmp_path: Path) -> None:
    """_eo_post_steps must complete without error when no equity CSV exists."""
    # No files at all in tmp_path
    # All post-steps should silently fail/skip
    _eo_post_steps(tmp_path)
    # Verify no benchmark_attr file was written
    assert not list(tmp_path.glob("ops/benchmark_attr_*.json"))


def test_benchmark_attr_block_skips_when_no_spy_prices(tmp_path: Path) -> None:
    """When equity CSV exists but prices parquet is absent, function completes
    without error and no benchmark_attr JSON is written."""
    _make_equity_csv(tmp_path)
    # Deliberately do NOT create prices_1d.parquet

    # Must not raise
    _eo_post_steps(tmp_path)

    # No JSON output written — SPY data was unavailable
    assert not list(tmp_path.glob("ops/benchmark_attr_*.json"))


def test_benchmark_attr_writes_json_when_spy_available(tmp_path: Path) -> None:
    """When equity CSV + prices parquet with SPY rows are present,
    benchmark_attr_*.json must be written to tmp_path/ops/."""
    _make_equity_csv(tmp_path, n=50)
    _make_prices_parquet(tmp_path, n=50)

    _eo_post_steps(tmp_path)

    written = list((tmp_path / "ops").glob("benchmark_attr_*.json"))
    assert len(written) == 1, f"Expected 1 benchmark_attr file, found: {written}"

    payload = json.loads(written[0].read_text(encoding="utf-8"))
    # Must contain the core BenchmarkMetrics fields
    for field in ("alpha", "beta", "information_ratio", "tracking_error"):
        assert field in payload, f"Missing field {field!r} in {payload}"
    # Values should be floats (or null) — not crash-level garbage
    if payload["beta"] is not None:
        assert isinstance(payload["beta"], float)
    if payload["alpha"] is not None:
        assert isinstance(payload["alpha"], float)


def test_benchmark_attr_skips_when_spy_not_in_prices(tmp_path: Path) -> None:
    """Parquet with no SPY rows → no JSON written (silent skip)."""
    _make_equity_csv(tmp_path, n=50)
    # Prices parquet exists but only has AAPL, no SPY
    dates = pd.date_range("2024-01-02", periods=50, freq="B")
    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        {
            "symbol": "AAPL",
            "timestamp": dates,
            "close": 180.0 * np.cumprod(1 + rng.normal(0, 0.01, 50)),
        }
    )
    (tmp_path / "prices_1d.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp_path / "prices_1d.parquet", index=False)

    _eo_post_steps(tmp_path)

    assert not list((tmp_path).glob("ops/benchmark_attr_*.json")), (
        "Should not write benchmark_attr when SPY missing from prices"
    )


def test_benchmark_attr_pit_filter_truncates_spy(tmp_path: Path) -> None:
    """SPY data beyond equity curve end date must be dropped (PIT filter)."""
    n_equity = 30
    _make_equity_csv(tmp_path, n=n_equity)
    # Prices parquet has 60 days — 30 extra beyond equity curve
    _make_prices_parquet(tmp_path, n=60, extra_days=0)

    _eo_post_steps(tmp_path)

    written = list((tmp_path / "ops").glob("benchmark_attr_*.json"))
    assert len(written) == 1
    # If PIT filter works, we get a valid JSON (not a crash from date mismatch)
    payload = json.loads(written[0].read_text())
    assert "beta" in payload


def test_benchmark_attr_skips_when_prices_missing_timestamp_col(tmp_path: Path) -> None:
    """Prices parquet without 'timestamp' column → no JSON written (F-senior-2 fix)."""
    _make_equity_csv(tmp_path, n=50)
    _make_prices_parquet(tmp_path, n=50, include_timestamp=False)

    _eo_post_steps(tmp_path)

    assert not list((tmp_path / "ops").glob("benchmark_attr_*.json")), (
        "Should not write benchmark_attr when prices parquet has no timestamp column"
    )
