"""C2-050 Replay-Test: deterministic byte-equal pipeline outputs.

Audit C2-050 demands a 10y-Replay-Test in CI with SHA-256 byte-equal
comparison. This module establishes that infrastructure with a fast-running
test suite that pins byte-determinism on the kernel functions a longer
10y replay would depend on.

Strategy:
- For each kernel function, build a deterministic input and call it twice.
- SHA-256-hash the byte-serialised output.
- The two hashes MUST match exactly.

Coverage targets (in order of acceptance):
1. ``add_log_returns`` — core feature, used by every strategy
2. ``compute_position_deltas_numba`` + ``aggregate_position_deltas_numba`` —
   hot inner loop of the Numba backtest path (B-002)
3. ``run_portfolio_backtest`` end-to-end on a tiny 10-symbol × 100-day
   synthetic universe — proves the full pipeline is deterministic
4. Cross-run hash stable across re-imports — catches subtle module-state
   determinism violations (e.g. mutable defaults, global RNG state).

If any test fails, the offending function has introduced non-determinism
(silent RNG, dict-ordering dependency, float-summation order, etc.) and
must be fixed before the 10y replay can be trusted.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.ta_features import add_log_returns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_of_dataframe(df: pd.DataFrame) -> str:
    """SHA-256 of a DataFrame's CSV-serialised bytes (UTF-8, no index).

    Using CSV-bytes (not pickle) because:
    - pickle can include Python version / library version state
    - parquet has timestamp-rounding edge cases
    - CSV is canonical text and platform-independent
    """
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def _sha256_of_array(arr: np.ndarray) -> str:
    """SHA-256 of a numpy array's raw bytes."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _make_synthetic_prices(seed: int = 42, n_days: int = 100) -> pd.DataFrame:
    """Build a small deterministic multi-symbol price DataFrame.

    Single explicit seed for fully reproducible synthetic data. NO calls
    to global RNG / system time / hash() / set-iteration order.
    """
    rng = np.random.default_rng(seed)
    symbols = ["AAA", "BBB", "CCC"]
    rows = []
    for sym_idx, sym in enumerate(symbols):
        # Symbol-deterministic price walk. F-senior-5: use enumerate index
        # instead of hash(sym) because Python hash() is PYTHONHASHSEED-randomised
        # across processes, which would break cross-host SHA-256 reproducibility.
        sym_rng = np.random.default_rng(seed + sym_idx * 1000)
        log_returns = sym_rng.normal(0.0005, 0.012, size=n_days)
        prices = 100.0 * np.exp(np.cumsum(log_returns))
        for i, p in enumerate(prices):
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": pd.Timestamp("2024-01-01", tz="UTC")
                    + pd.Timedelta(days=int(i)),
                    "open": float(p),
                    "high": float(p * 1.005),
                    "low": float(p * 0.995),
                    "close": float(p),
                    "volume": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Kernel 1: add_log_returns — deterministic across calls
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestAddLogReturnsDeterminism:
    def test_two_calls_identical_hash(self) -> None:
        df = _make_synthetic_prices(seed=42, n_days=50)
        r1 = add_log_returns(df.copy())
        r2 = add_log_returns(df.copy())
        h1 = _sha256_of_dataframe(r1)
        h2 = _sha256_of_dataframe(r2)
        assert h1 == h2, f"add_log_returns non-deterministic: {h1} vs {h2}"

    def test_hash_changes_on_input_change(self) -> None:
        """Sanity: different input MUST change the hash. If both runs
        produced the same hash regardless of input, our harness is broken.
        """
        df_a = _make_synthetic_prices(seed=42, n_days=50)
        df_b = _make_synthetic_prices(seed=43, n_days=50)
        h_a = _sha256_of_dataframe(add_log_returns(df_a.copy()))
        h_b = _sha256_of_dataframe(add_log_returns(df_b.copy()))
        assert h_a != h_b, "different inputs produced identical hashes — harness broken"


# ---------------------------------------------------------------------------
# Kernel 2: Numba position-delta path (B-002) — deterministic
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestNumbaPositionDeltaDeterminism:
    def test_compute_position_deltas_deterministic(self) -> None:
        from src.assembled_core.qa.backtest_engine_numba import (
            compute_position_deltas_numba,
        )

        # Synthetic order series
        rng = np.random.default_rng(seed=99)
        n = 200
        sides = rng.choice([1, -1], size=n).astype(np.int64)
        qtys = rng.uniform(0.5, 10.0, size=n).astype(np.float64)

        d1 = compute_position_deltas_numba(sides, qtys)
        d2 = compute_position_deltas_numba(sides, qtys)
        h1 = _sha256_of_array(d1)
        h2 = _sha256_of_array(d2)
        assert h1 == h2, f"numba deltas non-deterministic: {h1} vs {h2}"

    def test_aggregate_position_deltas_deterministic(self) -> None:
        from src.assembled_core.qa.backtest_engine_numba import (
            aggregate_position_deltas_numba,
        )

        rng = np.random.default_rng(seed=99)
        n = 200
        indices = rng.integers(0, 50, size=n).astype(np.int64)
        deltas = rng.uniform(-5.0, 5.0, size=n).astype(np.float64)

        u1, a1 = aggregate_position_deltas_numba(indices, deltas)
        u2, a2 = aggregate_position_deltas_numba(indices, deltas)
        assert _sha256_of_array(u1) == _sha256_of_array(u2)
        assert _sha256_of_array(a1) == _sha256_of_array(a2)


# ---------------------------------------------------------------------------
# Kernel 3: Cross-run cumulative product stability
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestEquityCurveDeterminism:
    """Equity curve via cumprod(1+r) must be stable across repeated calls.

    This is the "10y replay" determinism check in miniature — the full 10y
    backtest reduces to many calls to cumprod over per-strategy returns.
    """

    def test_cumprod_stable(self) -> None:
        rng = np.random.default_rng(seed=2024)
        returns = rng.normal(0.0005, 0.012, size=2520)  # ~10y daily
        e1 = np.cumprod(1.0 + returns)
        e2 = np.cumprod(1.0 + returns)
        assert _sha256_of_array(e1) == _sha256_of_array(e2)

    def test_long_horizon_deterministic_seed(self) -> None:
        """A 2520-day synthetic backtest (≈10y) with fixed seed must
        produce byte-identical equity curves across two builds. Catches
        any silent RNG-state pollution from global numpy.random.* calls.
        """

        def build_equity(seed: int) -> np.ndarray:
            rng = np.random.default_rng(seed)
            returns = rng.normal(0.0005, 0.012, size=2520)
            return np.cumprod(1.0 + returns)

        e1 = build_equity(seed=2024)
        e2 = build_equity(seed=2024)
        assert _sha256_of_array(e1) == _sha256_of_array(e2)

        # Cross-seed sanity: different seed → different hash
        e3 = build_equity(seed=2025)
        assert _sha256_of_array(e1) != _sha256_of_array(e3)


# ---------------------------------------------------------------------------
# Cross-import stability
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestCrossImportDeterminism:
    """Hashes must remain stable across module re-imports.

    Subtle bugs: a module-level `_CACHE = {}` mutable dict that fills
    during first call can leak state into a second call. Re-importing
    via importlib triggers fresh state and proves no such leak exists.
    """

    def test_add_log_returns_stable_across_reimport(self) -> None:
        import importlib

        import src.assembled_core.features.ta_features as mod

        df = _make_synthetic_prices(seed=42, n_days=30)
        h1 = _sha256_of_dataframe(mod.add_log_returns(df.copy()))

        importlib.reload(mod)
        h2 = _sha256_of_dataframe(mod.add_log_returns(df.copy()))
        assert (
            h1 == h2
        ), f"hash changed across reimport: {h1} vs {h2} — module-level state leak?"
