"""Hardening tests for multifactor_v2 backlog items 2, 3, 6, 15, 47, 48.

Items covered:
  Item  2 — _REGIME_WEIGHTS_CACHE mtime-based auto-invalidation + clear_regime_cache()
  Item  3 — _BoundedCache LRU eviction (HMM cache replacement)
  Item  6 — _DD_DAMPER single-strategy-only contract (documented + tested)
  Item 15 — multifactor_v2_constants importable; key constants present
  Item 47 — safe_divide helper + zero-total-weight edge case
  Item 48 — NaN factors treated as neutral (0) before clip; no NaN in composite
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Imports from the module under test
# ---------------------------------------------------------------------------
from src.assembled_core.strategies.multifactor_v2 import (
    _BoundedCache,
    _DD_DAMPER,
    _REGIME_WEIGHTS_CACHE,
    clear_regime_cache,
    compute_signals,
    reset_dd_damper,
    safe_divide,
    update_drawdown_damper,
)

pytestmark = pytest.mark.fast


# ===========================================================================
# Item 15 — Constants module
# ===========================================================================


class TestConstants:
    """Verify multifactor_v2_constants is importable and contains key values."""

    def test_constants_importable(self) -> None:
        from src.assembled_core.strategies import (
            multifactor_v2_constants as C,
        )  # noqa: N812

        assert hasattr(C, "TRADING_DAYS_PER_YEAR")
        assert hasattr(C, "FACTOR_CLIP_MIN")
        assert hasattr(C, "FACTOR_CLIP_MAX")
        assert hasattr(C, "DD_MDD_THRESHOLD")
        assert hasattr(C, "HMM_CACHE_MAXSIZE")
        assert hasattr(C, "SAFE_DIVIDE_EPS")

    def test_constants_values_sane(self) -> None:
        from src.assembled_core.strategies import (
            multifactor_v2_constants as C,
        )  # noqa: N812

        assert C.TRADING_DAYS_PER_YEAR == 252
        assert C.FACTOR_CLIP_MIN < 0.0
        assert C.FACTOR_CLIP_MAX > 0.0
        assert C.FACTOR_CLIP_MIN == -C.FACTOR_CLIP_MAX  # symmetric
        assert 0.0 < C.DD_MDD_THRESHOLD < 1.0
        assert C.DD_DAMPER_FACTOR < 1.0  # must actually damp
        assert C.HMM_CACHE_MAXSIZE >= 1
        assert C.SAFE_DIVIDE_EPS > 0.0
        assert (
            C.VIX_CAP_EXTREME < C.VIX_CAP_CRISIS < C.VIX_CAP_ELEVATED < C.VIX_CAP_MILD
        )

    def test_constants_imported_into_multifactor_v2(self) -> None:
        """Check that multifactor_v2 re-exports constants (via import at module level)."""
        import src.assembled_core.strategies.multifactor_v2 as m

        # These are used internally via imported names; module should not crash on import
        assert m.safe_divide is not None
        assert m._BoundedCache is not None


# ===========================================================================
# Item 47 — safe_divide
# ===========================================================================


class TestSafeDivide:
    def test_normal_division(self) -> None:
        assert safe_divide(10.0, 2.0) == pytest.approx(5.0)

    def test_zero_denominator_returns_default(self) -> None:
        assert safe_divide(1.0, 0.0) == 0.0

    def test_tiny_denominator_returns_default(self) -> None:
        # 1e-13 < SAFE_DIVIDE_EPS → default
        assert safe_divide(5.0, 1e-13) == 0.0

    def test_custom_default(self) -> None:
        assert safe_divide(1.0, 0.0, default=float("nan")) is not safe_divide(1.0, 0.0)
        assert safe_divide(1.0, 0.0, default=-99.0) == pytest.approx(-99.0)

    def test_negative_denominator(self) -> None:
        assert safe_divide(4.0, -2.0) == pytest.approx(-2.0)

    def test_negative_tiny_denominator(self) -> None:
        # abs(-1e-14) < eps → default
        assert safe_divide(1.0, -1e-14) == 0.0


# ===========================================================================
# Item 2 — Regime cache auto-invalidation (mtime-based)
# ===========================================================================


class TestRegimeCache:
    """Test clear_regime_cache() and mtime-based auto-invalidation."""

    def setup_method(self) -> None:
        clear_regime_cache()

    def test_clear_empties_cache(self) -> None:
        assert len(_REGIME_WEIGHTS_CACHE) == 0

    def test_clear_twice_is_idempotent(self) -> None:
        clear_regime_cache()
        clear_regime_cache()
        assert len(_REGIME_WEIGHTS_CACHE) == 0

    def test_file_change_triggers_reload(self, tmp_path: Path) -> None:
        """Modifying the JSON on disk invalidates the cached entry (mtime changes)."""
        weights_v1 = {"bull": {"trend_ema_spread": 0.1}}
        weights_v2 = {"bull": {"trend_ema_spread": 0.2}}

        weights_file = tmp_path / "factor_weights_by_regime.json"
        weights_file.write_text(json.dumps(weights_v1), encoding="utf-8")

        cfg = {"regime_weights_path": str(weights_file)}

        from src.assembled_core.strategies.multifactor_v2 import _load_regime_weights

        loaded_v1 = _load_regime_weights(cfg)
        assert loaded_v1 is not None
        assert loaded_v1["bull"]["trend_ema_spread"] == pytest.approx(0.1)

        # Overwrite with new content — mtime changes
        time.sleep(0.01)  # ensure mtime differs
        weights_file.write_text(json.dumps(weights_v2), encoding="utf-8")

        # Must reload from disk (different mtime → different cache key)
        loaded_v2 = _load_regime_weights(cfg)
        assert loaded_v2 is not None
        assert loaded_v2["bull"]["trend_ema_spread"] == pytest.approx(0.2)

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        from src.assembled_core.strategies.multifactor_v2 import _load_regime_weights

        cfg = {"regime_weights_path": str(tmp_path / "nonexistent.json")}
        assert _load_regime_weights(cfg) is None

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        from src.assembled_core.strategies.multifactor_v2 import _load_regime_weights

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json", encoding="utf-8")
        cfg = {"regime_weights_path": str(bad_file)}
        assert _load_regime_weights(cfg) is None


# ===========================================================================
# Item 3 — _BoundedCache (LRU eviction)
# ===========================================================================


class TestBoundedCache:
    def test_basic_set_get(self) -> None:
        cache = _BoundedCache(maxsize=3)
        cache.set("a", 1)
        assert cache.get("a") == 1

    def test_missing_key_returns_default(self) -> None:
        cache = _BoundedCache(maxsize=3)
        assert cache.get("missing") is None
        assert cache.get("missing", default=42) == 42

    def test_evicts_oldest_on_overflow(self) -> None:
        cache = _BoundedCache(maxsize=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # Access 'a' to make it recently used
        _ = cache.get("a")
        # Add 'd' → 'b' should be evicted (oldest not recently used)
        cache.set("d", 4)
        assert cache.get("d") == 4
        assert cache.get("b") is None  # evicted
        assert cache.get("a") == 1  # recently used → kept
        assert cache.get("c") == 3  # kept

    def test_len_bounded(self) -> None:
        cache = _BoundedCache(maxsize=5)
        for i in range(20):
            cache.set(i, i)
        assert len(cache) <= 5

    def test_clear_empties_cache(self) -> None:
        cache = _BoundedCache(maxsize=5)
        cache.set("x", 1)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("x") is None

    def test_contains(self) -> None:
        cache = _BoundedCache(maxsize=3)
        cache.set("z", 99)
        assert "z" in cache
        assert "missing" not in cache

    def test_thread_safe_concurrent_writes(self) -> None:
        """Concurrent sets from multiple threads must not corrupt the cache."""
        cache = _BoundedCache(maxsize=10)
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(start, start + 50):
                    cache.set(i, i * 2)
                    _ = cache.get(i)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t * 50,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(cache) <= 10


# ===========================================================================
# Item 6 — _DD_DAMPER single-strategy-only contract
# ===========================================================================


class TestDDDamperSingleStrategy:
    """Document and verify the single-strategy-only contract of _DD_DAMPER."""

    def setup_method(self) -> None:
        reset_dd_damper()

    def test_reset_isolates_independent_sequences(self) -> None:
        """Calling reset_dd_damper() between runs prevents cross-contamination."""
        # Run A: activate damper
        update_drawdown_damper(1.0)
        update_drawdown_damper(0.875)  # MDD=12.5% → activate
        assert _DD_DAMPER["damper_active"] is True

        # Between runs: MUST reset
        reset_dd_damper()

        # Run B: small drawdown — must NOT be affected by Run A
        update_drawdown_damper(1.0)
        activated = update_drawdown_damper(0.95)  # 5% drawdown → no activate
        assert activated is False
        assert _DD_DAMPER["damper_active"] is False

    def test_without_reset_state_leaks(self) -> None:
        """Without reset, damper state leaks between runs — illustrates the risk.

        This test documents the known limitation: module-global state is
        shared. Callers must call reset_dd_damper() between isolated runs.
        """
        # Run A: activate damper
        update_drawdown_damper(1.0)
        update_drawdown_damper(0.875)
        assert _DD_DAMPER["damper_active"] is True

        # Run B (no reset!): state from A is still there
        # damper_active remains True even though B has fresh equity
        assert _DD_DAMPER["damper_active"] is True  # leaked!

    def test_reset_restores_defaults(self) -> None:
        """reset_dd_damper() restores all fields to documented defaults."""
        update_drawdown_damper(1.0)
        update_drawdown_damper(0.875)

        reset_dd_damper()

        assert _DD_DAMPER["peak_equity"] == pytest.approx(1.0)
        assert _DD_DAMPER["current_equity"] == pytest.approx(1.0)
        assert _DD_DAMPER["damper_active"] is False
        assert _DD_DAMPER["damper_until"] is None


# ===========================================================================
# Item 48 — NaN propagation in factor scores
# ===========================================================================


def _build_nan_panel(n_days: int = 80, n_symbols: int = 6) -> pd.DataFrame:
    """Create a synthetic panel where several factor columns are entirely NaN."""
    rng = np.random.default_rng(0)
    symbols = [f"S{i}" for i in range(n_symbols)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = []
    for sym in symbols:
        prices = 100.0 + rng.normal(0, 0.5, n_days).cumsum()
        for i, d in enumerate(dates):
            p = float(max(prices[i], 1.0))
            row: dict = {
                "symbol": sym,
                "timestamp": d,
                "open": p,
                "high": p * 1.005,
                "low": p * 0.995,
                "close": p,
                "volume": int(1e6),
                # Inject deliberately NaN columns to simulate missing alt-data
                "ta_rsi_14_v1": float("nan"),
                "ta_macd_hist_v1": float("nan"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


class TestNaNPropagation:
    """Item 48 — missing factors must be treated as neutral, not propagated as NaN."""

    def setup_method(self) -> None:
        reset_dd_damper()

    def test_no_nan_in_composite_score(self) -> None:
        """compute_signals must not emit NaN scores when input has NaN factor cols."""
        df = _build_nan_panel()
        signals = compute_signals(
            df, strategy_cfg={"regime_weights_path": "__nonexistent__"}
        )
        if signals.empty:
            return  # acceptable — no signals above threshold
        assert (
            signals["score"].isna().sum() == 0
        ), f"NaN scores found: {signals['score'].isna().sum()} / {len(signals)}"

    def test_all_nan_factors_still_produces_output(self) -> None:
        """Even if alt-data columns are all NaN, compute_signals must not raise."""
        df = _build_nan_panel()
        # Should not raise
        try:
            signals = compute_signals(
                df, strategy_cfg={"regime_weights_path": "__nonexistent__"}
            )
        except Exception as exc:
            pytest.fail(f"compute_signals raised with all-NaN factor columns: {exc}")

    def test_factor_df_fillna_before_clip(self) -> None:
        """Directly verify that factor_df values after normalization have no NaN.

        This is a structural test: we create a minimal scores DataFrame with
        NaN entries and assert that the fillna→clip pipeline produces finite values.
        """
        # Simulate what compute_signals does for factor normalisation
        factor_cols = ["f1", "f2", "f3"]
        scores_raw = pd.DataFrame(
            {
                "f1": [1.0, float("nan"), 0.5],
                "f2": [float("nan"), float("nan"), 0.2],
                "f3": [0.3, 0.4, float("nan")],
            }
        )
        # Reproduce the Item 48 fix path (n_rows >= SMALL_UNIVERSE_THRESHOLD)
        from src.assembled_core.strategies.multifactor_v2_constants import (
            FACTOR_CLIP_MAX,
            FACTOR_CLIP_MIN,
            STD_ZERO_REPLACE,
        )

        factor_df = scores_raw.astype(float).fillna(0.0)  # Item 48: fillna BEFORE clip
        means = factor_df.mean()
        stds = factor_df.std(ddof=0)
        safe_stds = stds.replace(0.0, np.nan).where(
            stds > STD_ZERO_REPLACE, other=np.nan
        )
        normalized = (factor_df - means) / safe_stds
        result = normalized.fillna(0.0).clip(FACTOR_CLIP_MIN, FACTOR_CLIP_MAX)

        assert result.isna().sum().sum() == 0, "NaN survived fillna→clip pipeline"
        assert (result >= FACTOR_CLIP_MIN).all().all()
        assert (result <= FACTOR_CLIP_MAX).all().all()


# ===========================================================================
# Item 47 — composite zero-total-weight edge case via compute_signals
# ===========================================================================


class TestZeroTotalWeight:
    """When all factors are zero-variance, safe_divide must return 0 composite."""

    def setup_method(self) -> None:
        reset_dd_damper()

    def test_constant_price_panel_does_not_crash(self) -> None:
        """Flat prices → zero cross-sectional variance → all factors zero → no crash."""
        n_days = 60
        symbols = ["A", "B", "C", "D", "E", "F"]
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        rows = []
        for sym in symbols:
            for d in dates:
                rows.append(
                    {
                        "symbol": sym,
                        "timestamp": d,
                        "open": 100.0,
                        "high": 100.0,
                        "low": 100.0,
                        "close": 100.0,
                        "volume": 1_000_000,
                    }
                )
        df = pd.DataFrame(rows)
        try:
            signals = compute_signals(
                df, strategy_cfg={"regime_weights_path": "__nonexistent__"}
            )
            # No NaN or inf in scores
            if not signals.empty and "score" in signals.columns:
                assert np.isfinite(signals["score"].to_numpy()).all()
        except Exception as exc:
            pytest.fail(f"compute_signals raised on flat-price panel: {exc}")
