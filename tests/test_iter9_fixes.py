"""Regression tests for Iteration-9 fixes.

Covers:
  Fix 1 — prices_ingest.py: nullable-dtype volume does not raise false-positive ValueError
  Fix 2 — event_features.py: method='vectorized' with missing module falls back gracefully
  Fix 3 — _tc_execution.py: qty.abs() emits warning for negative qty (not silent)
  Fix 4 — batch_runner.py: reset_dd_damper() called between runs (smoke import check)
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Fix 1 — prices_ingest.py nullable-dtype volume coerce
# ---------------------------------------------------------------------------


class TestPricesIngestNullableDtype:
    """volume coerce check must not raise for legitimately missing nullable values."""

    def _write_tmp_parquet(self, df: pd.DataFrame, tmp_path: Path) -> Path:
        p = tmp_path / "test_prices.parquet"
        df.to_parquet(p, index=False)
        return p

    def _minimal_price_df(self, volume_col) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
                "symbol": ["AAPL", "AAPL"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": volume_col,
            }
        )

    def test_nullable_int64_nan_does_not_raise(self, tmp_path):
        """Nullable Int64 pd.NA volume should not trigger the ValueError."""
        from src.assembled_core.data.prices_ingest import load_eod_prices

        df = self._minimal_price_df(pd.array([pd.NA, 50000], dtype="Int64"))
        p = self._write_tmp_parquet(df, tmp_path)
        # Should not raise ValueError
        result = load_eod_prices(price_file=p)
        assert isinstance(result, pd.DataFrame)
        # The pd.NA row should have volume=0.0 (fillna(0))
        assert result["volume"].notna().all()

    def test_object_dtype_junk_string_still_raises(self, tmp_path):
        """A non-numeric string in object dtype volume MUST still raise ValueError."""
        from src.assembled_core.data.prices_ingest import load_eod_prices

        df = self._minimal_price_df(["N/A_JUNK", "50000"])
        p = self._write_tmp_parquet(df, tmp_path)
        with pytest.raises(ValueError, match="non-numeric volume cells"):
            load_eod_prices(price_file=p)

    def test_float64_nan_does_not_raise(self, tmp_path):
        """Standard float64 NaN volume (from CSV/parquet) must not raise."""
        from src.assembled_core.data.prices_ingest import load_eod_prices

        df = self._minimal_price_df([float("nan"), 50000.0])
        p = self._write_tmp_parquet(df, tmp_path)
        result = load_eod_prices(price_file=p)
        assert isinstance(result, pd.DataFrame)

    def test_numpy_bool_mask_correctness(self):
        """Unit-test the coerce mask logic directly."""
        # Simulate what the fixed code does
        vol_series = pd.Series([pd.NA, 100, "bad", np.nan], dtype=object)
        raw_volume = pd.to_numeric(vol_series, errors="coerce")
        pre_existing_nan = vol_series.isna().to_numpy(dtype=bool)
        coerced_to_nan = raw_volume.isna().to_numpy(dtype=bool) & ~pre_existing_nan
        # pd.NA (index 0): pre_existing=True → coerced_to_nan=False (no false positive)
        # 100 (index 1): pre_existing=False, raw=not-NaN → coerced_to_nan=False
        # "bad" (index 2): pre_existing=False, raw=NaN → coerced_to_nan=True
        # np.nan (index 3): pre_existing=True → coerced_to_nan=False
        assert not coerced_to_nan[0], "pd.NA should not be flagged as corrupt"
        assert not coerced_to_nan[1], "100 is valid"
        assert coerced_to_nan[2], "'bad' should be flagged as corrupt"
        assert not coerced_to_nan[3], "np.nan should not be flagged as corrupt"


# ---------------------------------------------------------------------------
# Fix 2 — event_features.py graceful degradation for missing vectorized module
# ---------------------------------------------------------------------------


class TestEventFeaturesGracefulDegradation:
    """method='vectorized' with unavailable module must fall back to legacy, not raise."""

    def _make_events(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "disclosure_date": pd.to_datetime(["2024-01-01"], utc=True),
                "event_date": pd.to_datetime(["2024-01-01"], utc=True),
                "symbol": ["AAPL"],
                "value": [1.0],
            }
        )

    def _make_prices(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-02", "2024-01-03", "2024-01-04"], utc=True
                ),
                "symbol": ["AAPL"] * 3,
                "close": [150.0, 151.0, 152.0],
            }
        )

    def test_vectorized_method_fallback_no_import_error(self):
        """When vectorized module is None, must fall back to legacy — never ImportError."""
        import src.assembled_core.features.event_features as ef

        original = ef.build_event_feature_panel_vectorized
        try:
            ef.build_event_feature_panel_vectorized = None
            # Must NOT raise ImportError
            result = ef.build_event_feature_panel(
                self._make_events(),
                self._make_prices(),
                as_of=pd.Timestamp("2024-01-05", tz="UTC"),
                method="vectorized",
            )
            assert isinstance(result, pd.DataFrame)
        finally:
            ef.build_event_feature_panel_vectorized = original

    def test_vectorized_method_fallback_emits_warning(self, caplog):
        """Fallback must emit a WARNING, not be silent."""
        import src.assembled_core.features.event_features as ef

        original = ef.build_event_feature_panel_vectorized
        try:
            ef.build_event_feature_panel_vectorized = None
            with caplog.at_level(logging.WARNING):
                ef.build_event_feature_panel(
                    self._make_events(),
                    self._make_prices(),
                    as_of=pd.Timestamp("2024-01-05", tz="UTC"),
                    method="vectorized",
                )
            assert any(
                "Vectorized implementation not available" in r.message
                for r in caplog.records
            ), "Expected warning about vectorized fallback"
        finally:
            ef.build_event_feature_panel_vectorized = original

    def test_legacy_method_direct_works(self):
        """method='legacy' must always work regardless of vectorized availability."""
        from src.assembled_core.features.event_features import build_event_feature_panel

        result = build_event_feature_panel(
            self._make_events(),
            self._make_prices(),
            as_of=pd.Timestamp("2024-01-05", tz="UTC"),
            method="legacy",
        )
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Fix 3 — _tc_execution.py: negative qty warning (not silent)
# ---------------------------------------------------------------------------


class TestTcExecutionNegativeQtyWarning:
    """Negative qty must emit a WARNING before abs() is applied."""

    def _make_orders_with_negative_qty(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "side": ["SELL", "BUY"],
                "qty": [-10.0, 5.0],
                "price": [150.0, 300.0],
            }
        )

    def test_negative_qty_triggers_warning(self, caplog):
        """When qty < 0, route_orders must log a WARNING."""
        from src.assembled_core.pipeline._tc_execution import route_orders
        from src.assembled_core.pipeline.trading_cycle_shared import TradingContext

        prices = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2025-01-10"] * 2, utc=True),
                "symbol": ["AAPL", "MSFT"],
                "close": [150.0, 300.0],
            }
        )
        targets = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "target_weight": [0.1, 0.05],
                "target_qty": [-10.0, 5.0],
            }
        )
        ctx = TradingContext(prices=prices, capital=10000.0)
        orders = self._make_orders_with_negative_qty()

        # Patch _generate_orders_default to return our pre-built orders
        with (
            patch(
                "src.assembled_core.pipeline._tc_execution._generate_orders_default",
                return_value=orders,
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = route_orders(targets, ctx)

        assert any("negative qty" in r.message.lower() for r in caplog.records), (
            "Expected a WARNING about negative qty"
        )
        # After abs(), all qty should be non-negative
        assert (result["qty"] >= 0).all()


# ---------------------------------------------------------------------------
# Fix 4 — batch_runner.py: reset_dd_damper smoke
# ---------------------------------------------------------------------------


class TestBatchRunnerDDDamperReset:
    """reset_dd_damper is importable and idempotent."""

    def test_reset_dd_damper_callable(self):
        from src.assembled_core.strategies.multifactor_v2 import (
            _DD_DAMPER,
            reset_dd_damper,
            update_drawdown_damper,
        )
        import datetime

        # Contaminate damper state
        update_drawdown_damper(8000.0, as_of=datetime.date(2024, 6, 1))
        update_drawdown_damper(7000.0, as_of=datetime.date(2024, 6, 2))

        # Reset should restore defaults
        reset_dd_damper()
        assert _DD_DAMPER["peak_equity"] == pytest.approx(1.0)
        assert _DD_DAMPER["current_equity"] == pytest.approx(1.0)
        assert _DD_DAMPER["damper_active"] is False

    def test_batch_runner_imports_reset_dd_damper(self):
        """Smoke test: batch_runner's lazy import of reset_dd_damper works."""
        # This exercises the exact import path added in Fix 4
        try:
            from src.assembled_core.strategies.multifactor_v2 import reset_dd_damper

            reset_dd_damper()
        except Exception as exc:
            pytest.fail(f"reset_dd_damper import/call failed: {exc}")
