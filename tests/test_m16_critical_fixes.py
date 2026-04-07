"""M16 Welle 1 — Tests for critical fixes.

Verifies:
- Factor weight renormalization when factors have NaN
- POSITIVE_SHOCKS correction
- NaN handling in stacking (tree vs linear)
- PIT guard module
- Fill model in paper trading engine
- Policy enforcement wiring (sector exposure, gross exposure)

Marker: phase12
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase12


# ---------------------------------------------------------------------------
# 1.2 Factor weight renormalization
# ---------------------------------------------------------------------------


class TestFactorWeightRenormalization:
    """Verify mf_score is properly renormalized when factors have NaN."""

    def _make_bundle(self):
        from src.assembled_core.config.factor_bundles import (
            FactorBundleConfig,
            FactorConfig,
            FactorBundleOptions,
        )

        return FactorBundleConfig(
            universe="test",
            factor_set="test_set",
            horizon_days=20,
            factors=[
                FactorConfig(name="f1", weight=0.5, direction="positive"),
                FactorConfig(name="f2", weight=0.5, direction="positive"),
            ],
            options=FactorBundleOptions(zscore=False, winsorize=False),
        )

    def test_full_data_no_change(self):
        """When all factors available, score = weighted sum."""
        from src.assembled_core.signals.multifactor_signal import build_multifactor_signal

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01"] * 2, utc=True),
            "symbol": ["A", "B"],
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
        })
        result = build_multifactor_signal(df, self._make_bundle())
        # A: 0.5*1 + 0.5*3 = 2.0, B: 0.5*2 + 0.5*4 = 3.0
        scores = result.df.set_index("symbol")["mf_score"]
        assert abs(scores["A"] - 2.0) < 1e-10
        assert abs(scores["B"] - 3.0) < 1e-10

    def test_nan_factor_renormalized(self):
        """When one factor is NaN, the other factor's weight scales up to 1.0."""
        from src.assembled_core.signals.multifactor_signal import build_multifactor_signal

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01"] * 2, utc=True),
            "symbol": ["A", "B"],
            "f1": [1.0, 2.0],
            "f2": [np.nan, 4.0],  # A has NaN for f2
        })
        result = build_multifactor_signal(df, self._make_bundle())
        scores = result.df.set_index("symbol")["mf_score"]
        # A: only f1 available, renormalized: 0.5*1.0 * (1.0/0.5) = 1.0
        # (equivalent to using f1 with full weight 1.0)
        assert abs(scores["A"] - 1.0) < 1e-10
        # B: both available: 0.5*2 + 0.5*4 = 3.0
        assert abs(scores["B"] - 3.0) < 1e-10

    def test_all_nan_produces_nan(self):
        """When all factors are NaN, score should be NaN."""
        from src.assembled_core.signals.multifactor_signal import build_multifactor_signal

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
            "symbol": ["A"],
            "f1": [np.nan],
            "f2": [np.nan],
        })
        result = build_multifactor_signal(df, self._make_bundle())
        assert pd.isna(result.df["mf_score"].iloc[0])


# ---------------------------------------------------------------------------
# 2.3 POSITIVE_SHOCKS correction
# ---------------------------------------------------------------------------


class TestPositiveShocksCorrection:
    def test_energy_price_spike_not_positive(self):
        """ENERGY_PRICE_SPIKE should NOT be in POSITIVE_SHOCKS."""
        from src.assembled_core.intel.shock_propagation import POSITIVE_SHOCKS
        from src.assembled_core.intel.models import ShockType

        assert ShockType.ENERGY_PRICE_SPIKE not in POSITIVE_SHOCKS

    def test_cyber_risk_not_positive(self):
        """CYBER_RISK should NOT be in POSITIVE_SHOCKS."""
        from src.assembled_core.intel.shock_propagation import POSITIVE_SHOCKS
        from src.assembled_core.intel.models import ShockType

        assert ShockType.CYBER_RISK not in POSITIVE_SHOCKS

    def test_defense_demand_surge_still_positive(self):
        """DEFENSE_DEMAND_SURGE should remain positive."""
        from src.assembled_core.intel.shock_propagation import POSITIVE_SHOCKS
        from src.assembled_core.intel.models import ShockType

        assert ShockType.DEFENSE_DEMAND_SURGE in POSITIVE_SHOCKS


# ---------------------------------------------------------------------------
# 6.2 PIT Guard
# ---------------------------------------------------------------------------


class TestPITGuard:
    def test_valid_data_passes(self):
        from src.assembled_core.data.pit_guard import PITGuard

        guard = PITGuard(as_of=pd.Timestamp("2024-06-15", tz="UTC"))
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-06-14", "2024-06-15"], utc=True),
            "value": [1, 2],
        })
        assert guard.validate(df) is True

    def test_future_data_raises(self):
        from src.assembled_core.data.pit_guard import PITGuard, PITViolationError

        guard = PITGuard(as_of=pd.Timestamp("2024-06-15", tz="UTC"))
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-06-14", "2024-06-16"], utc=True),
            "value": [1, 2],
        })
        with pytest.raises(PITViolationError, match="PIT violation"):
            guard.validate(df)

    def test_warn_mode_returns_false(self):
        from src.assembled_core.data.pit_guard import PITGuard

        guard = PITGuard(as_of=pd.Timestamp("2024-06-15", tz="UTC"), mode="warn")
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-06-16"], utc=True),
            "value": [1],
        })
        assert guard.validate(df) is False

    def test_truncate_removes_future(self):
        from src.assembled_core.data.pit_guard import PITGuard

        guard = PITGuard(as_of=pd.Timestamp("2024-06-15", tz="UTC"))
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(
                ["2024-06-14", "2024-06-15", "2024-06-16"], utc=True
            ),
            "value": [1, 2, 3],
        })
        result = guard.truncate(df)
        assert len(result) == 2
        assert result["value"].tolist() == [1, 2]

    def test_empty_df_passes(self):
        from src.assembled_core.data.pit_guard import PITGuard

        guard = PITGuard(as_of=pd.Timestamp("2024-06-15", tz="UTC"))
        df = pd.DataFrame(columns=["timestamp", "value"])
        assert guard.validate(df) is True


# ---------------------------------------------------------------------------
# 8.1 Fill model
# ---------------------------------------------------------------------------


class TestFillModel:
    def test_fill_model_import(self):
        from src.assembled_core.execution.paper_trading_engine import FillModel

        fm = FillModel()
        assert fm.half_spread_bps == 5.0

    def test_fill_model_buy_costs_more(self):
        from src.assembled_core.execution.paper_trading_engine import FillModel

        fm = FillModel(half_spread_bps=10.0)
        fill_px, costs = fm.compute_fill_price(100.0, "BUY", 1000)
        assert fill_px > 100.0
        assert costs["spread_cost"] > 0
        assert costs["impact_cost"] > 0

    def test_fill_model_sell_receives_less(self):
        from src.assembled_core.execution.paper_trading_engine import FillModel

        fm = FillModel(half_spread_bps=10.0)
        fill_px, costs = fm.compute_fill_price(100.0, "SELL", 1000)
        assert fill_px < 100.0

    def test_engine_with_fill_model(self):
        from src.assembled_core.execution.paper_trading_engine import (
            FillModel,
            PaperOrder,
            PaperTradingEngine,
        )

        engine = PaperTradingEngine(fill_model=FillModel())
        order = PaperOrder(
            order_id="test-1",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            status="NEW",
        )
        filled = engine.submit_orders([order])
        assert filled[0].status == "FILLED"
        assert filled[0].fill_price is not None
        assert filled[0].fill_price > 150.0  # BUY costs more
        assert filled[0].fill_cost_breakdown is not None

    def test_engine_without_fill_model(self):
        """Legacy behaviour: no fill model → fill at order price."""
        from src.assembled_core.execution.paper_trading_engine import (
            PaperOrder,
            PaperTradingEngine,
        )

        engine = PaperTradingEngine()
        order = PaperOrder(
            order_id="test-2",
            symbol="MSFT",
            side="BUY",
            quantity=50,
            price=400.0,
            status="NEW",
        )
        filled = engine.submit_orders([order])
        assert filled[0].fill_price == 400.0
        assert filled[0].fill_cost_breakdown is None

    def test_larger_orders_have_more_impact(self):
        """Market impact scales with sqrt(qty/adv)."""
        from src.assembled_core.execution.paper_trading_engine import FillModel

        fm = FillModel()
        _, costs_small = fm.compute_fill_price(100.0, "BUY", 100)
        _, costs_large = fm.compute_fill_price(100.0, "BUY", 100_000)
        assert costs_large["impact_cost"] > costs_small["impact_cost"]


# ---------------------------------------------------------------------------
# 6.1 Universe delisting handling
# ---------------------------------------------------------------------------


class TestUniverseDelisting:
    def test_null_end_date_still_active_default(self):
        """Default behaviour: end_date=NaT means still active (backwards compatible)."""
        from src.assembled_core.data.universe import get_universe_members

        # We test the logic directly - this verifies backwards compatibility
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            df = pd.DataFrame({
                "symbol": ["AAPL", "DEAD"],
                "start_date": ["2020-01-01", "2020-01-01"],
                "end_date": [None, None],
            })
            df.to_parquet(root / "test_uni.parquet", index=False)
            members = get_universe_members(
                as_of="2024-01-01", universe_name="test_uni", root=root
            )
            assert "AAPL" in members
            assert "DEAD" in members  # Still included by default

    def test_require_active_status_filters(self):
        """With require_active_status=True, only status='active' symbols with NaT end_date."""
        from src.assembled_core.data.universe import get_universe_members

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            df = pd.DataFrame({
                "symbol": ["AAPL", "DEAD"],
                "start_date": ["2020-01-01", "2020-01-01"],
                "end_date": [None, None],
                "status": ["active", "delisted"],
            })
            df.to_parquet(root / "test_uni.parquet", index=False)
            members = get_universe_members(
                as_of="2024-01-01",
                universe_name="test_uni",
                root=root,
                require_active_status=True,
            )
            assert "AAPL" in members
            assert "DEAD" not in members  # Excluded because status != active
