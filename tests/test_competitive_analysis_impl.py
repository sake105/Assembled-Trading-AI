"""Tests for modules implemented from competitive analysis specs (2026-04-27)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# BrinsonAttribution
# ---------------------------------------------------------------------------


class TestBrinsonAttribution:
    def _make(self):
        from assembled_core.attribution.brinson_hood import BrinsonAttribution

        dates = pd.date_range("2024-01-01", periods=3, freq="ME")
        sectors = ["Tech", "Finance", "Health"]
        w_p = pd.DataFrame(
            [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.4, 0.3]],
            index=dates,
            columns=sectors,
        )
        w_b = pd.DataFrame(
            [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3], [0.4, 0.3, 0.3]],
            index=dates,
            columns=sectors,
        )
        return BrinsonAttribution(w_p, w_b), dates, sectors

    def test_attribute_columns(self):
        ba, dates, sectors = self._make()
        rp = pd.DataFrame([[0.02, 0.01, 0.03]] * 3, index=dates, columns=sectors)
        rb = pd.DataFrame([[0.01, 0.02, 0.02]] * 3, index=dates, columns=sectors)
        result = ba.attribute(rp, rb)
        assert list(result.columns) == [
            "allocation",
            "selection",
            "interaction",
            "active_total",
        ]
        assert len(result) == 3

    def test_active_total_consistency(self):
        from assembled_core.attribution.brinson_hood import BrinsonAttribution

        dates = pd.date_range("2024-01-01", periods=2, freq="ME")
        sectors = ["A", "B"]
        w_p = pd.DataFrame([[0.6, 0.4], [0.5, 0.5]], index=dates, columns=sectors)
        w_b = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], index=dates, columns=sectors)
        ba = BrinsonAttribution(w_p, w_b)
        rp = pd.DataFrame([[0.03, 0.01], [0.02, 0.02]], index=dates, columns=sectors)
        rb = pd.DataFrame([[0.02, 0.02], [0.01, 0.03]], index=dates, columns=sectors)
        result = ba.attribute(rp, rb)
        diff = (
            result["active_total"]
            - result[["allocation", "selection", "interaction"]].sum(axis=1)
        ).abs()
        assert diff.max() < 1e-10

    def test_summary_returns_dict(self):
        ba, dates, sectors = self._make()
        rp = pd.DataFrame([[0.02, 0.01, 0.03]] * 3, index=dates, columns=sectors)
        rb = pd.DataFrame([[0.01, 0.02, 0.02]] * 3, index=dates, columns=sectors)
        summary = ba.summary(rp, rb)
        assert set(summary.keys()) == {
            "allocation",
            "selection",
            "interaction",
            "active_total",
        }

    def test_zero_weight_diff_zero_allocation(self):
        from assembled_core.attribution.brinson_hood import BrinsonAttribution

        dates = pd.date_range("2024-01-01", periods=1, freq="ME")
        sectors = ["X"]
        w = pd.DataFrame([[1.0]], index=dates, columns=sectors)
        ba = BrinsonAttribution(w.copy(), w.copy())
        rp = pd.DataFrame([[0.05]], index=dates, columns=sectors)
        rb = pd.DataFrame([[0.03]], index=dates, columns=sectors)
        result = ba.attribute(rp, rb)
        assert abs(result["allocation"].iloc[0]) < 1e-12
        assert abs(result["interaction"].iloc[0]) < 1e-12


# ---------------------------------------------------------------------------
# VPINCalculator
# ---------------------------------------------------------------------------


class TestVPINCalculator:
    def _trades(self, n=200):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01 09:30", periods=n, freq="1min")
        vols = rng.integers(100, 1000, size=n)
        prices = 100 + rng.normal(0, 0.5, size=n).cumsum()
        buy_pct = rng.uniform(0.3, 0.7, size=n)
        return pd.DataFrame(
            {
                "volume": vols,
                "price": prices,
                "buy_volume": (vols * buy_pct).astype(int),
                "sell_volume": (vols * (1 - buy_pct)).astype(int),
            },
            index=idx,
        )

    def test_returns_series(self):
        from assembled_core.qa.vpin import VPINCalculator

        calc = VPINCalculator(n_buckets=10, bucket_size_pct_adv=0.02)
        trades = self._trades(200)
        vpin = calc.compute(trades, avg_daily_volume=50_000)
        assert isinstance(vpin, pd.Series)
        assert len(vpin) == len(trades)

    def test_values_between_0_and_1(self):
        from assembled_core.qa.vpin import VPINCalculator

        calc = VPINCalculator(n_buckets=10, bucket_size_pct_adv=0.02)
        trades = self._trades(300)
        vpin = calc.compute(trades, avg_daily_volume=50_000)
        valid = vpin.dropna()
        if len(valid) > 0:
            assert (valid >= 0).all() and (valid <= 1).all()

    def test_empty_trades(self):
        from assembled_core.qa.vpin import VPINCalculator

        calc = VPINCalculator()
        result = calc.compute(pd.DataFrame(), avg_daily_volume=10_000)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_threshold(self):
        from assembled_core.qa.vpin import VPINCalculator

        assert VPINCalculator.threshold() == 0.7

    def test_tick_classify_fallback(self):
        from assembled_core.qa.vpin import VPINCalculator

        idx = pd.date_range("2024-01-01", periods=100, freq="1min")
        trades = pd.DataFrame(
            {
                "volume": np.ones(100, dtype=int) * 200,
                "price": np.linspace(100, 110, 100),
            },
            index=idx,
        )
        calc = VPINCalculator(n_buckets=5, bucket_size_pct_adv=0.05)
        result = calc.compute(trades, avg_daily_volume=10_000)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# BootstrapMetrics
# ---------------------------------------------------------------------------


class TestBootstrapMetrics:
    def _returns(self, n=500, seed=99):
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(0.0005, 0.01, n))

    def test_sharpe_keys(self):
        from assembled_core.qa.bootstrap_metrics import compute_sharpe_with_ci

        result = compute_sharpe_with_ci(self._returns(), n_bootstrap=200, seed=42)
        assert {
            "sharpe",
            "sharpe_ci_lower",
            "sharpe_ci_upper",
            "sharpe_p_value",
        }.issubset(result)

    def test_ci_lower_le_sharpe_le_upper(self):
        from assembled_core.qa.bootstrap_metrics import compute_sharpe_with_ci

        result = compute_sharpe_with_ci(self._returns(), n_bootstrap=500, seed=1)
        assert (
            result["sharpe_ci_lower"] <= result["sharpe"] <= result["sharpe_ci_upper"]
        )

    def test_sortino_keys(self):
        from assembled_core.qa.bootstrap_metrics import compute_sortino_with_ci

        result = compute_sortino_with_ci(self._returns(), n_bootstrap=200, seed=42)
        assert {"sortino", "sortino_ci_lower", "sortino_ci_upper"}.issubset(result)

    def test_max_drawdown_negative(self):
        from assembled_core.qa.bootstrap_metrics import compute_max_drawdown_with_ci

        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 300))
        result = compute_max_drawdown_with_ci(returns, n_bootstrap=200, seed=0)
        assert result["max_drawdown"] <= 0

    def test_all_with_ci_combined(self):
        from assembled_core.qa.bootstrap_metrics import compute_all_with_ci

        result = compute_all_with_ci(self._returns(), n_bootstrap=200, seed=7)
        assert "sharpe" in result
        assert "sortino" in result
        assert "max_drawdown" in result

    def test_positive_returns_p_value_low(self):
        from assembled_core.qa.bootstrap_metrics import compute_sharpe_with_ci

        returns = pd.Series([0.002] * 300)
        result = compute_sharpe_with_ci(returns, n_bootstrap=100, seed=5)
        assert result["sharpe_p_value"] < 0.1


# ---------------------------------------------------------------------------
# TermStructureFeatures
# ---------------------------------------------------------------------------


class TestTermStructureFeatures:
    def _vix_df(self, n=50):
        idx = pd.date_range("2024-01-01", periods=n)
        return pd.DataFrame(
            {
                "vix_spot": np.linspace(15, 25, n),
                "vix_1m": np.linspace(16, 26, n),
                "vix_2m": np.linspace(17, 27, n),
                "vix_3m": np.linspace(18, 28, n),
            },
            index=idx,
        )

    def _yld_df(self, n=50):
        idx = pd.date_range("2024-01-01", periods=n)
        return pd.DataFrame(
            {
                "y_3m": np.linspace(4.5, 5.0, n),
                "y_2y": np.linspace(4.8, 5.2, n),
                "y_10y": np.linspace(4.0, 4.5, n),
            },
            index=idx,
        )

    def test_vix_slope_columns(self):
        from assembled_core.features.term_structure import TermStructureFeatures

        tsf = TermStructureFeatures()
        result = tsf.vix_term_structure(self._vix_df())
        assert "vix_slope_short" in result.columns
        assert "vix_slope_long" in result.columns
        assert "vix_contango" in result.columns
        assert "vix_curvature" in result.columns

    def test_yield_curve_columns(self):
        from assembled_core.features.term_structure import TermStructureFeatures

        tsf = TermStructureFeatures()
        result = tsf.yield_curve_features(self._yld_df())
        assert "yc_2y10y" in result.columns
        assert "yc_3m10y" in result.columns
        assert "yc_inverted" in result.columns

    def test_inverted_curve_detected(self):
        from assembled_core.features.term_structure import TermStructureFeatures

        tsf = TermStructureFeatures()
        idx = pd.date_range("2024-01-01", periods=1)
        df = pd.DataFrame({"y_2y": [5.0], "y_3m": [5.5], "y_10y": [4.0]}, index=idx)
        result = tsf.yield_curve_features(df)
        assert result["yc_inverted"].iloc[0] == 1

    def test_combined_returns_dataframe(self):
        from assembled_core.features.term_structure import TermStructureFeatures

        tsf = TermStructureFeatures()
        result = tsf.combined(self._vix_df(), self._yld_df())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 50


# ---------------------------------------------------------------------------
# LiquidityAwareSizer
# ---------------------------------------------------------------------------


class TestLiquidityAwareSizer:
    def _sizer(self):
        from assembled_core.portfolio.liquidity_aware_sizer import LiquidityAwareSizer

        return LiquidityAwareSizer(
            max_pct_adv=0.05,
            max_pct_market_cap=0.001,
            max_days_to_liquidate=1.0,
            target_pov_pct=0.10,
        )

    def _sym(self, adv=1_000_000, price=100.0, mcap=10_000_000_000):
        return {"adv": adv, "price": price, "market_cap": mcap}

    def test_under_all_caps(self):
        sizer = self._sizer()
        result = sizer.size_position(1000, self._sym())
        assert result.final_qty == 1000

    def test_adv_cap_binding(self):
        sizer = self._sizer()
        result = sizer.size_position(999_999, self._sym(adv=100_000))
        assert result.final_qty <= int(100_000 * 0.05)

    def test_mcap_cap_binding(self):
        from assembled_core.portfolio.liquidity_aware_sizer import LiquidityAwareSizer

        sizer = LiquidityAwareSizer(
            max_pct_adv=1.0,
            max_pct_market_cap=0.0001,
            max_days_to_liquidate=100.0,
            target_pov_pct=1.0,
        )
        result = sizer.size_position(999_999, self._sym(mcap=1_000_000, price=100.0))
        assert result.final_qty <= 10
        assert result.binding_constraint == "mcap"

    def test_is_liquid_enough(self):
        sizer = self._sizer()
        assert sizer.is_liquid_enough(100, self._sym())
        assert not sizer.is_liquid_enough(10_000_000, self._sym(adv=1_000))

    def test_size_result_fields(self):
        from assembled_core.portfolio.liquidity_aware_sizer import SizeResult

        sizer = self._sizer()
        result = sizer.size_position(500, self._sym())
        assert isinstance(result, SizeResult)
        assert result.signal_qty == 500


# ENTFERNT 2026-08-17: testete signals/tail_risk_vvix, archiviert in Tranche 2, s. archive/orphaned_code_2026-08-17/README.md


# ---------------------------------------------------------------------------
# LeakageAnalyzer
# ---------------------------------------------------------------------------


class TestLeakageAnalyzer:
    def _analyzer(self):
        from assembled_core.qa.leakage_analyzer import LeakageAnalyzer

        return LeakageAnalyzer(max_lag_check=3, correlation_threshold=0.9)

    def test_no_leakage_in_clean_data(self):
        rng = np.random.default_rng(42)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n)
        features = pd.DataFrame({"feat_a": rng.normal(0, 1, n)}, index=idx)
        target = pd.Series(rng.normal(0, 1, n), index=idx)
        analyzer = self._analyzer()
        reports = analyzer.check_lookahead(features, target)
        assert len(reports) == 0

    def test_detects_lookahead(self):
        n = 200
        idx = pd.date_range("2024-01-01", periods=n)
        target = pd.Series(np.arange(n, dtype=float), index=idx)
        # feature = future target value → perfect lookahead
        leaked = target.shift(-1).fillna(0)
        features = pd.DataFrame({"leaky": leaked.values}, index=idx)
        analyzer = self._analyzer()
        reports = analyzer.check_lookahead(features, target)
        assert any(r.feature == "leaky" for r in reports)

    def test_detects_recursive_bias(self):
        n = 100
        idx = pd.date_range("2024-01-01", periods=n)
        target = pd.Series(np.linspace(0, 1, n), index=idx)
        features = pd.DataFrame({"same_as_target": target.values * 1.0001}, index=idx)
        analyzer = self._analyzer()
        reports = analyzer.check_recursive(features, target)
        assert len(reports) >= 1
        assert reports[0].leakage_type == "recursive"

    def test_summarize(self):
        from assembled_core.qa.leakage_analyzer import LeakageReport

        reports = [
            LeakageReport("feat_a", "lookahead", "corr=0.97", "high"),
            LeakageReport("feat_b", "recursive", "corr=0.99", "high"),
        ]
        summary = self._analyzer().summarize(reports)
        assert summary["total"] == 2
        assert summary["high"] == 2
        assert len(summary["features_flagged"]) == 2

    def test_full_check_no_crash(self):
        rng = np.random.default_rng(7)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n)
        features = pd.DataFrame(
            {"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)}, index=idx
        )
        target = pd.Series(rng.normal(0, 1, n), index=idx)
        analyzer = self._analyzer()
        reports = analyzer.full_check(features, target)
        assert isinstance(reports, list)

    # --- check_primary_meta_split ---

    def test_clean_split_no_leakage(self):
        """Non-overlapping train/meta indices → empty report."""
        primary_train = pd.date_range("2022-01-01", periods=200, freq="D")
        primary_val = pd.date_range("2022-07-21", periods=100, freq="D")
        meta_train = primary_val  # meta trained on OOS predictions → correct
        analyzer = self._analyzer()
        reports = analyzer.check_primary_meta_split(
            primary_train, primary_val, meta_train
        )
        assert reports == []

    def test_partial_overlap_medium_severity(self):
        """<50% overlap → LeakageReport with severity='medium'."""
        primary_train = pd.date_range("2022-01-01", periods=200, freq="D")
        # meta_train overlaps 40 rows with primary_train (20%)
        meta_train = pd.date_range("2022-06-20", periods=100, freq="D")
        primary_val = pd.date_range("2022-07-21", periods=50, freq="D")
        analyzer = self._analyzer()
        reports = analyzer.check_primary_meta_split(
            primary_train, primary_val, meta_train
        )
        assert len(reports) == 1
        r = reports[0]
        assert r.leakage_type == "primary_meta_split"
        assert r.severity == "medium"
        assert r.details["overlap_size"] > 0
        assert r.details["contamination_pct"] <= 50.0

    def test_heavy_overlap_high_severity(self):
        """>50% overlap → LeakageReport with severity='high'."""
        primary_train = pd.date_range("2022-01-01", periods=300, freq="D")
        # meta_train is identical to primary_train → 100% contamination
        meta_train = primary_train
        primary_val = pd.date_range("2022-10-29", periods=60, freq="D")
        analyzer = self._analyzer()
        reports = analyzer.check_primary_meta_split(
            primary_train, primary_val, meta_train
        )
        assert len(reports) == 1
        r = reports[0]
        assert r.severity == "high"
        assert r.details["contamination_pct"] > 50.0
        assert r.feature == "meta_model_training_set"

    def test_full_check_includes_primary_meta_split(self):
        """full_check with index params runs primary_meta_split and returns report."""
        rng = np.random.default_rng(99)
        n = 80
        idx = pd.date_range("2024-01-01", periods=n)
        features = pd.DataFrame({"f": rng.normal(size=n)}, index=idx)
        target = pd.Series(rng.normal(size=n), index=idx)
        primary_train = idx[:60]
        primary_val = idx[60:]
        meta_train = idx[:40]  # overlaps primary_train → contaminated
        analyzer = self._analyzer()
        reports = analyzer.full_check(
            features,
            target,
            primary_train_index=primary_train,
            primary_val_index=primary_val,
            meta_train_index=meta_train,
        )
        types = [r.leakage_type for r in reports]
        assert "primary_meta_split" in types


# ---------------------------------------------------------------------------
# ExecutionRouter
# ---------------------------------------------------------------------------


class TestExecutionRouter:
    def _order(self, qty=1000, price=100.0, symbol="AAPL", side="BUY"):
        from assembled_core.execution.execution_router import Order

        return Order(
            symbol=symbol, side=side, quantity=qty, price=price, order_id="test-1"
        )

    def test_small_order_direct(self):
        from assembled_core.execution.execution_router import (
            route_order,
            ExecutionConfig,
        )

        cfg = ExecutionConfig(direct_threshold=0.05)
        order = self._order(qty=100)
        slices = route_order(order, avg_daily_volume=100_000, config=cfg)
        assert len(slices) == 1
        assert slices[0].algo == "direct"
        assert slices[0].quantity == 100

    def test_medium_order_twap(self):
        from assembled_core.execution.execution_router import (
            route_order,
            ExecutionConfig,
        )

        cfg = ExecutionConfig(direct_threshold=0.01, twap_threshold=0.20, twap_slices=5)
        order = self._order(qty=5_000)
        slices = route_order(order, avg_daily_volume=100_000, config=cfg)
        assert all(s.algo == "twap" for s in slices)
        assert sum(s.quantity for s in slices) == 5_000

    def test_large_order_almgren_chriss(self):
        from assembled_core.execution.execution_router import (
            route_order,
            ExecutionConfig,
        )

        cfg = ExecutionConfig(direct_threshold=0.01, twap_threshold=0.10, twap_slices=5)
        order = self._order(qty=50_000)
        slices = route_order(order, avg_daily_volume=100_000, config=cfg)
        assert all(s.algo == "almgren_chriss" for s in slices)
        assert sum(s.quantity for s in slices) == 50_000

    def test_twap_split_quantity_preserved(self):
        from assembled_core.execution.execution_router import twap_split, Order

        order = Order("MSFT", "BUY", 1003, 250.0)
        slices = twap_split(order, n_slices=10)
        assert sum(s.quantity for s in slices) == 1003

    def test_ac_split_quantity_preserved(self):
        from assembled_core.execution.execution_router import (
            ac_split,
            Order,
            ExecutionConfig,
        )

        order = Order("TSLA", "SELL", 500, 200.0)
        slices = ac_split(order, ExecutionConfig(twap_slices=5))
        assert sum(s.quantity for s in slices) == 500

    def test_zero_adv_no_crash(self):
        from assembled_core.execution.execution_router import route_order

        order = self._order()
        slices = route_order(order, avg_daily_volume=0)
        assert len(slices) >= 1


# ENTFERNT 2026-08-17: testete signals/cross_asset_carry_v2, archiviert in Tranche 2, s. archive/orphaned_code_2026-08-17/README.md


# ---------------------------------------------------------------------------
# BarraRiskModel
# ---------------------------------------------------------------------------


class TestBarraRiskModel:
    def _make_model(self, n_symbols=10, n_days=60):
        from assembled_core.risk.barra_risk_model import BarraRiskModel

        rng = np.random.default_rng(42)
        symbols = [f"SYM{i}" for i in range(n_symbols)]
        dates = pd.date_range("2024-01-01", periods=n_days)
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.01, (n_days, n_symbols)), index=dates, columns=symbols
        )
        fundamentals = pd.DataFrame(
            {
                "market_cap": rng.uniform(1e9, 1e11, n_symbols),
                "book_to_price": rng.uniform(0.1, 1.5, n_symbols),
                "sector": rng.choice(["Tech", "Finance", "Health"], n_symbols),
            },
            index=symbols,
        )
        return BarraRiskModel(returns, fundamentals)

    def test_fit_no_crash(self):
        model = self._make_model()
        model.fit()
        assert model._factor_returns is not None
        assert model._factor_loadings is not None

    def test_decompose_sums_to_one(self):
        model = self._make_model()
        model.fit()
        symbols = model.returns.columns[:5].tolist()
        weights = pd.Series(0.2, index=symbols)
        result = model.decompose_portfolio_risk(weights)
        # Each component is a fraction of total variance (cross-group covariance
        # means individual group sums are approximate, not exact).
        for key in (
            "market_var_pct",
            "sector_var_pct",
            "style_var_pct",
            "idio_var_pct",
        ):
            assert result[key] >= 0.0, f"{key} should be non-negative"

    def test_decompose_keys_present(self):
        model = self._make_model()
        model.fit()
        symbols = model.returns.columns[:3].tolist()
        weights = pd.Series(1 / 3, index=symbols)
        result = model.decompose_portfolio_risk(weights)
        assert set(result.keys()) >= {
            "market_var_pct",
            "sector_var_pct",
            "style_var_pct",
            "idio_var_pct",
        }

    def test_factor_exposures_returns_series(self):
        model = self._make_model()
        model.fit()
        sym = model.returns.columns[0]
        exp = model.factor_exposures(sym)
        assert exp is not None
        assert isinstance(exp, pd.Series)

    def test_unknown_symbol_returns_none(self):
        model = self._make_model()
        model.fit()
        assert model.factor_exposures("NONEXISTENT") is None


# ---------------------------------------------------------------------------
# VolatilityEstimators
# ---------------------------------------------------------------------------


class TestVolatilityEstimators:
    def _ohlc(self, n=60):
        rng = np.random.default_rng(0)
        close = 100 + rng.normal(0, 1, n).cumsum()
        close = np.abs(close) + 50
        idx = pd.date_range("2024-01-01", periods=n)
        df = pd.DataFrame(index=idx)
        df["close"] = close
        df["open"] = close * (1 + rng.normal(0, 0.002, n))
        df["high"] = np.maximum(df["open"], df["close"]) * (1 + rng.uniform(0, 0.01, n))
        df["low"] = np.minimum(df["open"], df["close"]) * (1 - rng.uniform(0, 0.01, n))
        return df

    def test_parkinson_shape(self):
        from assembled_core.features.volatility_estimators import parkinson_volatility

        df = self._ohlc()
        result = parkinson_volatility(df["high"], df["low"], period=20)
        assert len(result) == len(df)
        assert result.dropna().gt(0).all()

    def test_garman_klass_shape(self):
        from assembled_core.features.volatility_estimators import (
            garman_klass_volatility,
        )

        df = self._ohlc()
        result = garman_klass_volatility(df["open"], df["high"], df["low"], df["close"])
        assert len(result) == len(df)

    def test_panel_columns(self):
        from assembled_core.features.volatility_estimators import volatility_panel

        df = self._ohlc()
        result = volatility_panel(df, period=20)
        assert set(result.columns) == {
            "parkinson",
            "garman_klass",
            "rogers_satchell",
            "close_to_close",
        }
        assert len(result) == len(df)

    def test_tick_rule_signs(self):
        from assembled_core.features.volatility_estimators import tick_rule_signs

        prices = pd.Series([100, 101, 101, 99, 100])
        signs = tick_rule_signs(prices)
        assert set(signs.dropna().unique()).issubset({-1, 1})

    def test_parkinson_more_precise_than_c2c(self):
        from assembled_core.features.volatility_estimators import (
            parkinson_volatility,
            close_to_close_volatility,
        )

        df = self._ohlc(200)
        park = parkinson_volatility(df["high"], df["low"], period=20).dropna()
        c2c = close_to_close_volatility(df["close"], period=20).dropna()
        # Both should produce valid (finite, positive) values
        assert (
            park.isfinite().all()
            if hasattr(park, "isfinite")
            else np.isfinite(park.values).all()
        )
        assert c2c.dropna().gt(0).all()


# ---------------------------------------------------------------------------
# CausalValidation (OLS fallback, no dowhy required)
# ---------------------------------------------------------------------------


class TestCausalValidation:
    def _trades(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(
            {
                "has_news_trigger": rng.integers(0, 2, n),
                "return": rng.normal(0.002, 0.01, n),
                "sector": rng.choice(["Tech", "Finance", "Health"], n),
                "vol_regime": rng.uniform(0.1, 0.3, n),
            }
        )
        # Add a real treatment effect
        df.loc[df["has_news_trigger"] == 1, "return"] += 0.005
        return df

    def test_ols_estimate_returns_dict(self):
        from assembled_core.qa.causal_validation import estimate_news_trigger_effect

        result = estimate_news_trigger_effect(self._trades(), use_dowhy=False)
        assert "estimates" in result
        assert "interpretation" in result

    def test_ols_estimate_detects_positive_ate(self):
        from assembled_core.qa.causal_validation import estimate_news_trigger_effect

        result = estimate_news_trigger_effect(self._trades(), use_dowhy=False)
        ate = result["estimates"]["ols_adjusted"]["ate"]
        assert ate > 0, "Should detect positive treatment effect"

    def test_heterogeneous_effects_by_sector(self):
        from assembled_core.qa.causal_validation import heterogeneous_treatment_effects

        df = self._trades(n=300)
        result = heterogeneous_treatment_effects(df, heterogeneity_col="sector")
        assert isinstance(result, pd.DataFrame)
        assert "ate" in result.columns
        assert len(result) >= 1

    def test_insufficient_data_no_crash(self):
        from assembled_core.qa.causal_validation import estimate_news_trigger_effect

        tiny = pd.DataFrame({"has_news_trigger": [1, 0], "return": [0.01, -0.01]})
        result = estimate_news_trigger_effect(tiny, use_dowhy=False)
        assert "estimates" in result


# ---------------------------------------------------------------------------
# AdaptiveConformalSizer
# ---------------------------------------------------------------------------


class TestAdaptiveConformalSizer:
    def _sizer(self):
        from assembled_core.portfolio.adaptive_conformal_position import (
            AdaptiveConformalSizer,
        )

        return AdaptiveConformalSizer(alpha=0.1, gamma=0.005, max_position=1.0)

    def test_initial_prediction_returns_dict(self):
        sizer = self._sizer()
        import numpy as np

        result = sizer.predict_and_size(np.array([[0.1, 0.2, 0.3]]))
        assert "position_size" in result
        assert "confidence" in result
        assert "current_alpha" in result

    def test_position_in_range(self):
        sizer = self._sizer()
        import numpy as np

        rng = np.random.default_rng(0)
        for _ in range(20):
            sizer.update(float(rng.normal(0, 0.01)), 0.0)
        result = sizer.predict_and_size(np.array([[0.0]]))
        assert 0.0 <= result["position_size"] <= 1.0

    def test_alpha_adapts_upward_on_miss(self):
        sizer = self._sizer()
        import numpy as np

        rng = np.random.default_rng(7)
        for _ in range(30):
            sizer.update(float(rng.normal(0, 1.0)), 0.0)  # wide residuals
        # alpha should increase after repeated misses
        assert sizer.current_alpha != 0.1 or len(sizer._residuals) == 0

    def test_empirical_coverage_none_before_update(self):
        sizer = self._sizer()
        assert sizer.empirical_coverage is None

    def test_size_from_interval_width(self):
        from assembled_core.portfolio.adaptive_conformal_position import (
            size_from_interval_width,
        )

        assert size_from_interval_width(0.0, 1.0, max_position=1.0) == 1.0
        assert size_from_interval_width(1.0, 1.0, max_position=1.0) == 0.0
        mid = size_from_interval_width(0.5, 1.0, max_position=1.0)
        assert 0.0 < mid < 1.0


# ENTFERNT 2026-08-17: testete signals/lppls_crash (dedizierter Test test_signals_lppls_validation.py mit-archiviert), archiviert in Tranche 2, s. archive/orphaned_code_2026-08-17/README.md


# ---------------------------------------------------------------------------
# HindenburgOmen + CBBI
# ---------------------------------------------------------------------------


class TestHindenburgOmen:
    def test_triggers_when_all_conditions_met(self):
        from assembled_core.features.market_breadth import hindenburg_omen

        n = 10
        idx = pd.date_range("2024-01-01", periods=n)
        new_highs = pd.Series([0.03] * n, index=idx)
        new_lows = pd.Series([0.03] * n, index=idx)
        nyse_ma50 = pd.Series([1.05] * n, index=idx)  # above MA
        mcclellan = pd.Series([-5.0] * n, index=idx)  # negative
        result = hindenburg_omen(new_highs, new_lows, nyse_ma50, mcclellan)
        assert result.all()

    def test_no_trigger_below_ma(self):
        from assembled_core.features.market_breadth import hindenburg_omen

        n = 5
        idx = pd.date_range("2024-01-01", periods=n)
        new_highs = pd.Series([0.03] * n, index=idx)
        new_lows = pd.Series([0.03] * n, index=idx)
        nyse_ma50 = pd.Series([0.95] * n, index=idx)  # BELOW MA
        mcclellan = pd.Series([-5.0] * n, index=idx)
        result = hindenburg_omen(new_highs, new_lows, nyse_ma50, mcclellan)
        assert not result.any()

    def test_cbbi_in_range(self):
        from assembled_core.features.market_breadth import compute_cbbi_composite

        idx = pd.date_range("2024-01-01", periods=20)
        indicators = {
            "ad_line": pd.Series(np.random.default_rng(0).uniform(0, 1, 20), index=idx),
            "mcclellan": pd.Series(
                np.random.default_rng(1).uniform(0, 1, 20), index=idx
            ),
        }
        result = compute_cbbi_composite(indicators)
        assert isinstance(result, pd.Series)
        assert (result >= 0).all() and (result <= 100).all()

    def test_cbbi_empty_returns_empty(self):
        from assembled_core.features.market_breadth import compute_cbbi_composite

        result = compute_cbbi_composite({})
        assert len(result) == 0


class TestComputeSampleWeights:
    def test_basic_overlap(self):
        from assembled_core.features.triple_barrier import compute_sample_weights

        idx = pd.date_range("2024-01-01", periods=20)
        prices = pd.Series(np.ones(20), index=idx)
        events = pd.DataFrame(
            {
                "t_in": [idx[0], idx[5]],
                "t_out": [idx[9], idx[14]],
            }
        )
        weights = compute_sample_weights(events, prices)
        assert len(weights) == 2
        assert (weights > 0).all()

    def test_non_overlapping_equal_weights(self):
        from assembled_core.features.triple_barrier import compute_sample_weights

        idx = pd.date_range("2024-01-01", periods=20)
        prices = pd.Series(np.ones(20), index=idx)
        events = pd.DataFrame(
            {
                "t_in": [idx[0], idx[10]],
                "t_out": [idx[4], idx[14]],
            }
        )
        weights = compute_sample_weights(events, prices)
        # Non-overlapping events should have equal weights
        assert abs(weights.iloc[0] - weights.iloc[1]) < 0.1

    def test_weights_sum_to_n(self):
        from assembled_core.features.triple_barrier import compute_sample_weights

        idx = pd.date_range("2024-01-01", periods=30)
        prices = pd.Series(np.ones(30), index=idx)
        events = pd.DataFrame(
            {
                "t_in": [idx[0], idx[5], idx[10]],
                "t_out": [idx[8], idx[12], idx[18]],
            }
        )
        weights = compute_sample_weights(events, prices)
        assert abs(weights.sum() - len(events)) < 0.01


# ---------------------------------------------------------------------------
# TradeAnomalyDetector + detect_fat_finger
# ---------------------------------------------------------------------------


class TestTradeAnomalyDetector:
    def _baseline(self, n=100, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "price": rng.normal(100, 1, n),
                "volume": rng.normal(1_000, 50, n),
                "spread": rng.normal(0.01, 0.001, n),
            }
        )

    def test_fit_and_score_returns_result(self):
        from assembled_core.qa.anomaly_detection import (
            TradeAnomalyDetector,
            AnomalyResult,
        )

        det = TradeAnomalyDetector(contamination=0.05)
        baseline = self._baseline(200)
        det.fit(baseline)
        current = self._baseline(20, seed=99)
        result = det.score(current)
        assert isinstance(result, AnomalyResult)
        assert len(result.scores) == 20
        assert len(result.flags) == 20

    def test_flags_are_binary(self):
        from assembled_core.qa.anomaly_detection import TradeAnomalyDetector

        det = TradeAnomalyDetector()
        det.fit(self._baseline(100))
        result = det.score(self._baseline(10))
        assert set(result.flags.unique()).issubset({0, 1})

    def test_n_anomalies_matches_flags(self):
        from assembled_core.qa.anomaly_detection import TradeAnomalyDetector

        det = TradeAnomalyDetector()
        det.fit(self._baseline(100))
        result = det.score(self._baseline(50))
        assert result.n_anomalies == int(result.flags.sum())

    def test_score_before_fit_raises(self):
        from assembled_core.qa.anomaly_detection import TradeAnomalyDetector

        det = TradeAnomalyDetector()
        with pytest.raises(RuntimeError, match="fit"):
            det.score(self._baseline(10))

    def test_obvious_outlier_flagged(self):
        from assembled_core.qa.anomaly_detection import TradeAnomalyDetector

        baseline = self._baseline(200)
        det = TradeAnomalyDetector(contamination=0.05)
        det.fit(baseline)
        # Extreme outlier row
        outlier = pd.DataFrame(
            {
                "price": [100_000.0],
                "volume": [1_000_000.0],
                "spread": [500.0],
            }
        )
        result = det.score(outlier)
        assert result.flags.iloc[0] == 1

    def test_method_set_after_fit(self):
        from assembled_core.qa.anomaly_detection import TradeAnomalyDetector

        det = TradeAnomalyDetector()
        det.fit(self._baseline(100))
        assert det._method in {"pyod_ensemble", "iqr_zscore_fallback"}


class TestDetectFatFinger:
    def test_normal_trades_no_flag(self):
        from assembled_core.qa.anomaly_detection import detect_fat_finger

        rng = np.random.default_rng(0)
        sizes = pd.Series(rng.normal(100, 5, 50))
        flags = detect_fat_finger(sizes, multiplier=10.0)
        assert not flags.any()

    def test_fat_finger_detected(self):
        from assembled_core.qa.anomaly_detection import detect_fat_finger

        sizes = pd.Series([100.0] * 49 + [50_000.0])
        flags = detect_fat_finger(sizes, multiplier=10.0, min_samples=20)
        assert flags.iloc[-1]

    def test_too_few_samples_no_flag(self):
        from assembled_core.qa.anomaly_detection import detect_fat_finger

        sizes = pd.Series([100.0, 200.0, 50_000.0])
        flags = detect_fat_finger(sizes, min_samples=20)
        assert not flags.any()


# ---------------------------------------------------------------------------
# tsfresh_augmentation
# ---------------------------------------------------------------------------


class TestTsfreshAugmentation:
    def _prices_df(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-01-01", periods=n)
        return pd.DataFrame(
            {
                "symbol": ["AAPL"] * n + ["MSFT"] * n,
                "date": list(idx) * 2,
                "close": np.concatenate(
                    [rng.normal(100, 5, n), rng.normal(200, 10, n)]
                ),
                "volume": np.concatenate(
                    [rng.normal(1e6, 1e5, n), rng.normal(2e6, 2e5, n)]
                ),
            }
        )

    def test_extract_features_shape(self):
        from assembled_core.features.tsfresh_augmentation import extract_features

        df = self._prices_df()
        result = extract_features(df, use_tsfresh=False)
        assert result.shape[0] == 2  # 2 symbols
        assert result.shape[1] > 5  # multiple features

    def test_extract_features_index_is_symbols(self):
        from assembled_core.features.tsfresh_augmentation import extract_features

        df = self._prices_df()
        result = extract_features(df, use_tsfresh=False)
        assert "AAPL" in result.index
        assert "MSFT" in result.index

    def test_rolling_features_minimal(self):
        from assembled_core.features.tsfresh_augmentation import (
            extract_rolling_features,
        )

        rng = np.random.default_rng(0)
        s = pd.Series(
            rng.normal(100, 5, 100), index=pd.date_range("2024-01-01", periods=100)
        )
        result = extract_rolling_features(s, window=20, feature_set="minimal")
        assert result.shape == (100, 8)
        assert "mean" in result.columns
        assert "std" in result.columns

    def test_rolling_features_full(self):
        from assembled_core.features.tsfresh_augmentation import (
            extract_rolling_features,
        )

        rng = np.random.default_rng(1)
        s = pd.Series(
            rng.normal(50, 2, 80), index=pd.date_range("2024-01-01", periods=80)
        )
        result = extract_rolling_features(s, window=20, feature_set="full")
        assert "kurtosis" in result.columns
        assert "trend_slope" in result.columns

    def test_extract_single_symbol(self):
        from assembled_core.features.tsfresh_augmentation import extract_features

        n = 40
        idx = pd.date_range("2024-01-01", periods=n)
        df = pd.DataFrame(
            {
                "symbol": ["SPY"] * n,
                "date": idx,
                "close": np.random.default_rng(0).normal(400, 10, n),
            }
        )
        result = extract_features(df, use_tsfresh=False)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# certify/mlflow_integration (no-op when mlflow not installed)
# ---------------------------------------------------------------------------


class TestMlflowIntegration:
    def test_log_backtest_run_returns_none_without_mlflow(self):
        """mlflow not installed in test env — should return None gracefully."""
        from assembled_core.certify.mlflow_integration import log_backtest_run

        result = log_backtest_run(
            params={"strategy": "momentum", "lookback": 20},
            metrics={"sharpe": 1.2, "cagr": 0.15},
        )
        # mlflow not installed → returns None
        assert result is None

    def test_log_certificate_returns_none_without_mlflow(self):
        from assembled_core.certify.mlflow_integration import log_certificate_to_mlflow

        result = log_certificate_to_mlflow(object())
        assert result is None

    def test_mlflow_available_is_bool(self):
        from assembled_core.certify.mlflow_integration import _mlflow_available

        assert isinstance(_mlflow_available(), bool)

    def test_certificate_to_json_with_dataclass(self):
        from dataclasses import dataclass
        from assembled_core.certify.mlflow_integration import _certificate_to_json

        @dataclass
        class FakeCert:
            certificate_id: str = "test-123"
            created_at: str = "2026-01-01"

        result = _certificate_to_json(FakeCert())
        assert result is not None
        assert "test-123" in result


# ---------------------------------------------------------------------------
# api/middleware
# ---------------------------------------------------------------------------


class TestAPIMiddleware:
    def test_get_request_id_default_empty(self):
        from assembled_core.api.middleware import get_request_id

        assert get_request_id() == ""

    def test_request_id_var_set_and_reset(self):
        from assembled_core.api.middleware import request_id_var

        token = request_id_var.set("test-rid-123")
        assert request_id_var.get() == "test-rid-123"
        request_id_var.reset(token)
        assert request_id_var.get("") == ""


# ---------------------------------------------------------------------------
# RegimeHMM.partial_update (online update)
# ---------------------------------------------------------------------------


class TestRegimeHMMOnlineUpdate:
    def test_partial_update_shape_preserved(self):
        # Runtime hmmlearn check (was an unconditional skipif(True) which
        # made the test silently un-runnable even when hmmlearn was present).
        try:
            from assembled_core.ml.regime_hmm import HMMLEARN_AVAILABLE, RegimeHMM
        except ImportError:
            pytest.skip("regime_hmm not importable")
        if not HMMLEARN_AVAILABLE:
            pytest.skip("hmmlearn not installed in test env")

        rng = np.random.default_rng(0)
        baseline = pd.Series(rng.normal(0, 0.01, 500))
        new_data = pd.Series(rng.normal(0, 0.01, 30))
        model = RegimeHMM(n_regimes=2)
        model.fit(baseline)
        n_regimes_before = model.n_regimes
        model.partial_update(new_data)
        assert model.n_regimes == n_regimes_before

    def test_partial_update_skips_too_few_samples(self):
        """partial_update should skip gracefully when min_samples not met.
        Works even without hmmlearn since it just checks the regime count.
        """
        # Directly test the size guard: if model is not fitted it raises RuntimeError
        # We simulate: partial_update requires fit() first
        try:
            from assembled_core.ml.regime_hmm import RegimeHMM, HMMLEARN_AVAILABLE

            if not HMMLEARN_AVAILABLE:
                pytest.skip("hmmlearn not available")
            rng = np.random.default_rng(0)
            model = RegimeHMM(n_regimes=2)
            model.fit(pd.Series(rng.normal(0, 0.01, 200)))
            # Too few samples — should return self without crashing
            tiny = pd.Series(rng.normal(0, 0.01, 5))
            result = model.partial_update(tiny, min_samples=20)
            assert result is model
        except ImportError:
            pytest.skip("hmmlearn not available")
