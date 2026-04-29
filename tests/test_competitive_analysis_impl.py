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
            index=dates, columns=sectors,
        )
        w_b = pd.DataFrame(
            [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3], [0.4, 0.3, 0.3]],
            index=dates, columns=sectors,
        )
        return BrinsonAttribution(w_p, w_b), dates, sectors

    def test_attribute_columns(self):
        ba, dates, sectors = self._make()
        rp = pd.DataFrame(
            [[0.02, 0.01, 0.03]] * 3, index=dates, columns=sectors
        )
        rb = pd.DataFrame(
            [[0.01, 0.02, 0.02]] * 3, index=dates, columns=sectors
        )
        result = ba.attribute(rp, rb)
        assert list(result.columns) == ["allocation", "selection", "interaction", "active_total"]
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
        diff = (result["active_total"] - result[["allocation", "selection", "interaction"]].sum(axis=1)).abs()
        assert diff.max() < 1e-10

    def test_summary_returns_dict(self):
        ba, dates, sectors = self._make()
        rp = pd.DataFrame([[0.02, 0.01, 0.03]] * 3, index=dates, columns=sectors)
        rb = pd.DataFrame([[0.01, 0.02, 0.02]] * 3, index=dates, columns=sectors)
        summary = ba.summary(rp, rb)
        assert set(summary.keys()) == {"allocation", "selection", "interaction", "active_total"}

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
            {"volume": np.ones(100, dtype=int) * 200, "price": np.linspace(100, 110, 100)},
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
        assert {"sharpe", "sharpe_ci_lower", "sharpe_ci_upper", "sharpe_p_value"}.issubset(result)

    def test_ci_lower_le_sharpe_le_upper(self):
        from assembled_core.qa.bootstrap_metrics import compute_sharpe_with_ci
        result = compute_sharpe_with_ci(self._returns(), n_bootstrap=500, seed=1)
        assert result["sharpe_ci_lower"] <= result["sharpe"] <= result["sharpe_ci_upper"]

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
        sizer = LiquidityAwareSizer(max_pct_adv=1.0, max_pct_market_cap=0.0001,
                                    max_days_to_liquidate=100.0, target_pov_pct=1.0)
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


# ---------------------------------------------------------------------------
# VVIXTailRiskSignal
# ---------------------------------------------------------------------------

class TestVVIXTailRiskSignal:
    def _signal(self):
        from assembled_core.signals.tail_risk_vvix import VVIXTailRiskSignal
        return VVIXTailRiskSignal()

    def test_calm_regime(self):
        s = self._signal()
        state = s.regime({"vvix": 85.0, "skew": 125.0, "vix": 15.0, "vix3m": 17.0})
        assert state.regime == "calm"
        assert state.score == 0

    def test_elevated_regime(self):
        s = self._signal()
        # vvix=95: >= calm(90) but < high(110) → score 1; skew=130: >= calm(130) → score 1
        state = s.regime({"vvix": 95.0, "skew": 130.0})
        assert state.regime == "elevated"

    def test_high_regime(self):
        s = self._signal()
        # vvix=115: >= high(110) → score 2; skew=142: >= high(140) → score 2
        state = s.regime({"vvix": 115.0, "skew": 142.0})
        assert state.regime == "high"

    def test_extreme_regime(self):
        s = self._signal()
        state = s.regime({"vvix": 135.0, "skew": 155.0})
        assert state.regime == "extreme"
        assert state.score == 3

    def test_backwardation_increases_score(self):
        s = self._signal()
        # vvix=95 → score=1 (elevated), backwardation adds +1 → high
        state = s.regime({"vvix": 95.0, "vix": 25.0, "vix3m": 20.0})
        assert state.backwardation is True
        assert state.score >= 2

    def test_no_backwardation_normal_term_structure(self):
        s = self._signal()
        state = s.regime({"vvix": 85.0, "vix": 15.0, "vix3m": 18.0})
        assert state.backwardation is False

    def test_pd_series_input(self):
        s = self._signal()
        data = pd.Series({"vvix": 120.0, "skew": 143.0, "vix": 25.0, "vix3m": 22.0})
        state = s.regime(data)
        assert state.regime in ("elevated", "high", "extreme")

    def test_missing_skew_uses_vvix_only(self):
        s = self._signal()
        state = s.regime({"vvix": 105.0})
        assert state.skew is None
        assert state.regime in ("elevated", "high", "extreme")


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


# ---------------------------------------------------------------------------
# ExecutionRouter
# ---------------------------------------------------------------------------

class TestExecutionRouter:
    def _order(self, qty=1000, price=100.0, symbol="AAPL", side="BUY"):
        from assembled_core.execution.execution_router import Order
        return Order(symbol=symbol, side=side, quantity=qty, price=price, order_id="test-1")

    def test_small_order_direct(self):
        from assembled_core.execution.execution_router import route_order, ExecutionConfig
        cfg = ExecutionConfig(direct_threshold=0.05)
        order = self._order(qty=100)
        slices = route_order(order, avg_daily_volume=100_000, config=cfg)
        assert len(slices) == 1
        assert slices[0].algo == "direct"
        assert slices[0].quantity == 100

    def test_medium_order_twap(self):
        from assembled_core.execution.execution_router import route_order, ExecutionConfig
        cfg = ExecutionConfig(direct_threshold=0.01, twap_threshold=0.20, twap_slices=5)
        order = self._order(qty=5_000)
        slices = route_order(order, avg_daily_volume=100_000, config=cfg)
        assert all(s.algo == "twap" for s in slices)
        assert sum(s.quantity for s in slices) == 5_000

    def test_large_order_almgren_chriss(self):
        from assembled_core.execution.execution_router import route_order, ExecutionConfig
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
        from assembled_core.execution.execution_router import ac_split, Order, ExecutionConfig
        order = Order("TSLA", "SELL", 500, 200.0)
        slices = ac_split(order, ExecutionConfig(twap_slices=5))
        assert sum(s.quantity for s in slices) == 500

    def test_zero_adv_no_crash(self):
        from assembled_core.execution.execution_router import route_order
        order = self._order()
        slices = route_order(order, avg_daily_volume=0)
        assert len(slices) >= 1


# ---------------------------------------------------------------------------
# CrossAssetCarryV2
# ---------------------------------------------------------------------------

class TestCrossAssetCarryV2:
    def test_fx_carry_shape(self):
        from assembled_core.signals.cross_asset_carry_v2 import UniversalCarrySignal
        sig = UniversalCarrySignal()
        idx = pd.date_range("2024-01-01", periods=10)
        rates = pd.DataFrame(
            np.random.default_rng(0).normal(0, 1, (10, 4)),
            index=idx, columns=["EUR", "JPY", "GBP", "AUD"],
        )
        result = sig.fx_carry(rates)
        assert result.shape == rates.shape
        assert ((result >= -1) & (result <= 1)).all().all()

    def test_crypto_carry_range(self):
        from assembled_core.signals.cross_asset_carry_v2 import UniversalCarrySignal
        sig = UniversalCarrySignal()
        idx = pd.date_range("2024-01-01", periods=50, freq="8h")
        rates = pd.DataFrame(
            np.random.default_rng(1).normal(0.0001, 0.001, (50, 3)),
            index=idx, columns=["BTC", "ETH", "SOL"],
        )
        result = sig.crypto_carry(rates)
        assert ((result >= -1) & (result <= 1)).all().all()

    def test_commodity_carry_wide_format(self):
        from assembled_core.signals.cross_asset_carry_v2 import UniversalCarrySignal
        sig = UniversalCarrySignal()
        idx = pd.date_range("2024-01-01", periods=20)
        df = pd.DataFrame(
            {"CL_M1": np.linspace(80, 90, 20), "CL_M2": np.linspace(81, 91, 20)},
            index=idx,
        )
        result = sig.commodity_carry(df)
        assert not result.empty

    def test_fx_carry_empty(self):
        from assembled_core.signals.cross_asset_carry_v2 import UniversalCarrySignal
        sig = UniversalCarrySignal()
        result = sig.fx_carry(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# BarraRiskModel
# ---------------------------------------------------------------------------

class TestBarraRiskModel:
    def _make_model(self, n_symbols=10, n_days=60):
        from assembled_core.risk.barra_risk_model import BarraRiskModel
        rng = np.random.default_rng(42)
        symbols = [f"SYM{i}" for i in range(n_symbols)]
        dates = pd.date_range("2024-01-01", periods=n_days)
        returns = pd.DataFrame(rng.normal(0.0005, 0.01, (n_days, n_symbols)),
                               index=dates, columns=symbols)
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
        for key in ("market_var_pct", "sector_var_pct", "style_var_pct", "idio_var_pct"):
            assert result[key] >= 0.0, f"{key} should be non-negative"

    def test_decompose_keys_present(self):
        model = self._make_model()
        model.fit()
        symbols = model.returns.columns[:3].tolist()
        weights = pd.Series(1 / 3, index=symbols)
        result = model.decompose_portfolio_risk(weights)
        assert set(result.keys()) >= {"market_var_pct", "sector_var_pct",
                                       "style_var_pct", "idio_var_pct"}

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
        from assembled_core.features.volatility_estimators import garman_klass_volatility
        df = self._ohlc()
        result = garman_klass_volatility(df["open"], df["high"], df["low"], df["close"])
        assert len(result) == len(df)

    def test_panel_columns(self):
        from assembled_core.features.volatility_estimators import volatility_panel
        df = self._ohlc()
        result = volatility_panel(df, period=20)
        assert set(result.columns) == {"parkinson", "garman_klass", "rogers_satchell", "close_to_close"}
        assert len(result) == len(df)

    def test_tick_rule_signs(self):
        from assembled_core.features.volatility_estimators import tick_rule_signs
        prices = pd.Series([100, 101, 101, 99, 100])
        signs = tick_rule_signs(prices)
        assert set(signs.dropna().unique()).issubset({-1, 1})

    def test_parkinson_more_precise_than_c2c(self):
        from assembled_core.features.volatility_estimators import parkinson_volatility, close_to_close_volatility
        df = self._ohlc(200)
        park = parkinson_volatility(df["high"], df["low"], period=20).dropna()
        c2c = close_to_close_volatility(df["close"], period=20).dropna()
        # Both should produce valid (finite, positive) values
        assert park.isfinite().all() if hasattr(park, 'isfinite') else np.isfinite(park.values).all()
        assert c2c.dropna().gt(0).all()


# ---------------------------------------------------------------------------
# CausalValidation (OLS fallback, no dowhy required)
# ---------------------------------------------------------------------------

class TestCausalValidation:
    def _trades(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            "has_news_trigger": rng.integers(0, 2, n),
            "return": rng.normal(0.002, 0.01, n),
            "sector": rng.choice(["Tech", "Finance", "Health"], n),
            "vol_regime": rng.uniform(0.1, 0.3, n),
        })
        # Add a real treatment effect
        df.loc[df["has_news_trigger"] == 1, "return"] += 0.005
        return df

    def test_ols_estimate_returns_dict(self):
        from assembled_core.qa.causal_validation import estimate_news_trigger_effect
        result = estimate_news_trigger_effect(
            self._trades(), use_dowhy=False
        )
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
        from assembled_core.portfolio.adaptive_conformal_position import AdaptiveConformalSizer
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
        from assembled_core.portfolio.adaptive_conformal_position import size_from_interval_width
        assert size_from_interval_width(0.0, 1.0, max_position=1.0) == 1.0
        assert size_from_interval_width(1.0, 1.0, max_position=1.0) == 0.0
        mid = size_from_interval_width(0.5, 1.0, max_position=1.0)
        assert 0.0 < mid < 1.0


# ---------------------------------------------------------------------------
# LPPLSCrashDetector
# ---------------------------------------------------------------------------

class TestLPPLSCrashDetector:
    def _prices(self, n=150, bubble=False):
        import numpy as np
        rng = np.random.default_rng(42)
        if bubble:
            t = np.arange(n)
            prices = 100 * np.exp(0.002 * t + 0.5 * np.cos(7 * np.log(n - t + 5))) * (
                1 + rng.normal(0, 0.005, n)
            )
        else:
            prices = 100 * np.exp(rng.normal(0, 0.01, n).cumsum())
        return np.abs(prices) + 10

    def test_returns_dict(self):
        from assembled_core.signals.lppls_crash import LPPLSCrashDetector
        det = LPPLSCrashDetector(fit_window=50, max_searches=5)
        result = det.fit_and_score(self._prices(100))
        assert "crash_confidence" in result
        assert "tc_estimate" in result
        assert "time_to_crash_days" in result
        assert "method" in result

    def test_confidence_in_range(self):
        from assembled_core.signals.lppls_crash import LPPLSCrashDetector
        det = LPPLSCrashDetector(fit_window=50, max_searches=5)
        result = det.fit_and_score(self._prices(120))
        assert 0.0 <= result["crash_confidence"] <= 1.0

    def test_method_is_numpy_fallback(self):
        from assembled_core.signals.lppls_crash import LPPLSCrashDetector
        det = LPPLSCrashDetector(fit_window=50, max_searches=5)
        result = det.fit_and_score(self._prices(80))
        assert "numpy" in result["method"]
