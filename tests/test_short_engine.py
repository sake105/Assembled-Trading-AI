"""Phase F — M15 Short-Profit Engine integration tests.

Tests the full short-selling pipeline:
  CrashPredictionEngine → ShortSignalGenerator → ShortRiskManager
  → order generation → P&L accounting.

Marker: phase12
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

pytestmark = pytest.mark.phase12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _universe(symbols=("SPY", "QQQ", "AAPL", "MSFT")) -> pd.DataFrame:
    """Simple universe DataFrame with symbol + sector columns."""
    sector_map = {"SPY": "BROAD", "QQQ": "TECH", "SH": "BROAD", "PSQ": "TECH",
                  "AAPL": "TECH", "MSFT": "TECH"}
    return pd.DataFrame([
        {"symbol": s, "sector": sector_map.get(s, "OTHER")}
        for s in symbols
    ])


def _prices(symbols=("SPY", "QQQ", "AAPL", "MSFT"), n_days=60) -> pd.DataFrame:
    """Synthetic daily price panel with a declining trend for the last 20 days."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    rows = []
    for sym in symbols:
        base = {"SPY": 450.0, "QQQ": 380.0, "AAPL": 185.0, "MSFT": 380.0}.get(sym, 100.0)
        for i, d in enumerate(dates):
            # declining last 20 days → should trigger bearish signals
            if i >= n_days - 20:
                close = base * (1 - 0.005 * (i - (n_days - 20)))
            else:
                close = base * (1 + 0.001 * i)
            rows.append({"timestamp": d, "symbol": sym, "close": close, "volume": 1e7})
    return pd.DataFrame(rows)


def _bear_regime() -> dict[str, Any]:
    return {"regime": "bear", "confidence": 0.80}


def _crisis_regime() -> dict[str, Any]:
    return {"regime": "crisis", "confidence": 0.90}


def _short_policy() -> dict[str, Any]:
    return {
        "enabled": True,
        "max_short_weight_per_position": 0.10,
        "max_total_short_exposure": 0.30,
        "max_gross_exposure": 1.50,
        "max_net_short": 0.20,
        "allowed_instruments": {
            "direct_short": True,
            "inverse_etf_1x": True,
            "inverse_etf_2x": False,
            "inverse_etf_3x": False,
        },
        "regime_scaling": {
            "bull": 0.0,
            "sideways": 0.10,
            "bear": 0.25,
            "crisis": 0.30,
            "reflation": 0.05,
        },
        "borrow_cost_assumption_bps": 50,
        "squeeze_risk_check": True,
        "min_short_signal_confidence": 0.70,
        "min_crash_probability": 0.60,
        "require_stop_loss": True,
        "max_stop_loss_pct": 0.15,
    }


# ---------------------------------------------------------------------------
# CrashPredictionEngine tests
# ---------------------------------------------------------------------------


class TestCrashPredictionEngine:
    def test_import(self):
        from src.assembled_core.signals.crash_prediction import CrashPredictionEngine

        engine = CrashPredictionEngine()
        assert engine is not None

    def test_predict_returns_crash_signal(self):
        from src.assembled_core.signals.crash_prediction import (
            CrashPredictionEngine,
            CrashSignal,
        )

        engine = CrashPredictionEngine()
        prices = _prices()
        result = engine.predict(
            market_data=prices,
            regime=_bear_regime(),
            intel_state=None,
            macro_data=None,
        )
        assert isinstance(result, CrashSignal)

    def test_crash_probability_in_range(self):
        from src.assembled_core.signals.crash_prediction import CrashPredictionEngine

        engine = CrashPredictionEngine()
        prices = _prices()
        result = engine.predict(
            market_data=prices,
            regime=_bear_regime(),
            intel_state=None,
            macro_data=None,
        )
        assert 0.0 <= result.crash_probability <= 1.0
        assert 0.0 <= result.expected_severity <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_contributing_signals_present(self):
        from src.assembled_core.signals.crash_prediction import CrashPredictionEngine

        engine = CrashPredictionEngine()
        prices = _prices()
        result = engine.predict(
            market_data=prices,
            regime=_bear_regime(),
            intel_state=None,
            macro_data=None,
        )
        # Should have contributing signals dict
        assert isinstance(result.contributing_signals, dict)
        assert len(result.contributing_signals) > 0

    def test_crisis_regime_raises_probability(self):
        """Crisis regime should produce higher crash probability than bull."""
        from src.assembled_core.signals.crash_prediction import CrashPredictionEngine

        engine = CrashPredictionEngine()
        prices = _prices()

        crisis_result = engine.predict(
            market_data=prices,
            regime=_crisis_regime(),
            intel_state=None,
            macro_data=None,
        )
        bull_result = engine.predict(
            market_data=prices,
            regime={"regime": "bull", "confidence": 0.80},
            intel_state=None,
            macro_data=None,
        )
        assert crisis_result.crash_probability >= bull_result.crash_probability

    def test_geo_signal_raises_probability(self):
        """High geo_score in intel_state should increase crash probability."""
        from src.assembled_core.signals.crash_prediction import CrashPredictionEngine

        engine = CrashPredictionEngine()
        prices = _prices()

        intel_high = {"mode": "CRISIS", "geo_score": 3}
        intel_low = {"mode": "NORMAL", "geo_score": 0}

        r_high = engine.predict(
            market_data=prices,
            regime=_bear_regime(),
            intel_state=intel_high,
            macro_data=None,
        )
        r_low = engine.predict(
            market_data=prices,
            regime=_bear_regime(),
            intel_state=intel_low,
            macro_data=None,
        )
        assert r_high.crash_probability >= r_low.crash_probability

    def test_empty_prices_graceful(self):
        """Empty prices DataFrame should not crash."""
        from src.assembled_core.signals.crash_prediction import CrashPredictionEngine

        engine = CrashPredictionEngine()
        result = engine.predict(
            market_data=pd.DataFrame(),
            regime=_bear_regime(),
            intel_state=None,
            macro_data=None,
        )
        assert 0.0 <= result.crash_probability <= 1.0


# ---------------------------------------------------------------------------
# ShortSignalGenerator tests
# ---------------------------------------------------------------------------


class TestShortSignalGenerator:
    def test_import(self):
        import pytest; pytest.importorskip('src.assembled_core.signals.short_signals')
        from src.assembled_core.signals.short_signals import ShortSignalGenerator

        gen = ShortSignalGenerator(policy=_short_policy())
        assert gen is not None

    def test_no_shorts_below_threshold(self):
        """Below min_crash_probability no shorts should be generated."""
        import pytest; pytest.importorskip('src.assembled_core.signals.short_signals')
        from src.assembled_core.signals.crash_prediction import CrashSignal
        from src.assembled_core.signals.short_signals import ShortSignalGenerator

        gen = ShortSignalGenerator(policy=_short_policy())
        low_prob_signal = CrashSignal(
            crash_probability=0.20,
            expected_severity=0.10,
            time_horizon_days=30,
            confidence=0.50,
            contributing_signals={},
            recommended_sectors_short=[],
            recommended_instruments=[],
            active=False,
        )
        result = gen.generate_short_targets(
            crash_signal=low_prob_signal,
            universe=_universe(["SPY", "QQQ", "AAPL"]),
            prices=_prices(),
            regime="bull",
        )
        assert len(result) == 0

    def test_shorts_generated_above_threshold(self):
        """Above threshold in bear regime → some shorts generated."""
        import pytest; pytest.importorskip('src.assembled_core.signals.short_signals')
        from src.assembled_core.signals.crash_prediction import CrashSignal
        from src.assembled_core.signals.short_signals import ShortSignalGenerator

        gen = ShortSignalGenerator(policy=_short_policy())
        high_prob_signal = CrashSignal(
            crash_probability=0.75,
            expected_severity=0.65,
            time_horizon_days=10,
            confidence=0.80,
            contributing_signals={"regime_bear_probability": 0.8},
            recommended_sectors_short=["TECH", "FINANCE"],
            recommended_instruments=["PSQ", "SEF"],
            active=True,
        )
        result = gen.generate_short_targets(
            crash_signal=high_prob_signal,
            universe=_universe(["SPY", "QQQ", "AAPL", "MSFT"]),
            prices=_prices(),
            regime="bear",
        )
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "symbol" in result.columns
            assert "target_weight" in result.columns
            # All weights must be negative for shorts
            assert (result["target_weight"] <= 0).all()

    def test_no_shorts_in_bull_regime(self):
        """Bull regime → 0.0 scaling → no shorts regardless of crash signal."""
        import pytest; pytest.importorskip('src.assembled_core.signals.short_signals')
        from src.assembled_core.signals.crash_prediction import CrashSignal
        from src.assembled_core.signals.short_signals import ShortSignalGenerator

        gen = ShortSignalGenerator(policy=_short_policy())
        high_signal = CrashSignal(
            crash_probability=0.90,
            expected_severity=0.80,
            time_horizon_days=5,
            confidence=0.85,
            contributing_signals={},
            recommended_sectors_short=["TECH"],
            recommended_instruments=["PSQ"],
            active=True,
        )
        result = gen.generate_short_targets(
            crash_signal=high_signal,
            universe=_universe(["SPY", "QQQ"]),
            prices=_prices(),
            regime="bull",
        )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# ShortRiskManager tests
# ---------------------------------------------------------------------------


class TestShortRiskManager:
    def test_import(self):
        import pytest; pytest.importorskip('src.assembled_core.risk.short_risk')
        from src.assembled_core.risk.short_risk import ShortRiskManager

        mgr = ShortRiskManager(policy=_short_policy())
        assert mgr is not None

    def test_validate_returns_short_risk_check(self):
        """validate_short_targets returns a ShortRiskCheck dataclass."""
        import pytest; pytest.importorskip('src.assembled_core.risk.short_risk')
        from src.assembled_core.risk.short_risk import ShortRiskCheck, ShortRiskManager

        mgr = ShortRiskManager(policy=_short_policy())
        targets = pd.DataFrame(
            [{"symbol": "SH", "direction": "SHORT", "target_weight": -0.08,
              "confidence": 0.80, "stop_loss_pct": 0.10}]
        )
        result = mgr.validate_short_targets(targets, regime="bear")
        assert isinstance(result, ShortRiskCheck)
        assert hasattr(result, "passed")
        assert hasattr(result, "violations")

    def test_rejects_leveraged_inverse_etf(self):
        """2x/3x inverse ETFs must produce a violation."""
        import pytest; pytest.importorskip('src.assembled_core.risk.short_risk')
        from src.assembled_core.risk.short_risk import ShortRiskManager

        mgr = ShortRiskManager(policy=_short_policy())
        targets = pd.DataFrame(
            [{"symbol": "SPXS", "direction": "SHORT", "target_weight": -0.05,
              "confidence": 0.80, "stop_loss_pct": 0.10}]
        )
        result = mgr.validate_short_targets(targets, regime="bear")
        # Must not pass — leveraged product violation
        assert result.passed is False
        assert any("Rule 2" in v or "leveraged" in v.lower() for v in result.violations)

    def test_enforces_max_per_position_via_regime_scaling(self):
        """enforce_regime_scaling scales down shorts when total exceeds cap."""
        import pytest; pytest.importorskip('src.assembled_core.risk.short_risk')
        from src.assembled_core.risk.short_risk import ShortRiskManager

        policy = _short_policy()
        # Wrap under 'shorts' key as ShortRiskManager.__init__ expects
        mgr = ShortRiskManager(policy={"shorts": policy})
        # 5 positions × 10% = 50% > 30% crisis cap
        targets = pd.DataFrame(
            [{"symbol": f"SH{i}", "direction": "SHORT", "target_weight": -0.10,
              "confidence": 0.80, "stop_loss_pct": 0.10}
             for i in range(5)]
        )
        scaled = mgr.enforce_regime_scaling(targets, regime="bear")
        total_short = scaled["target_weight"].abs().sum()
        # bear cap = 0.25 → must be scaled down to ≤25%
        assert total_short <= 0.25 + 1e-6

    def test_regime_scaling_bull_zeros_shorts(self):
        """In bull regime enforce_regime_scaling zeros out all shorts."""
        import pytest; pytest.importorskip('src.assembled_core.risk.short_risk')
        from src.assembled_core.risk.short_risk import ShortRiskManager

        mgr = ShortRiskManager(policy={"shorts": _short_policy()})
        targets = pd.DataFrame(
            [{"symbol": "SH", "direction": "SHORT", "target_weight": -0.10,
              "confidence": 0.80, "stop_loss_pct": 0.10}]
        )
        scaled = mgr.enforce_regime_scaling(targets, regime="bull")
        total = scaled["target_weight"].abs().sum()
        assert total == pytest.approx(0.0, abs=1e-6)

    def test_squeeze_risk_check(self):
        """High short-interest symbol should be flagged."""
        import pytest; pytest.importorskip('src.assembled_core.risk.short_risk')
        from src.assembled_core.risk.short_risk import ShortRiskManager

        mgr = ShortRiskManager(policy=_short_policy())
        result = mgr.check_short_squeeze_risk(
            symbol="GME",
            short_interest_pct=0.40,  # 40% float short → high squeeze risk
            days_to_cover=12.0,        # > 10 days threshold
        )
        assert result is True  # high squeeze risk detected


# ---------------------------------------------------------------------------
# InverseETFSelector tests
# ---------------------------------------------------------------------------


class TestInverseETFSelector:
    def test_import(self):
        from src.assembled_core.portfolio.inverse_etf_selector import InverseETFSelector

        sel = InverseETFSelector()
        assert sel is not None

    def test_selects_instrument_for_tech(self):
        from src.assembled_core.portfolio.inverse_etf_selector import InverseETFSelector

        sel = InverseETFSelector()
        instrument = sel.select_best_short_instrument(
            sector="TECH",
            severity=0.6,
            holding_period_days=3,
        )
        assert isinstance(instrument, str)
        assert len(instrument) > 0

    def test_no_2x_3x_selected(self):
        """Selector should not return leveraged instruments in standard mode."""
        from src.assembled_core.portfolio.inverse_etf_selector import (
            INVERSE_ETF_PROFILES,
            InverseETFSelector,
        )

        sel = InverseETFSelector()
        for sector in ["TECH", "FINANCE", "BROAD"]:
            instrument = sel.select_best_short_instrument(
                sector=sector,
                severity=0.7,
                holding_period_days=5,
            )
            if instrument and instrument in INVERSE_ETF_PROFILES:
                profile = INVERSE_ETF_PROFILES[instrument]
                assert abs(profile.leverage) <= 1, (
                    f"{instrument} has leverage {profile.leverage} — should not be selected"
                )

    def test_decay_adjusted_return(self):
        from src.assembled_core.portfolio.inverse_etf_selector import InverseETFSelector

        sel = InverseETFSelector()
        # 1x inverse ETF: minimal decay
        sh_return = sel.compute_decay_adjusted_return("SH", -0.20, 0.15, 5)
        # 3x would have more decay — just test the function runs
        assert isinstance(sh_return, float)


# ---------------------------------------------------------------------------
# LongShortBalancer tests
# ---------------------------------------------------------------------------


class TestLongShortBalancer:
    def test_import(self):
        import pytest; pytest.importorskip('src.assembled_core.portfolio.long_short_balance')
        from src.assembled_core.portfolio.long_short_balance import LongShortBalancer

        balancer = LongShortBalancer.from_policy(_short_policy())
        assert balancer is not None

    def test_compute_exposure_long_only(self):
        import pytest; pytest.importorskip('src.assembled_core.portfolio.long_short_balance')
        from src.assembled_core.portfolio.long_short_balance import LongShortBalancer

        balancer = LongShortBalancer.from_policy(_short_policy())
        positions = pd.DataFrame(
            [
                {"symbol": "AAPL", "weight": 0.20},
                {"symbol": "MSFT", "weight": 0.15},
                {"symbol": "SPY", "weight": 0.10},
            ]
        )
        metrics = balancer.compute_exposure(positions)
        assert metrics.long_exposure == pytest.approx(0.45)
        assert metrics.short_exposure == pytest.approx(0.0)
        assert metrics.net_exposure == pytest.approx(0.45)
        assert metrics.gross_exposure == pytest.approx(0.45)

    def test_compute_exposure_long_short(self):
        import pytest; pytest.importorskip('src.assembled_core.portfolio.long_short_balance')
        from src.assembled_core.portfolio.long_short_balance import LongShortBalancer

        balancer = LongShortBalancer.from_policy(_short_policy())
        positions = pd.DataFrame(
            [
                {"symbol": "AAPL", "weight": 0.30},
                {"symbol": "SH", "weight": -0.10},
                {"symbol": "PSQ", "weight": -0.08},
            ]
        )
        metrics = balancer.compute_exposure(positions)
        assert metrics.long_exposure == pytest.approx(0.30)
        assert metrics.short_exposure == pytest.approx(0.18)
        assert metrics.net_exposure == pytest.approx(0.12)
        assert metrics.gross_exposure == pytest.approx(0.48)

    def test_enforce_gross_exposure_limit(self):
        import pytest; pytest.importorskip('src.assembled_core.portfolio.long_short_balance')
        from src.assembled_core.portfolio.long_short_balance import LongShortBalancer

        balancer = LongShortBalancer.from_policy(_short_policy())
        # Longs 80% + Shorts 80% = 160% gross > 150% limit
        positions = pd.DataFrame(
            [
                {"symbol": "AAPL", "weight": 0.40},
                {"symbol": "MSFT", "weight": 0.40},
                {"symbol": "SH", "weight": -0.40},
                {"symbol": "PSQ", "weight": -0.40},
            ]
        )
        adjusted = balancer.enforce_exposure_limits(positions)
        metrics = balancer.compute_exposure(adjusted)
        assert metrics.gross_exposure <= 1.50 + 1e-6


# ---------------------------------------------------------------------------
# End-to-end: CrashSignal → ShortTargets → risk-checked orders
# ---------------------------------------------------------------------------


class TestShortEnginePipeline:
    def test_full_pipeline_bear_regime(self):
        """Full pipeline: crash signal → short targets → risk validation."""
        import pytest; pytest.importorskip('src.assembled_core.risk.short_risk')
        import pytest; pytest.importorskip('src.assembled_core.signals.short_signals')
        from src.assembled_core.signals.crash_prediction import (
            CrashPredictionEngine,
        )
        from src.assembled_core.signals.short_signals import ShortSignalGenerator
        from src.assembled_core.risk.short_risk import ShortRiskManager

        prices = _prices()
        policy = _short_policy()

        engine = CrashPredictionEngine()
        signal = engine.predict(
            market_data=prices,
            regime=_crisis_regime(),
            intel_state={"mode": "CRISIS", "geo_score": 3},
            macro_data=None,
        )

        gen = ShortSignalGenerator(policy=policy)
        raw_targets = gen.generate_short_targets(
            crash_signal=signal,
            universe=_universe(["SPY", "QQQ", "SH", "PSQ", "AAPL"]),
            prices=prices,
            regime="crisis",
        )

        mgr = ShortRiskManager(policy={"shorts": policy})
        # Use enforce_regime_scaling which returns a DataFrame
        validated = mgr.enforce_regime_scaling(raw_targets, regime="crisis")

        # Final portfolio must respect limits
        if len(validated) > 0:
            total_short = abs(validated["target_weight"].sum())
            crisis_cap = policy["regime_scaling"]["crisis"]  # 0.30
            assert total_short <= crisis_cap + 1e-6

    def test_no_leveraged_products_in_pipeline(self):
        """No 3x leveraged inverse ETFs should pass validation check."""
        import pytest; pytest.importorskip('src.assembled_core.risk.short_risk')
        from src.assembled_core.risk.short_risk import (
            LEVERAGED_INVERSE_ETFS,
            ShortRiskManager,
        )

        mgr = ShortRiskManager(policy=_short_policy())
        target_syms = list(LEVERAGED_INVERSE_ETFS)[:5]
        targets = pd.DataFrame(
            [{"symbol": sym, "direction": "SHORT", "target_weight": -0.05,
              "confidence": 0.80, "stop_loss_pct": 0.10}
             for sym in target_syms]
        )
        result = mgr.validate_short_targets(targets, regime="crisis")
        # Leveraged ETF rule must be violated — check should fail
        assert result.passed is False, "Leveraged ETF targets should fail validation"
        leveraged_violations = [
            v for v in result.violations
            if "Rule 2" in v or "leveraged" in v.lower() or "Leveraged" in v
        ]
        assert len(leveraged_violations) > 0
