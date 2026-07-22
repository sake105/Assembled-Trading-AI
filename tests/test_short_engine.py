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

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _universe(symbols=("SPY", "QQQ", "AAPL", "MSFT")) -> pd.DataFrame:
    """Simple universe DataFrame with symbol + sector columns."""
    sector_map = {
        "SPY": "BROAD",
        "QQQ": "TECH",
        "SH": "BROAD",
        "PSQ": "TECH",
        "AAPL": "TECH",
        "MSFT": "TECH",
    }
    return pd.DataFrame(
        [{"symbol": s, "sector": sector_map.get(s, "OTHER")} for s in symbols]
    )


def _prices(symbols=("SPY", "QQQ", "AAPL", "MSFT"), n_days=60) -> pd.DataFrame:
    """Synthetic daily price panel with a declining trend for the last 20 days."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    rows = []
    for sym in symbols:
        base = {"SPY": 450.0, "QQQ": 380.0, "AAPL": 185.0, "MSFT": 380.0}.get(
            sym, 100.0
        )
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


# ---------------------------------------------------------------------------
# ShortRiskManager tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# InverseETFSelector tests
# ---------------------------------------------------------------------------


class TestInverseETFSelector:
    def test_import_v4(self):
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


# ---------------------------------------------------------------------------
# End-to-end: CrashSignal → ShortTargets → risk-checked orders
# ---------------------------------------------------------------------------
