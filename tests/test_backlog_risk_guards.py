"""Targeted unit tests for critical risk guard modules.

Backlog item 4 — BLOCKER for live paper pilot.

Covers:
  - DD damper: update_drawdown_damper / reset_dd_damper (multifactor_v2)
  - VIX cap: via compute_signals with vix column in DataFrame
  - Conviction engine / trigger basket: smoke imports (files not yet present)
"""

from __future__ import annotations

import datetime as dt
import importlib

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Module-level import — fail fast with clear message if missing
# ---------------------------------------------------------------------------
from src.assembled_core.strategies.multifactor_v2 import (
    _DD_DAMPER,
    reset_dd_damper,
    update_drawdown_damper,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_state() -> None:
    """Reset damper between tests — prevents cross-contamination."""
    reset_dd_damper()


# ---------------------------------------------------------------------------
# DD Damper tests
# ---------------------------------------------------------------------------


class TestDDDamper:
    def setup_method(self) -> None:
        _fresh_state()

    def test_dd_damper_activates_at_threshold(self) -> None:
        """Drop equity 12.5% from peak → damper_active becomes True."""
        update_drawdown_damper(1.0)  # establish peak
        activated = update_drawdown_damper(0.875)  # 12.5% drawdown (>= 12%)
        assert activated is True
        assert _DD_DAMPER["damper_active"] is True

    def test_dd_damper_not_activated_below_threshold(self) -> None:
        """Drop equity only 10% → damper must NOT activate."""
        update_drawdown_damper(1.0)
        activated = update_drawdown_damper(0.90)  # 10% drawdown (< 12%)
        assert activated is False
        assert _DD_DAMPER["damper_active"] is False

    def test_dd_damper_expires_after_days(self) -> None:
        """Activate damper, advance past expiry with recovered equity → not re-armed.

        When equity recovers above the re-arm threshold (MDD < 12%) the damper
        must expire and not re-activate after damper_until passes.
        """
        today = dt.date(2025, 1, 1)
        update_drawdown_damper(1.0, as_of=today)
        update_drawdown_damper(0.875, as_of=today)  # activate (MDD=12.5%)
        assert _DD_DAMPER["damper_active"] is True

        # Advance past expiry with equity fully recovered — MDD=0% → no re-arm
        future = today + dt.timedelta(days=31)
        update_drawdown_damper(1.0, as_of=future)  # equity restored to peak
        assert _DD_DAMPER["damper_active"] is False

    def test_dd_damper_still_active_before_expiry(self) -> None:
        """Damper must remain active if date is within the damper window."""
        today = dt.date(2025, 1, 1)
        update_drawdown_damper(1.0, as_of=today)
        update_drawdown_damper(0.875, as_of=today)
        assert _DD_DAMPER["damper_active"] is True

        still_within = today + dt.timedelta(days=29)
        update_drawdown_damper(0.875, as_of=still_within)
        assert _DD_DAMPER["damper_active"] is True

    def test_dd_damper_no_zero_division(self) -> None:
        """update_drawdown_damper(0.0) must not raise."""
        result = update_drawdown_damper(0.0)
        # peak starts at 1.0, equity=0 → MDD=100% → activates
        assert result is True
        assert _DD_DAMPER["damper_active"] is True

    def test_dd_damper_no_zero_division_peak_zero(self) -> None:
        """If somehow peak is 0.0, must not raise ZeroDivisionError."""
        _DD_DAMPER["peak_equity"] = 0.0
        _DD_DAMPER["current_equity"] = 0.0
        try:
            update_drawdown_damper(0.0)
        except ZeroDivisionError:
            pytest.fail("update_drawdown_damper raised ZeroDivisionError with peak=0")

    def test_reset_dd_damper_clears_state(self) -> None:
        """Activate damper, then reset → damper_active=False and peak reset."""
        update_drawdown_damper(1.0)
        update_drawdown_damper(0.875)
        assert _DD_DAMPER["damper_active"] is True

        reset_dd_damper()

        assert _DD_DAMPER["damper_active"] is False
        assert _DD_DAMPER["damper_until"] is None
        assert _DD_DAMPER["peak_equity"] == 1.0
        assert _DD_DAMPER["current_equity"] == 1.0

    def test_dd_damper_no_cross_contamination(self) -> None:
        """Two independent sequences separated by reset must not share state."""
        # Sequence A: activate
        update_drawdown_damper(1.0)
        update_drawdown_damper(0.875)
        assert _DD_DAMPER["damper_active"] is True

        # Reset between sequences
        reset_dd_damper()

        # Sequence B: only 10% drawdown — must NOT activate
        update_drawdown_damper(1.0)
        activated_b = update_drawdown_damper(0.90)
        assert activated_b is False
        assert _DD_DAMPER["damper_active"] is False

    def test_dd_damper_double_activation_not_double_counted(self) -> None:
        """After activating, calling update again must not set a new damper_until."""
        today = dt.date(2025, 3, 1)
        update_drawdown_damper(1.0, as_of=today)
        update_drawdown_damper(0.875, as_of=today)
        until_first = _DD_DAMPER["damper_until"]

        # Another call with even lower equity — already active, no re-arm
        tomorrow = today + dt.timedelta(days=1)
        activated_again = update_drawdown_damper(0.80, as_of=tomorrow)
        assert activated_again is False  # not re-armed
        assert _DD_DAMPER["damper_until"] == until_first  # expiry unchanged

    def test_dd_damper_peak_tracks_new_highs(self) -> None:
        """Peak equity must update when new equity exceeds current peak."""
        update_drawdown_damper(1.0)
        update_drawdown_damper(1.20)  # new high
        assert _DD_DAMPER["peak_equity"] == pytest.approx(1.20)

        # Now drop 12.5% from 1.20 → 1.05; MDD = (1.20-1.05)/1.20 = 12.5%
        activated = update_drawdown_damper(1.05)
        assert activated is True


# ---------------------------------------------------------------------------
# VIX cap tests (via compute_signals)
# ---------------------------------------------------------------------------


def _make_minimal_df(
    symbols: list[str],
    vix: float | None,
    n_bars: int = 80,
) -> pd.DataFrame:
    """Build a minimal prices_with_features DataFrame for compute_signals tests."""
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="B")
    rows = []
    for sym in symbols:
        prices = pd.Series(100.0, index=range(n_bars)) * (
            1 + 0.0002 * pd.Series(range(n_bars))
        )
        for i, d in enumerate(dates):
            row: dict = {
                "symbol": sym,
                "timestamp": d,
                "open": float(prices[i]),
                "high": float(prices[i]) * 1.005,
                "low": float(prices[i]) * 0.995,
                "close": float(prices[i]),
                "volume": 1_000_000,
            }
            if vix is not None:
                row["vix"] = vix
            rows.append(row)
    return pd.DataFrame(rows)


class TestVIXCap:
    """Tests for VIX-tiered exposure cap in compute_signals."""

    def setup_method(self) -> None:
        reset_dd_damper()  # isolate from DD damper effects

    def _get_exposure_mult(self, df: pd.DataFrame) -> float:
        """Run compute_signals and return median |score| as proxy for exposure mult."""
        from src.assembled_core.strategies.multifactor_v2 import compute_signals

        cfg: dict = {"regime_weights_path": "__nonexistent__"}
        signals = compute_signals(df, strategy_cfg=cfg)
        if signals.empty or "score" not in signals.columns:
            return 0.0
        return float(signals["score"].abs().median())

    def test_vix_cap_high_vix_reduces_composite(self) -> None:
        """With extreme VIX=45, composite must be smaller than with VIX=12."""
        symbols = ["AAPL"]
        df_low_vix = _make_minimal_df(symbols, vix=12.0)
        df_high_vix = _make_minimal_df(symbols, vix=45.0)

        score_low = self._get_exposure_mult(df_low_vix)
        score_high = self._get_exposure_mult(df_high_vix)

        # VIX=45 cap is 0.25, VIX=12 cap is 1.0 → high-VIX composite must be lower
        assert (
            score_high <= score_low + 1e-9
        ), f"Expected high-VIX ({score_high:.4f}) <= low-VIX ({score_low:.4f})"

    def test_vix_cap_extreme_tier_at_40_plus(self) -> None:
        """VIX=42 hits extreme tier (cap=0.25) — signals must complete without error."""
        from src.assembled_core.strategies.multifactor_v2 import compute_signals

        df = _make_minimal_df(["SPY"], vix=42.0)
        result = compute_signals(
            df, strategy_cfg={"regime_weights_path": "__nonexistent__"}
        )
        assert isinstance(result, pd.DataFrame)

    def test_vix_cap_not_triggered_low_vix(self) -> None:
        """VIX=12 → cap=1.0, no cap message; signals should match no-vix baseline closely."""
        from src.assembled_core.strategies.multifactor_v2 import compute_signals

        df_no_vix = _make_minimal_df(["AAPL"], vix=None)
        df_low_vix = _make_minimal_df(["AAPL"], vix=12.0)

        cfg: dict = {"regime_weights_path": "__nonexistent__"}
        sig_no = compute_signals(df_no_vix, strategy_cfg=cfg)
        sig_low = compute_signals(df_low_vix, strategy_cfg=cfg)

        if not sig_no.empty and not sig_low.empty:
            s_no = float(sig_no["score"].iloc[0])
            s_low = float(sig_low["score"].iloc[0])
            # Both should be numerically identical (no cap applied at VIX<18)
            assert (
                abs(s_no - s_low) < 0.05
            ), f"Low-VIX score diverged unexpectedly: no_vix={s_no:.4f}, low_vix={s_low:.4f}"

    def test_vix_cap_zero_capital_safe(self) -> None:
        """DataFrame with VIX=25 and constant-zero prices must not raise."""
        from src.assembled_core.strategies.multifactor_v2 import compute_signals

        n = 80
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "symbol": "TEST",
                "timestamp": dates,
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 0.0,
                "volume": 0.0,
                "vix": 25.0,
            }
        )
        try:
            result = compute_signals(
                df, strategy_cfg={"regime_weights_path": "__nonexistent__"}
            )
            assert isinstance(result, pd.DataFrame)
        except Exception as exc:
            pytest.fail(f"compute_signals raised with zero prices: {exc}")

    def test_vix_cap_missing_vix_column_safe(self) -> None:
        """If vix column is absent, compute_signals must still return a DataFrame."""
        from src.assembled_core.strategies.multifactor_v2 import compute_signals

        df = _make_minimal_df(["AAPL"], vix=None)
        result = compute_signals(
            df, strategy_cfg={"regime_weights_path": "__nonexistent__"}
        )
        assert isinstance(result, pd.DataFrame)

    def test_vix_tiered_thresholds_ordering(self) -> None:
        """Higher VIX tiers must produce lower-or-equal composite than lower tiers."""
        from src.assembled_core.strategies.multifactor_v2 import compute_signals

        cfg: dict = {"regime_weights_path": "__nonexistent__"}
        vix_levels = [12.0, 20.0, 25.0, 35.0, 45.0]
        scores = []
        for v in vix_levels:
            df = _make_minimal_df(["AAPL"], vix=v)
            sig = compute_signals(df, strategy_cfg=cfg)
            score = float(sig["score"].abs().median()) if not sig.empty else 0.0
            scores.append(score)

        # Each tier should be <= previous (monotone non-increasing)
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1] + 1e-9, (
                f"VIX={vix_levels[i]} score ({scores[i]:.4f}) > "
                f"VIX={vix_levels[i-1]} score ({scores[i-1]:.4f}) — cap not monotone"
            )


# ---------------------------------------------------------------------------
# Yield-curve inversion cap tests (via compute_signals)
# ---------------------------------------------------------------------------


def _make_yc_df(
    symbols: list[str],
    yc_slope: float | None,
    n_bars: int = 80,
    n_positive_prefix: int = 0,
) -> pd.DataFrame:
    """Build a minimal prices_with_features DataFrame including yield_curve_slope.

    Args:
        yc_slope: value to fill into yield_curve_slope (None → column absent).
        n_positive_prefix: number of leading bars set to +0.5 (to simulate
            a mostly-positive history before an inversion tail).
    """
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="B")
    rows = []
    for sym in symbols:
        for i, d in enumerate(dates):
            price = 100.0 + i * 0.02
            row: dict = {
                "symbol": sym,
                "timestamp": d,
                "open": price,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "volume": 1_000_000,
            }
            if yc_slope is not None:
                if n_positive_prefix > 0 and i < n_positive_prefix:
                    row["yield_curve_slope"] = 0.5  # positive prefix
                else:
                    row["yield_curve_slope"] = yc_slope
            rows.append(row)
    return pd.DataFrame(rows)


class TestYieldCurveInversionCap:
    """Tests for yield_curve_inversion_cap guard in compute_signals."""

    _CFG: dict = {"regime_weights_path": "__nonexistent__"}

    def setup_method(self) -> None:
        reset_dd_damper()

    def _run(self, df: pd.DataFrame, extra_cfg: dict | None = None) -> pd.DataFrame:
        from src.assembled_core.strategies.multifactor_v2 import compute_signals

        cfg = {**self._CFG, **(extra_cfg or {})}
        return compute_signals(df, strategy_cfg=cfg)

    def test_yc_no_inversion_exposure_unchanged(self) -> None:
        """Normal case: positive yield_curve_slope → cap must NOT fire.

        The result should be identical to a run without any yc column, because
        no cap is applied (slope > 0).
        """
        from src.assembled_core.strategies.multifactor_v2 import compute_signals

        df_pos_yc = _make_yc_df(["AAPL"], yc_slope=0.50, n_bars=80)
        df_no_yc = _make_yc_df(["AAPL"], yc_slope=None, n_bars=80)

        sig_pos = self._run(df_pos_yc)
        sig_none = compute_signals(df_no_yc, strategy_cfg=self._CFG)

        if not sig_pos.empty and not sig_none.empty:
            score_pos = float(sig_pos["score"].abs().median())
            score_none = float(sig_none["score"].abs().median())
            # No cap applied in either case → scores must be numerically close
            assert (
                abs(score_pos - score_none) < 0.05
            ), f"Positive YC score diverged: yc={score_pos:.4f}, no-yc={score_none:.4f}"

    def test_yc_inversion_reduces_exposure(self) -> None:
        """Persistent negative yield_curve_slope → final score must be <= positive-slope version.

        All 80 bars inverted → persistence check passes (>= 52 of 30-bar tail inverted)
        → exposure cap 0.60 is applied → composite scores should be lower.
        """
        df_inv = _make_yc_df(["AAPL"], yc_slope=-0.30, n_bars=80)
        df_pos = _make_yc_df(["AAPL"], yc_slope=0.50, n_bars=80)

        sig_inv = self._run(df_inv)
        sig_pos = self._run(df_pos)

        if not sig_inv.empty and not sig_pos.empty:
            score_inv = float(sig_inv["score"].abs().median())
            score_pos = float(sig_pos["score"].abs().median())
            assert (
                score_inv <= score_pos + 1e-9
            ), f"Inverted YC score ({score_inv:.4f}) > positive YC ({score_pos:.4f}) — cap not applied"

    def test_yc_inversion_boundary_exactly_at_zero(self) -> None:
        """Boundary condition: yield_curve_slope == 0.0 → cap must NOT fire (condition is < 0)."""
        df_zero = _make_yc_df(["AAPL"], yc_slope=0.0, n_bars=80)
        df_pos = _make_yc_df(["AAPL"], yc_slope=0.50, n_bars=80)

        sig_zero = self._run(df_zero)
        sig_pos = self._run(df_pos)

        if not sig_zero.empty and not sig_pos.empty:
            score_zero = float(sig_zero["score"].abs().median())
            score_pos = float(sig_pos["score"].abs().median())
            # slope=0 not strictly < 0 → no cap → should be close to positive baseline
            assert (
                abs(score_zero - score_pos) < 0.05
            ), f"Boundary yc=0 score diverged: zero={score_zero:.4f}, pos={score_pos:.4f}"

    def test_yc_missing_column_no_crash(self) -> None:
        """No yield_curve_slope column at all → compute_signals must not raise."""
        df = _make_yc_df(["AAPL"], yc_slope=None, n_bars=80)
        assert "yield_curve_slope" not in df.columns

        try:
            result = self._run(df)
            assert isinstance(result, pd.DataFrame)
        except Exception as exc:
            pytest.fail(f"compute_signals raised with missing yc column: {exc}")

    def test_yc_policy_disabled_via_cfg_cap_not_applied(self) -> None:
        """Setting yield_curve_inversion_cap=1.0 in cfg effectively disables the cap.

        The cap guard only fires when _yc_cap < exposure_mult.  By setting
        yield_curve_inversion_cap=1.0 the cap value equals or exceeds any
        realistic exposure_mult, so the guard body is never entered and scores
        must match the no-cap baseline.
        """
        df_inv = _make_yc_df(["AAPL"], yc_slope=-0.30, n_bars=80)
        df_pos = _make_yc_df(["AAPL"], yc_slope=0.50, n_bars=80)

        # cap effectively disabled (1.0 ≥ any exposure_mult → condition never true)
        sig_disabled = self._run(df_inv, extra_cfg={"yield_curve_inversion_cap": 1.0})
        sig_pos = self._run(df_pos)

        if not sig_disabled.empty and not sig_pos.empty:
            score_disabled = float(sig_disabled["score"].abs().median())
            score_pos = float(sig_pos["score"].abs().median())
            # With cap disabled, inverted-slope scores should be close to positive baseline
            assert abs(score_disabled - score_pos) < 0.15, (
                f"Disabled-cap score ({score_disabled:.4f}) diverged too far from "
                f"positive-slope ({score_pos:.4f})"
            )


# ---------------------------------------------------------------------------
# Conviction engine / trigger basket — smoke imports
# ---------------------------------------------------------------------------


class TestConvictionEngineSmoke:
    """Conviction engine is not yet present in the repo — tests are skipped.

    If/when the module appears at one of the expected paths, remove the skip.
    """

    CANDIDATE_PATHS = [
        "src.assembled_core.ops.conviction_engine",
        "src.assembled_core.events.conviction_engine",
    ]

    def test_conviction_engine_importable_or_skip(self) -> None:
        """Import first available conviction engine path, or skip gracefully."""
        for path in self.CANDIDATE_PATHS:
            try:
                mod = importlib.import_module(path)
                assert mod is not None
                return
            except ImportError:
                continue
        pytest.skip(
            "conviction_engine not found at any expected path — "
            "module not yet implemented"
        )


class TestTriggerBasketSmoke:
    """trigger_basket.py is not yet present — test skips gracefully."""

    MODULE_PATH = "src.assembled_core.events.crisis_alpha.trigger_basket"

    def test_trigger_basket_importable_or_skip(self) -> None:
        try:
            mod = importlib.import_module(self.MODULE_PATH)
            assert mod is not None
        except ImportError:
            pytest.skip("trigger_basket not found — module not yet implemented")
