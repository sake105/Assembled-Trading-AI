"""Tests for M17 Wave 1: signal confidence, multi-channel propagation, alpha decay."""

import numpy as np
import pandas as pd
import pytest


def _scipy_available():
    try:
        import scipy  # noqa: F401

        return True
    except ImportError:
        return False


# ── Signal Confidence (1.9) ───────────────────────────────────────────


class TestSignalConfidence:
    def test_import(self):
        from src.assembled_core.signals.signal_confidence import (
            compute_signal_confidence,
        )

        assert compute_signal_confidence is not None

    def test_bayesian_update(self):
        from src.assembled_core.signals.signal_confidence import bayesian_update_normal

        # With many observations, posterior should shift toward sample mean
        obs = np.random.normal(5.0, 1.0, 100)
        post_mean, post_var = bayesian_update_normal(0.0, 10.0, obs)
        assert abs(post_mean - 5.0) < 1.0  # moved toward sample mean
        assert post_var < 10.0  # variance decreased

    @pytest.mark.skipif(not _scipy_available(), reason="scipy not installed")
    def test_signal_confidence_basic(self):
        from src.assembled_core.signals.signal_confidence import (
            compute_signal_confidence,
        )

        scores = pd.Series({"AAPL": 0.8, "MSFT": 0.3, "GOOG": -0.2, "AMZN": 0.5})

        result = compute_signal_confidence(scores)
        assert len(result) == 4
        for sym, conf in result.items():
            assert conf.ci_lower < conf.ci_upper
            assert conf.confidence_width > 0
            assert conf.n_obs == 4

    @pytest.mark.skipif(not _scipy_available(), reason="scipy not installed")
    def test_confidence_position_scaler(self):
        from src.assembled_core.signals.signal_confidence import (
            compute_signal_confidence,
            confidence_position_scaler,
        )

        scores = pd.Series({"A": 0.9, "B": 0.1})
        confidences = compute_signal_confidence(scores)

        for sym, conf in confidences.items():
            scale = confidence_position_scaler(conf)
            assert 0.2 <= scale <= 2.0


# ── Multi-Channel Propagation (4.1) ──────────────────────────────────


class TestMultiChannelPropagation:
    def test_import_v2(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.multichannel_propagation")
        from src.assembled_core.intel.multichannel_propagation import (
            propagate_multichannel,
        )

        assert propagate_multichannel is not None

    def test_exponential_decay(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.multichannel_propagation")
        from src.assembled_core.intel.multichannel_propagation import exponential_decay

        # At t=0, impact = magnitude
        assert exponential_decay(1.0, 0, 5.0) == 1.0
        # At t=half_life, impact ≈ 0.5
        assert abs(exponential_decay(1.0, 5.0, 5.0) - 0.5) < 0.01
        # At t=2*half_life, impact ≈ 0.25
        assert abs(exponential_decay(1.0, 10.0, 5.0) - 0.25) < 0.01

    def test_channel_impact_financial(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.multichannel_propagation")
        from src.assembled_core.intel.multichannel_propagation import (
            compute_channel_impact,
            PropagationChannel,
        )

        # Financial channel: fast (0-2d lag)
        impact = compute_channel_impact(
            initial_magnitude=0.8,
            channel=PropagationChannel.FINANCIAL,
            days_since_event=1,
            n_hops=1,
        )
        assert impact.current_impact > 0
        assert impact.channel == PropagationChannel.FINANCIAL

    def test_channel_impact_trade_delayed(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.multichannel_propagation")
        from src.assembled_core.intel.multichannel_propagation import (
            compute_channel_impact,
            PropagationChannel,
        )

        # Trade channel: slow (5-30d lag)
        impact_early = compute_channel_impact(
            initial_magnitude=0.8,
            channel=PropagationChannel.TRADE,
            days_since_event=1,  # before lag
            n_hops=1,
        )
        impact_peak = compute_channel_impact(
            initial_magnitude=0.8,
            channel=PropagationChannel.TRADE,
            days_since_event=20,  # near peak
            n_hops=1,
        )

        assert impact_early.current_impact == 0.0  # not yet started
        assert impact_peak.current_impact > 0  # active

    def test_propagate_multichannel(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.multichannel_propagation")
        from src.assembled_core.intel.multichannel_propagation import (
            propagate_multichannel,
        )

        result = propagate_multichannel(
            initial_magnitude=0.7,
            edge_types=["TRADE_DEPENDENT", "LENDS_TO"],
            days_since_event=5,
            n_hops=1,
        )
        assert result.total_impact > 0
        assert len(result.channel_impacts) >= 2
        assert result.dominant_channel in ("financial", "trade", "sentiment")

    def test_impact_timeline(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.multichannel_propagation")
        from src.assembled_core.intel.multichannel_propagation import (
            compute_impact_timeline,
        )

        timeline = compute_impact_timeline(
            initial_magnitude=0.8,
            edge_types=["TRADE_DEPENDENT"],
            n_hops=1,
            horizon_days=40,
        )
        assert len(timeline) == 41  # days 0-40
        # Impact should exist at some point
        assert max(timeline.values()) > 0

    def test_sentiment_instantaneous(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.multichannel_propagation")
        from src.assembled_core.intel.multichannel_propagation import (
            compute_channel_impact,
            PropagationChannel,
        )

        # Sentiment is instantaneous
        impact = compute_channel_impact(
            initial_magnitude=0.5,
            channel=PropagationChannel.SENTIMENT,
            days_since_event=0,
            n_hops=1,
        )
        assert impact.current_impact > 0  # immediate


# ── Alpha Decay Half-Life (1.4) ───────────────────────────────────────


class TestAlphaDecay:
    def test_import_v3(self):
        from src.assembled_core.qa.factor_analysis import estimate_alpha_decay_halflife

        assert estimate_alpha_decay_halflife is not None

    def test_decaying_ic(self):
        from src.assembled_core.qa.factor_analysis import estimate_alpha_decay_halflife

        # Simulate exponential decay with half-life ~10 days
        horizons = [1, 3, 5, 10, 20, 40, 60]
        ic_0 = 0.15
        true_hl = 10.0
        ic_values = [ic_0 * np.exp(-np.log(2) * h / true_hl) for h in horizons]

        ic_decay_df = pd.DataFrame(
            {
                "horizon_days": horizons,
                "ic_mean": ic_values,
            }
        )

        result = estimate_alpha_decay_halflife(ic_decay_df)
        assert abs(result["half_life_days"] - true_hl) < 2.0  # close to true
        assert result["r_squared"] > 0.9  # good fit
        assert result["ic_0"] > 0

    def test_flat_ic(self):
        from src.assembled_core.qa.factor_analysis import estimate_alpha_decay_halflife

        # Non-decaying IC (value factor)
        ic_decay_df = pd.DataFrame(
            {
                "horizon_days": [1, 5, 10, 20, 60],
                "ic_mean": [0.05, 0.05, 0.05, 0.05, 0.05],
            }
        )

        result = estimate_alpha_decay_halflife(ic_decay_df)
        # Should have very long or infinite half-life
        assert result["half_life_days"] > 100 or np.isinf(result["half_life_days"])

    def test_empty_df(self):
        from src.assembled_core.qa.factor_analysis import estimate_alpha_decay_halflife

        result = estimate_alpha_decay_halflife(pd.DataFrame())
        assert np.isnan(result["half_life_days"])


# ── IC Weights (1.1) ─────────────────────────────────────────────────


class TestICWeights:
    @pytest.mark.skipif(not _scipy_available(), reason="scipy not installed")
    def test_compute_ic_weights(self):
        from src.assembled_core.signals.multifactor_signal import compute_ic_weights

        np.random.seed(42)
        n_dates = 120
        n_symbols = 10
        dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")

        rows = []
        for d in dates:
            for s in range(n_symbols):
                # Factor with predictive power
                fwd = np.random.normal(0, 0.02)
                rows.append(
                    {
                        "timestamp": d,
                        "symbol": f"SYM{s}",
                        "factor_a": fwd + np.random.normal(0, 0.01),  # correlated
                        "factor_b": np.random.normal(0, 0.02),  # uncorrelated
                        "fwd_return": fwd,
                    }
                )
        df = pd.DataFrame(rows)

        result = compute_ic_weights(
            df,
            "fwd_return",
            ["factor_a", "factor_b"],
            ic_window=40,
        )
        assert not result.empty
        assert "weight_factor_a" in result.columns
        assert "weight_factor_b" in result.columns
        assert "aggregate_ic" in result.columns

        # Factor A should generally get higher weight (correlated with returns)
        last_row = result.iloc[-1]
        assert last_row["weight_factor_a"] >= last_row["weight_factor_b"]
