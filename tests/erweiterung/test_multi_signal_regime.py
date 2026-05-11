"""Tests für multi_signal_regime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.multi_signal_regime import (
    MultiSignalConfig,
    composite_stress_score,
    dispersion_signal,
    drawdown_signal,
    news_anomaly_signal,
    realized_vol_signal,
)


def _market(n: int = 400, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.0003, 0.01, n), index=pd.date_range("2022-01-01", periods=n)
    )


def _panel(n: int = 400, k: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0003, 0.012, (n, k)),
        index=pd.date_range("2022-01-01", periods=n),
        columns=[f"s{i}" for i in range(k)],
    )


def test_drawdown_signal_in_unit_range():
    s = drawdown_signal(_market())
    assert s.min() >= 0
    assert s.max() <= 1.0


def test_realized_vol_signal_in_unit_range():
    s = realized_vol_signal(_market())
    valid = s.dropna()
    assert (valid >= 0).all()
    assert (valid <= 1.0).all()


def test_dispersion_signal_returns_series():
    s = dispersion_signal(_panel())
    assert isinstance(s, pd.Series)
    assert len(s) == 400


def test_news_anomaly_none_returns_none():
    assert news_anomaly_signal(None) is None
    assert news_anomaly_signal(pd.DataFrame()) is None


def test_news_anomaly_with_sentiment_panel():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=10, tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "count": [3, 4, 5, 12, 30, 25, 5, 4, 3, 4],
            "sentiment_volume": [3, 4, 5, 12, 30, 25, 5, 4, 3, 4],
        }
    )
    sig = news_anomaly_signal(df, expected_baseline_count=5.0)
    assert sig is not None
    assert sig.max() > 0.5  # die 30-count Tage sind clearly stressig


def test_composite_stress_score_basic():
    out = composite_stress_score(_market(), _panel())
    assert "composite_score" in out.columns
    assert "regime" in out.columns
    assert set(out["regime"].unique()).issubset({"stress", "calm"})


def test_composite_stress_score_crash_triggers_stress():
    # Künstlicher Crash auslösen
    rng = np.random.default_rng(7)
    n = 400
    market_ret = pd.Series(
        rng.normal(0.0002, 0.005, n),
        index=pd.date_range("2022-01-01", periods=n),
    )
    market_ret.iloc[200:230] = np.linspace(-0.005, -0.05, 30)

    panel = pd.DataFrame(
        rng.normal(0.0002, 0.005, (n, 30)),
        index=market_ret.index,
        columns=[f"s{i}" for i in range(30)],
    )
    # Im Crash-Bereich die Cross-Section-Vol explodieren lassen
    panel.iloc[200:230] += rng.normal(0, 0.03, (30, 30))

    out = composite_stress_score(market_ret, panel)
    assert (out.loc[market_ret.index[210:225], "regime"] == "stress").sum() > 0


def test_composite_handles_missing_news():
    out = composite_stress_score(_market(), _panel(), sentiment_panel=None)
    # news_anomaly column existiert, ist aber komplett NaN
    assert "news_anomaly" in out.columns
    assert out["news_anomaly"].isna().all()
    assert out["composite_score"].notna().any()


def test_config_weights_renormalize_when_some_signals_missing():
    cfg = MultiSignalConfig(
        weights={
            "drawdown": 0.25,
            "realized_vol": 0.25,
            "dispersion": 0.25,
            "news_anomaly": 0.25,
        }
    )
    out = composite_stress_score(_market(), _panel(), sentiment_panel=None, config=cfg)
    # Da news_anomaly fehlt, sollten die übrigen 3 Signale die volle Gewichtung tragen
    assert out["composite_score"].max() > 0
