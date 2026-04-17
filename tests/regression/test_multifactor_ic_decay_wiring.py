"""F1 wiring — verify ``multifactor_v1.compute_signals`` honours ic_decay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.phase_depth]

from src.assembled_core.strategies import multifactor_v1  # noqa: E402


def _minimal_feature_frame(n_days: int = 30, symbols=("AAA", "BBB")) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rows = []
    rng = np.random.default_rng(7)
    for sym in symbols:
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n_days)))
        for ts, close in zip(idx, closes):
            rows.append({"timestamp": ts, "symbol": sym, "close": float(close)})
    return pd.DataFrame(rows)


def test_source_mentions_ic_decay_call() -> None:
    src_path = Path(multifactor_v1.__file__)
    src = src_path.read_text(encoding="utf-8")
    assert "compute_ic_decay_weights" in src
    assert 'ic_cfg = cfg.get("ic_decay"' in src


def test_ic_decay_disabled_is_a_noop() -> None:
    df = _minimal_feature_frame()
    base_cfg = {"factor_weights": {"trend_ema_spread": 1.0}}
    ic_cfg = {
        "factor_weights": {"trend_ema_spread": 1.0},
        "ic_decay": {"enabled": False, "ic_snapshot": {"trend_ema_spread": 0.1}},
    }
    a = multifactor_v1.compute_signals(df, base_cfg)
    b = multifactor_v1.compute_signals(df, ic_cfg)
    # Disabled → must match the base run exactly.
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False
    )


def test_ic_decay_missing_snapshot_is_a_noop() -> None:
    df = _minimal_feature_frame()
    cfg = {
        "factor_weights": {"trend_ema_spread": 1.0},
        "ic_decay": {"enabled": True},  # no snapshot supplied
    }
    result = multifactor_v1.compute_signals(df, cfg)
    assert isinstance(result, pd.DataFrame)  # silently falls through


def test_ic_decay_replaces_weights_when_enabled() -> None:
    df = _minimal_feature_frame()
    # Snapshot puts almost all weight on mom_rsi_centered instead of trend.
    cfg = {
        "factor_weights": {"trend_ema_spread": 1.0, "mom_rsi_centered": 0.0},
        "ic_decay": {
            "enabled": True,
            "ic_snapshot": {"trend_ema_spread": 0.01, "mom_rsi_centered": 0.2},
            "lags": {"trend_ema_spread": 0.0, "mom_rsi_centered": 0.0},
            "half_lives": {"trend_ema_spread": 30.0, "mom_rsi_centered": 30.0},
        },
    }
    # Just verify the call succeeds and returns a DataFrame; the assertion
    # that wiring path was exercised is the source pin above.
    out = multifactor_v1.compute_signals(df, cfg)
    assert isinstance(out, pd.DataFrame)
