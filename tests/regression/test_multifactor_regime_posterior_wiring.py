"""F2 wiring — verify ``multifactor_v1._compute_regime_multiplier`` honours posterior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.phase_depth]

from src.assembled_core.strategies import multifactor_v1  # noqa: E402


def _features(n_days: int = 40, symbols=("AAA", "BBB")) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rng = np.random.default_rng(3)
    rows = []
    for sym in symbols:
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n_days)))
        for ts, close in zip(idx, closes):
            rows.append({"timestamp": ts, "symbol": sym, "close": float(close)})
    return pd.DataFrame(rows)


def test_source_mentions_posterior_blend_call() -> None:
    src_path = Path(multifactor_v1.__file__)
    src = src_path.read_text(encoding="utf-8")
    assert "smooth_posterior" in src
    assert 'reg_cfg = (cfg or {}).get("regime_posterior")' in src


def test_bull_heavy_posterior_beats_bear_heavy_posterior() -> None:
    df = _features()
    bull_cfg = {
        "factor_weights": {"trend_ema_spread": 1.0},
        "regime_posterior": {
            "enabled": True,
            "posterior": {"bull": 0.9, "bear": 0.1},
        },
    }
    bear_cfg = {
        "factor_weights": {"trend_ema_spread": 1.0},
        "regime_posterior": {
            "enabled": True,
            "posterior": {"bull": 0.1, "bear": 0.9},
        },
    }
    bull_mult = multifactor_v1._compute_regime_multiplier(df, bull_cfg)
    bear_mult = multifactor_v1._compute_regime_multiplier(df, bear_cfg)
    # Plan-configured exposures: bull=1.0, bear=0.3 → bull path must give
    # a higher base multiplier than the bear path.
    assert bull_mult > bear_mult


def test_disabled_posterior_is_ignored() -> None:
    df = _features()
    cfg = {
        "factor_weights": {"trend_ema_spread": 1.0},
        "regime_posterior": {
            "enabled": False,
            "posterior": {"bull": 1.0},
        },
    }
    # Must not raise; should fall through to the legacy regime path.
    mult = multifactor_v1._compute_regime_multiplier(df, cfg)
    assert mult >= 0.0
