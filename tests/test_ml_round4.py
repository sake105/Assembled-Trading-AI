"""Tests für Round-4 ML-Erweiterungen (Online GB, Nested Meta, Wiring)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Online Gradient Boosting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Nested Meta Labeling
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# meta_model.py → predict_with_intervals
# ---------------------------------------------------------------------------


def test_meta_model_predict_with_intervals_no_calib():
    """Ohne Calib → confidence=1.0, intervals=point predictions."""
    pytest.importorskip("sklearn")
    from src.assembled_core.signals.meta_model import train_meta_model

    rng = np.random.default_rng(13)
    n = 200
    df = pd.DataFrame(
        {
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "label": rng.integers(0, 2, n),
        }
    )
    mm = train_meta_model(df, feature_cols=["f1", "f2"], label_col="label")

    result = mm.predict_with_intervals(df[["f1", "f2"]])
    assert "predictions" in result
    assert "lower" in result
    assert "upper" in result
    assert "confidence" in result
    assert (result["confidence"] == 1.0).all()
    assert result["half_width"] == 0.0


# ---------------------------------------------------------------------------
# build_factor_panel.py → triple-barrier wiring
# ---------------------------------------------------------------------------


def test_build_factor_panel_triple_barrier_flag(tmp_path):
    """Funktion build_full_factor_panel akzeptiert triple_barrier=True."""
    # Smoke-Test: Import-Check + Argumente
    from scripts.training.build_factor_panel import build_full_factor_panel

    import inspect

    sig = inspect.signature(build_full_factor_panel)
    assert "triple_barrier" in sig.parameters
    assert "tb_upper_mult" in sig.parameters
    assert "tb_lower_mult" in sig.parameters
