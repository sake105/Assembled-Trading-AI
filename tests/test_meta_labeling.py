"""Tests für meta_labeling.py (Phase 5 — Meta-Labeling)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_learning_store(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _make_records(n: int, closed_date: str = "2025-06-01") -> list[dict]:
    rng = np.random.default_rng(42)
    records = []
    for i in range(n):
        ret = float(rng.normal(0.002, 0.01))
        records.append(
            {
                "score": float(rng.uniform(-1, 1)),
                "direction": int(np.sign(rng.uniform(-1, 1))),
                "closed_at": closed_date + "T12:00:00",
                "closed_return": ret,
                "news_sentiment_mean": float(rng.uniform(-0.5, 0.5)),
                "news_velocity": float(rng.uniform(0.5, 2.0)),
                "regime_state": int(rng.integers(0, 3)),
                "vix_proxy": float(rng.uniform(10, 40)),
            }
        )
    return records


# ---------------------------------------------------------------------------
# scale_position
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PIT guard + min_samples
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training + predict_confidence
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# predict_proba_with_meta in meta_model.py
# ---------------------------------------------------------------------------


def test_predict_proba_with_meta_fallback():
    """meta_labeler=None → identische Ausgabe wie predict_proba."""
    pytest.importorskip("sklearn")

    from src.assembled_core.signals.meta_model import train_meta_model

    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "label": rng.integers(0, 2, n),
        }
    )
    meta_model = train_meta_model(df, feature_cols=["f1", "f2"], label_col="label")
    X = df[["f1", "f2"]]

    scores_base = meta_model.predict_proba(X)
    scores_meta = meta_model.predict_proba_with_meta(X, meta_labeler=None)

    pd.testing.assert_series_equal(scores_base, scores_meta)
