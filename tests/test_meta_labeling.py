"""Tests für meta_labeling.py (Phase 5 — Meta-Labeling)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase12


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
        records.append({
            "score": float(rng.uniform(-1, 1)),
            "direction": int(np.sign(rng.uniform(-1, 1))),
            "closed_at": closed_date + "T12:00:00",
            "closed_return": ret,
            "news_sentiment_mean": float(rng.uniform(-0.5, 0.5)),
            "news_velocity": float(rng.uniform(0.5, 2.0)),
            "regime_state": int(rng.integers(0, 3)),
            "vix_proxy": float(rng.uniform(10, 40)),
        })
    return records


# ---------------------------------------------------------------------------
# scale_position
# ---------------------------------------------------------------------------

def test_scale_position_below_threshold():
    from src.assembled_core.ml.meta_labeling import MetaLabeler
    labeler = MetaLabeler(confidence_threshold=0.55)
    assert labeler.scale_position(0.8, 0.40) == 0.0


def test_scale_position_above_threshold():
    from src.assembled_core.ml.meta_labeling import MetaLabeler
    labeler = MetaLabeler(confidence_threshold=0.55)
    result = labeler.scale_position(0.8, 0.70)
    assert abs(result - 0.56) < 1e-9


def test_scale_position_short():
    from src.assembled_core.ml.meta_labeling import MetaLabeler
    labeler = MetaLabeler(confidence_threshold=0.55)
    result = labeler.scale_position(-0.5, 0.65)
    assert abs(result - (-0.325)) < 1e-9


def test_scale_position_at_threshold():
    from src.assembled_core.ml.meta_labeling import MetaLabeler
    labeler = MetaLabeler(confidence_threshold=0.55)
    # exactly at threshold → scale (>= threshold, not strictly above)
    result = labeler.scale_position(1.0, 0.55)
    assert result == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# PIT guard + min_samples
# ---------------------------------------------------------------------------

def test_from_learning_store_pit_guard(tmp_path):
    """Records mit closed_at in der Zukunft werden NICHT verwendet."""
    from src.assembled_core.ml.meta_labeling import MetaLabeler

    pytest.importorskip("sklearn")

    store = tmp_path / "store.jsonl"
    future_date = "2099-01-01"
    records = _make_records(200, closed_date=future_date)
    _write_learning_store(store, records)

    as_of = pd.Timestamp("2025-01-01")
    with pytest.raises(ValueError, match="Nur 0 Records"):
        MetaLabeler.from_learning_store(store, as_of=as_of)


def test_from_learning_store_min_samples_error(tmp_path):
    """Weniger als 100 Records → ValueError."""
    from src.assembled_core.ml.meta_labeling import MetaLabeler

    store = tmp_path / "store.jsonl"
    records = _make_records(50, closed_date="2025-01-01")
    _write_learning_store(store, records)

    with pytest.raises(ValueError, match="Minimum: 100"):
        MetaLabeler.from_learning_store(store)


def test_from_learning_store_only_closed_records(tmp_path):
    """Nur Records mit closed_at UND closed_return werden gezählt."""
    from src.assembled_core.ml.meta_labeling import MetaLabeler

    store = tmp_path / "store.jsonl"
    # 150 valid + 50 without closed_at
    records = _make_records(150, closed_date="2025-01-01")
    records += [{"score": 0.5, "direction": 1} for _ in range(50)]  # kein closed_at
    _write_learning_store(store, records)

    pytest.importorskip("sklearn")

    labeler, report = MetaLabeler.from_learning_store(
        store,
        as_of=pd.Timestamp("2026-01-01"),
    )
    assert report["n_records"] == 150


# ---------------------------------------------------------------------------
# Training + predict_confidence
# ---------------------------------------------------------------------------

def test_labeler_fit_predict(tmp_path):
    pytest.importorskip("sklearn")

    from src.assembled_core.ml.meta_labeling import MetaLabeler

    store = tmp_path / "store.jsonl"
    records = _make_records(200, closed_date="2025-06-01")
    _write_learning_store(store, records)

    labeler, report = MetaLabeler.from_learning_store(
        store,
        as_of=pd.Timestamp("2026-01-01"),
    )

    assert "n_records" in report
    assert 0.0 <= report["hit_rate"] <= 1.0

    # predict_confidence on feature df
    features = pd.DataFrame({
        "primary_signal": [0.5, -0.3],
        "primary_direction": [1, -1],
        "news_sentiment_mean": [0.1, -0.2],
        "news_velocity": [1.2, 0.8],
        "regime_state": [1, 2],
        "vix_proxy": [20.0, 35.0],
    })
    conf = labeler.predict_confidence(features)
    assert len(conf) == 2
    assert (conf >= 0.0).all() and (conf <= 1.0).all()


def test_predict_confidence_no_model():
    """Ohne fit() → Fallback 0.5."""
    from src.assembled_core.ml.meta_labeling import MetaLabeler

    labeler = MetaLabeler()
    features = pd.DataFrame({
        "primary_signal": [0.5],
        "primary_direction": [1],
        "news_sentiment_mean": [0.0],
        "news_velocity": [1.0],
        "regime_state": [0],
        "vix_proxy": [20.0],
    })
    conf = labeler.predict_confidence(features)
    assert conf.iloc[0] == pytest.approx(0.5)


def test_labeler_save_load(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")

    from src.assembled_core.ml.meta_labeling import MetaLabeler

    store = tmp_path / "store.jsonl"
    _write_learning_store(store, _make_records(150, closed_date="2025-01-01"))

    labeler, _ = MetaLabeler.from_learning_store(
        store, as_of=pd.Timestamp("2026-01-01")
    )

    model_path = tmp_path / "labeler.joblib"
    labeler.save(model_path)
    assert model_path.exists()

    loaded = MetaLabeler.load(model_path)
    assert loaded.confidence_threshold == labeler.confidence_threshold


# ---------------------------------------------------------------------------
# predict_proba_with_meta in meta_model.py
# ---------------------------------------------------------------------------

def test_predict_proba_with_meta_fallback():
    """meta_labeler=None → identische Ausgabe wie predict_proba."""
    pytest.importorskip("sklearn")

    from src.assembled_core.signals.meta_model import train_meta_model

    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
        "label": rng.integers(0, 2, n),
    })
    meta_model = train_meta_model(df, feature_cols=["f1", "f2"], label_col="label")
    X = df[["f1", "f2"]]

    scores_base = meta_model.predict_proba(X)
    scores_meta = meta_model.predict_proba_with_meta(X, meta_labeler=None)

    pd.testing.assert_series_equal(scores_base, scores_meta)
