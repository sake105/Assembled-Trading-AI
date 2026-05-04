"""Tests for non-PAID autonome weiterarbeit modules.

Covers files 30-43 (excluding PAID 20-23):
  - signals/base.py + signals/registry.py  (33_EXECUTION §13)
  - signals/news_fusion.py                 (30_NEWS_TA_FUSION)
  - signals/composite_score.py             (31_COMPOSITE_SCORE)
  - qa/shadow_signal.py                    (32_VALIDIERUNG §32.6-7)
  - execution/pdt_counter.py               (41_PDT_REGEL)
  - execution/idempotency.py               (33_EXECUTION §33.2)
  - events/schema.py                       (42_EVENT_REPLAY)
  - events/store.py                        (42_EVENT_REPLAY)
  - events/replayer.py                     (42_EVENT_REPLAY)
  - certify/schema.py                      (43_REPRODUCIBILITY)
  - certify/generator.py                   (43_REPRODUCIBILITY)
  - data/quality_gate.py                   (37_DATA_QUALITY_GATE)
  - attribution/schemas.py                 (38_FEATURE_ATTRIBUTION)
  - attribution/storage.py                 (38_FEATURE_ATTRIBUTION)
  - attribution/composite.py               (38_FEATURE_ATTRIBUTION)
  - strategy/config.py                     (39_HYPERPARAMETER_GOVERNANCE)
  - strategy/experiment_tracker.py         (39_HYPERPARAMETER_GOVERNANCE)
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

# ==========================================================================
# signals/base.py + signals/registry.py  (33 §13)
# ==========================================================================


def test_signal_output_is_actionable():
    from src.assembled_core.signals.base import SignalOutput

    out = SignalOutput(symbol="AAPL", score=0.7, confidence=0.8)
    assert out.is_actionable(min_abs_score=0.5, min_confidence=0.7)
    assert not out.is_actionable(min_abs_score=0.9)


def test_signal_output_defaults():
    from src.assembled_core.signals.base import SignalOutput

    out = SignalOutput(symbol="GOOG", score=-0.3, confidence=0.6)
    assert out.symbol == "GOOG"
    assert isinstance(out.computed_at, datetime)


def test_signal_registry_empty():
    from src.assembled_core.signals.registry import SignalRegistry

    reg = SignalRegistry()
    assert len(reg) == 0
    assert reg.all() == []


def test_signal_registry_register_and_get():
    from src.assembled_core.signals.base import BaseSignal, SignalOutput
    from src.assembled_core.signals.registry import SignalRegistry

    class DummySignal(BaseSignal):
        name = "dummy_test"
        version = "0.1.0"

        async def compute(self, symbol, feature_store, now):
            return SignalOutput(symbol=symbol, score=0.1, confidence=0.9)

    reg = SignalRegistry()
    reg.register(DummySignal())
    assert reg.get("dummy_test") is not None
    assert len(reg) == 1


def test_signal_registry_duplicate_raises():
    from src.assembled_core.signals.base import BaseSignal
    from src.assembled_core.signals.registry import SignalRegistry

    class DupSignal(BaseSignal):
        name = "dup_signal"

        async def compute(self, symbol, feature_store, now):
            return None

    reg = SignalRegistry()
    reg.register(DupSignal())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(DupSignal())


def test_signal_registry_load_all_no_eps():
    """load_all with no entry-points registered returns 0."""
    from src.assembled_core.signals.registry import SignalRegistry

    reg = SignalRegistry()
    n = reg.load_all()
    assert isinstance(n, int)


# ==========================================================================
# signals/news_fusion.py  (30_NEWS_TA_FUSION)
# ==========================================================================


def _news_feats(**kw):
    base = {
        k: 0.0
        for k in (
            "sentiment_vw",
            "novelty",
            "surprise",
            "event_volume_z",
            "velocity",
            "dispersion",
        )
    }
    base.update(kw)
    return base


def test_news_z_score_range():
    from src.assembled_core.signals.news_fusion import news_z_score

    for _ in range(50):
        feats = _news_feats(
            sentiment_vw=np.random.uniform(-2, 2), velocity=np.random.uniform(-2, 2)
        )
        nz = news_z_score(feats)
        assert -3.0 <= nz <= 3.0, nz


def test_news_z_score_dispersion_penalty():
    from src.assembled_core.signals.news_fusion import news_z_score

    high_disp = news_z_score(_news_feats(sentiment_vw=1.0, dispersion=2.0))
    low_disp = news_z_score(_news_feats(sentiment_vw=1.0, dispersion=0.0))
    assert high_disp < low_disp


def test_size_from_meta_below_threshold():
    from src.assembled_core.signals.news_fusion import size_from_meta

    assert size_from_meta(0.50, theta_meta=0.55) == 0.0


def test_size_from_meta_above_threshold():
    from src.assembled_core.signals.news_fusion import size_from_meta

    s = size_from_meta(0.80, theta_meta=0.55)
    assert 0.0 < s <= 1.0


def test_news_veto_true():
    from src.assembled_core.signals.news_fusion import news_veto

    # strong negative news, positive primary
    assert news_veto(news_z=-2.0, primary_side=1.0, tau_veto=1.5) is True


def test_news_veto_false_same_sign():
    from src.assembled_core.signals.news_fusion import news_veto

    assert news_veto(news_z=2.0, primary_side=1.0, tau_veto=1.5) is False


def test_bayesian_update_range():
    from src.assembled_core.signals.news_fusion import bayesian_update

    for ta, nz in [(-1, -3), (0, 0), (1, 3), (-0.5, 2)]:
        p = bayesian_update(ta, nz)
        assert 0.0 <= p <= 1.0, p


def test_agreement_multiplier_agreement():
    from src.assembled_core.signals.news_fusion import agreement_multiplier

    m = agreement_multiplier(ta_score=0.8, news_z=2.0)
    assert m >= 1.0  # both positive → boost


def test_agreement_multiplier_conflict():
    from src.assembled_core.signals.news_fusion import agreement_multiplier

    m = agreement_multiplier(ta_score=0.8, news_z=-2.0)
    assert m == 0.5  # strong conflict → reduce


def test_decide_trade_skip_meta():
    from src.assembled_core.signals.news_fusion import decide_trade

    r = decide_trade(
        composite_score=0.5, news_features=_news_feats(), meta_probability=0.40
    )
    assert r["action"] == "skip"
    assert r["reason"] == "meta_below_threshold"


def test_decide_trade_skip_veto():
    from src.assembled_core.signals.news_fusion import decide_trade

    # positive primary, strong negative news → veto
    # Need nz < -1.5: sentiment_vw=-3 (-0.90) + velocity=-3 (-0.45) + novelty=-3 (-0.45) = -1.80
    r = decide_trade(
        composite_score=0.6,
        news_features=_news_feats(sentiment_vw=-3.0, velocity=-3.0, novelty=-3.0),
        meta_probability=0.70,
    )
    assert r["action"] == "skip"
    assert r["reason"] == "news_veto"


def test_decide_trade_long():
    from src.assembled_core.signals.news_fusion import decide_trade

    r = decide_trade(
        composite_score=0.6,
        news_features=_news_feats(sentiment_vw=1.0),
        meta_probability=0.75,
    )
    assert r["action"] == "long"
    assert 0.0 < r["size"] <= 1.0


# ==========================================================================
# signals/composite_score.py  (31_COMPOSITE_SCORE)
# ==========================================================================


def test_composite_weights_sum_to_one():
    from src.assembled_core.signals.composite_score import COMPOSITE_WEIGHTS_BY_REGIME

    for regime, weights in COMPOSITE_WEIGHTS_BY_REGIME.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"{regime}: sum={total}"


def test_composite_score_range():
    from src.assembled_core.signals.composite_score import composite_score

    for regime in ("calm", "normal", "elevated", "crisis"):
        scores = [np.random.uniform(-1, 1) for _ in range(9)]
        result, dims = composite_score(regime, *scores)
        assert -1.0 <= result <= 1.0, result
        assert len(dims) == 9


def test_composite_score_returns_dims():
    from src.assembled_core.signals.composite_score import composite_score

    result, dims = composite_score(
        "normal", 0.1, 0.2, 0.0, 0.1, 0.0, -0.1, 0.1, 0.05, 0.15
    )
    assert set(dims.keys()) == {
        "mtf",
        "classical_ta",
        "microstructure",
        "volume_profile",
        "chart_pattern",
        "vol_surface",
        "breadth",
        "seasonality",
        "news",
    }


def test_mtf_alignment_score_range():
    from src.assembled_core.signals.composite_score import mtf_alignment_score

    close = pd.Series(np.random.uniform(100, 200, 250))
    s = mtf_alignment_score(close, macd_hist_15m=0.02, rsi_5m=60.0, adx_daily=30.0)
    assert -1.0 <= s <= 1.0


def test_classical_ta_score_range():
    from src.assembled_core.signals.composite_score import classical_ta_score

    s = classical_ta_score(rsi=45.0, macd_hist=0.01, bb_percent=0.4, regime="normal")
    assert -1.0 <= s <= 1.0


def test_seasonality_score_turn_of_month():
    from src.assembled_core.signals.composite_score import seasonality_score

    # Turn-of-month effect should boost score
    tom = seasonality_score(date(2024, 1, 2), overnight_gap=0.0)
    mid = seasonality_score(date(2024, 1, 15), overnight_gap=0.0)
    assert tom > mid


def test_vol_surface_score_low_iv_bullish():
    from src.assembled_core.signals.composite_score import vol_surface_score

    s_low_iv = vol_surface_score(iv_rank=10.0)
    s_high_iv = vol_surface_score(iv_rank=90.0)
    assert s_low_iv > s_high_iv


def test_breadth_intermarket_score_range():
    from src.assembled_core.signals.composite_score import breadth_intermarket_score

    s = breadth_intermarket_score(
        mcclellan=50.0,
        xly_xlp_ratio_change=1.05,
        hyg_tlt_change=1.01,
        dxy_change_20d=0.01,
    )
    assert -1.0 <= s <= 1.0


# ==========================================================================
# qa/shadow_signal.py  (32_VALIDIERUNG §32.6-7)
# ==========================================================================


def test_shadow_signal_emits_shadow_flag():
    from src.assembled_core.qa.shadow_signal import ShadowSignal

    sig = ShadowSignal(
        "test", lambda ctx: {"score": 0.5, "side": 1, "return_next": 0.01}
    )
    r = sig.emit({"price": 100})
    assert r["shadow"] is True
    assert r["signal_name"] == "test"


def test_shadow_signal_live_flag():
    from src.assembled_core.qa.shadow_signal import ShadowSignal

    sig = ShadowSignal(
        "test2", lambda ctx: {"score": 0.3, "side": 1, "return_next": 0.005}, live=True
    )
    r = sig.emit({})
    assert r["shadow"] is False


def test_shadow_signal_rolling_ic_insufficient():
    from src.assembled_core.qa.shadow_signal import ShadowSignal

    sig = ShadowSignal(
        "test3", lambda ctx: {"score": 0.3, "side": 1, "return_next": 0.0}
    )
    assert np.isnan(sig.rolling_ic())


def test_canary_size_shadow_phase():
    from src.assembled_core.qa.shadow_signal import canary_size

    assert canary_size(3, sharpe_15d=1.0, drawdown_ratio=0.5) == 0.0


def test_canary_size_first_phase():
    from src.assembled_core.qa.shadow_signal import canary_size

    assert canary_size(10, sharpe_15d=1.0, drawdown_ratio=0.5) == pytest.approx(0.10)


def test_canary_size_pause_on_bad_sharpe():
    from src.assembled_core.qa.shadow_signal import canary_size

    assert canary_size(10, sharpe_15d=0.2, drawdown_ratio=0.5) == 0.0


def test_auto_rollback_triggers():
    from src.assembled_core.qa.shadow_signal import auto_rollback

    assert auto_rollback(0.20, 0.09) is True  # 0.20 > 2 × 0.09


def test_auto_rollback_no_trigger():
    from src.assembled_core.qa.shadow_signal import auto_rollback

    assert auto_rollback(0.10, 0.09) is False  # 0.10 < 2 × 0.09


def test_detect_wf_drift_ok():
    from src.assembled_core.qa.shadow_signal import detect_wf_drift

    sharpes = [0.8, 0.9, 0.85, 0.88, 0.82, 0.87, 0.83, 0.86]
    assert detect_wf_drift(sharpes) == "OK"


def test_detect_wf_drift_alarm():
    from src.assembled_core.qa.shadow_signal import detect_wf_drift

    sharpes = [0.8, 0.9, 0.85, 0.88, 0.82, -0.5, -0.6, -0.7]
    assert detect_wf_drift(sharpes) == "DRIFT"


# ==========================================================================
# execution/pdt_counter.py  (41_PDT_REGEL)
# ==========================================================================


def test_pdt_counter_empty():
    from src.assembled_core.execution.pdt_counter import PDTCounter

    c = PDTCounter()
    assert c.count_in_window() == 0
    assert c.would_trigger_pdt() is False


def test_pdt_counter_add_and_count():
    from src.assembled_core.execution.pdt_counter import PDTCounter

    c = PDTCounter()
    today = date.today()
    c.add_day_trade("AAPL", today)
    c.add_day_trade("MSFT", today)
    assert c.count_in_window() == 2


def test_pdt_counter_triggers_at_3():
    from src.assembled_core.execution.pdt_counter import PDTCounter

    c = PDTCounter()
    today = date.today()
    for sym in ("AAPL", "MSFT", "GOOG"):
        c.add_day_trade(sym, today)
    assert c.would_trigger_pdt() is True


def test_pdt_pre_order_check_high_equity():
    from src.assembled_core.execution.pdt_counter import PDTCounter

    c = PDTCounter()
    today = date.today()
    for sym in ("A", "B", "C"):
        c.add_day_trade(sym, today)
    allowed, reason = c.pre_order_check("D", "buy", True, account_equity=50_000.0)
    assert allowed is True  # above $25k threshold


def test_pdt_pre_order_check_blocked():
    import os

    os.environ["PDT_RULE_ACTIVE"] = "true"
    # reimport to pick up env var
    import importlib
    import src.assembled_core.execution.pdt_counter as mod

    importlib.reload(mod)
    c = mod.PDTCounter()
    today = date.today()
    for sym in ("A", "B", "C"):
        c.add_day_trade(sym, today)
    allowed, reason = c.pre_order_check("D", "buy", True, account_equity=10_000.0)
    assert allowed is False
    assert reason == "pdt_risk"


def test_pdt_summary():
    from src.assembled_core.execution.pdt_counter import PDTCounter

    c = PDTCounter()
    summary = c.summary()
    assert "day_trades_in_window" in summary
    assert "would_trigger_pdt" in summary


# ==========================================================================
# execution/idempotency.py  (33 §33.2)
# ==========================================================================


def test_compute_intent_hash_deterministic():
    from src.assembled_core.execution.idempotency import compute_intent_hash

    h1 = compute_intent_hash("AAPL", "buy", 100.0, "market")
    h2 = compute_intent_hash("AAPL", "buy", 100.0, "market")
    assert h1 == h2


def test_compute_intent_hash_differs_on_side():
    from src.assembled_core.execution.idempotency import compute_intent_hash

    h_buy = compute_intent_hash("AAPL", "buy", 100.0, "market")
    h_sell = compute_intent_hash("AAPL", "sell", 100.0, "market")
    assert h_buy != h_sell


def test_build_client_order_id_length():
    from src.assembled_core.execution.idempotency import (
        build_client_order_id,
        compute_intent_hash,
    )

    h = compute_intent_hash("AAPL", "buy", 100.0, "market")
    coid = build_client_order_id("signal-001", h)
    assert len(coid) <= 48
    assert coid.startswith("ata-")


def test_build_client_order_id_deterministic():
    from src.assembled_core.execution.idempotency import (
        build_client_order_id,
        compute_intent_hash,
    )

    h = compute_intent_hash("MSFT", "sell", 50.0, "limit", limit_price=420.0)
    c1 = build_client_order_id("sig-x", h, attempt=0)
    c2 = build_client_order_id("sig-x", h, attempt=0)
    assert c1 == c2


def test_build_client_order_id_attempt_differs():
    from src.assembled_core.execution.idempotency import (
        build_client_order_id,
        compute_intent_hash,
    )

    h = compute_intent_hash("GOOG", "buy", 5.0, "market")
    c0 = build_client_order_id("sig-y", h, attempt=0)
    c1 = build_client_order_id("sig-y", h, attempt=1)
    assert c0 != c1


def test_is_duplicate_error():
    from src.assembled_core.execution.idempotency import is_duplicate_error

    assert is_duplicate_error("Duplicate client_order_id detected") is True
    assert is_duplicate_error("Insufficient buying power") is False


# ==========================================================================
# events/schema.py + store.py + replayer.py  (42_EVENT_REPLAY)
# ==========================================================================


def test_event_schema_base_event():
    from src.assembled_core.events.schema import BaseEvent, EventSource

    evt = BaseEvent(
        event_type="test",
        source=EventSource.SYSTEM,
        session_id="s1",
        sequence=1,
    )
    assert evt.event_type == "test"
    assert isinstance(evt.occurred_at, datetime)


def test_event_schema_make_market_tick():
    from src.assembled_core.events.schema import make_market_tick

    evt = make_market_tick("sess1", 1, "AAPL", 180.0, 180.05, 180.02, 1000)
    assert evt.event_type == "market_tick_received"
    assert evt.payload["symbol"] == "AAPL"


def test_event_schema_make_news_event():
    from src.assembled_core.events.schema import make_news_event

    evt = make_news_event("sess1", 2, "Apple beats estimates", "AAPL", 0.8)
    assert evt.event_type == "news_received"
    assert evt.payload["ticker"] == "AAPL"


def test_event_schema_to_json():
    from src.assembled_core.events.schema import make_order_filled

    evt = make_order_filled("sess1", 3, "AAPL", "buy", 100.0, 182.5, "ata-abc")
    j = evt.to_json()
    import json

    d = json.loads(j)
    assert d["event_type"] == "order_filled"


def test_event_store_append_and_load(tmp_path):
    from src.assembled_core.events.store import EventStore
    from src.assembled_core.events.schema import make_market_tick

    store = EventStore(tmp_path / "test.db")
    evt = make_market_tick("sess1", 1, "AAPL", 180.0, 180.05, 180.02, 1000)
    store.append(evt)
    rows = store.load_session("sess1")
    assert len(rows) == 1
    assert rows[0]["event_type"] == "market_tick_received"


def test_event_store_append_batch(tmp_path):
    from src.assembled_core.events.store import EventStore
    from src.assembled_core.events.schema import make_market_tick

    store = EventStore(tmp_path / "batch.db")
    evts = [
        make_market_tick("sess2", i, "GOOG", 170.0, 170.1, 170.05, 500)
        for i in range(5)
    ]
    n = store.append_batch(evts)
    assert n == 5
    rows = store.load_session("sess2")
    assert len(rows) == 5


def test_event_store_session_stats(tmp_path):
    from src.assembled_core.events.store import EventStore
    from src.assembled_core.events.schema import make_clock_tick

    store = EventStore(tmp_path / "stats.db")
    now = datetime.now(tz=timezone.utc)
    store.append(make_clock_tick("sess3", 1, now))
    stats = store.session_stats("sess3")
    assert stats["n_events"] == 1


def test_event_store_idempotent(tmp_path):
    from src.assembled_core.events.store import EventStore
    from src.assembled_core.events.schema import make_market_tick

    store = EventStore(tmp_path / "idem.db")
    evt = make_market_tick("sess4", 1, "AAPL", 180.0, 180.05, 180.02, 1000)
    store.append(evt)
    store.append(evt)  # duplicate — should be silently ignored
    rows = store.load_session("sess4")
    assert len(rows) == 1


def test_replayer_handlers(tmp_path):
    from src.assembled_core.events.store import EventStore
    from src.assembled_core.events.schema import make_market_tick
    from src.assembled_core.events.replayer import Replayer

    store = EventStore(tmp_path / "replay.db")
    for i in range(3):
        store.append(make_market_tick("sess5", i, "AAPL", 180.0, 180.05, 180.02, 100))
    r = Replayer(store)
    collected = []
    r.register_handler("market_tick_received", lambda e: collected.append(e))
    res = r.replay_session("sess5")
    assert len(collected) == 3
    assert res.n_events_replayed == 3
    assert res.duration_seconds is not None


def test_replayer_wildcard_handler(tmp_path):
    from src.assembled_core.events.store import EventStore
    from src.assembled_core.events.schema import make_market_tick, make_news_event
    from src.assembled_core.events.replayer import Replayer

    store = EventStore(tmp_path / "wild.db")
    store.append(make_market_tick("sess6", 1, "AAPL", 180.0, 180.05, 180.02, 100))
    store.append(make_news_event("sess6", 2, "headline", "AAPL", 0.5))
    r = Replayer(store)
    all_events = []
    r.register_handler("*", lambda e: all_events.append(e["event_type"]))
    r.replay_session("sess6")
    assert "market_tick_received" in all_events
    assert "news_received" in all_events


# ==========================================================================
# certify/schema.py + certify/generator.py  (43)
# ==========================================================================


def test_certificate_schema_roundtrip():
    from src.assembled_core.certify.schema import ReproducibilityCertificate

    cert = ReproducibilityCertificate(certificate_id="test-cert", notes="unit test")
    d = cert.to_dict()
    restored = ReproducibilityCertificate.from_dict(d)
    assert restored.certificate_id == "test-cert"
    assert restored.notes == "unit test"


def test_certificate_to_json():
    from src.assembled_core.certify.schema import ReproducibilityCertificate
    import json

    cert = ReproducibilityCertificate(certificate_id="abc")
    j = cert.to_json()
    d = json.loads(j)
    assert d["certificate_id"] == "abc"


def test_file_sha256_not_found():
    from src.assembled_core.certify.generator import file_sha256

    assert file_sha256("/nonexistent/path/abc.parquet") == "NOT_FOUND"


def test_file_sha256_actual_file(tmp_path):
    from src.assembled_core.certify.generator import file_sha256

    f = tmp_path / "test.txt"
    f.write_text("hello world")
    h = file_sha256(f)
    assert len(h) == 64
    assert h == file_sha256(f)  # deterministic


def test_object_sha256():
    from src.assembled_core.certify.generator import object_sha256

    h = object_sha256({"key": "value", "n": 42})
    assert len(h) == 64
    assert h == object_sha256({"n": 42, "key": "value"})  # sorted keys


def test_generate_certificate_empty():
    from src.assembled_core.certify.generator import generate_certificate

    cert = generate_certificate(notes="test run")
    assert cert.certificate_id != ""
    assert cert.notes == "test run"
    assert cert.environment.python_version != ""


def test_save_and_load_certificate(tmp_path):
    from src.assembled_core.certify.generator import (
        generate_certificate,
        save_certificate,
    )
    from src.assembled_core.certify.schema import ReproducibilityCertificate
    import json

    cert = generate_certificate(notes="save test")
    out = tmp_path / "cert.json"
    save_certificate(cert, out)
    assert out.exists()
    with open(out) as f:
        d = json.load(f)
    restored = ReproducibilityCertificate.from_dict(d)
    assert restored.certificate_id == cert.certificate_id


# ==========================================================================
# data/quality_gate.py  (37_DATA_QUALITY_GATE)
# ==========================================================================


def _make_ohlcv(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    close = 100.0 + rng.standard_normal(n).cumsum()
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, n))
    vol = rng.integers(100_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def test_quality_gate_passes_clean_data():
    from src.assembled_core.data.quality_gate import validate_ohlcv, QualityStatus

    df = _make_ohlcv(50)
    r = validate_ohlcv(df, ticker="AAPL")
    assert r.status in (QualityStatus.PASS, QualityStatus.WARN)


def test_quality_gate_fails_zero_price():
    from src.assembled_core.data.quality_gate import validate_ohlcv, QualityStatus

    df = _make_ohlcv(20)
    df.loc[df.index[5], "Close"] = 0.0
    r = validate_ohlcv(df, ticker="AAPL")
    assert r.status == QualityStatus.FAIL
    assert any("zero_price" in c for c in r.checks_failed)


def test_quality_gate_fails_negative_price():
    from src.assembled_core.data.quality_gate import validate_ohlcv, QualityStatus

    df = _make_ohlcv(20)
    df.loc[df.index[3], "Open"] = -1.0
    r = validate_ohlcv(df, ticker="MSFT")
    assert r.status == QualityStatus.FAIL


def test_quality_gate_fails_empty():
    from src.assembled_core.data.quality_gate import validate_ohlcv

    r = validate_ohlcv(pd.DataFrame(), ticker="EMPTY")
    assert r.blocked is True


def test_quality_gate_warns_null_price():
    from src.assembled_core.data.quality_gate import validate_ohlcv

    df = _make_ohlcv(20)
    df.loc[df.index[10], "Close"] = np.nan
    r = validate_ohlcv(df, ticker="GOOG")
    # null price → warning
    assert any("null" in c for c in r.checks_warned)


def test_quality_gate_fails_high_lt_low():
    from src.assembled_core.data.quality_gate import validate_ohlcv, QualityStatus

    df = _make_ohlcv(20)
    df.loc[df.index[2], "High"] = df.loc[df.index[2], "Low"] - 1.0
    r = validate_ohlcv(df, ticker="BAD")
    assert r.status == QualityStatus.FAIL
    assert any("high_lt_low" in c for c in r.checks_failed)


def test_quality_gate_result_metadata():
    from src.assembled_core.data.quality_gate import validate_ohlcv

    df = _make_ohlcv(30)
    r = validate_ohlcv(df, ticker="TEST")
    assert r.ticker == "TEST"
    assert r.n_rows == 30


def test_quality_gate_quarantine(tmp_path):
    from src.assembled_core.data.quality_gate import validate_ohlcv, QualityStatus

    df = _make_ohlcv(20)
    df.loc[df.index[0], "Close"] = 0.0
    r = validate_ohlcv(df, ticker="QUAR", quarantine_dir=str(tmp_path))
    assert r.status == QualityStatus.FAIL
    assert "quarantine_path" in r.metadata


# ---------------------------------------------------------------------------
# 38 — Feature Attribution (attribution module)
# ---------------------------------------------------------------------------


def test_composite_attribution_build():
    from src.assembled_core.attribution.composite import build_attribution

    attr = build_attribution(
        ticker="AAPL",
        composite_score=0.42,
        dimension_raw_scores={"mtf": 0.6, "news": 0.3, "classical_ta": 0.5},
        dimension_weights={"mtf": 0.15, "news": 0.10, "classical_ta": 0.20},
        regime="normal",
    )
    assert attr.ticker == "AAPL"
    assert abs(attr.composite_score - 0.42) < 1e-9
    assert "mtf" in attr.dimension_contributions
    assert abs(attr.dimension_contributions["mtf"] - 0.15 * 0.6) < 1e-9


def test_composite_attribution_top_contributors():
    from src.assembled_core.attribution.composite import build_attribution

    attr = build_attribution(
        ticker="MSFT",
        composite_score=0.30,
        dimension_raw_scores={"news": 0.9, "mtf": 0.5, "classical_ta": 0.1},
        dimension_weights={"news": 0.10, "mtf": 0.15, "classical_ta": 0.20},
        regime="elevated",
    )
    top = attr.top_contributors(n=2)
    assert len(top) == 2
    # mtf: 0.15*0.5=0.075, classical_ta: 0.20*0.1=0.02, news: 0.10*0.9=0.09
    # largest absolute: mtf(0.075) and news(0.09)
    assert set(top.keys()) == {"mtf", "news"}


def test_composite_attribution_to_dict():
    from src.assembled_core.attribution.composite import (
        build_attribution,
        attribution_to_dict,
    )

    attr = build_attribution(
        ticker="GOOG",
        composite_score=0.1,
        dimension_raw_scores={"news": 0.5},
        dimension_weights={"news": 0.10},
        regime="calm",
    )
    d = attribution_to_dict(attr)
    assert d["ticker"] == "GOOG"
    assert isinstance(d["timestamp"], str)


def test_attribution_store_save_load(tmp_path):
    from datetime import timezone
    from src.assembled_core.attribution.storage import AttributionStore
    from src.assembled_core.attribution.schemas import CompositeAttribution
    import datetime as dt

    store = AttributionStore(db_path=str(tmp_path / "attr.db"))
    attr = CompositeAttribution(
        timestamp=dt.datetime(2026, 4, 1, tzinfo=timezone.utc),
        ticker="AAPL",
        composite_score=0.7,
        dimension_contributions={"news": 0.07, "mtf": 0.09},
        dimension_raw_scores={"news": 0.7, "mtf": 0.6},
        dimension_weights={"news": 0.10, "mtf": 0.15},
        strategy_id="s1",
        model_version="v1",
        regime="normal",
    )
    store.save(attr)
    results = store.load_for_ticker("AAPL")
    assert len(results) == 1
    assert results[0].ticker == "AAPL"
    assert abs(results[0].composite_score - 0.7) < 1e-9


def test_attribution_store_date_filter(tmp_path):
    from datetime import timezone
    from src.assembled_core.attribution.storage import AttributionStore
    from src.assembled_core.attribution.schemas import CompositeAttribution
    import datetime as dt

    store = AttributionStore(db_path=str(tmp_path / "attr2.db"))
    for day in [1, 5, 10]:
        store.save(
            CompositeAttribution(
                timestamp=dt.datetime(2026, 4, day, tzinfo=timezone.utc),
                ticker="SPY",
                composite_score=0.1 * day,
                dimension_contributions={},
                dimension_raw_scores={},
                dimension_weights={},
                strategy_id="s1",
                model_version="v1",
                regime="normal",
            )
        )
    results = store.load_for_ticker(
        "SPY",
        start=dt.datetime(2026, 4, 3, tzinfo=timezone.utc),
        end=dt.datetime(2026, 4, 7, tzinfo=timezone.utc),
    )
    assert len(results) == 1
    assert abs(results[0].composite_score - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# 39 — Strategy Config + Experiment Tracker
# ---------------------------------------------------------------------------


def test_strategy_config_defaults():
    from src.assembled_core.strategy.config import StrategyConfig

    cfg = StrategyConfig(strategy_id="test_v1")
    assert cfg.strategy_id == "test_v1"
    total = sum(cfg.composite_weights.model_dump().values())
    assert 0.98 <= total <= 1.02


def test_strategy_config_from_dict():
    from src.assembled_core.strategy.config import StrategyConfig

    data = {
        "strategy_id": "trend_v2",
        "description": "test",
        "composite_weights": {
            "mtf": 0.15,
            "classical_ta": 0.20,
            "microstructure": 0.10,
            "volume_profile": 0.10,
            "chart_pattern": 0.05,
            "vol_surface": 0.10,
            "breadth": 0.15,
            "seasonality": 0.05,
            "news": 0.10,
        },
        "thresholds": {"buy": 0.55, "sell": -0.55},
        "risk": {
            "max_position_pct_of_equity": 0.04,
            "max_daily_loss_pct": 0.015,
            "kill_switch_loss_pct": 0.05,
        },
    }
    cfg = StrategyConfig.from_dict(data)
    assert cfg.thresholds.buy == 0.55
    assert cfg.risk.max_position_pct_of_equity == 0.04


def test_strategy_config_weights_must_sum_to_one():
    from pydantic import ValidationError
    from src.assembled_core.strategy.config import CompositeWeights

    with pytest.raises(ValidationError):
        CompositeWeights(
            mtf=0.50,
            classical_ta=0.50,
            microstructure=0.50,  # way over 1
            volume_profile=0.10,
            chart_pattern=0.05,
            vol_surface=0.10,
            breadth=0.15,
            seasonality=0.05,
            news=0.10,
        )


def test_strategy_config_to_dict():
    from src.assembled_core.strategy.config import StrategyConfig

    cfg = StrategyConfig(strategy_id="v3")
    d = cfg.to_dict()
    assert "strategy_id" in d
    assert "composite_weights" in d


def test_experiment_tracker_local(tmp_path):
    from src.assembled_core.strategy.experiment_tracker import start_run

    store = str(tmp_path / "runs")
    with start_run("unit_test_run", local_store_dir=store) as run:
        run.log_params({"lr": 0.001, "n_estimators": 200})
        run.log_metrics({"accuracy": 0.88})
        run.set_tag("env", "test")
    files = list((tmp_path / "runs").glob("*.json"))
    assert len(files) == 1
    import json

    data = json.loads(files[0].read_text())
    assert data["params"]["lr"] == 0.001
    assert data["metrics"]["accuracy"] == 0.88
    assert data["status"] == "FINISHED"


def test_experiment_tracker_logs_strategy_config(tmp_path):
    from src.assembled_core.strategy.experiment_tracker import (
        start_run,
        log_strategy_config,
    )
    from src.assembled_core.strategy.config import StrategyConfig

    cfg = StrategyConfig(strategy_id="cfg_test")
    store = str(tmp_path / "runs2")
    with start_run("cfg_log_test", local_store_dir=store) as run:
        log_strategy_config(run, cfg)
    import json

    files = list((tmp_path / "runs2").glob("*.json"))
    data = json.loads(files[0].read_text())
    assert "strategy_id" in data["params"]


# ---------------------------------------------------------------------------
# 50 — Tax lots (accounting/tax_lots.py)
# ---------------------------------------------------------------------------


def test_tax_lot_open_lot():
    from datetime import date, datetime, timezone
    from src.assembled_core.accounting.tax_lots import TaxLot

    lot = TaxLot.open_lot(
        "AAPL",
        qty=10,
        price_usd=150.0,
        usd_eur_rate=0.93,
        trade_date=date(2026, 1, 3),
        trade_timestamp=datetime(2026, 1, 3, 15, 0, tzinfo=timezone.utc),
    )
    assert lot.symbol == "AAPL"
    assert abs(lot.price_eur - 150.0 * 0.93) < 1e-9
    assert lot.status == "open"


def test_tax_lot_fifo_full_close():
    from datetime import date, datetime, timezone
    from src.assembled_core.accounting.tax_lots import TaxLot, match_fifo

    lot = TaxLot.open_lot(
        "MSFT",
        qty=5,
        price_usd=300.0,
        usd_eur_rate=0.93,
        trade_date=date(2026, 1, 5),
        trade_timestamp=datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc),
    )
    result = match_fifo(
        [lot],
        qty_to_close=5,
        exit_price_usd=310.0,
        usd_eur_rate=0.93,
        exit_date=date(2026, 2, 1),
    )
    assert result.qty_remaining == 0.0
    assert len(result.lots_closed) == 1
    # P&L = (310 - 300) * 5 * 0.93 = 46.5 EUR
    assert abs(result.total_pnl_eur - 46.5) < 0.01


def test_tax_lot_fifo_partial_close():
    from datetime import date, datetime, timezone
    from src.assembled_core.accounting.tax_lots import TaxLot, match_fifo

    lot = TaxLot.open_lot(
        "NVDA",
        qty=10,
        price_usd=500.0,
        usd_eur_rate=0.92,
        trade_date=date(2026, 1, 10),
        trade_timestamp=datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc),
    )
    result = match_fifo(
        [lot],
        qty_to_close=3,
        exit_price_usd=520.0,
        usd_eur_rate=0.92,
        exit_date=date(2026, 1, 20),
    )
    assert result.qty_remaining == 0.0
    assert abs(result.lots_closed[0]["qty"] - 3.0) < 1e-9


def test_tax_lot_store_roundtrip(tmp_path):
    from datetime import date, datetime, timezone
    from src.assembled_core.accounting.tax_lots import TaxLot, TaxLotStore

    store = TaxLotStore(db_path=str(tmp_path / "tax.db"))
    lot = TaxLot.open_lot(
        "AAPL",
        qty=10,
        price_usd=155.0,
        usd_eur_rate=0.93,
        trade_date=date(2026, 3, 1),
        trade_timestamp=datetime(2026, 3, 1, 15, 0, tzinfo=timezone.utc),
    )
    store.add_lot(lot)
    open_lots = store.open_lots_for("AAPL")
    assert len(open_lots) == 1
    assert open_lots[0].symbol == "AAPL"


def test_tax_lot_store_close_and_pnl(tmp_path):
    from datetime import date, datetime, timezone
    from src.assembled_core.accounting.tax_lots import TaxLot, TaxLotStore

    store = TaxLotStore(db_path=str(tmp_path / "tax2.db"))
    lot = TaxLot.open_lot(
        "GOOG",
        qty=2,
        price_usd=170.0,
        usd_eur_rate=0.93,
        trade_date=date(2026, 2, 1),
        trade_timestamp=datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc),
    )
    store.add_lot(lot)
    result = store.close_lots(
        "GOOG",
        qty_to_close=2,
        exit_price_usd=180.0,
        usd_eur_rate=0.93,
        exit_date=date(2026, 4, 1),
    )
    assert result.qty_remaining == 0.0
    pnl_2026 = store.realized_pnl_for_year(2026)
    assert pnl_2026 > 0  # sold higher than bought


# ---------------------------------------------------------------------------
# 60 — Signal Dispatcher (pipeline/dispatcher.py)
# ---------------------------------------------------------------------------


def test_dispatcher_legacy_mode():
    from src.assembled_core.pipeline.dispatcher import SignalDispatcher, Pipeline

    d = SignalDispatcher(Pipeline.LEGACY, legacy_fn=lambda x: x * 2)
    assert d.run(5) == 10


def test_dispatcher_modern_mode():
    from src.assembled_core.pipeline.dispatcher import SignalDispatcher, Pipeline

    d = SignalDispatcher(Pipeline.MODERN, registry=lambda x: x + 1)
    assert d.run(4) == 5


def test_dispatcher_shadow_returns_legacy():
    from src.assembled_core.pipeline.dispatcher import SignalDispatcher, Pipeline

    d = SignalDispatcher(
        Pipeline.SHADOW,
        legacy_fn=lambda x: {"score": x},
        registry=lambda x: {"score": x + 0.01},
        record_diffs=True,
    )
    result = d.run(0.5)
    assert result == {"score": 0.5}  # legacy wins


def test_dispatcher_shadow_records_divergence():
    from src.assembled_core.pipeline.dispatcher import SignalDispatcher, Pipeline

    d = SignalDispatcher(
        Pipeline.SHADOW,
        legacy_fn=lambda x: {"a": 1},
        registry=lambda x: {"a": 2},
        record_diffs=True,
    )
    d.run(None)
    assert d.divergence_rate() == 1.0


def test_dispatcher_shadow_no_divergence():
    from src.assembled_core.pipeline.dispatcher import SignalDispatcher, Pipeline

    d = SignalDispatcher(
        Pipeline.SHADOW,
        legacy_fn=lambda x: {"a": x},
        registry=lambda x: {"a": x},
        record_diffs=True,
    )
    d.run(42)
    assert d.divergence_rate() == 0.0


def test_dispatcher_promote_to_modern():
    from src.assembled_core.pipeline.dispatcher import SignalDispatcher, Pipeline

    d = SignalDispatcher(
        Pipeline.SHADOW, legacy_fn=lambda x: x, registry=lambda x: x, record_diffs=True
    )
    d.run(1)
    d.promote_to_modern()
    assert d.mode == Pipeline.MODERN
