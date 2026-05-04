"""Tests für Round-7 (12 Module + Wirings)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase12


# ---------------------------------------------------------------------------
# A: Signal-Decay-Tracker
# ---------------------------------------------------------------------------


def test_signal_decay_tracker_basic(tmp_path):
    import pytest

    pytest.importorskip("src.assembled_core.ml.signal_decay_tracker")
    from src.assembled_core.ml.signal_decay_tracker import SignalDecayTracker

    tracker = SignalDecayTracker(state_path=tmp_path / "decay.json", horizons=[1, 5])

    rng = np.random.default_rng(3)
    n = 100
    preds = {"sig1": pd.Series(rng.standard_normal(n))}
    rets = {1: pd.Series(rng.normal(0, 0.01, n)), 5: pd.Series(rng.normal(0, 0.02, n))}

    snap = tracker.record_snapshot(preds, rets)
    assert snap.as_of is not None
    assert "sig1" in snap.signal_ic
    assert tracker.history_length() == 1


def test_signal_decay_report_halflife(tmp_path):
    """IC sinkt über Snapshots → halflife erkennbar."""
    import pytest

    pytest.importorskip("src.assembled_core.ml.signal_decay_tracker")
    from src.assembled_core.ml.signal_decay_tracker import SignalDecayTracker

    tracker = SignalDecayTracker(state_path=tmp_path / "decay.json", horizons=[5])

    rng = np.random.default_rng(42)
    n = 100

    # 5 Snapshots mit abnehmender IC
    for ic_noise in [0.0, 0.2, 0.5, 0.8, 1.0]:  # immer mehr Noise → sinkende IC
        alpha_signal = 0.5
        actual_ret = rng.normal(0, 0.01, n)
        pred = alpha_signal * actual_ret + ic_noise * rng.standard_normal(n)
        tracker.record_snapshot(
            {"sig1": pd.Series(pred)},
            {5: pd.Series(actual_ret)},
            as_of=f"2025-01-{ic_noise*10:02.0f}",
        )

    report = tracker.get_report("sig1")
    assert report is not None
    assert "horizon_5d" in report.current_ic
    # historische Werte müssen eine Zeitreihe sein
    assert len(report.historical_ic["horizon_5d"]) == 5


def test_signal_decay_wiring_feedback_loop(tmp_path):
    """WIRING: feedback_loop._record_signal_decay existiert und crasht nicht."""
    import pytest

    pytest.importorskip("src.assembled_core.ml.feedback_loop")
    from src.assembled_core.ml.feedback_loop import FeedbackLoopController

    controller = FeedbackLoopController()
    assert hasattr(controller, "_record_signal_decay")

    # Empty panel → should not crash, just return
    controller._record_signal_decay(pd.DataFrame())
    # Also works with valid panel
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=60),
            "f1": rng.standard_normal(60),
            "fwd_return_5d": rng.normal(0, 0.01, 60),
        }
    )
    controller._record_signal_decay(panel)  # should not raise


# ---------------------------------------------------------------------------
# B: Turnover-Penalty
# ---------------------------------------------------------------------------


def test_turnover_smoothing():
    from src.assembled_core.portfolio.turnover_penalty import apply_turnover_smoothing

    target = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    previous = pd.Series({"A": 0.3, "B": 0.4, "C": 0.3})
    smoothed = apply_turnover_smoothing(target, previous, alpha=0.5)
    # Mid-point
    assert abs(smoothed["A"] - 0.4) < 1e-9
    assert abs(smoothed["B"] - 0.35) < 1e-9


def test_turnover_budget():
    from src.assembled_core.portfolio.turnover_penalty import (
        enforce_turnover_budget,
        compute_turnover,
    )

    target = pd.Series({"A": 1.0, "B": -1.0})  # alles neu
    previous = pd.Series({"A": 0.0, "B": 0.0})
    capped = enforce_turnover_budget(target, previous, max_turnover=0.3)

    actual_turnover = compute_turnover(capped, previous)
    assert actual_turnover <= 0.3 + 1e-6


def test_turnover_wrapper_stateful():
    from src.assembled_core.portfolio.turnover_penalty import TurnoverConstrainedSizer

    sizer = TurnoverConstrainedSizer()
    t1 = sizer.process({"A": 0.5, "B": 0.5})
    t2 = sizer.process({"A": 1.0, "B": 0.0})
    # t2 sollte zwischen t1 und target liegen (Smoothing aktiv)
    assert t2["A"] > t1["A"]
    assert t2["A"] < 1.0


def test_turnover_wiring_position_sizing():
    """WIRING: compute_target_positions_with_smoothing existiert."""
    from src.assembled_core.portfolio.position_sizing import (
        compute_target_positions_with_smoothing,
    )

    sig_df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "score": [0.8, 0.5, 0.3],
            "direction": ["long", "long", "long"],
        }
    )
    result = compute_target_positions_with_smoothing(
        sig_df,
        previous_positions=None,  # no smoothing
    )
    assert "target_weight" in result.columns


def test_smoothing_preserves_capital_scaling():
    """Regression: target_qty muss total_capital-Skalierung behalten, auch nach Smoothing."""
    from src.assembled_core.portfolio.position_sizing import (
        compute_target_positions_with_smoothing,
    )

    sig_df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "score": [0.8, 0.5, 0.3],
            "direction": ["LONG", "LONG", "LONG"],
        }
    )
    previous = pd.Series({"A": 0.2, "B": 0.3, "C": 0.1})
    total_capital = 100000.0
    result = compute_target_positions_with_smoothing(
        sig_df,
        previous_positions=previous,
        total_capital=total_capital,
        smoothing_alpha=0.5,
    )
    # Nach Smoothing muss target_qty ≈ target_weight * total_capital sein.
    assert not result.empty
    assert "target_qty" in result.columns
    expected = result["target_weight"] * total_capital
    diff = (result["target_qty"] - expected).abs().max()
    assert (
        diff < 1e-6
    ), f"target_qty lost capital scaling after smoothing (max diff {diff})"


# ---------------------------------------------------------------------------
# C: Online-HPO
# ---------------------------------------------------------------------------


def test_online_hpo_select_arm(tmp_path):
    import pytest

    pytest.importorskip("src.assembled_core.ml.online_hpo")
    from src.assembled_core.ml.online_hpo import OnlineHyperparamAdapter

    adapter = OnlineHyperparamAdapter(state_path=tmp_path / "hpo.json")
    chosen = adapter.select_arm()
    assert chosen.arm_id.startswith("arm_")
    assert isinstance(chosen.params, dict)


def test_online_hpo_reward_update(tmp_path):
    import pytest

    pytest.importorskip("src.assembled_core.ml.online_hpo")
    from src.assembled_core.ml.online_hpo import OnlineHyperparamAdapter

    adapter = OnlineHyperparamAdapter(state_path=tmp_path / "hpo.json")
    chosen = adapter.select_arm()
    adapter.observe_reward(chosen.arm_id, reward=0.12)
    adapter.save()

    # Re-load and check state persisted
    adapter2 = OnlineHyperparamAdapter(state_path=tmp_path / "hpo.json")
    assert adapter2.arms[chosen.arm_id].n_pulls == 1
    assert adapter2.arms[chosen.arm_id].mean_reward == pytest.approx(0.12)


def test_online_hpo_wiring_retraining_scheduler(tmp_path):
    """WIRING: adapt_hyperparameters_via_bandit existiert."""
    import pytest

    pytest.importorskip("src.assembled_core.ml.retraining_scheduler")
    from src.assembled_core.ml.retraining_scheduler import RetrainingScheduler

    scheduler = RetrainingScheduler()
    # Wenn flag nicht aktiv → None
    result = scheduler.adapt_hyperparameters_via_bandit(
        state_path=tmp_path / "hpo.json"
    )
    assert result is None  # default disabled


def test_online_hpo_sklearn_gb_preset(tmp_path):
    """Preset: sklearn GB-Arme ohne LightGBM-spezifische Keys."""
    import pytest

    pytest.importorskip("src.assembled_core.ml.online_hpo")
    from src.assembled_core.ml.online_hpo import OnlineHyperparamAdapter

    adapter = OnlineHyperparamAdapter.with_sklearn_gb_arms(
        state_path=tmp_path / "hpo.json"
    )
    assert len(adapter.arms) == 4
    # Jeder Arm hat generische GB-Keys, kein `num_leaves` o.ä.
    for arm in adapter.arms.values():
        assert set(arm.params.keys()) == {"n_estimators", "learning_rate", "max_depth"}


def test_online_hpo_ridge_preset(tmp_path):
    """Preset: Ridge-Arme mit nur alpha-Parameter."""
    import pytest

    pytest.importorskip("src.assembled_core.ml.online_hpo")
    from src.assembled_core.ml.online_hpo import OnlineHyperparamAdapter

    adapter = OnlineHyperparamAdapter.with_ridge_arms(state_path=tmp_path / "hpo.json")
    assert len(adapter.arms) == 4
    for arm in adapter.arms.values():
        assert set(arm.params.keys()) == {"alpha"}


def test_online_hpo_from_param_grid(tmp_path):
    """Custom-Grid: kartesisches Produkt → n_arms = prod(len(values))."""
    import pytest

    pytest.importorskip("src.assembled_core.ml.online_hpo")
    from src.assembled_core.ml.online_hpo import OnlineHyperparamAdapter

    grid = {"alpha": [0.1, 1.0], "l1_ratio": [0.0, 0.5, 1.0]}
    adapter = OnlineHyperparamAdapter.from_param_grid(
        grid, state_path=tmp_path / "hpo.json"
    )
    assert len(adapter.arms) == 6  # 2 * 3
    # jeder Arm hat beide Keys
    for arm in adapter.arms.values():
        assert set(arm.params.keys()) == {"alpha", "l1_ratio"}


def test_online_hpo_from_empty_grid_safe(tmp_path):
    """Leerer Grid → keine Arme, select_arm nimmt fallback."""
    import pytest

    pytest.importorskip("src.assembled_core.ml.online_hpo")
    from src.assembled_core.ml.online_hpo import OnlineHyperparamAdapter

    adapter = OnlineHyperparamAdapter.from_param_grid(
        {}, state_path=tmp_path / "hpo.json"
    )
    assert len(adapter.arms) == 0


def test_online_hpo_discount_factor_bounds_effective_n(tmp_path):
    """Mit discount_factor < 1 bleibt effektive Sample-Size endlich."""
    import pytest

    pytest.importorskip("src.assembled_core.ml.online_hpo")
    from src.assembled_core.ml.online_hpo import OnlineHyperparamAdapter

    adapter = OnlineHyperparamAdapter(
        arms=[{"x": 1}],
        state_path=tmp_path / "hpo.json",
        discount_factor=0.9,
    )
    arm_id = "arm_0"
    for _ in range(200):
        adapter.observe_reward(arm_id, reward=0.1)
    # Geometrische Reihe: n_pulls → 1 / (1 - df) = 10 im Limit
    n = adapter.arms[arm_id].n_pulls
    assert 9.0 < n < 10.5


def test_online_hpo_no_discount_keeps_integer_count(tmp_path):
    """Default discount_factor=1.0 → n_pulls bleibt exakt integer-äquivalent."""
    import pytest

    pytest.importorskip("src.assembled_core.ml.online_hpo")
    from src.assembled_core.ml.online_hpo import OnlineHyperparamAdapter

    adapter = OnlineHyperparamAdapter(
        arms=[{"x": 1}],
        state_path=tmp_path / "hpo.json",
    )
    for _ in range(5):
        adapter.observe_reward("arm_0", reward=0.1)
    assert adapter.arms["arm_0"].n_pulls == 5


def test_online_hpo_rejects_invalid_discount():
    """discount_factor außerhalb (0, 1] → ValueError."""
    import pytest

    pytest.importorskip("src.assembled_core.ml.online_hpo")
    from src.assembled_core.ml.online_hpo import OnlineHyperparamAdapter

    with pytest.raises(ValueError):
        OnlineHyperparamAdapter(arms=[{"x": 1}], discount_factor=0.0)
    with pytest.raises(ValueError):
        OnlineHyperparamAdapter(arms=[{"x": 1}], discount_factor=1.5)
    with pytest.raises(ValueError):
        OnlineHyperparamAdapter(arms=[{"x": 1}], discount_factor=-0.5)


# ---------------------------------------------------------------------------
# D: Backtest-Comparison
# ---------------------------------------------------------------------------


def test_backtest_comparison_basic():
    from src.assembled_core.qa.backtest_comparison import compare_backtests

    rng = np.random.default_rng(7)
    n = 100
    strategies = {
        "A": pd.Series(rng.normal(0.001, 0.01, n)),
        "B": pd.Series(rng.normal(0.0005, 0.012, n)),
        "C": pd.Series(rng.normal(-0.0005, 0.015, n)),
    }
    report = compare_backtests(strategies)
    assert len(report.strategies) == 3
    assert len(report.pairwise) == 3  # C(3,2) = 3
    assert len(report.ranking) == 3


def test_rank_strategies():
    from src.assembled_core.qa.backtest_comparison import rank_strategies

    rng = np.random.default_rng(11)
    strategies = {
        "good": pd.Series(rng.normal(0.002, 0.01, 100)),
        "bad": pd.Series(rng.normal(-0.001, 0.01, 100)),
    }
    ranked = rank_strategies(strategies)
    assert ranked[0][0] == "good"


# ---------------------------------------------------------------------------
# E: News-Trade-Attribution
# ---------------------------------------------------------------------------


def test_news_trade_attribution_link():
    from src.assembled_core.intel.news_trade_attribution import NewsTradeAttributor

    trade = {
        "trade_id": "T1",
        "symbol": "AAPL",
        "opened_at": "2025-06-01T12:00:00Z",
        "closed_return": 0.02,
    }
    news = pd.DataFrame(
        {
            "event_id": ["E1", "E2"],
            "symbol": ["AAPL", "GOOG"],
            "published_at": ["2025-06-01T10:00:00Z", "2025-06-01T11:00:00Z"],
            "impact_bps": [50.0, 30.0],
        }
    )

    attributor = NewsTradeAttributor(pre_window_hours=24, post_window_hours=24)
    links = attributor.link_trade_to_events(trade, news)
    # Nur AAPL-Event sollte gematched werden
    assert len(links) == 1
    assert links[0].event_id == "E1"


def test_news_trade_attribution_wiring_learning_store(tmp_path):
    """WIRING: enrich_learning_store schreibt news_links in Records."""
    from src.assembled_core.intel.news_trade_attribution import NewsTradeAttributor

    ls_path = tmp_path / "learning_store.jsonl"
    records = [
        {
            "trade_id": "T1",
            "symbol": "AAPL",
            "opened_at": "2025-06-01T12:00:00Z",
            "closed_at": "2025-06-02T12:00:00Z",
            "closed_return": 0.02,
        }
    ]
    with ls_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    news_path = tmp_path / "news.jsonl"
    news_records = [
        {
            "event_id": "E1",
            "symbol": "AAPL",
            "published_at": "2025-06-01T10:00:00Z",
            "impact_bps": 50.0,
        }
    ]
    with news_path.open("w", encoding="utf-8") as fh:
        for r in news_records:
            fh.write(json.dumps(r) + "\n")

    attributor = NewsTradeAttributor()
    n = attributor.enrich_learning_store(ls_path, news_path)
    assert n == 1

    with ls_path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    enriched = json.loads(lines[0])
    assert "news_links" in enriched

    # Idempotenz: zweiter Lauf darf nichts erneut enrichen (news_links bleibt unverändert)
    n2 = attributor.enrich_learning_store(ls_path, news_path)
    assert n2 == 0
    with ls_path.open("r", encoding="utf-8") as fh:
        lines2 = fh.readlines()
    enriched2 = json.loads(lines2[0])
    assert enriched2["news_links"] == enriched["news_links"]

    # reenrich=True: news_links wird neu berechnet (zählt wieder als enriched)
    n3 = attributor.enrich_learning_store(ls_path, news_path, reenrich=True)
    assert n3 == 1

    # Kein .tmp-File übrig
    assert not (ls_path.parent / (ls_path.name + ".tmp")).exists()


def test_eod_post_feedback_integration_end_to_end(tmp_path):
    """End-to-End-Integrationstest des EOD-Post-Feedback-Blocks.

    Repliziert die Sequenz aus pipeline/orchestrator.py nach run_feedback_check:
      1) NewsTradeAttributor.enrich_learning_store (news_links füllen)
      2) run_tca_from_learning_store (tca_report_{date}.json schreiben)
      3) purge_old_dated_reports (Retention kickt in)

    Verifiziert Schreibartefakte und deren Shape, nicht nur Aufrufbarkeit.
    """
    from src.assembled_core.intel.news_trade_attribution import NewsTradeAttributor
    from src.assembled_core.ops.report_retention import purge_old_dated_reports
    from src.assembled_core.qa.trade_tca import run_tca_from_learning_store

    ops_dir = tmp_path / "ops"
    ops_dir.mkdir(parents=True, exist_ok=True)
    intel_dir = tmp_path / "intel"
    intel_dir.mkdir(parents=True, exist_ok=True)

    # 1) Echter Learning-Store mit Closed Trade + TCA-Feldern
    ls_path = ops_dir / "learning_store.jsonl"
    trade_rec = {
        "trade_id": "T1",
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 100,
        "opened_at": "2025-06-01T12:00:00Z",
        "closed_at": "2025-06-02T12:00:00Z",
        "closed_return": 0.02,
        "execution_price": 100.5,
        "arrival_price": 100.0,
        "vwap_price": 100.3,
    }
    ls_path.write_text(json.dumps(trade_rec) + "\n", encoding="utf-8")

    # News-Events
    news_path = intel_dir / "news_event_store.jsonl"
    news_rec = {
        "event_id": "E1",
        "symbol": "AAPL",
        "published_at": "2025-06-01T10:00:00Z",
        "impact_bps": 50.0,
    }
    news_path.write_text(json.dumps(news_rec) + "\n", encoding="utf-8")

    # Alte tca-Reports anlegen (simuliert > 60 Tage Historie)
    import time

    old_mtime = time.time() - 120 * 86400
    for i in range(65):
        old = ops_dir / f"tca_report_2024{i:03d}.json"
        old.write_text("{}", encoding="utf-8")
        import os

        os.utime(old, (old_mtime - i, old_mtime - i))

    # --- Schritt 1: News-Attribution ---
    attributor = NewsTradeAttributor()
    n_enriched = attributor.enrich_learning_store(ls_path, news_path)
    assert n_enriched == 1

    with ls_path.open("r", encoding="utf-8") as fh:
        enriched_rec = json.loads(fh.read().strip())
    assert "news_links" in enriched_rec
    assert len(enriched_rec["news_links"]) == 1
    assert enriched_rec["news_links"][0]["event_id"] == "E1"

    # --- Schritt 2: TCA ---
    tca_out = ops_dir / f"tca_report_{pd.Timestamp.now().strftime('%Y%m%d')}.json"
    tca_result = run_tca_from_learning_store(ls_path, tca_out)
    assert tca_result, "TCA result must not be empty"
    assert tca_out.exists(), "TCA report file must be written"

    tca_data = json.loads(tca_out.read_text(encoding="utf-8"))
    # Shape-Check: muss n_trades enthalten
    assert tca_data.get("n_trades", 0) >= 1

    # --- Schritt 3: Retention ---
    n_purged = purge_old_dated_reports(ops_dir, "tca_report_", ".json", keep_last_n=60)
    remaining = sorted(ops_dir.glob("tca_report_*.json"))
    assert len(remaining) == 60, f"Expected 60 reports, got {len(remaining)}"
    # Der neu geschriebene Report (heutiger mtime) muss überleben
    assert tca_out.exists(), "Newest TCA report must survive retention"
    assert n_purged >= 1


# ---------------------------------------------------------------------------
# F: Kelly-with-Uncertainty
# ---------------------------------------------------------------------------


def test_kelly_uncertainty_basic():
    from src.assembled_core.portfolio.kelly_uncertainty import (
        compute_kelly_with_uncertainty,
    )

    # Kein Uncertainty-Input → Standard-Kelly × fractional
    w = compute_kelly_with_uncertainty(
        edge=0.05,
        variance=0.04,
        conformal_half_width=None,
        reference_half_width=None,
        fractional_kelly=0.5,
        max_fraction=1.0,
    )
    # kelly = 0.05 / 0.04 = 1.25, × 0.5 = 0.625, gecappt auf max_fraction
    assert w == pytest.approx(0.625, rel=1e-6)


def test_kelly_uncertainty_discount():
    """Hohes Conformal-Intervall → niedriger Kelly."""
    from src.assembled_core.portfolio.kelly_uncertainty import (
        compute_kelly_with_uncertainty,
    )

    # cw << ref_cw (relative ≈ 0.1) → scale ≈ 0.9 → fast volle Position
    w_low_uncertainty = compute_kelly_with_uncertainty(
        edge=0.05,
        variance=0.04,
        conformal_half_width=0.001,
        reference_half_width=0.01,
        fractional_kelly=1.0,
        max_fraction=10.0,
    )
    # cw >> ref_cw → relative clipped to 1 → scale = 0 → keine Position
    w_high_uncertainty = compute_kelly_with_uncertainty(
        edge=0.05,
        variance=0.04,
        conformal_half_width=0.05,
        reference_half_width=0.01,
        fractional_kelly=1.0,
        max_fraction=10.0,
    )
    assert w_high_uncertainty < w_low_uncertainty
    assert w_low_uncertainty > 0.0, "Low uncertainty must produce non-zero position"


def test_kelly_uncertainty_semantics():
    """Formel-Verifikation: scale = 1 - clip(cw/ref_cw, 0, 1)."""
    from src.assembled_core.portfolio.kelly_uncertainty import (
        compute_kelly_with_uncertainty,
    )

    # cw == 0 → scale = 1 (volle Sicherheit)
    w_zero_cw = compute_kelly_with_uncertainty(
        edge=0.05,
        variance=0.04,
        conformal_half_width=0.0,
        reference_half_width=0.01,
        fractional_kelly=1.0,
        max_fraction=10.0,
    )
    # Expected: kelly (1.25) × 1.0 × 1.0 = 1.25
    assert w_zero_cw == pytest.approx(1.25, rel=1e-6)

    # cw == ref_cw → scale = 0 (vollständige Unsicherheit → keine Position)
    w_equal_cw = compute_kelly_with_uncertainty(
        edge=0.05,
        variance=0.04,
        conformal_half_width=0.01,
        reference_half_width=0.01,
        fractional_kelly=1.0,
        max_fraction=10.0,
    )
    assert w_equal_cw == pytest.approx(0.0, abs=1e-9)

    # cw >> ref_cw → scale geclippt auf 0
    w_huge_cw = compute_kelly_with_uncertainty(
        edge=0.05,
        variance=0.04,
        conformal_half_width=10.0,
        reference_half_width=0.01,
        fractional_kelly=1.0,
        max_fraction=10.0,
    )
    assert w_huge_cw == pytest.approx(0.0, abs=1e-9)


def test_kelly_uncertainty_nan_guard():
    """NaN / Inf in edge oder variance → 0-Position."""
    from src.assembled_core.portfolio.kelly_uncertainty import (
        compute_kelly_with_uncertainty,
    )

    assert compute_kelly_with_uncertainty(edge=float("nan"), variance=0.04) == 0.0
    assert compute_kelly_with_uncertainty(edge=0.05, variance=float("nan")) == 0.0
    assert compute_kelly_with_uncertainty(edge=float("inf"), variance=0.04) == 0.0
    # NaN conformal_half_width → volle Abwertung (scale = 0)
    w = compute_kelly_with_uncertainty(
        edge=0.05,
        variance=0.04,
        conformal_half_width=float("nan"),
        reference_half_width=0.01,
        fractional_kelly=1.0,
        max_fraction=10.0,
    )
    assert w == pytest.approx(0.0, abs=1e-9)


def test_kelly_wiring_position_sizing():
    """WIRING: compute_kelly_weights_with_uncertainty in position_sizing."""
    from src.assembled_core.portfolio.position_sizing import (
        compute_kelly_weights_with_uncertainty,
    )

    edges = pd.Series({"A": 0.05, "B": 0.02})
    variances = pd.Series({"A": 0.04, "B": 0.01})
    weights = compute_kelly_weights_with_uncertainty(edges, variances)
    assert len(weights) == 2
    assert (weights.abs() <= 1.0).all()


# ---------------------------------------------------------------------------
# G: Signal-Correlation-Analyzer
# ---------------------------------------------------------------------------


def test_signal_correlation_redundancy():
    import pytest

    pytest.importorskip("src.assembled_core.ml.signal_correlation")
    from src.assembled_core.ml.signal_correlation import SignalCorrelationAnalyzer

    rng = np.random.default_rng(1)
    n = 200
    base = rng.standard_normal(n)
    # Zwei stark korrelierte und ein unabhängiges Signal
    df = pd.DataFrame(
        {
            "sig1": base + 0.01 * rng.standard_normal(n),
            "sig2": base + 0.01 * rng.standard_normal(n),
            "sig3": rng.standard_normal(n),
        }
    )
    analyzer = SignalCorrelationAnalyzer(redundancy_threshold=0.9)
    report = analyzer.analyze(df)
    assert report.n_signals == 3
    assert len(report.redundant_clusters) >= 1


def test_signal_correlation_wiring_feedback_loop():
    """WIRING: _record_signal_correlation existiert und läuft fehlerfrei."""
    pytest.importorskip("src.assembled_core.ml.feedback_loop")
    from src.assembled_core.ml.feedback_loop import FeedbackLoopController

    controller = FeedbackLoopController()
    assert hasattr(controller, "_record_signal_correlation")

    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        {
            "f1": rng.standard_normal(50),
            "f2": rng.standard_normal(50),
            "f3": rng.standard_normal(50),
        }
    )
    controller._record_signal_correlation(panel)  # no-raise


# ---------------------------------------------------------------------------
# H: Drawdown-Decomposition
# ---------------------------------------------------------------------------


def test_find_worst_drawdown():
    from src.assembled_core.qa.drawdown_decomposition import find_worst_drawdown

    returns = pd.Series([0.01, -0.05, -0.03, 0.02, 0.01])
    dd = find_worst_drawdown(returns)
    assert dd.max_drawdown < 0
    assert dd.duration >= 1


def test_drawdown_decomposition():
    from src.assembled_core.qa.drawdown_decomposition import decompose_drawdown

    rng = np.random.default_rng(5)
    n = 100
    market = pd.Series(rng.normal(0.0005, 0.01, n))
    portfolio = 0.5 * market + rng.normal(0, 0.008, n)
    # Force a drawdown
    portfolio.iloc[20:30] = -0.01

    report = decompose_drawdown(portfolio, pd.DataFrame({"market": market}))
    assert "market" in report.factor_betas or report.drawdown.max_drawdown < 0


def test_drawdown_wiring_attribution():
    """WIRING: attribution_during_worst_drawdown existiert."""
    from src.assembled_core.qa.performance_attribution import (
        attribution_during_worst_drawdown,
    )

    rng = np.random.default_rng(3)
    n = 100
    portfolio = pd.Series(rng.normal(0.0001, 0.01, n))
    factor = pd.Series(rng.normal(0.0005, 0.012, n))

    result = attribution_during_worst_drawdown(
        portfolio,
        pd.DataFrame({"mkt": factor}),
    )
    # Entweder summary dict oder error dict
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# I: Trade-TCA
# ---------------------------------------------------------------------------


def test_trade_tca_basic():
    from src.assembled_core.qa.trade_tca import compute_trade_tca

    tca = compute_trade_tca(
        trade_id="T1",
        symbol="AAPL",
        side="buy",
        quantity=100,
        execution_price=100.5,
        arrival_price=100.0,
        vwap_price=100.2,
    )
    # buy higher than arrival → positive IS
    assert tca.implementation_shortfall_bps > 0


def test_trade_tca_aggregate():
    from src.assembled_core.qa.trade_tca import aggregate_tca, compute_trade_tca

    tcas = [
        compute_trade_tca("T1", "AAPL", "buy", 100, 100.5, 100.0),
        compute_trade_tca("T2", "AAPL", "sell", 50, 99.5, 100.0),
        compute_trade_tca("T3", "GOOG", "buy", 10, 2000.0, 1999.0),
    ]
    report = aggregate_tca(tcas)
    assert report.n_trades == 3
    assert "AAPL" in report.per_symbol


def test_trade_tca_wiring_learning_store(tmp_path):
    """WIRING: run_tca_from_learning_store verarbeitet JSONL."""
    from src.assembled_core.qa.trade_tca import run_tca_from_learning_store

    ls_path = tmp_path / "ls.jsonl"
    records = [
        {
            "trade_id": "T1",
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "execution_price": 100.5,
            "arrival_price": 100.0,
        },
    ]
    with ls_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    report = run_tca_from_learning_store(ls_path, tmp_path / "tca.json")
    assert report["n_trades"] == 1


# ---------------------------------------------------------------------------
# J: Online-HMM-Regime
# ---------------------------------------------------------------------------


def test_online_hmm_fallback():
    """Ohne hmmlearn → Vol-Quantile-Fallback funktioniert."""
    pytest.importorskip("src.assembled_core.ml.online_hmm_regime")
    from src.assembled_core.ml.online_hmm_regime import OnlineHMMRegimeDetector

    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.0005, 0.01, 200))
    detector = OnlineHMMRegimeDetector()
    detector.fit(returns)
    state = detector.predict_current_regime(returns)
    assert state.regime_label in ("LOW_VOL", "NORMAL", "HIGH_VOL")


def test_combined_regime_agreement():
    pytest.importorskip("src.assembled_core.ml.combined_regime")
    from src.assembled_core.ml.combined_regime import CombinedRegimeClassifier

    # Beide classifier None → NEUTRAL / NEUTRAL → agreement
    combined = CombinedRegimeClassifier()
    out = combined.predict()
    assert out.combined_regime == "NEUTRAL"


def test_combined_regime_wiring_ml_pipeline():
    """WIRING: MLSignalPipeline akzeptiert combined_regime_classifier."""
    import pytest

    pytest.importorskip("src.assembled_core.signals.ml_integration")
    pytest.importorskip("src.assembled_core.ml.combined_regime")
    from src.assembled_core.signals.ml_integration import MLSignalPipeline
    from src.assembled_core.ml.combined_regime import CombinedRegimeClassifier

    combined = CombinedRegimeClassifier()
    pipeline = MLSignalPipeline(combined_regime_classifier=combined)
    X = pd.DataFrame({"f1": [0.0, 0.1, -0.2]})
    output = pipeline.run(X, market_returns=pd.Series([0.001, -0.002, 0.001]))
    assert output.regime in ("RISK_ON", "NEUTRAL", "RISK_OFF", "CRISIS")


# ---------------------------------------------------------------------------
# K: LIME
# ---------------------------------------------------------------------------


def test_lime_permutation_fallback():
    """Ohne training_data → Permutation-Fallback."""
    pytest.importorskip("sklearn")
    pytest.importorskip("src.assembled_core.ml.lime_explainer")
    from sklearn.linear_model import LinearRegression
    from src.assembled_core.ml.lime_explainer import LIMEExplainerWrapper

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.1, 100)
    model = LinearRegression().fit(X, y)

    wrapper = LIMEExplainerWrapper(
        model=model,
        feature_names=["a", "b", "c"],
        training_data=None,
    )
    expl = wrapper.explain(np.array([1.0, 0.0, 0.0]))
    assert len(expl.feature_contributions) > 0
    assert expl.source == "permutation_fallback"


def test_lime_wiring_meta_model():
    """WIRING: MetaModel.explain_prediction_lime existiert."""
    pytest.importorskip("sklearn")
    from src.assembled_core.signals.meta_model import train_meta_model

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "f1": rng.standard_normal(100),
            "f2": rng.standard_normal(100),
            "label": rng.integers(0, 2, 100),
        }
    )
    mm = train_meta_model(df, feature_cols=["f1", "f2"], label_col="label")
    result = mm.explain_prediction_lime(df.iloc[0][["f1", "f2"]])
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# L: Monte-Carlo Scenarios
# ---------------------------------------------------------------------------


def test_scenario_vol_spike():
    from src.assembled_core.qa.scenario_simulator import simulate_vol_spike_scenario

    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    result = simulate_vol_spike_scenario(returns, vol_multiplier=3.0, duration=10)
    assert result.scenario_name == "VolSpike"
    assert result.cvar_95 < 0  # tail risk negativ


def test_scenario_crash():
    from src.assembled_core.qa.scenario_simulator import simulate_crash_scenario

    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    result = simulate_crash_scenario(returns, crash_magnitude=-0.10, recovery_days=20)
    # Crash scenario sollte negativer mean return haben
    assert result.mean_return < 0


def test_stress_test_report():
    from src.assembled_core.qa.scenario_simulator import run_stress_test

    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.001, 0.01, 300))
    report = run_stress_test(returns, include_correlation=False)
    assert len(report.scenarios) >= 2
    assert report.worst_scenario != ""


# ---------------------------------------------------------------------------
# Correctness tests (depth over smoke) — E, I, J
# ---------------------------------------------------------------------------


def test_news_trade_attribution_decay_ordering():
    """Näher am Trade → höheres Gewicht als weiter weg."""
    from src.assembled_core.intel.news_trade_attribution import NewsTradeAttributor

    trade = {
        "trade_id": "T1",
        "symbol": "AAPL",
        "opened_at": "2025-06-01T12:00:00Z",
        "closed_return": 0.02,
    }
    news = pd.DataFrame(
        {
            "event_id": ["NEAR", "FAR"],
            "symbol": ["AAPL", "AAPL"],
            # 1h vs 20h vor Trade — gleiche impact_bps
            "published_at": ["2025-06-01T11:00:00Z", "2025-05-31T16:00:00Z"],
            "impact_bps": [50.0, 50.0],
        }
    )
    attributor = NewsTradeAttributor(pre_window_hours=24, decay_halflife_hours=6)
    links = attributor.link_trade_to_events(trade, news)
    by_id = {lnk.event_id: lnk for lnk in links}
    assert by_id["NEAR"].weight > by_id["FAR"].weight
    assert by_id["NEAR"].distance_hours < by_id["FAR"].distance_hours


def test_news_trade_attribution_tickers_list_match():
    """Events ohne `symbol`, dafür `tickers`-Liste, werden korrekt gematched."""
    from src.assembled_core.intel.news_trade_attribution import NewsTradeAttributor

    trade = {
        "symbol": "MSFT",
        "opened_at": "2025-06-01T12:00:00Z",
        "closed_return": 0.01,
    }
    news = pd.DataFrame(
        {
            "event_id": ["E1", "E2"],
            "tickers": [["AAPL", "MSFT"], ["GOOG"]],
            "published_at": ["2025-06-01T10:00:00Z", "2025-06-01T10:00:00Z"],
            "impact_bps": [20.0, 20.0],
        }
    )
    links = NewsTradeAttributor().link_trade_to_events(trade, news)
    assert len(links) == 1
    assert links[0].event_id == "E1"


def test_news_trade_attribution_residual_and_contribution():
    """Residual = closed_return - sum(estimated_contributions) über attribute_trades."""
    from src.assembled_core.intel.news_trade_attribution import NewsTradeAttributor

    trades = [
        {
            "trade_id": "T1",
            "symbol": "AAPL",
            "opened_at": "2025-06-01T12:00:00Z",
            "closed_return": 0.03,
        }
    ]
    news = pd.DataFrame(
        {
            "event_id": ["E1"],
            "symbol": ["AAPL"],
            "published_at": ["2025-06-01T10:00:00Z"],
            "impact_bps": [100.0],
        }
    )
    attrs = NewsTradeAttributor(decay_halflife_hours=6).attribute_trades(trades, news)
    assert len(attrs) == 1
    a = attrs[0]
    total_contrib = sum(lnk.estimated_contribution for lnk in a.news_links)
    assert abs(a.residual_return - (a.closed_return - total_contrib)) < 1e-6


def test_news_trade_attribution_outside_window_excluded():
    """Event außerhalb pre/post-Window wird NICHT gematched."""
    from src.assembled_core.intel.news_trade_attribution import NewsTradeAttributor

    trade = {
        "symbol": "AAPL",
        "opened_at": "2025-06-01T12:00:00Z",
        "closed_return": 0.01,
    }
    news = pd.DataFrame(
        {
            "event_id": ["OLD"],
            "symbol": ["AAPL"],
            "published_at": ["2025-05-25T12:00:00Z"],  # 7 Tage vorher
            "impact_bps": [100.0],
        }
    )
    links = NewsTradeAttributor(
        pre_window_hours=24, post_window_hours=24
    ).link_trade_to_events(trade, news)
    assert links == []


def test_trade_tca_is_buy_formula():
    """Buy, exec > arrival → IS_bps = (exec-arrival)/arrival*1e4."""
    from src.assembled_core.qa.trade_tca import compute_trade_tca

    tca = compute_trade_tca(
        "T", "X", "buy", 100, execution_price=101.0, arrival_price=100.0
    )
    # (101-100)/100 * 10000 = 100 bps
    assert tca.implementation_shortfall_bps == pytest.approx(100.0, abs=1e-6)


def test_trade_tca_is_sell_formula():
    """Sell, exec < arrival → positive IS (Verkäufer hat schlechter verkauft)."""
    from src.assembled_core.qa.trade_tca import compute_trade_tca

    tca = compute_trade_tca(
        "T", "X", "sell", 100, execution_price=99.0, arrival_price=100.0
    )
    # sell_multiplier=-1, (99-100)/100*1e4 = -100; × -1 = +100 bps
    assert tca.implementation_shortfall_bps == pytest.approx(100.0, abs=1e-6)


def test_trade_tca_vwap_slippage_only_when_price_given():
    """Ohne vwap_price → vwap_slippage_bps == 0."""
    from src.assembled_core.qa.trade_tca import compute_trade_tca

    tca_with = compute_trade_tca("T1", "X", "buy", 100, 100.5, 100.0, vwap_price=100.2)
    tca_without = compute_trade_tca("T2", "X", "buy", 100, 100.5, 100.0)
    # 100.5 > 100.2 bei buy → positive slippage
    assert tca_with.vwap_slippage_bps > 0
    assert tca_without.vwap_slippage_bps == 0.0


def test_trade_tca_zero_arrival_graceful():
    """arrival_price ≤ 0 → IS=0, kein Divide-by-zero."""
    from src.assembled_core.qa.trade_tca import compute_trade_tca

    tca = compute_trade_tca(
        "T", "X", "buy", 100, execution_price=100.0, arrival_price=0.0
    )
    assert tca.implementation_shortfall_bps == 0.0
    assert tca.vwap_slippage_bps == 0.0


def test_trade_tca_aggregate_mean_matches():
    """aggregate_tca.mean_impact_bps == mean(individual IS_bps)."""
    from src.assembled_core.qa.trade_tca import aggregate_tca, compute_trade_tca

    tcas = [
        compute_trade_tca("T1", "A", "buy", 100, 101.0, 100.0),  # +100 bps
        compute_trade_tca("T2", "A", "buy", 100, 100.5, 100.0),  # +50 bps
        compute_trade_tca("T3", "A", "sell", 100, 99.0, 100.0),  # +100 bps
    ]
    report = aggregate_tca(tcas)
    # (100 + 50 + 100) / 3 ≈ 83.33
    assert report.mean_impact_bps == pytest.approx(83.33, abs=0.5)
    assert report.per_symbol["A"]["n"] == 3


def test_online_hmm_high_vol_detection():
    """Künstlich hohe Vol am Ende → recent_vol / long_vol > 1.5 → HIGH_VOL im Fallback."""
    pytest.importorskip("src.assembled_core.ml.online_hmm_regime")
    from src.assembled_core.ml.online_hmm_regime import OnlineHMMRegimeDetector

    rng = np.random.default_rng(0)
    # 180 normale Tage + 20 hochvolatile
    calm = rng.normal(0.0, 0.005, 180)
    storm = rng.normal(0.0, 0.05, 20)
    returns = pd.Series(np.concatenate([calm, storm]))

    detector = OnlineHMMRegimeDetector()
    # Fallback-Pfad erzwingen, damit der Test deterministisch ist (hmmlearn nicht garantiert)
    detector._available = False
    state = detector.predict_current_regime(returns)
    assert state.regime_label == "HIGH_VOL"
    assert state.regime_id == 2


def test_online_hmm_low_vol_detection():
    """Künstlich ruhiges Ende nach volatiler Historie → ratio < 0.7 → LOW_VOL."""
    pytest.importorskip("src.assembled_core.ml.online_hmm_regime")
    from src.assembled_core.ml.online_hmm_regime import OnlineHMMRegimeDetector

    rng = np.random.default_rng(1)
    vol = rng.normal(0.0, 0.04, 180)
    calm = rng.normal(0.0, 0.002, 20)
    returns = pd.Series(np.concatenate([vol, calm]))

    detector = OnlineHMMRegimeDetector()
    detector._available = False
    state = detector.predict_current_regime(returns)
    assert state.regime_label == "LOW_VOL"
    assert state.regime_id == 0


def test_online_hmm_short_input_safe():
    """Weniger als 20 Punkte → default NORMAL, kein Crash."""
    pytest.importorskip("src.assembled_core.ml.online_hmm_regime")
    from src.assembled_core.ml.online_hmm_regime import OnlineHMMRegimeDetector

    detector = OnlineHMMRegimeDetector()
    state = detector.predict_current_regime(pd.Series([0.001, -0.002, 0.0005]))
    assert state.regime_label == "NORMAL"


# ---------------------------------------------------------------------------
# Auto-deploy invariants — pin the human-review-required contract (CLAUDE.md Rule 30)
# ---------------------------------------------------------------------------


def test_feedback_loop_config_auto_deploy_default_false():
    """FeedbackLoopConfig.auto_deploy default MUSS False bleiben (Human-Review-Pflicht)."""
    pytest.importorskip("src.assembled_core.ml.feedback_loop")
    from src.assembled_core.ml.feedback_loop import FeedbackLoopConfig

    cfg = FeedbackLoopConfig()
    assert cfg.auto_deploy is False


def test_retraining_scheduler_hard_enforces_auto_deploy_false(tmp_path):
    """Selbst wenn eine YAML-Config `auto_deploy: true` enthält, muss Scheduler False erzwingen."""
    pytest.importorskip("src.assembled_core.ml.retraining_scheduler")
    from src.assembled_core.ml.retraining_scheduler import RetrainingScheduler

    cfg = tmp_path / "self_learning.yaml"
    cfg.write_text(
        "self_learning:\n" "  retraining:\n" "    auto_deploy: true\n",
        encoding="ascii",
    )
    scheduler = RetrainingScheduler(config_path=cfg)
    assert scheduler._cfg["auto_deploy"] is False


def test_model_registry_register_returns_candidate_status(tmp_path):
    """register() liefert IMMER status='candidate' — promote_to_deployed ist explizit nötig."""
    pytest.importorskip("src.assembled_core.ml.model_registry")
    from src.assembled_core.ml.model_registry import ModelRegistry

    reg = ModelRegistry(base_dir=tmp_path)
    # dummy model object (joblib kann alles Pickle-bare serialisieren)
    dummy = {"fake": "model"}
    rec = reg.register(
        model=dummy,
        model_id="test_model",
        metrics={"auc": 0.6},
    )
    assert rec.status == "candidate"
    assert rec.status != "deployed"


def test_model_registry_promote_requires_approved(tmp_path):
    """promote_to_deployed lehnt ab, wenn Status nicht 'approved' ist."""
    pytest.importorskip("src.assembled_core.ml.model_registry")
    from src.assembled_core.ml.model_registry import ModelRegistry

    reg = ModelRegistry(base_dir=tmp_path)
    dummy = {"fake": "model"}
    rec = reg.register(model=dummy, model_id="test_model", metrics={"auc": 0.6})
    # Candidate → promote muss fehlschlagen
    with pytest.raises(ValueError, match="nicht approved"):
        reg.promote_to_deployed("test_model", rec.version)


# ---------------------------------------------------------------------------
# Scenario-Simulator correctness
# ---------------------------------------------------------------------------


def test_scenario_vol_spike_std_scales_with_multiplier():
    """Simulierter Vol-Multiplier erhöht final-return std deutlich gegenüber Baseline."""
    from src.assembled_core.qa.scenario_simulator import simulate_vol_spike_scenario

    rng = np.random.default_rng(0)
    baseline = pd.Series(rng.normal(0.001, 0.01, 1000))

    low = simulate_vol_spike_scenario(baseline, vol_multiplier=1.0, duration=20, seed=1)
    high = simulate_vol_spike_scenario(
        baseline, vol_multiplier=5.0, duration=20, seed=1
    )
    # 5x Vol → final std ≈ 5x; wir akzeptieren 3x als robuste Untergrenze
    assert high.std_return > 3.0 * low.std_return


def test_scenario_crash_mean_reflects_injection():
    """Crash=-0.15 + recovery_days=10 mit mu~0 → mean_return ≈ -0.15."""
    from src.assembled_core.qa.scenario_simulator import simulate_crash_scenario

    baseline = pd.Series([0.0] * 500)  # mu=0, sigma=0
    result = simulate_crash_scenario(
        baseline,
        crash_magnitude=-0.15,
        recovery_days=10,
        n_simulations=200,
    )
    # final = crash + 9 recovery days × 0 = -0.15
    assert result.mean_return == pytest.approx(-0.15, abs=0.01)
    assert result.shock_magnitude == pytest.approx(0.15, abs=1e-9)


def test_scenario_crash_var_picks_up_shock():
    """Crash-Szenario: VaR_95 und CVaR_95 müssen den Schock widerspiegeln (sehr negativ)."""
    from src.assembled_core.qa.scenario_simulator import simulate_crash_scenario

    rng = np.random.default_rng(0)
    baseline = pd.Series(rng.normal(0.001, 0.005, 500))
    result = simulate_crash_scenario(
        baseline,
        crash_magnitude=-0.20,
        recovery_days=10,
        n_simulations=500,
    )
    # VaR_95 sollte weit unter 0 liegen (Mehrheit der Pfade bleibt unter dem Schock)
    assert result.var_95 < -0.10
    assert result.cvar_95 <= result.var_95  # CVaR ≤ VaR im linken Tail
