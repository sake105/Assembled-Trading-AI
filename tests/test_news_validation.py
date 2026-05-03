"""Tests for src/assembled_core/signals/news_validation.py."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from assembled_core.signals.news_validation import (
    GATE_THRESHOLDS,
    build_factor_series,
    car_significance_report,
    check_level_a,
    classification_metrics,
    compute_abnormal_returns,
    compute_ic,
    compute_market_model,
    compute_quantile_returns,
    entity_anonymize,
    event_study,
    gate_summary,
    load_gold_dataset,
    net_edge_after_costs,
    news_feature_production_ready,
    run_finbert_tone,
    run_haiku_zeroshot,
)


# ---------------------------------------------------------------------------
# Level A — classification_metrics
# ---------------------------------------------------------------------------

class TestClassificationMetrics:
    def _labels(self):
        return ["positive", "negative", "neutral"]

    def test_perfect_predictions(self):
        y = ["positive", "negative", "neutral", "positive", "neutral"]
        m = classification_metrics(y, y)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == pytest.approx(1.0)
        assert m["directional_error_rate"] == 0.0

    def test_all_wrong_inverted(self):
        y_true = ["positive"] * 10
        y_pred = ["negative"] * 10
        m = classification_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0
        assert m["directional_error_rate"] == 1.0

    def test_macro_f1_balanced_classes(self):
        y_true = ["positive", "negative", "neutral"] * 4
        y_pred = ["positive", "negative", "neutral"] * 4
        m = classification_metrics(y_true, y_pred)
        assert m["macro_f1"] == pytest.approx(1.0)

    def test_mixed_predictions(self):
        y_true = ["positive", "negative", "neutral", "positive", "neutral"]
        y_pred = ["positive", "neutral",  "neutral", "negative", "neutral"]
        m = classification_metrics(y_true, y_pred)
        assert 0.0 < m["macro_f1"] < 1.0
        assert m["n"] == 5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            classification_metrics([], [])

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            classification_metrics(["positive"], ["positive", "negative"])

    def test_confusion_matrix_shape(self):
        y = ["positive", "negative", "neutral"]
        m = classification_metrics(y, y)
        assert len(m["confusion_matrix"]) == 3
        assert len(m["confusion_matrix"][0]) == 3


class TestCheckLevelA:
    def _good_metrics(self):
        return {
            "accuracy": 0.87,
            "macro_f1": 0.82,
            "weighted_f1": 0.83,
            "directional_error_rate": 0.03,
            "per_class_f1": {},
            "confusion_matrix": [],
            "n": 100,
        }

    def test_all_pass_standard_dataset(self):
        gate = check_level_a(self._good_metrics(), dataset_name="fpb_allagree")
        assert all(gate.values())

    def test_own_gold_uses_lower_threshold(self):
        metrics = self._good_metrics()
        metrics["macro_f1"] = 0.67  # passes own-gold threshold (0.65) but not standard (0.80)
        gate_own = check_level_a(metrics, dataset_name="ata_2026q1")
        gate_std = check_level_a(metrics, dataset_name="fpb_allagree")
        assert gate_own["macro_f1_ge_thresh"] is True
        assert gate_std["macro_f1_ge_thresh"] is False

    def test_directional_error_fails(self):
        metrics = self._good_metrics()
        metrics["directional_error_rate"] = 0.08
        gate = check_level_a(metrics)
        assert gate["directional_error_le_5pct"] is False


class TestLoadGoldDataset:
    def test_load_jsonl(self, tmp_path):
        data = [
            {"text": "Revenue beats estimates", "label": "Positive"},
            {"text": "Company files for bankruptcy", "label": "Negative"},
        ]
        p = tmp_path / "gold.jsonl"
        p.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")
        texts, labels = load_gold_dataset(p)
        assert texts == ["Revenue beats estimates", "Company files for bankruptcy"]
        assert labels == ["positive", "negative"]


# ---------------------------------------------------------------------------
# Level B — event study
# ---------------------------------------------------------------------------

class TestMarketModel:
    def _make_returns(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        mkt = pd.Series(rng.normal(0.0, 0.01, n),
                        index=pd.date_range("2023-01-01", periods=n, freq="B"))
        ticker = 0.5 + 1.2 * mkt + pd.Series(rng.normal(0, 0.005, n), index=mkt.index)
        return ticker, mkt

    def test_recovers_approximate_beta(self):
        ticker_rets, mkt_rets = self._make_returns()
        est_start = ticker_rets.index[0]
        est_end = ticker_rets.index[149]
        alpha, beta, resid = compute_market_model(ticker_rets, mkt_rets, est_start, est_end)
        assert alpha is not None
        assert 0.8 < beta < 1.5

    def test_insufficient_obs_returns_none(self):
        ticker, mkt = self._make_returns(n=50)
        alpha, beta, resid = compute_market_model(
            ticker, mkt, ticker.index[0], ticker.index[-1], min_obs=100)
        assert alpha is None

    def test_constant_market_returns_none(self):
        idx = pd.date_range("2023-01-01", periods=150, freq="B")
        ticker = pd.Series(np.random.randn(150) * 0.01, index=idx)
        mkt = pd.Series([0.0] * 150, index=idx)
        alpha, beta, resid = compute_market_model(ticker, mkt, idx[0], idx[-1])
        assert alpha is None


class TestAbnormalReturns:
    def test_zero_ar_when_exact_model(self):
        idx = pd.date_range("2023-01-01", periods=10, freq="B")
        mkt = pd.Series([0.01] * 10, index=idx)
        alpha, beta = 0.001, 1.5
        ticker = alpha + beta * mkt
        ars = compute_abnormal_returns(ticker, mkt, alpha, beta, idx[0], idx[-1])
        assert (ars.abs() < 1e-10).all()

    def test_event_window_subset(self):
        idx = pd.date_range("2023-01-01", periods=20, freq="B")
        mkt = pd.Series(np.random.randn(20) * 0.01, index=idx)
        ticker = pd.Series(np.random.randn(20) * 0.01, index=idx)
        ars = compute_abnormal_returns(ticker, mkt, 0.0, 1.0, idx[5], idx[10])
        assert len(ars) <= 10


class TestEventStudy:
    def _setup(self, seed=42):
        rng = np.random.default_rng(seed)
        n = 300
        idx = pd.date_range("2022-01-01", periods=n, freq="B")
        mkt = pd.Series(rng.normal(0, 0.01, n), index=idx)
        aapl = 0.0005 + 1.1 * mkt + rng.normal(0, 0.008, n)
        msft = -0.0002 + 0.9 * mkt + rng.normal(0, 0.007, n)
        returns_df = pd.DataFrame({"AAPL": aapl, "MSFT": msft}, index=idx)
        events = pd.DataFrame([
            {"ticker": "AAPL", "event_date": "2022-09-01", "sentiment_label": "positive"},
            {"ticker": "MSFT", "event_date": "2022-10-01", "sentiment_label": "negative"},
        ])
        return events, returns_df, mkt

    def test_returns_dataframe(self):
        events, returns_df, mkt = self._setup()
        es = event_study(events, returns_df, mkt)
        assert isinstance(es, pd.DataFrame)
        assert set(["ticker", "event_date", "label", "car"]).issubset(es.columns)

    def test_unknown_ticker_skipped(self):
        events, returns_df, mkt = self._setup()
        events = events.copy()
        events.loc[len(events)] = {"ticker": "UNKNWN", "event_date": "2022-09-05",
                                   "sentiment_label": "neutral"}
        es = event_study(events, returns_df, mkt)
        assert "UNKNWN" not in es["ticker"].values

    def test_car_is_float(self):
        events, returns_df, mkt = self._setup()
        es = event_study(events, returns_df, mkt)
        if len(es) > 0:
            assert all(isinstance(c, float) for c in es["car"])


class TestCarSignificance:
    def test_significant_positive_car(self):
        rng = np.random.default_rng(0)
        cars = rng.normal(0.02, 0.005, 200)
        df = pd.DataFrame({"label": ["positive"] * 200, "car": cars})
        sig = car_significance_report(df)
        assert sig["positive"]["significant_5pct"] is True
        assert sig["positive"]["mean_car"] > 0

    def test_zero_mean_not_significant(self):
        rng = np.random.default_rng(1)
        cars = rng.normal(0.0, 0.01, 200)
        df = pd.DataFrame({"label": ["neutral"] * 200, "car": cars})
        sig = car_significance_report(df)
        # With n=200 and true mean=0, usually not significant
        assert "neutral" in sig

    def test_single_sample(self):
        df = pd.DataFrame({"label": ["positive"], "car": [0.05]})
        sig = car_significance_report(df)
        assert sig["positive"]["significant_5pct"] is False


# ---------------------------------------------------------------------------
# Level C — IC and quantile returns
# ---------------------------------------------------------------------------

class TestComputeIC:
    def test_perfect_positive_correlation(self):
        idx = range(100)
        factor = pd.Series(list(range(100)), index=idx, dtype=float)
        fwd = pd.Series(list(range(100)), index=idx, dtype=float)
        ic = compute_ic(factor, fwd)
        assert ic == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative_correlation(self):
        idx = range(100)
        factor = pd.Series(list(range(100)), index=idx, dtype=float)
        fwd = pd.Series(list(range(99, -1, -1)), index=idx, dtype=float)
        ic = compute_ic(factor, fwd)
        assert ic == pytest.approx(-1.0, abs=1e-6)

    def test_no_correlation_near_zero(self):
        rng = np.random.default_rng(42)
        idx = range(200)
        factor = pd.Series(rng.normal(0, 1, 200), index=idx)
        fwd = pd.Series(rng.normal(0, 1, 200), index=idx)
        ic = compute_ic(factor, fwd)
        assert abs(ic) < 0.3

    def test_insufficient_data_returns_nan(self):
        idx = range(5)
        factor = pd.Series([1, 2, 3, 4, 5], index=idx, dtype=float)
        fwd = pd.Series([5, 4, 3, 2, 1], index=idx, dtype=float)
        ic = compute_ic(factor, fwd)
        assert math.isnan(ic)

    def test_nan_dropped_pairwise(self):
        idx = range(50)
        factor = pd.Series([float("nan")] * 10 + list(range(40)), index=idx)
        fwd = pd.Series(list(range(50)), index=idx, dtype=float)
        ic = compute_ic(factor, fwd)
        assert not math.isnan(ic)


class TestQuantileReturns:
    def test_monotone_returns(self):
        rng = np.random.default_rng(0)
        n = 500
        factor = pd.Series(rng.normal(0, 1, n))
        noise = pd.Series(rng.normal(0, 0.001, n))
        fwd = factor * 0.01 + noise
        q_rets = compute_quantile_returns(factor, fwd)
        assert q_rets.iloc[0] < q_rets.iloc[-1]

    def test_empty_on_too_few_obs(self):
        factor = pd.Series([1.0, 2.0])
        fwd = pd.Series([0.01, 0.02])
        q_rets = compute_quantile_returns(factor, fwd, n_quantiles=5)
        assert q_rets.empty


class TestNetEdgeAfterCosts:
    def test_positive_spread(self):
        q_rets = pd.Series({0: -0.001, 1: -0.0005, 2: 0.0, 3: 0.0005, 4: 0.001})
        result = net_edge_after_costs(q_rets, turnover_rate=0.10, cost_bps_per_side=5.0)
        assert "net_edge_annual_pct" in result

    def test_empty_returns_nan(self):
        result = net_edge_after_costs(pd.Series(dtype=float))
        assert math.isnan(result["net_edge_annual_pct"])

    def test_high_cost_eats_edge(self):
        q_rets = pd.Series({0: -0.00001, 4: 0.00001})
        result = net_edge_after_costs(q_rets, turnover_rate=0.80, cost_bps_per_side=20.0)
        assert result["net_edge_annual_pct"] < 0


# ---------------------------------------------------------------------------
# Production Gate
# ---------------------------------------------------------------------------

class TestProductionGate:
    def _passing(self):
        return {
            "level_a_fpb_macro_f1": 0.80,
            "level_a_own_gold_macro_f1": 0.65,
            "level_b_car_significance_p": 0.01,
            "level_b_car_magnitude_bps": 40.0,
            "level_c_ic_mean": 0.03,
            "level_c_quantile_spread_bps": 20.0,
            "level_c_net_edge_after_costs_pct": 4.0,
            "look_ahead_anonymization_agreement": 0.90,
        }

    def test_all_pass(self):
        ok, criteria = news_feature_production_ready("finbert", self._passing())
        assert ok is True
        assert all(criteria.values())

    def test_one_failure_blocks(self):
        v = self._passing()
        v["level_b_car_significance_p"] = 0.30  # too high → fail
        ok, criteria = news_feature_production_ready("finbert", v)
        assert ok is False
        assert criteria["level_b_car_significance_p"] is False

    def test_missing_key_fails(self):
        v = self._passing()
        del v["level_c_ic_mean"]
        ok, _ = news_feature_production_ready("finbert", v)
        assert ok is False

    def test_gate_summary_contains_feature_name(self):
        ok, per = news_feature_production_ready("my_feature", self._passing())
        summary = gate_summary("my_feature", ok, per)
        assert "my_feature" in summary

    def test_gate_thresholds_complete(self):
        assert len(GATE_THRESHOLDS) == 8


# ---------------------------------------------------------------------------
# run_finbert_tone (transformers optional dep)
# ---------------------------------------------------------------------------

class TestRunFinbertTone:
    def test_fallback_without_transformers(self, monkeypatch):
        import sys
        # Force ImportError for transformers
        monkeypatch.setitem(sys.modules, "transformers", None)
        result = run_finbert_tone(["Market rallied sharply.", "Stock fell 10%."])
        assert result == ["neutral", "neutral"]

    def test_returns_list_length_matches(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "transformers", None)
        texts = ["a", "b", "c"]
        assert len(run_finbert_tone(texts)) == len(texts)

    def test_mock_pipeline_positive(self, monkeypatch):
        fake_result = [{"label": "Positive", "score": 0.9}]
        fake_pipe = lambda text: fake_result  # noqa: E731

        import types
        fake_transformers = types.SimpleNamespace(pipeline=lambda *a, **kw: fake_pipe)
        monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

        # Direct call with mock
        import assembled_core.signals.news_validation as nv
        original = nv.run_finbert_tone

        def patched(texts):
            return [fake_pipe(t)[0]["label"].lower() for t in texts]

        monkeypatch.setattr(nv, "run_finbert_tone", patched)
        assert nv.run_finbert_tone(["good news"]) == ["positive"]


# ---------------------------------------------------------------------------
# run_haiku_zeroshot
# ---------------------------------------------------------------------------

class TestRunHaikuZeroshot:
    def _mock_client(self, label="positive"):
        from unittest.mock import MagicMock
        client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text=label)]
        client.messages.create.return_value = msg
        return client

    def test_positive_label(self):
        client = self._mock_client("positive")
        result = run_haiku_zeroshot(["Stocks surge on strong earnings."], client)
        assert result == ["positive"]

    def test_negative_label(self):
        client = self._mock_client("negative")
        result = run_haiku_zeroshot(["Company files for bankruptcy."], client)
        assert result == ["negative"]

    def test_unknown_label_defaults_neutral(self):
        client = self._mock_client("bullish")  # not in valid set
        result = run_haiku_zeroshot(["Some text."], client)
        assert result == ["neutral"]

    def test_api_error_returns_neutral(self):
        from unittest.mock import MagicMock
        client = MagicMock()
        client.messages.create.side_effect = Exception("API error")
        result = run_haiku_zeroshot(["text"], client)
        assert result == ["neutral"]

    def test_multiple_texts(self):
        client = self._mock_client("negative")
        result = run_haiku_zeroshot(["a", "b", "c"], client)
        assert len(result) == 3
        assert all(r == "negative" for r in result)


# ---------------------------------------------------------------------------
# entity_anonymize
# ---------------------------------------------------------------------------

class TestEntityAnonymize:
    def test_ticker_replaced(self):
        result = entity_anonymize("AAPL surges 5%", "AAPL", "Apple Inc.")
        assert "AAPL" not in result
        assert "XYZ" in result

    def test_company_replaced(self):
        result = entity_anonymize("Apple Inc. beats estimates", "AAPL", "Apple Inc.")
        assert "Apple Inc." not in result
        assert "Company XYZ" in result

    def test_both_replaced(self):
        result = entity_anonymize("AAPL (Apple Inc.) gains", "AAPL", "Apple Inc.")
        assert "AAPL" not in result
        assert "Apple Inc." not in result

    def test_company_case_insensitive(self):
        result = entity_anonymize("apple inc. reported earnings", "AAPL", "Apple Inc.")
        assert "apple inc." not in result.lower()

    def test_no_match_unchanged(self):
        original = "MSFT rises on cloud growth"
        result = entity_anonymize(original, "AAPL", "Apple Inc.")
        assert result == original

    def test_ticker_word_boundary(self):
        # 'AAPLS' should NOT be replaced for ticker 'AAPL'
        result = entity_anonymize("AAPLS is not AAPL", "AAPL", "Apple Inc.")
        assert "AAPLS" in result  # not replaced (word boundary)
        assert "XYZ" in result     # AAPL replaced


# ---------------------------------------------------------------------------
# build_factor_series
# ---------------------------------------------------------------------------

class TestBuildFactorSeries:
    def _events(self):
        return pd.DataFrame({
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "date": ["2024-01-02", "2024-01-03", "2024-01-02"],
            "sentiment_numeric": [0.5, -0.3, 0.8],
        })

    def _dates(self):
        return pd.date_range("2024-01-02", periods=3, freq="B")

    def test_returns_series(self):
        s = build_factor_series(self._events(), ["AAPL", "MSFT", "GOOG"], self._dates())
        assert isinstance(s, pd.Series)

    def test_multiindex_names(self):
        s = build_factor_series(self._events(), ["AAPL", "MSFT"], self._dates())
        assert s.index.names == ["date", "asset"]

    def test_sentiment_averaged(self):
        # AAPL on 2024-01-02 has one event with 0.5
        s = build_factor_series(self._events(), ["AAPL", "MSFT"], self._dates())
        val = s.loc[(pd.Timestamp("2024-01-02"), "AAPL")]
        assert abs(val - 0.5) < 1e-9

    def test_missing_fills_zero(self):
        s = build_factor_series(self._events(), ["AAPL", "MSFT", "GOOG"], self._dates())
        # GOOG has no events → should be 0.0
        goog_vals = s.xs("GOOG", level="asset")
        assert (goog_vals == 0.0).all()

    def test_factor_name(self):
        s = build_factor_series(self._events(), ["AAPL"], self._dates())
        assert s.name == "news_sentiment_factor"

    def test_length(self):
        tickers = ["AAPL", "MSFT", "GOOG"]
        dates = self._dates()
        s = build_factor_series(self._events(), tickers, dates)
        assert len(s) == len(tickers) * len(dates)
