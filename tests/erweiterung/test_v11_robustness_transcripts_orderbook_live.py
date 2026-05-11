"""Tests für Wave-11: Robustness + Transcripts + OrderBook + Live-Pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.live_pipeline.edgar_stream import (
    EdgarFiling,
    EdgarStreamState,
    filings_to_dataframe,
)
from erweiterung.live_pipeline.filing_signal_mapper import (
    aggregate_filings_to_portfolio,
    map_classification_to_signal,
    signals_to_dataframe,
)
from erweiterung.live_pipeline.material_event_classifier import (
    classify_filing,
    classify_filing_text,
    extract_items,
    filter_high_material_events,
)
from erweiterung.orderbook.iex_deep import (
    DEEPMessage,
    parse_quote,
    replay_messages_to_lob,
)
from erweiterung.orderbook.lob_state import LOBState
from erweiterung.orderbook.microprice import (
    microprice_from_lob,
    microprice_series,
    order_book_imbalance_signal,
)
from erweiterung.orderbook.queue_model import (
    adverse_selection_cost,
    expected_fill_time,
    naive_fill_probability,
    optimal_placement_choice,
)
from erweiterung.robustness.anchored_walk_forward import (
    WalkForwardConfig,
    compare_anchored_vs_rolling,
    walk_forward,
)
from erweiterung.robustness.parameter_sensitivity import (
    best_robust_parameter,
    parameter_sweep,
    smoothness_score,
    stability_score,
)
from erweiterung.robustness.regime_conditional import (
    conditional_sharpe_breakdown,
    regime_expected_duration,
    regime_transition_matrix,
    returns_by_regime,
)
from erweiterung.robustness.robustness_report import robustness_score
from erweiterung.robustness.sub_period import (
    STANDARD_EPOCHS_US_EQUITY,
    consistency_score,
    sub_period_metrics,
    worst_period_sharpe,
)
from erweiterung.transcripts.earnings_call_tone import (
    call_summary,
    classify_speaker_role,
    parse_transcript,
    score_segments,
)
from erweiterung.transcripts.fomc_tone import (
    fomc_change_signal,
    hawkish_dovish_score,
    score_fomc_statements,
)
from erweiterung.transcripts.loughran_mcdonald import (
    lm_count_tokens,
    lm_score_documents,
    lm_sentiment_score,
    tokenize,
)


# ===== ROBUSTNESS =====


def _toy_returns_with_epochs() -> pd.Series:
    """Returns covering all standard epochs."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2005-01-01", "2024-12-31", freq="B", tz="UTC")
    return pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)


def test_sub_period_metrics():
    r = _toy_returns_with_epochs()
    df = sub_period_metrics(r)
    assert "epoch" in df.columns
    assert len(df) == len(STANDARD_EPOCHS_US_EQUITY)


def test_worst_period_sharpe():
    r = _toy_returns_with_epochs()
    worst = worst_period_sharpe(r)
    assert "sharpe" in worst


def test_consistency_score():
    r = _toy_returns_with_epochs()
    s = consistency_score(r)
    assert np.isfinite(s) or pd.isna(s)


def test_returns_by_regime():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 500))
    regime = pd.Series(rng.integers(0, 3, 500))
    df = returns_by_regime(r, regime)
    assert "regime" in df.columns
    assert (df["frequency"] >= 0).all()


def test_regime_transition_matrix():
    regime = pd.Series([0, 0, 1, 1, 1, 2, 2, 0, 0, 1])
    M = regime_transition_matrix(regime)
    assert M.shape[0] == M.shape[1]
    # rows must sum ≈ 1
    assert (M.sum(axis=1) - 1.0).abs().max() < 1e-9


def test_regime_expected_duration():
    regime = pd.Series([0, 0, 0, 1, 1, 2, 2, 2, 0, 0])
    durs = regime_expected_duration(regime)
    assert (durs >= 1).all()


def test_conditional_sharpe_breakdown():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    regime = pd.Series(rng.integers(0, 2, 500))
    summary = conditional_sharpe_breakdown(r, regime)
    assert "best_regime" in summary


def test_parameter_sweep():
    rng = np.random.default_rng(0)

    def fake_backtest(window: int) -> pd.Series:
        return pd.Series(rng.normal(0.0005, 0.01 + 0.0001 * window, 200))

    df = parameter_sweep(fake_backtest, [5, 10, 20, 30, 60])
    assert "sharpe" in df.columns


def test_stability_score():
    df = pd.DataFrame({"sharpe": [1.0, 1.1, 0.95, 1.05]})
    stab = stability_score(df)
    assert np.isfinite(stab)


def test_smoothness_score():
    df = pd.DataFrame({"sharpe": [1.0, 1.1, 0.95, 1.05, 1.0]})
    smooth = smoothness_score(df)
    assert np.isfinite(smooth)


def test_best_robust_parameter():
    df = pd.DataFrame(
        {"param": [5, 10, 20, 30, 60], "sharpe": [0.5, 1.0, 1.2, 1.1, 0.8]}
    )
    best = best_robust_parameter(df)
    assert "param" in best
    assert best["param"] in [10, 20, 30]


def test_walk_forward_anchored():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=600, freq="B")

    def strat(train_idx, test_idx):
        return pd.Series(rng.normal(0, 0.01, len(test_idx)), index=test_idx)

    df = walk_forward(
        idx, strat, WalkForwardConfig(min_train_size=200, test_size=20, mode="anchored")
    )
    assert "fold_id" in df.columns


def test_compare_anchored_vs_rolling():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=600, freq="B")

    def strat(train_idx, test_idx):
        return pd.Series(rng.normal(0, 0.01, len(test_idx)), index=test_idx)

    res = compare_anchored_vs_rolling(idx, strat, min_train=200, test_size=20)
    assert "sharpe_anchored" in res


def test_robustness_score():
    r = _toy_returns_with_epochs()
    sc = robustness_score(r)
    assert "composite_score" in sc
    assert 0 <= sc["composite_score"] <= 1


# ===== TRANSCRIPTS =====


def test_lm_tokenize():
    toks = tokenize("Apple's earnings exceeded expectations.")
    assert "apple's" in [t for t in toks]
    assert "earnings" in toks


def test_lm_count_tokens():
    text = "The company exceeded expectations with strong growth despite weakness in some segments."
    counts = lm_count_tokens(text)
    assert counts["positive"] > 0
    assert counts["negative"] > 0


def test_lm_sentiment_score():
    pos = lm_sentiment_score("Outstanding strong performance positive growth")
    neg = lm_sentiment_score("Loss decline weakness bankruptcy fraud")
    assert pos["sentiment"] > 0
    assert neg["sentiment"] < 0


def test_lm_score_documents():
    docs = ["growth strong positive", "decline negative loss", "neutral text here"]
    df = lm_score_documents(docs)
    assert df.iloc[0]["sentiment"] > df.iloc[1]["sentiment"]


def test_hawkish_dovish_score():
    hawkish_txt = (
        "The committee will raise rates to fight inflation. Tightening monetary policy."
    )
    dovish_txt = "We will lower rates and support accommodative policy."
    h = hawkish_dovish_score(hawkish_txt)
    d = hawkish_dovish_score(dovish_txt)
    assert h["hd_score"] > 0
    assert d["hd_score"] < 0


def test_negation_aware():
    # "will not raise" should NOT count as hawkish
    txt = "The committee will not raise rates."
    res = hawkish_dovish_score(txt, negation_aware=True)
    # 'raise' is flipped → dovish
    assert res["dovish_count"] >= 1


def test_score_fomc_statements():
    statements = [
        "We will raise rates to combat inflation.",
        "We are patient and accommodative.",
    ]
    df = score_fomc_statements(statements, dates=["2024-01-01", "2024-02-01"])
    assert df.iloc[0]["hd_score"] > df.iloc[1]["hd_score"]


def test_fomc_change_signal():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "hd_score": [0.2, 0.5, 0.1],
        }
    )
    sig = fomc_change_signal(df)
    assert len(sig) == 3


def test_classify_speaker_role():
    assert classify_speaker_role("Operator") == "operator"
    assert classify_speaker_role("Tim Cook (CEO Apple Inc.)") == "executive"
    assert classify_speaker_role("Toni Sacconaghi - Bernstein Analyst") == "analyst"


def test_parse_transcript():
    text = (
        "Operator: Good morning and welcome.\n"
        "Tim Cook (CEO): We had an outstanding quarter with strong growth.\n"
        "Operator: We will now begin the question and answer session.\n"
        "Toni Sacconaghi (Bernstein Analyst): What about competition?\n"
        "Tim Cook (CEO): We believe our products remain competitive."
    )
    segments = parse_transcript(text)
    assert len(segments) >= 2
    qa_seg = [s for s in segments if s.section == "qa"]
    pr_seg = [s for s in segments if s.section == "prepared_remarks"]
    assert len(qa_seg) >= 1
    assert len(pr_seg) >= 1


def test_score_segments():
    from erweiterung.transcripts.earnings_call_tone import TranscriptSegment

    segs = [
        TranscriptSegment(
            "Tim Cook",
            "executive",
            "prepared_remarks",
            "Strong growth positive outlook",
        ),
        TranscriptSegment(
            "Analyst", "analyst", "qa", "Concerned about weakness in china."
        ),
    ]
    df = score_segments(segs)
    assert "sentiment" in df.columns


def test_call_summary():
    from erweiterung.transcripts.earnings_call_tone import TranscriptSegment

    segs = [
        TranscriptSegment(
            "Cook", "executive", "prepared_remarks", "Strong outperform growth"
        ),
        TranscriptSegment("Cook", "executive", "qa", "We are concerned but optimistic"),
        TranscriptSegment("Analyst", "analyst", "qa", "Question about weakness"),
    ]
    s = call_summary(segs)
    assert "executive_qa_sentiment" in s or "executive_prepared_remarks_sentiment" in s


# ===== ORDER BOOK =====


def test_lob_basic():
    lob = LOBState()
    lob.add_order("buy", 100.0, 500)
    lob.add_order("buy", 99.5, 800)
    lob.add_order("sell", 100.5, 300)
    lob.add_order("sell", 101.0, 600)
    bb, ba = lob.best_bid_ask()
    assert bb == 100.0
    assert ba == 100.5
    assert lob.mid_price() == 100.25
    assert lob.spread() == 0.5


def test_lob_imbalance():
    lob = LOBState()
    lob.add_order("buy", 100.0, 1000)
    lob.add_order("sell", 100.5, 300)
    imb = lob.imbalance(1)
    assert imb is not None
    assert imb > 0  # bid volume > ask volume


def test_lob_trade():
    lob = LOBState()
    lob.add_order("sell", 100.5, 500)
    lob.trade("buy", 100.5, 200)
    assert lob.asks[100.5] == 300


def test_microprice():
    lob = LOBState()
    lob.add_order("buy", 100.0, 800)
    lob.add_order("sell", 100.5, 200)
    mp = microprice_from_lob(lob)
    # bid_volume » ask_volume → microprice nearer to ask
    assert mp > 100.25  # above midprice


def test_microprice_series():
    n = 50
    bp = pd.Series(np.full(n, 100.0))
    bv = pd.Series(np.linspace(100, 1000, n))
    ap = pd.Series(np.full(n, 100.5))
    av = pd.Series(np.linspace(1000, 100, n))
    mp = microprice_series(bp, bv, ap, av)
    assert (mp > 100.0).all()
    assert (mp < 100.5).all()


def test_order_book_imbalance_signal():
    n = 100
    bv = pd.Series(np.random.default_rng(0).integers(500, 2000, n))
    av = pd.Series(np.random.default_rng(1).integers(500, 2000, n))
    obi = order_book_imbalance_signal(bv, av, lookback=20)
    valid = obi.dropna()
    assert ((valid >= -1) & (valid <= 1)).all()


def test_queue_naive_fill_prob():
    p = naive_fill_probability(my_position_ahead=200, queue_total=1000)
    assert 0.5 < p < 1.0


def test_expected_fill_time():
    t = expected_fill_time(my_position_ahead=500, arrival_rate=100, cancel_rate=50)
    assert t > 0


def test_optimal_placement():
    choice = optimal_placement_choice(
        spread=0.01, my_size=100, queue_bid=5000, queue_ask=5000
    )
    assert choice in ("aggressive", "patient_join", "patient_inside")


def test_adverse_selection():
    c = adverse_selection_cost(fill_price=100.5, future_mid=100.3, side=1)
    assert c > 0  # bought 100.5, mid moved to 100.3 → cost


def test_iex_deep_parsing():
    msg = DEEPMessage(
        msg_type="Q",
        timestamp_ns=1000,
        symbol="AAPL",
        payload={
            "bid_price": 100.0,
            "bid_size": 500,
            "ask_price": 100.5,
            "ask_size": 300,
        },
    )
    parsed = parse_quote(msg)
    assert parsed["bid_price"] == 100.0
    assert parsed["symbol"] == "AAPL"


def test_iex_deep_replay():
    msgs = [
        DEEPMessage(
            "Q",
            1,
            "AAPL",
            {"bid_price": 100.0, "bid_size": 500, "ask_price": 100.5, "ask_size": 300},
        ),
        DEEPMessage("T", 2, "AAPL", {"price": 100.5, "size": 100, "side": "buy"}),
    ]
    state, trades = replay_messages_to_lob(iter(msgs))
    assert isinstance(state, LOBState)
    assert len(trades) == 1


# ===== LIVE PIPELINE =====


def test_edgar_filings_to_df():
    filings = [
        EdgarFiling(
            accession="0001234-25-000001",
            cik="0000320193",
            company="Apple Inc.",
            form_type="8-K",
            filed_at=pd.Timestamp("2025-01-01", tz="UTC"),
            primary_doc_url="https://example.com/doc.htm",
        )
    ]
    df = filings_to_dataframe(filings)
    assert "accession" in df.columns
    assert len(df) == 1


def test_edgar_stream_state_save_load(tmp_path):
    state = EdgarStreamState(state_file=tmp_path / "stream_state.json")
    state.last_seen_accession = {"acc1", "acc2"}
    state.save()
    new_state = EdgarStreamState(state_file=tmp_path / "stream_state.json")
    new_state.load()
    assert "acc1" in new_state.last_seen_accession


def test_extract_items():
    txt = "Item 2.02 Results of Operations and Item 9.01 Financial Statements"
    items = extract_items(txt)
    assert "2.02" in items
    assert "9.01" in items


def test_classify_filing():
    res = classify_filing(["3.01", "9.01"])
    # 3.01 = delisting = strongly negative
    assert res.expected_direction < 0


def test_classify_filing_positive():
    res = classify_filing(["5.01"])  # change in control = M&A
    assert res.expected_direction > 0


def test_classify_filing_text():
    txt = "Item 2.06 Material Impairment of Assets"
    res = classify_filing_text(txt)
    assert "material_impairment" in res.categories


def test_filter_high_material_events():
    filings_df = pd.DataFrame(
        {
            "accession": ["a", "b", "c"],
            "company": ["X", "Y", "Z"],
            "form_type": ["8-K"] * 3,
        }
    )
    classifications = [
        classify_filing(["3.01"], "a"),  # high
        classify_filing(["9.01"], "b"),  # low
        classify_filing(["5.01"], "c"),  # high
    ]
    out = filter_high_material_events(filings_df, classifications, score_threshold=0.7)
    assert len(out) == 2  # delisting + M&A


def test_filing_signal_mapping():
    classification = classify_filing(["5.01"], "test_acc")  # M&A = +1
    sig = map_classification_to_signal(
        classification, symbol="AAPL", filing_age_hours=1.0, market_session="intraday"
    )
    assert sig is not None
    assert sig.direction == 1
    assert sig.conviction > 0.2


def test_filing_signal_age_decay():
    classification = classify_filing(["5.01"], "test_acc")
    fresh = map_classification_to_signal(classification, "AAPL", filing_age_hours=1.0)
    stale = map_classification_to_signal(classification, "AAPL", filing_age_hours=40.0)
    assert fresh is not None
    assert stale is None or stale.conviction < fresh.conviction


def test_aggregate_filings_to_portfolio():
    from erweiterung.live_pipeline.filing_signal_mapper import FilingSignal

    signals = [
        FilingSignal("AAPL", 1, 0.8, 0.04, "M&A", "a1"),
        FilingSignal("MSFT", -1, 0.5, -0.02, "delisting", "a2"),
        FilingSignal("AAPL", 1, 0.6, 0.03, "earnings", "a3"),
    ]
    w = aggregate_filings_to_portfolio(signals)
    assert "AAPL" in w
    assert w["AAPL"] > 0
    assert w["MSFT"] < 0


def test_signals_to_dataframe():
    from erweiterung.live_pipeline.filing_signal_mapper import FilingSignal

    signals = [FilingSignal("AAPL", 1, 0.5, 0.02, "test", "acc")]
    df = signals_to_dataframe(signals)
    assert "symbol" in df.columns
