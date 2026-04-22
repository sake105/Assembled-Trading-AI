"""Tests for Part B News→Signal bridge."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.assembled_core.signals.news_signal_bridge import (
    _extract_tickers,
    _sentiment_sign,
    _urgency,
    compute_news_deltas,
    load_and_apply_news_signals,
)


def _base_signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-04-22"), "symbol": "AAPL", "direction": "LONG", "score": 0.5},
            {"timestamp": pd.Timestamp("2026-04-22"), "symbol": "MSFT", "direction": "LONG", "score": 0.3},
        ]
    )


def _triggers(sym: str = "AAPL", sev: int = 2, urg: float = 0.8, sent: str = "positive") -> list[dict]:
    return [{"tickers": [sym], "severity": sev, "urgency": urg, "sentiment": sent}]


def test_sentiment_sign_strings():
    assert _sentiment_sign({"sentiment": "positive"}) == 1
    assert _sentiment_sign({"sentiment": "bullish"}) == 1
    assert _sentiment_sign({"sentiment": "negative"}) == -1
    assert _sentiment_sign({"sentiment": "bearish"}) == -1
    assert _sentiment_sign({"sentiment": "neutral"}) == 0
    assert _sentiment_sign({"sentiment": "garbled"}) == 0
    assert _sentiment_sign({}) == 0


def test_sentiment_sign_numeric():
    assert _sentiment_sign({"sentiment": 0.5}) == 1
    assert _sentiment_sign({"sentiment": -0.5}) == -1
    assert _sentiment_sign({"sentiment": 0.05}) == 0


def test_urgency_bounds():
    assert _urgency({"urgency": 0.5}) == 0.5
    assert _urgency({"urgency": 1.5}) == 1.0
    assert _urgency({"urgency": -0.2}) == 0.0
    assert _urgency({"urgency": "bad"}) == 0.5
    assert _urgency({}) == 0.5


def test_extract_tickers_variants():
    assert _extract_tickers({"tickers": ["aapl", "msft"]}) == ["AAPL", "MSFT"]
    assert _extract_tickers({"symbols": [" nvda "]}) == ["NVDA"]
    assert _extract_tickers({"affected_assets": ["SPY"]}) == ["SPY"]
    assert _extract_tickers({}) == []


def test_compute_news_deltas_positive():
    deltas = compute_news_deltas(_triggers())
    # sev=2, urg=0.8, sign=+1 → delta=1.6
    assert deltas["AAPL"] == pytest.approx(1.6)


def test_compute_news_deltas_blocks_short_when_disallowed():
    triggers = _triggers(sent="negative")
    deltas = compute_news_deltas(triggers, allow_short=False)
    assert deltas == {}


def test_compute_news_deltas_neutral_gets_nudge():
    triggers = _triggers(sent="neutral")
    deltas = compute_news_deltas(triggers)
    # sev=2, urg=0.8, sign=0 → fallback 0.1 * 2 * 0.8 = 0.16
    assert deltas["AAPL"] == pytest.approx(0.16)


def test_compute_news_deltas_respects_min_severity():
    triggers = _triggers(sev=1)
    assert compute_news_deltas(triggers, min_severity=2) == {}


def test_load_and_apply_disabled(tmp_path):
    signals = _base_signals()
    policy = {"intel": {"news_signal_bridge": {"enabled": False}}}
    out, meta = load_and_apply_news_signals(signals, root=tmp_path, policy=policy)
    assert meta["enabled"] is False
    assert out.equals(signals)


def test_load_and_apply_missing_artifact(tmp_path):
    signals = _base_signals()
    policy = {"intel": {"news_signal_bridge": {"enabled": True, "weight": 0.1}}}
    out, meta = load_and_apply_news_signals(signals, root=tmp_path, policy=policy)
    assert meta["applied"] == 0
    assert meta["added"] == 0
    assert len(out) == len(signals)


def test_load_and_apply_boosts_existing(tmp_path):
    art = tmp_path / "output" / "intel" / "news" / "triggers_latest.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps({"triggers": _triggers()}), encoding="utf-8")

    signals = _base_signals()
    policy = {"intel": {"news_signal_bridge": {"enabled": True, "weight": 0.1}}}
    out, meta = load_and_apply_news_signals(signals, root=tmp_path, policy=policy)
    aapl_score = float(out.loc[out["symbol"] == "AAPL", "score"].iloc[0])
    # base 0.5 + 1.6 * 0.1 = 0.66
    assert aapl_score == pytest.approx(0.66)
    assert meta["applied"] == 1


def test_load_and_apply_adds_new_symbol(tmp_path):
    art = tmp_path / "output" / "intel" / "news" / "triggers_latest.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps({"triggers": _triggers(sym="NVDA")}), encoding="utf-8")

    signals = _base_signals()
    policy = {"intel": {"news_signal_bridge": {"enabled": True, "weight": 0.1}}}
    out, meta = load_and_apply_news_signals(signals, root=tmp_path, policy=policy)
    assert "NVDA" in out["symbol"].values
    assert meta["added"] == 1


def test_load_and_apply_corrupt_json(tmp_path):
    art = tmp_path / "output" / "intel" / "news" / "triggers_latest.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text("not json", encoding="utf-8")

    signals = _base_signals()
    policy = {"intel": {"news_signal_bridge": {"enabled": True}}}
    out, meta = load_and_apply_news_signals(signals, root=tmp_path, policy=policy)
    assert meta["applied"] == 0
    assert len(out) == len(signals)


def test_load_and_apply_allow_short_false_blocks_short_adds(tmp_path):
    art = tmp_path / "output" / "intel" / "news" / "triggers_latest.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(
        json.dumps({"triggers": _triggers(sym="NVDA", sent="negative")}),
        encoding="utf-8",
    )

    signals = _base_signals()
    policy = {"intel": {"news_signal_bridge": {"enabled": True, "allow_short": False}}}
    out, meta = load_and_apply_news_signals(signals, root=tmp_path, policy=policy)
    assert "NVDA" not in out["symbol"].values
    assert meta["added"] == 0
