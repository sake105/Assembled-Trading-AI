"""Unit tests for Part B ctx wiring helper."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import pandas as pd

from src.assembled_core.paper.intel_context import (
    MIN_SHOCK_SEVERITY,
    TOPIC_TO_SHOCKS,
    _populate_insider_data,
    active_shocks_from_triggers,
    persist_historical_scores,
    populate_ctx_from_artifacts,
)


def test_empty_triggers_returns_empty():
    assert active_shocks_from_triggers([]) == []


def test_below_severity_filtered():
    items = [{"topic_id": "energy_crisis", "severity": 1}]
    assert active_shocks_from_triggers(items) == []


def test_single_high_severity_shock_mapped():
    items = [{"topic_id": "energy_crisis", "severity": 2}]
    result = active_shocks_from_triggers(items)
    assert "oil_supply_risk" in result
    assert "energy_price_spike" in result


def test_unknown_topic_ignored():
    items = [
        {"topic_id": "weather_report", "severity": 3},
        {"topic_id": "energy_crisis", "severity": 2},
    ]
    result = active_shocks_from_triggers(items)
    assert result == sorted({"oil_supply_risk", "energy_price_spike"})


def test_duplicates_deduplicated():
    items = [
        {"topic_id": "geopolitical_conflict", "severity": 3},
        {"topic_id": "market_crash", "severity": 3},
    ]
    result = active_shocks_from_triggers(items)
    # both map to global_risk_off — deduplicated
    assert result.count("global_risk_off") == 1
    assert "defense_demand_surge" in result


def test_missing_severity_skipped():
    items = [{"topic_id": "energy_crisis"}]  # no severity
    assert active_shocks_from_triggers(items) == []


def test_malformed_severity_skipped():
    items = [{"topic_id": "energy_crisis", "severity": "high"}]
    assert active_shocks_from_triggers(items) == []


def test_custom_min_severity():
    items = [{"topic_id": "central_bank", "severity": 1}]
    # default cutoff (=2) would filter
    assert active_shocks_from_triggers(items) == []
    # lowered cutoff passes through
    assert "rate_shock" in active_shocks_from_triggers(items, min_severity=1)


def test_all_curated_topics_map_to_known_shocks():
    # Sanity: every value in TOPIC_TO_SHOCKS must be a SHOCK_BENEFICIARY_MAP key
    import pytest

    pytest.importorskip("src.assembled_core.signals.intel_signal_adapter")
    from src.assembled_core.signals.intel_signal_adapter import SHOCK_BENEFICIARY_MAP

    known = set(SHOCK_BENEFICIARY_MAP.keys())
    for topic, shocks in TOPIC_TO_SHOCKS.items():
        unknown = [s for s in shocks if s not in known]
        assert not unknown, f"{topic} maps to unknown shocks: {unknown}"


def test_populate_ctx_no_artifact(tmp_path: Path):
    ctx = SimpleNamespace()
    populate_ctx_from_artifacts(ctx, tmp_path)
    # No attribute set when artifact missing — trading_cycle uses getattr default
    assert not hasattr(ctx, "intel_active_shocks")


def test_populate_ctx_empty_items(tmp_path: Path):
    artifact = tmp_path / "output" / "intel" / "news" / "triggers_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({"items": []}), encoding="utf-8")

    ctx = SimpleNamespace()
    populate_ctx_from_artifacts(ctx, tmp_path)
    assert not hasattr(ctx, "intel_active_shocks")


def test_populate_ctx_active_shocks(tmp_path: Path):
    artifact = tmp_path / "output" / "intel" / "news" / "triggers_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "items": [
                    {"topic_id": "energy_crisis", "severity": 2},
                    {"topic_id": "weather_report", "severity": 3},  # unknown, dropped
                    {"topic_id": "nuclear_risk", "severity": 3},
                ]
            }
        ),
        encoding="utf-8",
    )

    ctx = SimpleNamespace()
    populate_ctx_from_artifacts(ctx, tmp_path)

    assert hasattr(ctx, "intel_active_shocks")
    shocks = set(ctx.intel_active_shocks)
    assert "oil_supply_risk" in shocks
    assert "energy_price_spike" in shocks
    assert "nuclear_escalation_risk" in shocks


def test_populate_ctx_corrupt_artifact(tmp_path: Path):
    artifact = tmp_path / "output" / "intel" / "news" / "triggers_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{not valid json", encoding="utf-8")

    ctx = SimpleNamespace()
    # Must not raise; warning is logged
    populate_ctx_from_artifacts(ctx, tmp_path)
    assert not hasattr(ctx, "intel_active_shocks")


def test_populate_ctx_explicit_path(tmp_path: Path):
    custom = tmp_path / "custom_triggers.json"
    custom.write_text(
        json.dumps({"items": [{"topic_id": "market_crash", "severity": 2}]}),
        encoding="utf-8",
    )

    ctx = SimpleNamespace()
    populate_ctx_from_artifacts(ctx, tmp_path, news_triggers_path=str(custom))

    assert ctx.intel_active_shocks == ["global_risk_off"]


def test_min_shock_severity_constant():
    assert MIN_SHOCK_SEVERITY == 2


def _make_prices_with_sector_etfs(rows_per_symbol: int = 130) -> pd.DataFrame:
    """Build a long-format price panel with SPY + 4 sector ETFs over enough history."""
    import numpy as np

    symbols = ["SPY", "XLK", "XLF", "XLE", "XLV"]
    dates = pd.date_range("2025-10-01", periods=rows_per_symbol, freq="B", tz="UTC")
    rows = []
    rng = np.random.default_rng(42)
    for sym in symbols:
        base = 100.0 + rng.uniform(-5, 5)
        walk = base + rng.normal(0, 1.0, rows_per_symbol).cumsum()
        for ts, px in zip(dates, walk):
            rows.append({"timestamp": ts, "symbol": sym, "close": float(max(px, 1.0))})
    return pd.DataFrame(rows)


def test_populate_insider_prefers_form4(tmp_path: Path):
    """When both feeds exist, the real EDGAR Form 4 file wins over legacy."""
    pd.DataFrame(
        {"symbol": ["FORM4SYM"], "filing_date": [pd.Timestamp("2026-01-01")]}
    ).to_parquet(tmp_path / "insider_form4.parquet", index=False)
    pd.DataFrame(
        {"symbol": ["LEGACYSYM"], "filing_date": [pd.Timestamp("2026-01-01")]}
    ).to_parquet(tmp_path / "insider_trading.parquet", index=False)
    ctx = SimpleNamespace(as_of=pd.Timestamp("2026-06-01", tz="UTC"))
    _populate_insider_data(ctx, tmp_path)
    assert list(ctx.insider_data["symbol"]) == ["FORM4SYM"]


def test_populate_insider_falls_back_to_legacy(tmp_path: Path):
    """Only the retired legacy file present -> still read (back-compat)."""
    pd.DataFrame(
        {"symbol": ["LEGACYSYM"], "filing_date": [pd.Timestamp("2026-01-01")]}
    ).to_parquet(tmp_path / "insider_trading.parquet", index=False)
    ctx = SimpleNamespace(as_of=pd.Timestamp("2026-06-01", tz="UTC"))
    _populate_insider_data(ctx, tmp_path)
    assert list(ctx.insider_data["symbol"]) == ["LEGACYSYM"]


def test_populate_sector_scores_requires_history():
    """Too little history → silent no-op, attribute not set."""
    ctx = SimpleNamespace(
        prices=_make_prices_with_sector_etfs(rows_per_symbol=20),
        as_of=pd.Timestamp("2025-11-01", tz="UTC"),
    )
    populate_ctx_from_artifacts(ctx, Path("."))
    assert not hasattr(ctx, "sector_rotation_scores")


def test_populate_sector_scores_sets_attribute_when_sufficient_history(tmp_path: Path):
    ctx = SimpleNamespace(
        prices=_make_prices_with_sector_etfs(rows_per_symbol=140),
        as_of=pd.Timestamp("2026-04-01", tz="UTC"),
    )
    populate_ctx_from_artifacts(ctx, tmp_path)
    assert hasattr(ctx, "sector_rotation_scores")
    scores = ctx.sector_rotation_scores
    score_keys = [k for k in scores.index if k.endswith("_score")]
    assert len(score_keys) >= 3


def test_populate_sector_scores_requires_spy():
    """No SPY in universe → no-op."""
    df = _make_prices_with_sector_etfs(rows_per_symbol=140)
    df = df[df["symbol"] != "SPY"]
    ctx = SimpleNamespace(prices=df, as_of=pd.Timestamp("2026-04-01", tz="UTC"))
    populate_ctx_from_artifacts(ctx, Path("."))
    assert not hasattr(ctx, "sector_rotation_scores")


def test_populate_earnings_calendar_cache_missing(tmp_path: Path):
    ctx = SimpleNamespace()
    populate_ctx_from_artifacts(ctx, tmp_path)
    assert not hasattr(ctx, "earnings_calendar")


def test_populate_earnings_calendar_from_cache(tmp_path: Path):
    cache_dir = tmp_path / "output" / "intel" / "earnings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cal = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "earnings_date": pd.to_datetime(["2026-05-01", "2026-05-03"]),
            "eps_estimate": [2.10, 3.00],
            "eps_actual": [float("nan"), float("nan")],
            "surprise_pct": [float("nan"), float("nan")],
        }
    )
    try:
        cal.to_parquet(cache_dir / "calendar_latest.parquet", index=False)
    except Exception:
        pytest.skip("pyarrow/fastparquet not available")

    ctx = SimpleNamespace()
    populate_ctx_from_artifacts(ctx, tmp_path)
    assert hasattr(ctx, "earnings_calendar")
    assert len(ctx.earnings_calendar) == 2


def test_persist_and_load_historical_scores(tmp_path: Path):
    scores1 = pd.Series([0.1, 0.2, 0.3, 0.4], index=list("ABCD"))
    scores2 = pd.Series([0.5, 0.6, 0.7], index=list("EFG"))

    persist_historical_scores(scores1, tmp_path)
    persist_historical_scores(scores2, tmp_path)

    cache = tmp_path / "output" / "intel" / "signals" / "historical_scores.jsonl"
    assert cache.exists()
    assert cache.read_text(encoding="utf-8").count("\n") >= 2

    ctx = SimpleNamespace()
    populate_ctx_from_artifacts(ctx, tmp_path)
    assert hasattr(ctx, "signal_historical_scores")
    assert len(ctx.signal_historical_scores) >= 2


def test_persist_historical_scores_trims_old_entries(tmp_path: Path, monkeypatch):
    """Old entries past window_days should be dropped on write."""
    cache = tmp_path / "output" / "intel" / "signals" / "historical_scores.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Pre-seed with an ancient entry
    old_ts = (pd.Timestamp.now("UTC") - pd.Timedelta(days=400)).isoformat()
    cache.write_text(
        json.dumps({"ts": old_ts, "mean": 99.0, "n": 5}) + "\n",
        encoding="utf-8",
    )

    persist_historical_scores(pd.Series([1.0, 2.0]), tmp_path, window_days=90)
    body = cache.read_text(encoding="utf-8")
    assert "99.0" not in body  # old entry trimmed
    assert body.strip().count("\n") == 0  # exactly one new entry
