"""Tests for the Step 2.15 order-book-imbalance merge in _tc_features (E-059 #6).

Covers:
- latest_imbalance_by_symbol(): per-symbol latest ImbalanceFeatures with the
  documented l1/vw formulas (reuse of compute_imbalance_features).
- E2E through build_features(): with the policy gate enabled and
  ctx.order_book_snapshots set, the columns ob_imbalance / ob_vw_imbalance
  are merged into the feature panel with numerically correct values.
- Disabled gate / missing snapshots: no ob_* columns appear.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.features.order_book_imbalance import (
    ImbalanceFeatures,
    latest_imbalance_by_symbol,
)
from src.assembled_core.pipeline._tc_features import build_features
from src.assembled_core.pipeline.trading_cycle_shared import TradingContext

pytestmark = pytest.mark.fast


def _snap(
    symbol: str,
    ts: float,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> dict:
    return {
        "symbol": symbol,
        "timestamp": ts,
        "bids": [{"price": p, "size": s} for p, s in bids],
        "asks": [{"price": p, "size": s} for p, s in asks],
    }


def _two_symbol_snapshots() -> list[dict]:
    """AAA has two snapshots (latest must win), BBB has one 2-level book."""
    return [
        # AAA old snapshot: bid-heavy (l1 = +0.5) — must be superseded
        _snap("AAA", 1_700_000_000.0, bids=[(100.0, 300.0)], asks=[(100.1, 100.0)]),
        # AAA latest snapshot: ask-heavy → l1 = (100-300)/400 = -0.5
        _snap("AAA", 1_700_000_060.0, bids=[(100.0, 100.0)], asks=[(100.1, 300.0)]),
        # BBB: l1 = (200-100)/300 = 1/3
        # vw: bid_w = 200/1 + 100/2 = 250, ask_w = 100 → 150/350 = 3/7
        _snap(
            "BBB",
            1_700_000_030.0,
            bids=[(50.0, 200.0), (49.9, 100.0)],
            asks=[(50.1, 100.0)],
        ),
    ]


AAA_L1_EXPECTED = -0.5
AAA_VW_EXPECTED = -0.5  # single level each side → vw == l1
BBB_L1_EXPECTED = (200.0 - 100.0) / 300.0
BBB_VW_EXPECTED = (250.0 - 100.0) / 350.0


class TestLatestImbalanceBySymbol:
    def test_per_symbol_latest_with_correct_fields(self):
        result = latest_imbalance_by_symbol(_two_symbol_snapshots())

        assert set(result.keys()) == {"AAA", "BBB"}
        assert all(isinstance(v, ImbalanceFeatures) for v in result.values())

        # AAA: latest (ask-heavy) snapshot wins over the older bid-heavy one
        assert result["AAA"].timestamp == pytest.approx(1_700_000_060.0)
        assert result["AAA"].l1_imbalance == pytest.approx(AAA_L1_EXPECTED)
        assert result["AAA"].vw_imbalance == pytest.approx(AAA_VW_EXPECTED)

        assert result["BBB"].l1_imbalance == pytest.approx(BBB_L1_EXPECTED)
        assert result["BBB"].vw_imbalance == pytest.approx(BBB_VW_EXPECTED)

    def test_equal_timestamp_later_entry_wins(self):
        snaps = [
            _snap("AAA", 1.0, bids=[(100.0, 300.0)], asks=[(100.1, 100.0)]),
            _snap("AAA", 1.0, bids=[(100.0, 100.0)], asks=[(100.1, 300.0)]),
        ]
        result = latest_imbalance_by_symbol(snaps)
        assert result["AAA"].l1_imbalance == pytest.approx(-0.5)

    def test_empty_input(self):
        assert latest_imbalance_by_symbol([]) == {}


def _make_ctx(
    policy: dict, snapshots: list[dict] | None
) -> tuple[pd.DataFrame, TradingContext]:
    """Backtest ctx with a precomputed panel (2 symbols x 3 days).

    Uses the precomputed-features path of build_features so no core feature
    engine runs; only the policy-gated enrichment steps execute.
    """
    ts = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    rows = []
    for sym, base in (("AAA", 100.0), ("BBB", 50.0)):
        for i, t in enumerate(ts):
            rows.append({"timestamp": t, "symbol": sym, "close": base + i})
    panel = pd.DataFrame(rows)

    ctx = TradingContext(
        prices=panel,
        as_of=ts[-1],
        mode="backtest",
        precomputed_prices_with_features=panel,
        use_factor_store=False,
    )
    # policy cache prevents load_policy() fallback; dynamic attrs as in prod
    ctx._policy_cache = policy
    if snapshots is not None:
        ctx.order_book_snapshots = snapshots
    return panel, ctx


class TestBuildFeaturesObImbalanceE2E:
    def test_enabled_merges_ob_columns(self, tmp_path, monkeypatch):
        # chdir: keeps default-on file-based steps (macro/news parquet) inert
        monkeypatch.chdir(tmp_path)
        policy = {"features": {"order_book_imbalance": {"enabled": True}}}
        panel, ctx = _make_ctx(policy, _two_symbol_snapshots())

        pwf, _ = build_features(panel, ctx)

        assert "ob_imbalance" in pwf.columns
        assert "ob_vw_imbalance" in pwf.columns

        aaa = pwf[pwf["symbol"] == "AAA"]
        bbb = pwf[pwf["symbol"] == "BBB"]
        assert not aaa.empty and not bbb.empty
        assert aaa["ob_imbalance"].unique() == pytest.approx([AAA_L1_EXPECTED])
        assert aaa["ob_vw_imbalance"].unique() == pytest.approx([AAA_VW_EXPECTED])
        assert bbb["ob_imbalance"].unique() == pytest.approx([BBB_L1_EXPECTED])
        assert bbb["ob_vw_imbalance"].unique() == pytest.approx([BBB_VW_EXPECTED])
        # snapshot mode: one row per symbol; merge must not duplicate rows
        assert len(pwf) == 2

    def test_disabled_no_ob_columns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        policy = {"features": {"order_book_imbalance": {"enabled": False}}}
        panel, ctx = _make_ctx(policy, _two_symbol_snapshots())

        pwf, _ = build_features(panel, ctx)

        assert "ob_imbalance" not in pwf.columns
        assert "ob_vw_imbalance" not in pwf.columns

    def test_enabled_without_snapshots_no_ob_columns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        policy = {"features": {"order_book_imbalance": {"enabled": True}}}
        panel, ctx = _make_ctx(policy, snapshots=None)

        pwf, _ = build_features(panel, ctx)

        assert "ob_imbalance" not in pwf.columns
        assert "ob_vw_imbalance" not in pwf.columns

    def test_multi_date_panel_skips_merge_pit_guard(
        self, tmp_path, monkeypatch, caplog
    ):
        """PIT guard (review H-3): on a multi-date panel the latest-snapshot
        imbalance must NOT be broadcast into historical rows — the merge is
        skipped with a loud warning instead."""
        import logging

        monkeypatch.chdir(tmp_path)
        policy = {"features": {"order_book_imbalance": {"enabled": True}}}
        panel, ctx = _make_ctx(policy, _two_symbol_snapshots())
        # Full-panel backtest mode: keep all dates per symbol (no snapshot
        # reduction) -> the guard must refuse the merge.
        ctx.backtest_use_snapshot = False

        with caplog.at_level(logging.WARNING):
            pwf, _ = build_features(panel, ctx)

        assert pwf.groupby("symbol")["timestamp"].nunique().max() > 1
        assert "ob_imbalance" not in pwf.columns
        assert "ob_vw_imbalance" not in pwf.columns
        assert any("PIT guard" in r.message for r in caplog.records)
