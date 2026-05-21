"""F-RX-1 regression guard: per-symbol staleness drop in run_live_paper._load_prices.

The global cache freshness check in _load_prices uses cache_prices["timestamp"].max(),
a single global value. As long as one symbol is fresh, the entire batch passes
the age check — but cross-sectional signal generation then runs on heterogeneous
as-of dates (audit 2026-05-21).

_drop_per_symbol_stale_rows is the per-symbol filter that drops any symbol whose
own latest bar is older than max_age_days, applied to every return path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_live_paper.py"


def _load_module():
    # Ensure repo root is on path so the script's own absolute imports work
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("run_live_paper_mod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_prices(per_sym_latest: dict[str, str]) -> pd.DataFrame:
    """Build a tiny prices frame where each symbol's latest bar = per_sym_latest[sym]."""
    rows = []
    for sym, latest in per_sym_latest.items():
        dates = pd.date_range(end=pd.Timestamp(latest, tz="UTC"), periods=3, freq="D")
        for d in dates:
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "adj_close": 100.5,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_drop_per_symbol_stale_rows_drops_stale_keeps_fresh():
    mod = _load_module()
    # Today is 2026-05-21 in the session; we use ages relative to "now" so
    # construct the fixture with dates close to today.
    today = pd.Timestamp.now("UTC").normalize()
    fresh = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # 1d old
    stale = (today - pd.Timedelta(days=20)).strftime("%Y-%m-%d")  # 20d old
    delisted = (today - pd.Timedelta(days=60)).strftime("%Y-%m-%d")  # 60d old

    prices = _make_prices(
        {
            "AAPL": fresh,
            "MSFT": fresh,
            "KO": stale,
            "EXAS": delisted,
        }
    )
    out = mod._drop_per_symbol_stale_rows(prices, max_age_days=3)
    surviving = set(out["symbol"].unique())
    assert surviving == {"AAPL", "MSFT"}, f"expected AAPL+MSFT, got {surviving}"


def test_drop_per_symbol_stale_rows_empty_panel_is_safe():
    mod = _load_module()
    empty = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    out = mod._drop_per_symbol_stale_rows(empty)
    assert out.empty


def test_drop_per_symbol_stale_rows_returns_unchanged_when_all_fresh():
    mod = _load_module()
    today = pd.Timestamp.now("UTC").normalize()
    fresh = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prices = _make_prices({"AAPL": fresh, "MSFT": fresh})
    out = mod._drop_per_symbol_stale_rows(prices, max_age_days=3)
    assert set(out["symbol"].unique()) == {"AAPL", "MSFT"}
    assert len(out) == len(prices)


def test_drop_per_symbol_stale_rows_respects_max_age_days_param():
    """Setting max_age_days=10 keeps a 5d-stale symbol that would otherwise drop."""
    mod = _load_module()
    today = pd.Timestamp.now("UTC").normalize()
    five_d = (today - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    prices = _make_prices({"AAPL": five_d})

    out_default = mod._drop_per_symbol_stale_rows(prices, max_age_days=3)
    assert out_default.empty, "5d age with max=3 should drop AAPL"

    out_loose = mod._drop_per_symbol_stale_rows(prices, max_age_days=10)
    assert "AAPL" in set(out_loose["symbol"].unique())


def test_drop_per_symbol_stale_rows_handles_tz_naive_timestamps():
    """Production cache stores tz-naive timestamps; helper must localize."""
    mod = _load_module()
    today = pd.Timestamp.now("UTC").normalize()
    fresh = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    stale = (today - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    prices = _make_prices({"AAPL": fresh, "KO": stale})
    # Strip tz to simulate cache format
    prices = prices.assign(timestamp=prices["timestamp"].dt.tz_convert(None))
    out = mod._drop_per_symbol_stale_rows(prices, max_age_days=3)
    assert set(out["symbol"].unique()) == {"AAPL"}
