"""Tests for the CLI wiring added to scripts/run_backtest_strategy.py.

WHY THIS FILE EXISTS
--------------------
Both BLOCKERs of the 2026-08-15 data-inventory step lived in exactly this code,
and both got through because only the underlying MODULE was tested, never the
CLI branch:

  1. ``--pit-prices`` sat as an ``elif`` after the ``--data-source`` branch and
     read ``symbols`` / ``start_date`` / ``end_date``, which only exist when the
     FIRST branch runs. Every invocation died with ``UnboundLocalError``; with
     ``--data-source`` set the branch was unreachable. The flag had never been
     executed once.
  2. After that was fixed, ``--symbols`` (argparse ``nargs="+"``, i.e. a LIST)
     was parsed with ``str(args.symbols).split(",")``. That stringified the list
     to ``"['AAPL,BSC,SIVB']"`` and matched exactly one symbol by accident — a
     silently WRONG universe rather than an error.

E-149 states the lesson: "die Kette prueft Diffs, nicht Lauffaehigkeit". These
tests call the real functions with a real (tiny) panel. No network, no
production paths.

MUTATION-VERIFIED (2026-08-16): with the PIT branch disabled
(``if False and getattr(args, "pit_prices", False)``) exactly these six tests
go red — the other seven (guard + leakage builder) do not depend on the branch
and rightly stay green:

  - test_pit_symbol_parsing_accepts_every_shape  (3 parametrisations)
  - test_pit_without_symbols_loads_the_whole_panel
  - test_pit_branch_wins_over_data_source
  - test_pit_frame_carries_the_operational_schema

The last two were BLIND in an earlier version (they passed against the real
operational cache because ``out=None`` and symbol-set-only assertions did not
pin the fixture). Panel-specific assertions (open==close, close==10.0,
len==4) are what made them mutation-sensitive; keep them when editing.

Side note found while writing these: ``load_price_data`` is annotated
``-> pd.DataFrame`` but returns a 3-tuple ``(prices, qa_block_trading,
qa_block_reason)``. Pre-existing, and ``scripts/`` is outside the mypy gate, so
nothing catches it — the first caller to trust the annotation gets a tuple.
Left as a follow-up rather than changed here (shared function, unrelated to this
step's scope).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _panel(tmp_path: Path) -> Path:
    """A stand-in for prices_verdict.parquet: close only, several symbols."""
    rows = []
    for sym in ("AAPL", "BSC", "SIVB"):
        for day in pd.date_range("2008-01-02", periods=4, freq="D", tz="UTC"):
            rows.append({"timestamp": day, "symbol": sym, "close": 10.0, "volume": 5})
    path = tmp_path / "prices_verdict.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _args(out: Path | str | None = None, **over) -> argparse.Namespace:
    """Namespace for load_price_data.

    ``out`` MUST be given a tmp path by every caller: with ``out=None`` the
    function writes qc_report.json into the real ``output/`` tree, and two of
    these tests passed even with the PIT branch deleted entirely — they were
    asserting on a frame that came from somewhere else. A test that cannot go
    red is not a guard, and this file exists precisely because untested CLI
    wiring produced two BLOCKERs.
    """
    base = dict(
        pit_prices=True,
        allow_synthetic_ohlc_risk=True,
        symbols=None,
        symbols_file=None,
        universe=None,
        start_date=None,
        end_date=None,
        data_source=None,
        price_file=None,
        freq="1d",
        output_dir=None,
        # load_price_data reads args.out before the branch chain; listed
        # explicitly rather than using a defaulting Namespace, so a newly
        # required attribute shows up as a failure instead of silently
        # resolving to None.
        out=None if out is None else str(out),
    )
    base.update(over)
    return argparse.Namespace(**base)


# --- BLOCKER 2: symbol parsing ------------------------------------------


@pytest.mark.parametrize(
    "symbols",
    [
        ["AAPL", "BSC"],  # argparse nargs="+" gives a list
        ["AAPL,BSC"],  # a single comma-joined element
        "AAPL,BSC",  # a bare string, for callers that pass one
    ],
)
def test_pit_symbol_parsing_accepts_every_shape(tmp_path, monkeypatch, symbols):
    """The regression: str(list).split(",") matched ONE symbol and looked fine."""
    import src.assembled_core.data.pit_prices as pp

    monkeypatch.setattr(pp, "DEFAULT_PIT_PANEL", _panel(tmp_path))

    from scripts.run_backtest_strategy import load_price_data

    prices, _blocked, _reason = load_price_data(_args(out=tmp_path, symbols=symbols))

    assert set(prices["symbol"].unique()) == {"AAPL", "BSC"}, (
        "both requested symbols must be loaded, not one by accident"
    )


def test_pit_without_symbols_loads_the_whole_panel(tmp_path, monkeypatch):
    import src.assembled_core.data.pit_prices as pp

    monkeypatch.setattr(pp, "DEFAULT_PIT_PANEL", _panel(tmp_path))

    from scripts.run_backtest_strategy import load_price_data

    prices, _blocked, _reason = load_price_data(_args(out=tmp_path))

    assert set(prices["symbol"].unique()) == {"AAPL", "BSC", "SIVB"}


# --- BLOCKER 1: the branch must be reachable at all ----------------------


def test_pit_branch_wins_over_data_source(tmp_path, monkeypatch):
    """--pit-prices is an explicit statement about which universe is allowed.

    As an `elif` after --data-source it was unreachable whenever that flag was
    set; the run then silently used the survivorship-biased panel instead.
    """
    import src.assembled_core.data.pit_prices as pp

    monkeypatch.setattr(pp, "DEFAULT_PIT_PANEL", _panel(tmp_path))

    from scripts.run_backtest_strategy import load_price_data

    prices, _blocked, _reason = load_price_data(
        _args(out=tmp_path, symbols=["AAPL"], data_source="local")
    )

    # Panel-specific, not just "AAPL arrived": the local data source could
    # serve an AAPL too (it is among the 220 operational symbols), and with
    # the PIT branch deleted this test used to pass against that cache. Only
    # the PIT fixture has synthetic OHLC and exactly this close value.
    assert set(prices["symbol"].unique()) == {"AAPL"}
    assert len(prices) == 4, "must be the 4-bar PIT fixture, not the real cache"
    assert (prices["open"] == prices["close"]).all()
    assert (prices["high"] == prices["low"]).all()
    assert prices["close"].eq(10.0).all(), "must come from the PIT panel fixture"


def test_pit_frame_carries_the_operational_schema(tmp_path, monkeypatch):
    import src.assembled_core.data.pit_prices as pp

    monkeypatch.setattr(pp, "DEFAULT_PIT_PANEL", _panel(tmp_path))

    from scripts.run_backtest_strategy import load_price_data

    prices, _blocked, _reason = load_price_data(_args(out=tmp_path))

    for col in ("timestamp", "symbol", "open", "high", "low", "close", "volume"):
        assert col in prices.columns
    # Schema alone is not panel-specific — the operational cache has the same
    # columns. Pin the fixture's synthetic OHLC so this test cannot silently
    # pass against a frame from anywhere else.
    assert (prices["open"] == prices["close"]).all()
    assert prices["close"].eq(10.0).all()


# --- the synthetic-OHLC guard -------------------------------------------


def test_guard_refuses_when_trailing_stops_are_enabled(monkeypatch):
    """ATR measures ~61% of true on synthetic OHLC — stops would be ~39% tight."""
    import logging

    import scripts.run_backtest_strategy as rbs

    monkeypatch.setattr(
        rbs,
        "load_policy",
        lambda *a, **k: {"trailing_stops": {"enabled": True}},
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "src.assembled_core.config.policy_loader",
        type(
            "M",
            (),
            {
                "load_policy": staticmethod(
                    lambda *a, **k: {"trailing_stops": {"enabled": True}}
                )
            },
        )(),
    )

    with pytest.raises(SystemExit, match="REFUSING TO RUN"):
        rbs._pit_guard_after_load(
            _args(allow_synthetic_ohlc_risk=False), logging.getLogger("t")
        )


def test_guard_can_be_overridden_knowingly(monkeypatch):
    import logging

    import scripts.run_backtest_strategy as rbs

    monkeypatch.setitem(
        sys.modules,
        "src.assembled_core.config.policy_loader",
        type(
            "M",
            (),
            {
                "load_policy": staticmethod(
                    lambda *a, **k: {"trailing_stops": {"enabled": True}}
                )
            },
        )(),
    )

    # Must not raise: the operator accepted the bias explicitly.
    rbs._pit_guard_after_load(
        _args(allow_synthetic_ohlc_risk=True), logging.getLogger("t")
    )


# --- the leakage frame builder ------------------------------------------


def test_leakage_frame_is_armed_when_all_columns_exist(tmp_path):
    from src.assembled_core.qa.leakage_frame import build_leakage_frame

    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-02"], utc=True),
            "disclosure_date": pd.to_datetime(["2026-01-01"], utc=True),
            "eps_surprise_pct": [0.1],
        }
    ).to_parquet(tmp_path / "events_earnings.parquet", index=False)

    frame, col, reason = build_leakage_frame(tmp_path)

    assert frame is not None
    assert col == "eps_surprise_pct"
    assert "armed" in reason


@pytest.mark.parametrize("drop", ["timestamp", "disclosure_date", "eps_surprise_pct"])
def test_leakage_frame_stays_none_when_a_column_is_missing(tmp_path, drop):
    """None keeps the gate SKIPPED. Guessing a column name would BLOCK the
    pilot via qa_block.json — failing to check is recoverable, halting on a
    bookkeeping mistake is not."""
    from src.assembled_core.qa.leakage_frame import build_leakage_frame

    data = {
        "timestamp": pd.to_datetime(["2026-01-02"], utc=True),
        "disclosure_date": pd.to_datetime(["2026-01-01"], utc=True),
        "eps_surprise_pct": [0.1],
    }
    del data[drop]
    pd.DataFrame(data).to_parquet(tmp_path / "events_earnings.parquet", index=False)

    frame, col, reason = build_leakage_frame(tmp_path)

    assert frame is None
    assert col is None
    assert "no leakage frame" in reason


def test_leakage_frame_reports_a_missing_file_instead_of_guessing(tmp_path):
    from src.assembled_core.qa.leakage_frame import build_leakage_frame

    frame, col, reason = build_leakage_frame(tmp_path)

    assert frame is None
    assert "does not exist" in reason
