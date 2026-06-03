"""Batch 9 — signal/exit honesty fixes (discriminating tests).

Three findings, each test fails pre-fix and passes post-fix:

  F1 (signals/sector_rotation.py):
      generate_sector_rotation_signals() no longer fabricates
      pd.Timestamp.now() as the SectorSignals.date. Precedence is
      as_of → Series.name (non-None) → pd.NaT (+ one-shot WARNING).

  F2 (events/news_alpha/signal_generator.py):
      a batch whose topics match NO routing topic_id yields zero signals AND
      emits exactly ONE dormancy WARNING (one-shot); a matching topic produces
      a signal and emits NO dormancy WARNING.

  F3 (events/news_alpha/exit_rules.py):
      the documented reversal exit (trigger 4) is implemented: a fresh trigger
      resolving to the SAME theme but OPPOSITE direction closes an open
      position; same-direction / different-topic / None / [] are inert.
"""

from __future__ import annotations

import logging
import sys as _sys

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Module handles
#
# ``src.assembled_core.signals`` re-exports a function ``sector_rotation_signal``
# on the package, which can shadow the same-named submodule attribute on a plain
# ``import ... as`` binding. Pull the real module object from sys.modules (mirror
# of tests/test_observability_hardening.py).
# ---------------------------------------------------------------------------

import src.assembled_core.signals.sector_rotation  # noqa: F401  (registers module)

sector_rotation = _sys.modules["src.assembled_core.signals.sector_rotation"]

import src.assembled_core.events.news_alpha.signal_generator as signal_generator

from src.assembled_core.events.news_alpha.exit_rules import check_exits
from src.assembled_core.events.news_alpha.models import NewsAlphaSignal
from src.assembled_core.signals.sector_rotation import (
    generate_sector_rotation_signals,
)

SR_LOGGER = "src.assembled_core.signals.sector_rotation"
SG_LOGGER = "src.assembled_core.events.news_alpha.signal_generator"


@pytest.fixture(autouse=True)
def _reset_one_shot_state():
    """Reset module-global one-shot guards before and after each test."""
    sector_rotation._SECTOR_DATE_UNKNOWN_WARNED = False
    signal_generator._DORMANCY_WARNED = False
    yield
    sector_rotation._SECTOR_DATE_UNKNOWN_WARNED = False
    signal_generator._DORMANCY_WARNED = False


def _scores_dict() -> dict:
    """A valid score dict with at least one finite ETF score."""
    return {
        f"{etf}_score": 0.01 * i for i, etf in enumerate(sector_rotation.SECTOR_ETFS)
    }


# ---------------------------------------------------------------------------
# F1 — PIT date stamp
# ---------------------------------------------------------------------------


def test_f1_dict_input_does_not_stamp_now(caplog) -> None:
    """dict input → date is NaT (NOT a ~now timestamp) + one-shot WARNING."""
    before = pd.Timestamp.now()
    with caplog.at_level(logging.WARNING, logger=SR_LOGGER):
        sig = generate_sector_rotation_signals(_scores_dict())
    after = pd.Timestamp.now()

    assert pd.isna(sig.date), f"expected NaT, got {sig.date!r}"
    # Discriminating: pre-fix this would be a now() timestamp inside [before, after].
    assert not (isinstance(sig.date, pd.Timestamp) and before <= sig.date <= after), (
        "date must not be a fabricated now() timestamp"
    )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "look-ahead" in warnings[0].getMessage().lower()


def test_f1_explicit_as_of_wins() -> None:
    """as_of overrides everything, including a Series.name."""
    as_of = pd.Timestamp("2021-03-04")
    row = pd.Series(_scores_dict(), name=pd.Timestamp("2019-01-01"))
    sig = generate_sector_rotation_signals(row, as_of=as_of)
    assert sig.date == as_of


def test_f1_series_name_preserved(caplog) -> None:
    """A Series whose .name is a date → that date is used, no WARNING."""
    stamp = pd.Timestamp("2020-06-15")
    row = pd.Series(_scores_dict(), name=stamp)
    with caplog.at_level(logging.WARNING, logger=SR_LOGGER):
        sig = generate_sector_rotation_signals(row)
    assert sig.date == stamp
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_f1_series_name_none_warns_and_nat(caplog) -> None:
    """Series with name=None → NaT + one-shot WARNING (no now())."""
    before = pd.Timestamp.now()
    row = pd.Series(_scores_dict(), name=None)
    with caplog.at_level(logging.WARNING, logger=SR_LOGGER):
        sig = generate_sector_rotation_signals(row)
    after = pd.Timestamp.now()

    assert pd.isna(sig.date)
    assert not (isinstance(sig.date, pd.Timestamp) and before <= sig.date <= after)
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_f1_warning_is_one_shot(caplog) -> None:
    """Second dateless call does not re-WARN (one-shot guard)."""
    with caplog.at_level(logging.WARNING, logger=SR_LOGGER):
        generate_sector_rotation_signals(_scores_dict())
        first = len([r for r in caplog.records if r.levelno == logging.WARNING])
        generate_sector_rotation_signals(_scores_dict())
        second = len([r for r in caplog.records if r.levelno == logging.WARNING])
    assert first == 1
    assert second == 1  # unchanged → no re-warn


# ---------------------------------------------------------------------------
# F2 — EOD dormancy observability
# ---------------------------------------------------------------------------


def test_f2_all_unmatched_topics_warn_once_and_zero_signals(caplog) -> None:
    items = [
        {"severity": 3, "topic": "totally_unknown_topic_aaa", "source": "x"},
        {"severity": 3, "topic": "totally_unknown_topic_bbb", "source": "y"},
    ]
    with caplog.at_level(logging.WARNING, logger=SG_LOGGER):
        sigs = signal_generator.generate_signals(items)
    assert sigs == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "dormant" in msg.lower()
    assert "totally_unknown_topic_aaa" in msg


def test_f2_dormancy_warning_is_one_shot(caplog) -> None:
    items = [{"severity": 3, "topic": "no_such_topic_zzz", "source": "x"}]
    with caplog.at_level(logging.WARNING, logger=SG_LOGGER):
        signal_generator.generate_signals(items)
        first = len([r for r in caplog.records if r.levelno == logging.WARNING])
        signal_generator.generate_signals(items)
        second = len([r for r in caplog.records if r.levelno == logging.WARNING])
    assert first == 1
    assert second == 1  # one-shot → no re-warn on second call


def test_f2_matching_topic_no_dormancy_warning_and_produces_signal(caplog) -> None:
    items = [
        {
            "severity": 3,
            "topic": "shipping_disruption",
            "source": "reuters",
            "event_id": "ev1",
        }
    ]
    with caplog.at_level(logging.WARNING, logger=SG_LOGGER):
        sigs = signal_generator.generate_signals(items)
    assert len(sigs) > 0
    dormancy = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "dormant" in r.getMessage().lower()
    ]
    assert dormancy == []


# ---------------------------------------------------------------------------
# F3 — reversal exit
# ---------------------------------------------------------------------------


def _open_sig(topic_id: str, symbol: str = "XLF") -> NewsAlphaSignal:
    """Open position that has NOT hit any time/price exit (entry_day=current)."""
    return NewsAlphaSignal(
        event_id="e1",
        topic_id=topic_id,
        trigger_type="macro",
        source="reuters",
        symbol=symbol,
        direction="long",
        raw_weight=0.10,
        severity=3,
        hold_days=5,
        entry_day=0,
    )


def test_f3_opposite_trigger_reverses(caplog) -> None:
    """Open central_bank_hike + fresh cut trigger → reversal exit."""
    sig = _open_sig("central_bank_hike")
    fresh = [
        {"severity": 3, "topic": "central_bank", "source": "Fed announces dovish cut"}
    ]
    with caplog.at_level(
        logging.INFO, logger="src.assembled_core.events.news_alpha.exit_rules"
    ):
        exits = check_exits([sig], current_day=1, new_trigger_items=fresh)
    assert len(exits) == 1
    assert exits[0][0] is sig
    assert exits[0][1].startswith("reversal")


def test_f3_same_direction_trigger_does_not_reverse() -> None:
    """A fresh trigger of the SAME direction must NOT exit."""
    sig = _open_sig("central_bank_hike")
    fresh = [
        {"severity": 3, "topic": "central_bank", "source": "Fed surprise rate hike"}
    ]
    exits = check_exits([sig], current_day=1, new_trigger_items=fresh)
    assert exits == []


def test_f3_none_trigger_items_inert() -> None:
    sig = _open_sig("central_bank_hike")
    assert check_exits([sig], current_day=1, new_trigger_items=None) == []


def test_f3_empty_trigger_items_inert() -> None:
    sig = _open_sig("central_bank_hike")
    assert check_exits([sig], current_day=1, new_trigger_items=[]) == []


def test_f3_different_topic_does_not_reverse() -> None:
    """A fresh trigger for a DIFFERENT theme must not exit the open position."""
    sig = _open_sig("central_bank_hike")
    fresh = [{"severity": 3, "topic": "shipping_disruption", "source": "reuters"}]
    exits = check_exits([sig], current_day=1, new_trigger_items=fresh)
    assert exits == []


def test_f3_reversal_first_match_no_double_append() -> None:
    """Two opposite fresh triggers → exactly one reversal exit for the sig."""
    sig = _open_sig("central_bank_cut", symbol="TLT")
    fresh = [
        {"severity": 3, "topic": "central_bank", "source": "Fed hawkish hike"},
        {"severity": 3, "topic": "central_bank", "source": "ECB rate hike tighten"},
    ]
    exits = check_exits([sig], current_day=1, new_trigger_items=fresh)
    assert len(exits) == 1
    assert exits[0][1].startswith("reversal")
